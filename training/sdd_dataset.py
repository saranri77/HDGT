"""
SDD Dataset Loader - HDGT Style (Trajectory Prediction)

Samples follow the trajectories already cut in the pickles; incomplete histories and futures
are handled as in the original HDGT (preprocess/preprocess_waymo.py):
- Each metaId is one agent trajectory of 20 timesteps: 8 observed, 12 predicted
- One sample per distinct trajectory start frame. Its window is the 20 timesteps of the
  trajectories starting there, on their own SDD frames (start + 12*k)
- Targets: every trajectory, in the window that starts with it, so each is predicted exactly
  once. Only parked cars/buses/carts are not predicted (see PARKED_TYPES)
- Neighbours: every other agent with at least one sample during the observation, except
  parked cars/buses/carts. Observed steps it has no sample for are filled as HDGT does (see
  fill_observation)
- Future steps a target has no sample for are masked out of the loss and metrics (label_mask)
- Multiple agents interact in the same scene simultaneously
- Heterogeneous graphs with 6 agent types (Pedestrian, Biker, Car, Bus, Cart, Skater)
- No map features (the drawn obstacle shapes are only used to find parked vehicles)

Expected Data Format (in the sdd folder given by --data_folder; by default ../sdd, the parent
directory of these scripts):
- sdd/train/: 52 pickle files. The 8 videos in VAL_SCENES are held out as the validation
  set and the remaining 44 are used for training (see obtain_dataset)
- sdd/test/: 8 pickle files, only evaluated by test_sdd.py
- sdd/shapes/: one pickle of shapes per video, drawn with draw_shapes.py (the same files
  sdd_dataset_goal.py uses); only the obstacle shapes are read here
- Each track pickle contains a pandas DataFrame with columns:
  ['trackId', 'xmin', 'ymin', 'xmax', 'ymax', 'frame', 'label', 'x', 'y', 'sceneId', 'metaId']
- trackId identifies an agent; metaId identifies one 20-timestep trajectory of that agent
- Trajectories are sampled at 2.5 FPS (every 12 frames at 30 FPS = 0.4 seconds), each from
  its own first frame, so different agents can be sampled on different frames
"""

import os
import random
import torch
from torch.utils.data import Dataset, DataLoader, BatchSampler
import numpy as np
import pandas as pd
from functools import partial
import dgl


SCALE_FACTOR = 50.0
FRAME_STEP = 12  # Raw frames between two timesteps (30 FPS video -> 2.5 FPS trajectories)

# Videos held out of sdd/train/ to form the validation set (one video per SDD location).
# They are removed from the training files, so train and val never share a video.
# train.py selects best_min_ade.pt / best_min_fde.pt on this split; sdd/test/ is left untouched for test_sdd.py.
VAL_SCENES = [
    'bookstore_video4',
    'coupa_video1',
    'deathCircle_video2',
    'gates_video1',
    'hyang_video1',
    'little_video2',
    'nexus_video9',
    'quad_video0',
]


# Parked vehicles: the same rule in every version (keep them in sync). A car, bus or cart is
# parked when EVERY position of its trackId in the video lies inside the drawn obstacle
# shapes. It is an obstacle, not an agent: neither a prediction target nor anyone's
# neighbour. A vehicle outside the obstacles at even one position is an agent, so a moving
# one whose annotation strays into an obstacle (a camera jump, a misplaced box) is still
# predicted. A parked car split over several trackIds is caught track by track. This loader
# has no map, so a parked vehicle's rows are simply left out of the video.
PARKED_TYPES = (3, 4, 5)   # Car, Bus, Cart


def load_obstacle_shapes(shapes_folder, scene_id, scale_factor=SCALE_FACTOR):
    """Obstacle shapes of one video as (S, 6) rows [cx, cy, half_w, half_h, theta, is_circle], SCALED"""
    path = os.path.join(shapes_folder, scene_id + '.pickle')
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"No shapes for {scene_id}: expected {path}. Parked vehicles are found with the "
            f"obstacle shapes drawn by draw_shapes.py (the same files sdd_dataset_goal.py uses).")
    shapes = pd.read_pickle(path)
    shapes = shapes[shapes['label'] == 'obstacle']
    is_circle = (shapes['shape'].values == 'circle')
    radius = shapes['radius'].values.astype(np.float64)
    half_w = np.where(is_circle, radius, shapes['half_w'].values.astype(np.float64))
    half_h = np.where(is_circle, radius, shapes['half_h'].values.astype(np.float64))
    theta = np.where(is_circle, 0.0, shapes['theta'].values.astype(np.float64))
    return np.stack([shapes['cx'].values / scale_factor, shapes['cy'].values / scale_factor,
                     half_w / scale_factor, half_h / scale_factor, theta,
                     is_circle.astype(np.float64)], axis=-1).reshape(-1, 6)


def shape_sdf_np(points, geom):
    """Signed distance from points (..., 2) to shapes (S, 6), negative inside -> (..., S)"""
    p = np.asarray(points, dtype=np.float64)[..., np.newaxis, :]
    dx = p[..., 0] - geom[:, 0]
    dy = p[..., 1] - geom[:, 1]
    c, s = np.cos(geom[:, 4]), np.sin(geom[:, 4])
    u = dx * c + dy * s
    v = -dx * s + dy * c
    qx = np.abs(u) - geom[:, 2]
    qy = np.abs(v) - geom[:, 3]
    sd_rect = (np.hypot(np.maximum(qx, 0.0), np.maximum(qy, 0.0))
               + np.minimum(np.maximum(qx, qy), 0.0))
    sd_circle = np.hypot(dx, dy) - geom[:, 2]
    return np.where(geom[:, 5] > 0.5, sd_circle, sd_rect)


def parked_track_ids(df, type_map, obstacles, scale_factor=SCALE_FACTOR):
    """trackIds of one video's parked vehicles: the Car, Bus and Cart tracks whose every row
    lies inside one of the obstacle shapes (S, 6), SCALED"""
    vehicles = df[df['label'].map(type_map).isin(PARKED_TYPES)]
    if obstacles.shape[0] == 0 or len(vehicles) == 0:
        return set()
    inside = (shape_sdf_np(vehicles[['x', 'y']].values / scale_factor, obstacles) < 0).any(axis=1)
    all_inside = pd.Series(inside).groupby(vehicles['trackId'].values).all()
    return set(all_inside.index[all_inside.values])


def fill_observation(frames, positions, obs_frames):
    """Positions at the observed frames from an agent's own samples, the way HDGT fills an
    incomplete history

    HDGT keeps every agent seen at least once in the observation and fills the steps it has
    no state for (preprocess/preprocess_waymo.py, lines 157-190): it interpolates between
    states, extrapolates with the agent's velocity before its first and after its last state,
    and keeps a mask of the filled steps. The same is done here on SDD frames; SDD stores no
    velocities, so the interpolation is linear in position. Steps fall between an agent's
    samples when it is missing for a while, or when it is sampled on other frames than the
    window (each SDD trajectory is sampled from its own first frame).

    Args:
        frames: (n,) sorted frames of the agent's samples inside the observation period, n >= 1
        positions: (n, 2) positions at those frames
        obs_frames: (obs_len,) frames of the observation period
    Returns:
        (obs_len, 2) positions and (obs_len,) mask: 1 where the agent has a sample at exactly
        that frame, 0 where the position was filled. Only samples inside the observation period
        are used, so nothing from the future leaks into the inputs.
    """
    filled = np.stack([np.interp(obs_frames, frames, positions[:, d]) for d in range(2)], axis=-1)
    # np.interp holds the end values outside the samples: replace them with constant-velocity
    # extrapolation (with a single sample the velocity is unknown and stays zero)
    if len(frames) > 1:
        before = obs_frames < frames[0]
        after = obs_frames > frames[-1]
        first_velocity = (positions[1] - positions[0]) / (frames[1] - frames[0])  # per frame
        last_velocity = (positions[-1] - positions[-2]) / (frames[-1] - frames[-2])
        filled[before] = positions[0] + (obs_frames[before] - frames[0])[:, np.newaxis] * first_velocity
        filled[after] = positions[-1] + (obs_frames[after] - frames[-1])[:, np.newaxis] * last_velocity
    mask = np.isin(obs_frames, frames).astype(np.float32)
    return filled, mask


class SDDDataset(Dataset):
    def __init__(self, pickle_files, obs_len=8, pred_len=12, shapes_folder=None):
        """
        Args:
            pickle_files: List of pickle file paths
            obs_len: Number of observation frames (default 8)
            pred_len: Number of prediction frames (default 12)
            shapes_folder: Folder of the per-video shape pickles, used to find parked vehicles.
                Default: "shapes" next to the folder holding each track pickle (sdd/shapes)
        """
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.total_len = obs_len + pred_len  # 20
        self.time_step = 0.4  # SDD: 2.5 FPS (12 frames at 30 FPS)
        self.scale_factor = SCALE_FACTOR
        self.shapes_folder = shapes_folder
        # Agent type mapping (6 types for SDD)
        self.type_map = {
            'Pedestrian': 1,
            'Biker': 2,
            'Car': 3,
            'Bus': 4,
            'Cart': 5,
            'Skater': 6
        }

        # Load statistics, printed at the end of loading
        self.num_trajectories = 0       # trajectories (metaIds) in the pickles
        self.num_targets = 0            # trajectories predicted
        self.num_parked = 0             # trajectories of parked vehicles, not predicted
        self.num_incomplete_future = 0  # targets with masked future steps
        self.num_neighbours = 0         # (window, neighbour) pairs
        self.num_filled_steps = 0       # observed steps of neighbours that were filled

        self.samples = []
        self._load_all_data(pickle_files)

    def _load_all_data(self, pickle_files):
        """Load all pickle files and extract multi-agent samples per time window

        The pickles are already cut into 20-timestep trajectories (metaIds). A window is the
        20 frames of the trajectories that start on its first frame, so every trajectory is a
        prediction target in exactly one sample, and trajectories that start on the same frame
        are predicted together.
        """
        print("Loading SDD dataset...")

        for pkl_idx, pkl_file in enumerate(pickle_files):
            df = pd.read_pickle(pkl_file)
            scene_id = os.path.basename(pkl_file).replace('.pickle', '')
            shapes_folder = self.shapes_folder or os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(pkl_file))), 'shapes')
            obstacles = load_obstacle_shapes(shapes_folder, scene_id, self.scale_factor)
            self.num_trajectories += df['metaId'].nunique()

            # Parked vehicles are obstacles, not agents: their rows go before any window is cut
            parked = df['trackId'].isin(parked_track_ids(df, self.type_map, obstacles, self.scale_factor))
            self.num_parked += df.loc[parked, 'metaId'].nunique()
            df = df[~parked]
            tracks = self._tracks_of_scene(df)

            # All rows sorted by frame, to find the agents seen during an observation period
            order = np.argsort(df['frame'].values, kind='stable')
            row_frames = df['frame'].values[order]
            row_track_ids = df['trackId'].values[order]

            # Start frame and agent of every trajectory (metaId) in the scene
            trajectories = df.groupby('metaId').agg(start_frame=('frame', 'min'), trackId=('trackId', 'first'))

            # One multi-agent sample per distinct start frame
            for start_frame, starting_here in trajectories.groupby('start_frame'):
                window_frames = start_frame + FRAME_STEP * np.arange(self.total_len)
                obs_frames = window_frames[:self.obs_len]

                # Agents with at least one sample during the observation: HDGT keeps exactly these
                first = np.searchsorted(row_frames, obs_frames[0], side='left')
                last = np.searchsorted(row_frames, obs_frames[-1], side='right')
                seen_track_ids = np.unique(row_track_ids[first:last])

                sample = self._build_multi_agent_sample(
                    tracks, seen_track_ids, window_frames, scene_id,
                    set(starting_here['trackId']))

                if sample is not None:
                    self.samples.append(sample)

            print(f"  Loaded {pkl_idx+1}/{len(pickle_files)}: {scene_id} - {len(self.samples)} total samples")

        print(f"Total samples loaded: {len(self.samples)}")
        print(f"Trajectories: {self.num_trajectories}, predicted: {self.num_targets}, "
              f"parked vehicles (not predicted): {self.num_parked}, "
              f"predicted with masked future steps: {self.num_incomplete_future}")
        if self.num_neighbours > 0:
            print(f"Neighbours: {self.num_neighbours} (window, agent) pairs, "
                  f"{100.0 * self.num_filled_steps / (self.num_neighbours * self.obs_len):.1f}% "
                  f"of their observed steps filled")

    def _tracks_of_scene(self, df):
        """Per agent (trackId): its frames, positions and box sizes sorted by frame (SCALED), and its type"""
        tracks = {}
        for track_id, rows in df.sort_values('frame', kind='stable').groupby('trackId', sort=False):
            tracks[track_id] = {
                'frames': rows['frame'].values,
                'positions': rows[['x', 'y']].values / self.scale_factor,
                # [width, length] = [xmax - xmin, ymax - ymin]
                'sizes': np.stack([(rows['xmax'] - rows['xmin']).values,
                                   (rows['ymax'] - rows['ymin']).values], axis=-1) / self.scale_factor,
                'type': self.type_map.get(rows['label'].iloc[0], 1),
            }
        return tracks

    def _agent_features(self, obs_positions, size):
        """Observation features [x, y, vx, vy, yaw, width, length] (obs_len, 7)"""
        obs_velocities = np.diff(obs_positions, axis=0)  # scaled displacement per timestep
        obs_velocities = np.vstack([obs_velocities[0], obs_velocities])  # (8, 2)
        obs_yaw = np.arctan2(obs_velocities[:, 1], obs_velocities[:, 0])  # (8,)

        features = np.zeros((self.obs_len, 7))
        features[:, 0:2] = obs_positions
        features[:, 2:4] = obs_velocities
        features[:, 4] = obs_yaw
        features[:, 5:7] = size  # Same width and length for all timesteps
        return features

    def _build_multi_agent_sample(self, tracks, seen_track_ids, window_frames, scene_id,
                                  starting_track_ids):
        """
        Build a multi-agent sample for one time window

        Args:
            tracks: Per-agent arrays of the scene (see _tracks_of_scene), parked vehicles
                already left out (see parked_track_ids)
            seen_track_ids: Agents with at least one sample in the observation period
            window_frames: (20,) frames of this window, 12 apart
            scene_id: Scene identifier
            starting_track_ids: Agents whose trajectory (metaId) starts at window_frames[0],
                the targets of this window
        """
        obs_frames = window_frames[:self.obs_len]
        pred_frames = window_frames[self.obs_len:]

        target_agents = []
        target_types = []
        target_labels = []
        target_label_masks = []
        target_auxiliary = []
        target_obs_masks = []
        target_track_ids = []
        context_agents = []
        context_types = []
        context_obs_masks = []

        for track_id in seen_track_ids:
            track = tracks[track_id]
            frames = track['frames']
            agent_type = track['type']

            # The agent's own samples inside the observation period (HDGT only uses the history)
            first = np.searchsorted(frames, obs_frames[0], side='left')
            last = np.searchsorted(frames, obs_frames[-1], side='right')
            obs_positions, obs_mask = fill_observation(frames[first:last], track['positions'][first:last], obs_frames)
            # Median width/length over its real samples (HDGT averages the size of the observed states)
            size = np.median(track['sizes'][first:last], axis=0)
            features = self._agent_features(obs_positions, size)

            is_target = track_id in starting_track_ids
            if is_target:
                # Future labels only from the agent's own samples. Steps without a sample are
                # masked out of the loss and metrics, as HDGT does with an incomplete future, and
                # a target needs at least one valid future step. A trajectory starting here has
                # all 20 steps, so in practice nothing is masked.
                label_mask = np.isin(pred_frames, frames)
                if not label_mask.any():
                    is_target = False

            if is_target:
                pred_positions = np.zeros((self.pred_len, 2))
                pred_positions[label_mask] = track['positions'][np.isin(frames, pred_frames)]
                # Prepend last observation position as bridge so pred_vel[0] = p8-p7,
                # not p9-p8 — eliminates the discontinuity at the obs/pred boundary
                pred_velocities = np.diff(np.vstack([obs_positions[-1:], pred_positions]), axis=0)
                # A velocity needs both of its positions: zero where either one is masked
                velocity_valid = label_mask & np.concatenate([[True], label_mask[:-1]])
                pred_velocities[~velocity_valid] = 0.0
                pred_yaw = np.arctan2(pred_velocities[:, 1], pred_velocities[:, 0])  # (12,)

                auxiliary_label = np.zeros((self.pred_len, 3))
                auxiliary_label[:, 0:2] = pred_velocities
                auxiliary_label[:, 2] = pred_yaw

                target_agents.append(features)
                target_types.append(agent_type)
                target_labels.append(pred_positions)
                target_label_masks.append(label_mask)
                target_auxiliary.append(auxiliary_label)
                target_obs_masks.append(obs_mask)
                target_track_ids.append(track_id)
                self.num_incomplete_future += int(not label_mask.all())
            else:
                # Neighbour (context agent): only its observation is used
                context_agents.append(features)
                context_types.append(agent_type)
                context_obs_masks.append(obs_mask)

        # Must have at least one target agent
        if len(target_agents) == 0:
            return None
        self.num_neighbours += len(context_agents)
        self.num_filled_steps += int(sum((mask == 0).sum() for mask in context_obs_masks))

        # Combine target agents + context agents
        all_features = target_agents + context_agents
        all_types = target_types + context_types

        agent_features = np.stack(all_features, axis=0)  # (num_agents, 8, 7)
        agent_types = np.array(all_types)  # (num_agents,)
        pred_num = len(target_agents)
        self.num_targets += pred_num

        # Stack labels
        labels = np.stack(target_labels, axis=0)  # (pred_num, 12, 2)
        auxiliary_labels = np.stack(target_auxiliary, axis=0)  # (pred_num, 12, 3)
        label_masks = np.stack(target_label_masks, axis=0).astype(np.float64)  # (pred_num, 12)

        sample = {
            'agent_feature': agent_features,
            'agent_type': agent_types,
            # (num_agents, 8): 1 = real sample, 0 = filled. Like HDGT, which keeps this mask but
            # whose model does not read it, the model's input features do not include it
            'agent_obs_mask': np.stack(target_obs_masks + context_obs_masks, axis=0),
            'pred_num': pred_num,
            'label': labels,
            'label_mask': label_masks,
            'auxiliary_label': auxiliary_labels,
            'scene_id': scene_id,
            'obejct_id_lis': target_track_ids,
            'map_fea': [[], []]  # Empty map features for SDD
        }

        return sample

    def __getitem__(self, idx):
        """Return preprocessed sample"""
        return self.samples[idx]

    def __len__(self):
        return len(self.samples)


class BalancedBatchSampler(BatchSampler):
    """Batches with about the same number of agents, so GPU memory stays steady from batch to
    batch (the original HDGT sampler, training/waymo_dataset.py)

    Samples are sorted by size (number of agents here; agents plus map elements in HDGT) and
    cut into batch_size equal groups, from largest to smallest. Each batch takes one sample from
    every group. In training the groups are reshuffled every epoch and each GPU takes its own
    slice of the batches. As in the original, the few samples left over by the equal split are
    not used in that epoch (different ones each epoch, since the groups are reshuffled).

    keep_remainder (single GPU only) adds the left-over samples as one last, smaller batch, so
    that validation evaluates every sample; the original skips them.
    """
    def __init__(self, input_size_lis, seed_num, gpu, gpu_cnt, batch_size, is_train, keep_remainder=False):
        assert not keep_remainder or gpu_cnt == 1, "keep_remainder needs a single GPU"
        self.batch_size = batch_size
        sorted_index = input_size_lis.argsort()[::-1].tolist()  # Largest sample first
        self.index_lis = []
        self.is_train = is_train
        for i in range(self.batch_size):
            self.index_lis.append(sorted_index[int(len(sorted_index)//self.batch_size * i):int(len(sorted_index)//self.batch_size * (i+1))])
        if len(sorted_index)//self.batch_size * self.batch_size < len(sorted_index):
            self.index_lis[-1] = self.index_lis[-1] + sorted_index[len(sorted_index)//self.batch_size * self.batch_size:]
        self.seed_num = seed_num
        self.gpu = gpu
        self.sample_per_gpu = len(self.index_lis[0])//gpu_cnt  # Batches per GPU per epoch
        self.keep_remainder = keep_remainder

    def _remainder(self):
        """Samples that no regular batch takes: the tail of the last group"""
        return [index for group in self.index_lis for index in group[self.sample_per_gpu:]]

    def __iter__(self):
        if self.is_train:
            for i in range(len(self.index_lis)):
                random.Random(self.seed_num+i).shuffle(self.index_lis[i])
        self.seed_num += 1
        for i in range(int(self.gpu*self.sample_per_gpu), int((self.gpu+1)*self.sample_per_gpu)):
            yield [self.index_lis[j][i] for j in range(self.batch_size)]
        if self.keep_remainder and len(self._remainder()) > 0:
            yield self._remainder()

    def __len__(self):
        return self.sample_per_gpu + int(self.keep_remainder and len(self._remainder()) > 0)


def split_files(data_folder, split):
    """Pickle files of one split. The single definition of the splits, used by train.py (through
    obtain_dataset) and test_sdd.py, so the two cannot disagree.

    Splits (by whole video, so no video is shared between splits):
        train: sdd/train/ without VAL_SCENES (44 videos)
        val:   VAL_SCENES, taken out of sdd/train/ (8 videos), used for model selection
        test:  sdd/test/ (8 videos), the final holdout
    """
    if split not in ('train', 'val', 'test'):
        raise ValueError(f"split must be train, val or test, got {split!r}")
    split_dir = os.path.join(data_folder, 'test' if split == 'test' else 'train')
    if not os.path.isdir(split_dir):
        raise FileNotFoundError(f"Dataset folder not found: {split_dir}")
    files = sorted([os.path.join(split_dir, f) for f in os.listdir(split_dir) if f.endswith('.pickle')])
    if split == 'test':
        return files

    def scene_name(path):
        return os.path.basename(path).replace('.pickle', '')

    # Fail instead of silently validating on fewer videos than intended
    missing_val_scenes = sorted(set(VAL_SCENES) - {scene_name(f) for f in files})
    if missing_val_scenes:
        raise FileNotFoundError(f"Validation scenes not found in {split_dir}: {missing_val_scenes}")

    # The validation videos are held out of the training files
    if split == 'val':
        return [f for f in files if scene_name(f) in VAL_SCENES]
    return [f for f in files if scene_name(f) not in VAL_SCENES]


def obtain_dataset(gpu, gpu_count, seed_num, args):
    """Load SDD dataset and create the train and validation dataloaders (see split_files; the
    test split is not loaded here, only test_sdd.py evaluates it)

    Returns:
        train_dataloader, val_dataloader (None on GPUs other than rank 0),
        train_sample_num, val_sample_num (0 on GPUs other than rank 0)
    """
    dataset_path = args.data_folder

    print(f"GPU {gpu}, Seed: {seed_num}", flush=True)

    train_files = split_files(dataset_path, 'train')
    val_files = split_files(dataset_path, 'val')

    print(f"Found {len(train_files)} train files and {len(val_files)} val files")

    # Shapes used to find parked vehicles; default: <data_folder>/shapes
    shapes_folder = getattr(args, 'shapes_folder', None)

    # Create datasets
    train_dataset = SDDDataset(train_files, obs_len=args.obs_len, pred_len=args.pred_len,
                               shapes_folder=shapes_folder)

    # Balanced batches as in the original HDGT (see BalancedBatchSampler). With several GPUs each
    # one takes its own slice of the batches; all processes share seed_num, so they shuffle alike.
    train_sizes = np.array([sample['agent_feature'].shape[0] for sample in train_dataset.samples])
    train_sampler = BalancedBatchSampler(train_sizes, seed_num=seed_num, gpu=gpu, gpu_cnt=gpu_count,
                                         batch_size=args.batch_size, is_train=True)
    train_dataloader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        pin_memory=True,
        collate_fn=partial(SDD_collate_fn, args=args, is_train=True),
        num_workers=args.num_worker
    )

    # Validation only runs on rank 0
    val_dataloader = None
    val_sample_num = 0
    if gpu == 0:
        val_dataset = SDDDataset(val_files, obs_len=args.obs_len, pred_len=args.pred_len,
                                 shapes_folder=shapes_folder)
        # Same balanced batches, without shuffling, plus one last batch with the left-over
        # samples so that every validation trajectory is evaluated
        val_sizes = np.array([sample['agent_feature'].shape[0] for sample in val_dataset.samples])
        val_sampler = BalancedBatchSampler(val_sizes, seed_num=seed_num, gpu=0, gpu_cnt=1,
                                           batch_size=args.val_batch_size, is_train=False, keep_remainder=True)
        val_dataloader = DataLoader(
            val_dataset,
            batch_sampler=val_sampler,
            pin_memory=True,
            collate_fn=partial(SDD_collate_fn, args=args, is_train=False),
            num_workers=args.num_worker
        )
        val_sample_num = len(val_dataset)
        print(f"Train samples: {len(train_dataset)}, Val samples: {val_sample_num}", flush=True)
        print('Data loaded', flush=True)

    train_sample_num = len(train_dataset)

    return train_dataloader, val_dataloader, train_sample_num, val_sample_num


def euclid_np(label, pred):
    """Euclidean distance between 2D points"""
    return np.sqrt((label[..., 0] - pred[..., 0])**2 + (label[..., 1] - pred[..., 1])**2)


def generate_heterogeneous_graph(agent_fea, agent_map_size_lis):
    """Build agent-only heterogeneous graph with proximity-based edges"""
    max_in_edge_per_type = 32
    uv_dic = {}

    # Self-loop edges for all agents
    uv_dic[("agent", "self", "agent")] = [
        list(range(agent_fea.shape[0])),
        list(range(agent_fea.shape[0]))
    ]
    num_of_agent = agent_fea.shape[0]

    # Agent-to-agent edges based on proximity
    uv_dic[("agent", "other", "agent")] = [[], []]
    for agent_index_i in range(num_of_agent):
        # Compute distance to all other agents at last observation timestep
        final_dist_between_agent = euclid_np(
            agent_fea[agent_index_i, -1, :][np.newaxis, :2],
            agent_fea[:, -1, :2]
        )

        # Find agents within interaction range
        nearby_agent_index = np.where(
            final_dist_between_agent < np.maximum(
                agent_map_size_lis[agent_index_i][np.newaxis],
                agent_map_size_lis
            )
        )[0]

        # Remove self
        nearby_agent_index = np.delete(
            nearby_agent_index,
            obj=np.where(nearby_agent_index == agent_index_i)
        )

        # Limit to max edges
        if len(nearby_agent_index) > max_in_edge_per_type:
            final_dist_between_agent_sorted_nearby_index = np.argsort(
                final_dist_between_agent[nearby_agent_index]
            )
            nearby_agent_index = nearby_agent_index[
                final_dist_between_agent_sorted_nearby_index
            ][:max_in_edge_per_type]

        nearby_agent_index = nearby_agent_index.tolist()
        if len(nearby_agent_index) > 0:
            uv_dic[("agent", "other", "agent")][0] += [agent_index_i] * len(nearby_agent_index)
            uv_dic[("agent", "other", "agent")][1] += nearby_agent_index

    # Convert to tensors
    for key in uv_dic:
        uv_dic[key] = (torch.LongTensor(uv_dic[key][0]), torch.LongTensor(uv_dic[key][1]))

    output_dic = {"uv_dic": uv_dic}
    return output_dic


def rotate(data, cos_theta, sin_theta):
    """Rotate 2D coordinates"""
    data[..., 0], data[..., 1] = (
        data[..., 0] * cos_theta - data[..., 1] * sin_theta,
        data[..., 1] * cos_theta + data[..., 0] * sin_theta
    )
    return data


def normal_agent_feature(feature, ref_coor, ref_psi, cos_theta, sin_theta):
    """Normalize agent features to ego-centric coordinates
    Input: [x, y, vx, vy, yaw, width, length] (7D)
    Output: [x', y', vx', vy', cos(yaw'), sin(yaw'), width, length] (8D)
    """
    # Translate positions
    feature[..., :2] -= ref_coor[:, np.newaxis, :]
    # Rotate positions
    feature[..., 0], feature[..., 1] = (
        feature[..., 0] * cos_theta - feature[..., 1] * sin_theta,
        feature[..., 1] * cos_theta + feature[..., 0] * sin_theta
    )
    # Rotate velocities
    feature[..., 2], feature[..., 3] = (
        feature[..., 2] * cos_theta - feature[..., 3] * sin_theta,
        feature[..., 3] * cos_theta + feature[..., 2] * sin_theta
    )
    # Normalize heading
    feature[..., 4] -= ref_psi
    cos_psi = np.cos(feature[..., 4])
    sin_psi = np.sin(feature[..., 4])
    # Output: [x, y, vx, vy, cos_psi, sin_psi, width, length]
    # Width and length (features 5, 6) are kept as-is (not rotated/normalized)
    feature = np.concatenate([
        feature[..., :4],
        cos_psi[..., np.newaxis],
        sin_psi[..., np.newaxis],
        feature[..., 5:]  # width, length
    ], axis=-1)
    return feature


def return_rel_e_feature(src_ref_coor, dst_ref_coor, src_ref_psi, dst_ref_psi):
    """Compute relative edge features
    Output: [rel_x, rel_y, sin(Δθ), cos(Δθ)] (4D)
    """
    rel_coor = src_ref_coor - dst_ref_coor
    if rel_coor.ndim == 0 or rel_coor.ndim == 1:
        rel_coor = np.atleast_1d(rel_coor)[np.newaxis, :]
    rel_coor = rotate(rel_coor, np.cos(-dst_ref_psi), np.sin(-dst_ref_psi))
    rel_psi = np.atleast_1d(src_ref_psi - dst_ref_psi)[:, np.newaxis]
    rel_sin_theta = np.sin(rel_psi)
    rel_cos_theta = np.cos(rel_psi)
    return np.concatenate([rel_coor, rel_sin_theta, rel_cos_theta], axis=-1)


# Map size (interaction range) based on agent type for SDD (6 agent types)
map_size_lis = {
    1: 3.0,   # Pedestrian: 150/50 = 3.0 (unchanged)
    2: 4.5,   # Biker: 225/50 = 4.5 (increased from 4.0)
    3: 5.0,   # Car: 250/50 = 5.0 (unchanged)
    4: 5.0,   # Bus: 250/50 = 5.0 (unchanged)
    5: 4.0,   # Cart: 200/50 = 4.0 (increased from 3.0)
    6: 4.5    # Skater: 225/50 = 4.5 (increased from 4.0)
}
@torch.no_grad()
def SDD_collate_fn(batch, args, is_train):
    """Collate function to batch samples and build graphs"""
    # Filter out None samples
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None

    agent_drop = args.agent_drop if hasattr(args, 'agent_drop') else 0.0

    agent_feature_lis = [item["agent_feature"] for item in batch]
    agent_type_lis = [item["agent_type"] for item in batch]
    agent_obs_mask_lis = [item["agent_obs_mask"] for item in batch]
    pred_num_lis = np.array([item["pred_num"] for item in batch])
    label_lis = [item["label"] for item in batch]
    auxiliary_label_lis = [item["auxiliary_label"] for item in batch]
    label_mask_lis = [item["label_mask"] for item in batch]
    case_id_lis = [item["scene_id"] for item in batch]
    object_id_lis = [item["obejct_id_lis"] for item in batch]

    # Agent dropout for regularization during training
    if agent_drop > 0 and is_train:
        for i in range(len(agent_feature_lis)):
            keep_index = (np.random.random(agent_feature_lis[i].shape[0]) > agent_drop)
            # Ensure at least one target agent is kept
            while keep_index[:pred_num_lis[i]].sum() == 0:
                keep_index = (np.random.random(agent_feature_lis[i].shape[0]) > agent_drop)

            origin_pred_num = pred_num_lis[i]
            target_keep_index = keep_index[:origin_pred_num]

            agent_feature_lis[i] = agent_feature_lis[i][keep_index]
            agent_type_lis[i] = agent_type_lis[i][keep_index]
            agent_obs_mask_lis[i] = agent_obs_mask_lis[i][keep_index]
            pred_num_lis[i] = int(target_keep_index.sum())

            label_lis[i] = label_lis[i][target_keep_index]
            auxiliary_label_lis[i] = auxiliary_label_lis[i][target_keep_index]
            label_mask_lis[i] = label_mask_lis[i][target_keep_index]

    neighbor_size = np.array([int(agent_feature_lis[i].shape[0])
                              for i in range(len(agent_feature_lis))])

    out_normal_lis = []
    out_graph_lis = []
    out_label_lis = []
    out_label_mask_lis = []
    out_auxiliary_label_lis = []
    out_auxiliary_label_future_lis = []

    for i in range(len(agent_feature_lis)):
        # Compute agent map size (interaction range based on velocity and type)
        all_agent_obs_final_v = np.sqrt(
            agent_feature_lis[i][:, -1, 2]**2 +
            agent_feature_lis[i][:, -1, 3]**2
        )
        all_agent_map_size = np.vectorize(map_size_lis.__getitem__)(agent_type_lis[i])
        # Base range of the type + 2 x the displacement per timestep: the same interaction range
        # as hdgt_goal and every other version. (HDGT itself adds the distance covered over the
        # whole horizon, which here would be displacement * pred_len.)
        all_agent_map_size = all_agent_obs_final_v * 2.0 + all_agent_map_size
        # Generate heterogeneous graph
        graph_dic = generate_heterogeneous_graph(agent_feature_lis[i], all_agent_map_size)
        g = dgl.heterograph(data_dict=graph_dic["uv_dic"])

        now_agent_feature = agent_feature_lis[i]
        now_agent_type = agent_type_lis[i]

        ### Self-loop edges
        type0_u, type0_v = g.edges(etype="self")
        now_t0_v_feature = now_agent_feature[type0_v, :, :]
        now_t0_e_feature = now_agent_feature[type0_u].copy()
        if len(type0_v) == 1:
            now_t0_v_feature = now_t0_v_feature[np.newaxis, :, :]
            now_t0_e_feature = now_t0_e_feature[np.newaxis, :, :]

        now_t0_e_feature = return_rel_e_feature(
            now_t0_e_feature[:, -1, :2], now_t0_v_feature[:, -1, :2],
            now_t0_e_feature[:, -1, 4], now_t0_v_feature[:, -1, 4]
        )
        g.edges["self"].data['a_e_fea'] = torch.as_tensor(now_t0_e_feature.astype(np.float32))
        g.edges["self"].data['a_e_type'] = torch.as_tensor(
            (now_agent_type[type0_u].ravel() - 1).astype(np.int32)
        ).long()

        ### Agent-to-agent edges
        type1_u, type1_v = g.edges(etype="other")
        if len(type1_v) > 0:
            now_t1_v_feature = now_agent_feature[type1_v, :, :]
            now_t1_e_feature = now_agent_feature[type1_u].copy()
            if len(type1_v) == 1:
                now_t1_v_feature = now_t1_v_feature[np.newaxis, :, :]
                now_t1_e_feature = now_t1_e_feature[np.newaxis, :, :]

            now_t1_e_feature = return_rel_e_feature(
                now_t1_e_feature[:, -1, :2], now_t1_v_feature[:, -1, :2],
                now_t1_e_feature[:, -1, 4], now_t1_v_feature[:, -1, 4]
            )
            g.edges["other"].data['a_e_fea'] = torch.as_tensor(now_t1_e_feature.astype(np.float32))
            g.edges["other"].data['a_e_type'] = torch.as_tensor(
                (now_agent_type[type1_u].ravel() - 1).astype(np.int32)
            ).long()
        else:
            g.edges["other"].data['a_e_fea'] = torch.zeros((0, 4))
            g.edges["other"].data['a_e_type'] = torch.zeros((0,)).long()

        # Multi-agent prediction: normalize each target agent to its own reference frame
        now_pred_num = pred_num_lis[i]
        selected_pred_indices = list(range(0, now_pred_num))
        non_pred_indices = list(range(now_pred_num, now_agent_feature.shape[0]))

        ## Process target agents (agents with labels)
        now_full_agent_n_feature = now_agent_feature[selected_pred_indices].copy()
        ref_coor = now_full_agent_n_feature[:, -1, :2].copy()  # (pred_num, 2)
        now_label = label_lis[i][selected_pred_indices].copy()
        now_auxiliary_label = auxiliary_label_lis[i][selected_pred_indices].copy()

        # Translate labels to each agent's own frame
        now_label = now_label - ref_coor[:, np.newaxis, :]
        ref_psi = now_full_agent_n_feature[:, -1, 4:5].copy()  # (pred_num, 1)
        normal_val = np.concatenate([ref_coor, ref_psi], axis=-1)
        out_normal_lis.append(normal_val)

        sin_theta = np.sin(-ref_psi)
        cos_theta = np.cos(-ref_psi)
        rotate(now_label, cos_theta, sin_theta)
        rotate(now_auxiliary_label, cos_theta, sin_theta)
        now_auxiliary_label[..., 2] = now_auxiliary_label[..., 2] - ref_psi

        # Normalize target agents to their own reference frames
        now_full_agent_n_feature = normal_agent_feature(
            now_full_agent_n_feature, ref_coor, ref_psi, cos_theta, sin_theta
        )
        now_auxiliary_label_future = now_auxiliary_label.copy()
        now_auxiliary_label = np.stack([
            now_full_agent_n_feature[..., 2],  # vx
            now_full_agent_n_feature[..., 3],  # vy
            now_agent_feature[selected_pred_indices, :, 4] - ref_psi,  # normalized yaw
        ], axis=-1)

        # Process non-target agents (context agents) - normalize to their own frames
        now_all_agent_n_feature = now_full_agent_n_feature
        if now_pred_num < now_agent_feature.shape[0]:
            now_other_agent_n_feature = now_agent_feature[non_pred_indices].copy()
            ref_coor = now_other_agent_n_feature[:, -1, :2]
            ref_psi = now_other_agent_n_feature[:, -1, 4:5].copy()
            sin_theta = np.sin(-ref_psi)
            cos_theta = np.cos(-ref_psi)
            now_other_agent_n_feature = normal_agent_feature(
                now_other_agent_n_feature, ref_coor, ref_psi, cos_theta, sin_theta
            )
            now_all_agent_n_feature = np.concatenate([now_all_agent_n_feature, now_other_agent_n_feature], axis=0)

        # Assign to graph
        g.nodes["agent"].data["a_n_fea"] = torch.as_tensor(now_all_agent_n_feature.astype(np.float32))
        g.nodes["agent"].data["a_n_type"] = torch.as_tensor((now_agent_type - 1).astype(np.int32)).long()

        out_graph_lis.append(g)
        out_label_lis.append(now_label)
        out_label_mask_lis.append(label_mask_lis[i][selected_pred_indices])
        out_auxiliary_label_lis.append(now_auxiliary_label)
        out_auxiliary_label_future_lis.append(now_auxiliary_label_future)

    # Batch everything
    output_dic = {}
    output_dic["cuda_tensor_lis"] = [
        "graph_lis", "label_lis", "label_mask_lis",
        "auxiliary_label_lis", "auxiliary_label_future_lis"
    ]
    output_dic["label_lis"] = torch.as_tensor(np.concatenate(out_label_lis, axis=0).astype(np.float32))
    output_dic["auxiliary_label_lis"] = torch.as_tensor(np.concatenate(out_auxiliary_label_lis, axis=0).astype(np.float32))
    output_dic["auxiliary_label_future_lis"] = torch.as_tensor(np.concatenate(out_auxiliary_label_future_lis, axis=0).astype(np.float32))
    output_dic["label_mask_lis"] = torch.as_tensor(np.concatenate(out_label_mask_lis, axis=0).astype(np.float32))

    # Batch graphs
    output_g = dgl.batch(out_graph_lis)

    # Create edge type dictionary (6 agent types for SDD)
    a_e_type_dict = {}
    for out_etype in ["self", "other"]:
        a_e_type_dict[out_etype] = []
        for agent_type_index in range(6):
            a_e_type_dict[out_etype].append(
                torch.where(output_g.edges[out_etype].data["a_e_type"] == agent_type_index)[0]
            )

    # Create node type list (6 agent types for SDD)
    a_n_type_lis = [
        torch.where(output_g.nodes["agent"].data["a_n_type"] == _)[0]
        for _ in range(6)
    ]

    output_dic["a_e_type_dict"] = a_e_type_dict
    output_dic["a_n_type_lis"] = a_n_type_lis
    output_dic["graph_lis"] = output_g
    output_dic["neighbor_size_lis"] = neighbor_size
    output_dic["pred_num_lis"] = pred_num_lis
    output_dic["case_id_lis"] = case_id_lis
    output_dic["object_id_lis"] = object_id_lis
    output_dic["normal_lis"] = np.concatenate(out_normal_lis, axis=0)
    # (num_agents_in_batch, obs_len) in graph node order: 1 = real sample, 0 = filled step.
    # Kept on the CPU and not used by the model (HDGT's model does not use it either)
    output_dic["agent_obs_mask_lis"] = torch.as_tensor(np.concatenate(agent_obs_mask_lis, axis=0).astype(np.float32))

    del batch
    return output_dic
