"""
SDD Dataset Loader for Diffusion Model - HDGT Style (Trajectory Prediction)

CORRECTED VERSION:
- ALL velocities are PER-FRAME (pixels/frame = displacements)
- Velocity convention: vel[t] = pos[t] - pos[t-1] (BACKWARD-LOOKING)
- This ensures NO discontinuity between observation and prediction velocities
- Coordinates are scaled down by SCALE_FACTOR for numerical stability
- Agent distribution tracking and speed-normalized metrics

Expected Data Format:
- Train: 52 pickle files in sdd/train/, 8 of them (VAL_VIDEOS) held out as validation
- Test: 8 pickle files in sdd/test/, the final holdout
- Shapes: sdd/shapes/<video>.pickle, drawn with draw_shapes.py; the obstacles find parked vehicles
- Each pickle file contains a pandas DataFrame with columns:
  ['trackId', 'frame', 'label', 'x', 'y', 'sceneId', 'metaId', 'xmin', 'ymin', 'xmax', 'ymax']
- Trajectories are sampled at 2.5 FPS (every 12 frames at 30 FPS = 0.4 seconds)
- Each metaId is one 20-timestep trajectory. One sample per distinct trajectory start frame,
  whose targets are the trajectories starting there (the same windows as original_hdgt_on_sdd)
- Neighbours as in HDGT (and original_hdgt_on_sdd): every other agent seen during the
  observation, its observed steps without a sample filled (see fill_observation)
- Parked cars/buses/carts (every position inside the drawn obstacles) are not agents (see PARKED_TYPES)
"""

import os
import random
import torch
from torch.utils.data import Dataset, DataLoader, BatchSampler
import numpy as np
import pandas as pd
from functools import partial
import dgl


# =============================================================================
# COORDINATE SCALING
# SDD uses pixel coordinates which can be very large (hundreds to thousands).
# We scale down by this factor for numerical stability during diffusion training.
# Metrics are reported in original pixel coordinates (scaled back up).
# =============================================================================
SCALE_FACTOR = 50.0

# Agent type name mapping
AGENT_TYPE_NAMES = {
    1: "Pedestrian",
    2: "Biker",
    3: "Car",
    4: "Bus",
    5: "Cart",
    6: "Skater"
}

# =============================================================================
# TRAIN / VAL SPLIT
# Model selection, dev-time reports and visualizations run on VAL. The test
# folder is the final holdout and should be touched once, at the end.
#
# Split by whole VIDEO, never by window: windows span 228 frames and start at
# every trajectory's first frame, so neighbouring windows share most of their
# frames and agents. A per-window split would put near-copies in both sets.
#
# One video per scene, held out of ../sdd/train/. Chosen from agent-type counts
# alone (never from model results), targeting ~10% of each type. The same 8
# videos in every version: 1549 val samples (1947 target agents).
# =============================================================================
VAL_VIDEOS = (
    "bookstore_video4",
    "coupa_video1",
    "deathCircle_video2",
    "gates_video1",
    "hyang_video1",
    "little_video2",
    "nexus_video9",
    "quad_video0",
)


def split_train_val(train_files):
    """Partition the train folder's pickles into (train, val) by VAL_VIDEOS."""
    val_set = set(VAL_VIDEOS)
    val = [f for f in train_files
           if os.path.basename(f).replace(".pickle", "") in val_set]
    train = [f for f in train_files
             if os.path.basename(f).replace(".pickle", "") not in val_set]
    missing = val_set - {os.path.basename(f).replace(".pickle", "") for f in val}
    if missing:
        raise FileNotFoundError(
            f"VAL_VIDEOS not found in the train folder: {sorted(missing)}")
    return train, val


def resolve_split_files(data_folder, split):
    """Pickle paths for 'train', 'val' or 'test'. One source of truth so the
    train, test and visualization scripts cannot drift apart."""
    if split not in ("train", "val", "test"):
        raise ValueError(f"split must be train/val/test, got {split!r}")
    if split == "test":
        d = os.path.join(data_folder, "test")
        return sorted(os.path.join(d, f) for f in os.listdir(d)
                      if f.endswith(".pickle"))
    d = os.path.join(data_folder, "train")
    all_train = sorted(os.path.join(d, f) for f in os.listdir(d)
                       if f.endswith(".pickle"))
    tr, va = split_train_val(all_train)
    return tr if split == "train" else va


# =============================================================================
# PARKED VEHICLES -- the same rule in every version (keep them in sync)
# A car, bus or cart is parked when EVERY position of its trackId in the video
# lies inside the obstacle shapes drawn on the scene image (sdd/shapes,
# draw_shapes.py). It is an obstacle, not an agent: neither predicted nor
# anyone's neighbour, so it is in no metric. A vehicle outside the obstacles at
# even one position is an agent, so a moving one whose annotation strays into an
# obstacle (a camera jump, a misplaced box) is still predicted. A parked car
# split over several trackIds is caught track by track. This loader has no map,
# so a parked vehicle's rows are simply left out of the video.
# =============================================================================
PARKED_TYPES = (3, 4, 5)   # Car, Bus, Cart


def load_obstacle_shapes(shapes_folder, scene_id, scale_factor=SCALE_FACTOR):
    """Obstacle shapes of one video as (S, 6) rows [cx, cy, half_w, half_h, theta, is_circle], SCALED"""
    path = os.path.join(shapes_folder, scene_id + '.pickle')
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"No shapes for {scene_id}: expected {path}. Parked vehicles are found with the "
            f"obstacle shapes drawn by draw_shapes.py.")
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
    incomplete history -- the same neighbours in every version (keep them in sync)

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
                Default: "shapes" next to the folder holding each track pickle (../sdd/shapes)
        """
        self.shapes_folder = shapes_folder
        self.parked_count = 0  # parked Car/Bus/Cart trajectories, not agents
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.total_len = obs_len + pred_len  # 20
        self.time_step = 0.4  # SDD: 2.5 FPS (12 frames at 30 FPS = 0.4 seconds per frame)
        self.scale_factor = SCALE_FACTOR
        self.prediction_horizon_seconds = pred_len * self.time_step  # 4.8 seconds

        # Agent type mapping (6 types for SDD)
        self.type_map = {
            'Pedestrian': 1,
            'Biker': 2,
            'Car': 3,
            'Bus': 4,
            'Cart': 5,
            'Skater': 6
        }

        self.samples = []
        
        # Track agent type distribution
        self.agent_counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}
        
        self._load_all_data(pickle_files)

    def _load_all_data(self, pickle_files):
        """Load all pickle files and extract multi-agent samples per time window"""
        print(f"\n{'='*80}")
        print(f"Loading SDD dataset for diffusion model")
        print(f"Scale factor: {self.scale_factor}")
        print(f"Prediction horizon: {self.pred_len} frames = {self.prediction_horizon_seconds:.1f} seconds")
        print(f"{'='*80}\n")

        for pkl_idx, pkl_file in enumerate(pickle_files):
            df = pd.read_pickle(pkl_file)
            scene_id = os.path.basename(pkl_file).replace('.pickle', '')
            shapes_folder = self.shapes_folder or os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(pkl_file))), 'shapes')
            obstacles = load_obstacle_shapes(shapes_folder, scene_id, self.scale_factor)

            # Parked vehicles are obstacles, not agents: their rows go before any window is cut
            parked = df['trackId'].isin(parked_track_ids(df, self.type_map, obstacles, self.scale_factor))
            self.parked_count += df.loc[parked, 'metaId'].nunique()
            df = df[~parked]
            tracks = self._tracks_of_scene(df)

            # All rows sorted by frame, to find the agents seen during an observation period
            order = np.argsort(df['frame'].values, kind='stable')
            row_frames = df['frame'].values[order]
            row_track_ids = df['trackId'].values[order]

            # Windows as in original_hdgt_on_sdd: each metaId is one 20-timestep trajectory,
            # sampled every 12 frames from its own first frame. One sample per distinct
            # trajectory start frame; its window is start + 12*k, and the trajectories that
            # start there are its targets, so every trajectory is predicted exactly once.
            trajectories = df.groupby('metaId').agg(start_frame=('frame', 'min'),
                                                    trackId=('trackId', 'first'))
            for start_frame, starting_here in trajectories.groupby('start_frame'):
                window_frames = (start_frame + 12 * np.arange(self.total_len)).tolist()
                obs_frames = window_frames[:self.obs_len]

                # Agents with at least one sample during the observation: HDGT keeps exactly these
                first = np.searchsorted(row_frames, obs_frames[0], side='left')
                last = np.searchsorted(row_frames, obs_frames[-1], side='right')
                seen_track_ids = np.unique(row_track_ids[first:last])

                sample = self._build_multi_agent_sample(df, tracks, seen_track_ids, window_frames,
                                                        obs_frames, scene_id,
                                                        set(starting_here['trackId']))
                if sample is not None:
                    self.samples.append(sample)

            print(f"  [{pkl_idx+1:2d}/{len(pickle_files)}] {scene_id:40s} -> {len(self.samples):6d} total samples")

        print(f"\n{'='*80}")
        print(f"Dataset loading complete: {len(self.samples)} total samples")
        print(f"Parked cars/buses/carts (every position inside the drawn obstacles, not agents): "
              f"{self.parked_count} trajectories")
        self._print_agent_distribution()
        print(f"{'='*80}\n")

    def _print_agent_distribution(self):
        """Print agent type distribution statistics"""
        total_agents = sum(self.agent_counts.values())
        
        print(f"\nAgent Type Distribution:")
        print(f"{'-'*60}")
        print(f"{'Agent Type':<15} {'Count':>10} {'Percentage':>12}")
        print(f"{'-'*60}")

        for agent_type in sorted(self.agent_counts.keys()):
            count = self.agent_counts[agent_type]
            percentage = (count / total_agents * 100) if total_agents > 0 else 0
            name = AGENT_TYPE_NAMES[agent_type]
            print(f"{name:<15} {count:>10,} {percentage:>11.2f}%")

        print(f"{'-'*60}")
        print(f"{'TOTAL':<15} {total_agents:>10,} {100.0:>11.2f}%")
        print(f"{'-'*60}")

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

    def _compute_backward_velocities(self, positions):
        """
        Compute backward-looking velocities: vel[t] = pos[t] - pos[t-1]
        This ensures NO discontinuity between observation and prediction velocities.

        Args:
            positions: (N, 2) array of positions

        Returns:
            (N, 2) array of velocities where:
            - vel[0] = vel[1] (replicated for initial timestep)
            - vel[t] = pos[t] - pos[t-1] for t >= 1
        """
        # Calculate backward velocities: vel[t] = pos[t] - pos[t-1]
        backward_vels = np.diff(positions, axis=0)  # (N-1, 2)
        # vel[1] = pos[1]-pos[0], vel[2] = pos[2]-pos[1], ..., vel[N-1] = pos[N-1]-pos[N-2]

        # For vel[0], replicate vel[1] (initial velocity estimate)
        velocities = np.vstack([backward_vels[0:1], backward_vels])  # (N, 2)

        return velocities

    def _build_multi_agent_sample(self, df, tracks, seen_track_ids, window_frames, obs_frames,
                                  scene_id, starting_track_ids):
        """
        Build a multi-agent sample for one window: the trajectories starting on its first
        frame (starting_track_ids) are the targets, and every other agent seen during the
        observation (seen_track_ids) is a neighbour, its missing observed steps filled as
        HDGT does (see fill_observation). Agents go in trackId order, targets first.
        Parked vehicles are already out of df (see parked_track_ids).
        All coordinates are scaled down by SCALE_FACTOR for numerical stability.
        All velocities are PER-FRAME (displacements).
        """
        window_agents = df[df['frame'].isin(window_frames)]

        target_agents = []
        target_types = []
        target_labels_pos = []  # Position labels for evaluation
        target_labels_vel = []  # Velocity labels for diffusion training
        target_goals = []  # Goal positions (final timestep) for goal-conditioned prediction
        target_track_ids = []
        context_agents = []
        context_types = []

        for track_id in seen_track_ids:
            agent_data = (window_agents[window_agents['trackId'] == track_id]
                          if track_id in starting_track_ids else None)

            # Target: the trajectory (metaId) starting on this window's first frame, which has
            # all 20 timesteps
            if agent_data is not None and len(agent_data) == self.total_len:
                agent_data = agent_data.sort_values('frame')
                agent_type = self.type_map.get(agent_data['label'].iloc[0], 1)

                # Get positions and SCALE DOWN
                positions = agent_data[['x', 'y']].values / self.scale_factor  # (20, 2) SCALED

                # Extract bounding box data and calculate width/length, SCALED
                obs_bbox_data = agent_data[['xmin', 'ymin', 'xmax', 'ymax']].values[:self.obs_len] / self.scale_factor
                agent_width = np.median(obs_bbox_data[:, 2] - obs_bbox_data[:, 0])   # SCALED
                agent_length = np.median(obs_bbox_data[:, 3] - obs_bbox_data[:, 1])  # SCALED

                # Observation features: [x, y, vx, vy, yaw, width, length] - all SCALED
                # Velocities are BACKWARD-LOOKING: vel[t] = pos[t] - pos[t-1]
                obs_positions = positions[:self.obs_len]  # (8, 2)

                # Track agent count
                self.agent_counts[agent_type] += 1

                # Calculate backward-looking velocities (NO discontinuity)
                # vel[0] = vel[1], vel[1] = pos[1]-pos[0], ..., vel[7] = pos[7]-pos[6]
                obs_velocities = self._compute_backward_velocities(obs_positions)  # (8, 2)
                
                obs_yaw = np.arctan2(obs_velocities[:, 1], obs_velocities[:, 0])

                features = np.zeros((self.obs_len, 7))
                features[:, 0:2] = obs_positions      # SCALED positions
                features[:, 2:4] = obs_velocities     # SCALED per-frame velocities (displacements)
                features[:, 4] = obs_yaw              # Angle (radians)
                features[:, 5] = agent_width          # SCALED
                features[:, 6] = agent_length         # SCALED

                # Prediction labels - positions (for evaluation metrics) - SCALED
                pred_positions = positions[self.obs_len:]  # (12, 2) SCALED

                # Prediction labels - velocities (for diffusion training) - SCALED, BACKWARD-LOOKING
                # vel[0] = pos[8]-pos[7], vel[1] = pos[9]-pos[8], ..., vel[11] = pos[19]-pos[18]
                # This naturally continues from obs vel[7] = pos[7]-pos[6] with NO discontinuity
                pred_velocities = self._compute_backward_velocities(
                    np.vstack([positions[self.obs_len-1:self.obs_len], pred_positions])
                )[1:]  # (12, 2) - skip the replicated first velocity

                # GOAL POSITION: Final position (timestep 19) relative to last observation
                # This is already in ego-centric coordinates (relative to pos[7])
                goal_position = pred_positions[-1, :]  # (2,) - Last timestep position

                target_agents.append(features)
                target_types.append(agent_type)
                target_labels_pos.append(pred_positions)
                target_labels_vel.append(pred_velocities)
                target_goals.append(goal_position)
                target_track_ids.append(track_id)

            else:
                # Neighbour (context agent, only its observation is used), as in HDGT: its
                # observed steps come from its own samples, the missing ones filled (see
                # fill_observation), and its size is the median over those real samples
                track = tracks[track_id]
                frames = track['frames']
                first = np.searchsorted(frames, obs_frames[0], side='left')
                last = np.searchsorted(frames, obs_frames[-1], side='right')
                obs_positions, _ = fill_observation(frames[first:last], track['positions'][first:last],
                                                    np.asarray(obs_frames))  # SCALED
                agent_type = track['type']
                agent_width, agent_length = np.median(track['sizes'][first:last], axis=0)  # SCALED

                # Calculate backward-looking velocities (NO discontinuity)
                obs_velocities = self._compute_backward_velocities(obs_positions)  # (8, 2)

                obs_yaw = np.arctan2(obs_velocities[:, 1], obs_velocities[:, 0])

                features = np.zeros((self.obs_len, 7))
                features[:, 0:2] = obs_positions      # SCALED positions
                features[:, 2:4] = obs_velocities     # SCALED per-frame velocities (displacements)
                features[:, 4] = obs_yaw              # Angle (radians)
                features[:, 5] = agent_width          # SCALED
                features[:, 6] = agent_length         # SCALED

                context_agents.append(features)
                context_types.append(agent_type)

        # Must have at least one target agent
        if len(target_agents) == 0:
            return None

        # Combine target agents + context agents
        all_features = target_agents + context_agents
        all_types = target_types + context_types

        agent_features = np.stack(all_features, axis=0)
        agent_types = np.array(all_types)
        pred_num = len(target_agents)

        # Stack labels (all SCALED)
        labels_pos = np.stack(target_labels_pos, axis=0)  # (pred_num, 12, 2) SCALED positions
        labels_vel = np.stack(target_labels_vel, axis=0)  # (pred_num, 12, 2) SCALED per-frame velocities
        goal_pos = np.stack(target_goals, axis=0)  # (pred_num, 2) SCALED goal positions
        label_masks = np.ones((pred_num, self.pred_len))

        sample = {
            'agent_feature': agent_features,
            'agent_type': agent_types,
            'pred_num': pred_num,
            'label_pos': labels_pos,      # Position labels for evaluation (SCALED)
            'label_vel': labels_vel,      # Velocity labels for diffusion (SCALED, per-frame)
            'goal_pos': goal_pos,         # Goal positions (final timestep) for goal-conditioned prediction (SCALED)
            'label_mask': label_masks,
            'scene_id': scene_id,
            'object_id_lis': target_track_ids,
        }

        return sample

    def __getitem__(self, idx):
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


def obtain_dataset(gpu, gpu_count, seed_num, args, test_only=False):
    """Load SDD dataset and create dataloaders.

    Returns (train_loader, eval_loader, train_n, eval_n). The eval loader is the
    VAL split by default, so checkpoint selection never sees the test folder.

    Args:
        test_only: If True, skip the training set (for evaluation only)
    """
    dataset_path = args.data_folder

    print(f"GPU {gpu}, Seed: {seed_num}", flush=True)

    # Model selection runs on VAL (videos held out of the train folder), so the
    # test folder stays an untouched final holdout. --use_val_split False
    # restores the old select-on-test behavior, which leaks.
    use_val_split = getattr(args, "use_val_split", True)
    if isinstance(use_val_split, str):
        use_val_split = (use_val_split == "True")
    eval_split = "val" if use_val_split else "test"

    train_files = resolve_split_files(dataset_path, "train") if use_val_split \
        else sorted(os.path.join(dataset_path, "train", f)
                    for f in os.listdir(os.path.join(dataset_path, "train"))
                    if f.endswith(".pickle"))
    eval_files = resolve_split_files(dataset_path, eval_split)

    print(f"\n{'='*80}")
    print(f"Found {len(train_files)} train files and "
          f"{len(eval_files)} {eval_split} files")
    if use_val_split:
        print(f"Validation videos held out of training: {', '.join(VAL_VIDEOS)}")
        print("Test folder is NOT touched during training.")
    else:
        print("!" * 80)
        print("WARNING: --use_val_split False -- selecting checkpoints on the "
              "TEST set.\n         Reported test numbers will be optimistic. "
              "Do not use for final results.")
        print("!" * 80)
    print(f"{'='*80}")

    # Load train set (skip if test_only)
    if not test_only:
        print(f"\n{'#'*80}")
        print(f"# LOADING TRAINING SET")
        print(f"{'#'*80}")
        train_dataset = SDDDataset(train_files, obs_len=args.obs_len, pred_len=args.pred_len)
        
        # Balanced batches as in the original HDGT, the same batches in every version (see
        # BalancedBatchSampler). With several GPUs each one takes its own slice of the batches;
        # all processes share seed_num, so they shuffle alike.
        train_sizes = np.array([sample['agent_feature'].shape[0] for sample in train_dataset.samples])
        train_sampler = BalancedBatchSampler(train_sizes, seed_num=seed_num, gpu=gpu, gpu_cnt=max(gpu_count, 1),
                                             batch_size=args.batch_size, is_train=True)
        train_dataloader = DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            pin_memory=True,
            collate_fn=partial(SDD_collate_fn, args=args, is_train=True),
            num_workers=args.num_worker
        )
        train_sample_num = len(train_dataset)
    else:
        # Create dummy values for train
        train_dataloader = None
        train_sample_num = 0
        print("\nSkipping training set (test_only=True)")
    
    print(f"\n{'#'*80}")
    print(f"# LOADING {eval_split.upper()} SET")
    print(f"{'#'*80}")
    eval_dataset = SDDDataset(eval_files, obs_len=args.obs_len, pred_len=args.pred_len)

    eval_dataloader = None
    if gpu == 0 or test_only:
        # Same balanced batches, without shuffling, plus one last batch with the left-over
        # samples so that every validation trajectory is evaluated
        eval_sizes = np.array([sample['agent_feature'].shape[0] for sample in eval_dataset.samples])
        eval_sampler = BalancedBatchSampler(eval_sizes, seed_num=seed_num, gpu=0, gpu_cnt=1,
                                            batch_size=args.val_batch_size, is_train=False, keep_remainder=True)
        eval_dataloader = DataLoader(
            eval_dataset,
            batch_sampler=eval_sampler,
            pin_memory=True,
            collate_fn=partial(SDD_collate_fn, args=args, is_train=False),
            num_workers=args.num_worker
        )
        print(f"\n{'='*80}")
        print(f"FINAL DATASET SUMMARY")
        print(f"{'='*80}")
        if not test_only:
            print(f"Train samples: {len(train_dataset):,}")
        print(f"{eval_split.capitalize():<5} samples: {len(eval_dataset):,}"
              f"   <- checkpoints are selected on this")
        print(f"{'='*80}\n")
        print('Data loaded successfully!\n', flush=True)

    return train_dataloader, eval_dataloader, train_sample_num, len(eval_dataset)

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
        final_dist_between_agent = euclid_np(
            agent_fea[agent_index_i, -1, :][np.newaxis, :2],
            agent_fea[:, -1, :2]
        )

        nearby_agent_index = np.where(
            final_dist_between_agent < np.maximum(
                agent_map_size_lis[agent_index_i][np.newaxis],
                agent_map_size_lis
            )
        )[0]

        nearby_agent_index = np.delete(
            nearby_agent_index,
            obj=np.where(nearby_agent_index == agent_index_i)
        )

        if len(nearby_agent_index) > max_in_edge_per_type:
            sorted_idx = np.argsort(final_dist_between_agent[nearby_agent_index])
            nearby_agent_index = nearby_agent_index[sorted_idx][:max_in_edge_per_type]

        nearby_agent_index = nearby_agent_index.tolist()
        if len(nearby_agent_index) > 0:
            uv_dic[("agent", "other", "agent")][0] += [agent_index_i] * len(nearby_agent_index)
            uv_dic[("agent", "other", "agent")][1] += nearby_agent_index

    for key in uv_dic:
        uv_dic[key] = (torch.LongTensor(uv_dic[key][0]), torch.LongTensor(uv_dic[key][1]))

    return {"uv_dic": uv_dic}


def rotate(data, cos_theta, sin_theta):
    """Rotate 2D coordinates"""
    x_new = data[..., 0] * cos_theta - data[..., 1] * sin_theta
    y_new = data[..., 1] * cos_theta + data[..., 0] * sin_theta
    data[..., 0] = x_new
    data[..., 1] = y_new
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
    rel_coor = rotate(rel_coor.copy(), np.cos(-dst_ref_psi), np.sin(-dst_ref_psi))
    rel_psi = np.atleast_1d(src_ref_psi - dst_ref_psi)[:, np.newaxis]
    rel_sin_theta = np.sin(rel_psi)
    rel_cos_theta = np.cos(rel_psi)
    return np.concatenate([rel_coor, rel_sin_theta, rel_cos_theta], axis=-1)


# =============================================================================
# Map size (interaction range) based on agent type - SCALED by SCALE_FACTOR
# INCREASED ranges for Bikers, Carts, and Skaters for better neighbor context
# Original pixel values scaled down by 50
# =============================================================================
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
    """Collate function to batch samples and build graphs for diffusion model"""
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None

    agent_drop = args.agent_drop if hasattr(args, 'agent_drop') else 0.0

    agent_feature_lis = [item["agent_feature"] for item in batch]
    agent_type_lis = [item["agent_type"] for item in batch]
    pred_num_lis = np.array([item["pred_num"] for item in batch])
    label_pos_lis = [item["label_pos"] for item in batch]
    label_vel_lis = [item["label_vel"] for item in batch]
    goal_pos_lis = [item["goal_pos"] for item in batch]  # Goal positions for goal-conditioned prediction
    label_mask_lis = [item["label_mask"] for item in batch]
    case_id_lis = [item["scene_id"] for item in batch]
    object_id_lis = [item["object_id_lis"] for item in batch]

    # Agent dropout for regularization during training
    if agent_drop > 0 and is_train:
        for i in range(len(agent_feature_lis)):
            keep_index = (np.random.random(agent_feature_lis[i].shape[0]) > agent_drop)
            while keep_index[:pred_num_lis[i]].sum() == 0:
                keep_index = (np.random.random(agent_feature_lis[i].shape[0]) > agent_drop)

            origin_pred_num = pred_num_lis[i]
            target_keep_index = keep_index[:origin_pred_num]

            agent_feature_lis[i] = agent_feature_lis[i][keep_index]
            agent_type_lis[i] = agent_type_lis[i][keep_index]
            pred_num_lis[i] = int(target_keep_index.sum())
            label_pos_lis[i] = label_pos_lis[i][target_keep_index]
            label_vel_lis[i] = label_vel_lis[i][target_keep_index]
            goal_pos_lis[i] = goal_pos_lis[i][target_keep_index]
            label_mask_lis[i] = label_mask_lis[i][target_keep_index]

    neighbor_size = np.array([int(agent_feature_lis[i].shape[0]) for i in range(len(agent_feature_lis))])

    out_normal_lis = []
    out_graph_lis = []
    out_label_pos_lis = []
    out_label_vel_lis = []
    out_goal_pos_lis = []
    out_label_mask_lis = []

    for i in range(len(agent_feature_lis)):
        # Compute agent map size (values are already in SCALED coordinates)
        # Velocities are per-frame displacements
        all_agent_obs_final_v = np.sqrt(
            agent_feature_lis[i][:, -1, 2]**2 +
            agent_feature_lis[i][:, -1, 3]**2
        )
        all_agent_map_size = np.vectorize(map_size_lis.__getitem__)(agent_type_lis[i])
        # Expand interaction range based on velocity
        all_agent_map_size = all_agent_obs_final_v * 2.0 + all_agent_map_size

        # Generate graph
        graph_dic = generate_heterogeneous_graph(agent_feature_lis[i], all_agent_map_size)
        g = dgl.heterograph(data_dict=graph_dic["uv_dic"])

        now_agent_feature = agent_feature_lis[i]
        now_agent_type = agent_type_lis[i]

        # Self-loop edges
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
        g.edges["self"].data['a_e_type'] = torch.as_tensor((now_agent_type[type0_u].ravel() - 1).astype(np.int32)).long()

        # Agent-to-agent edges
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
            g.edges["other"].data['a_e_type'] = torch.as_tensor((now_agent_type[type1_u].ravel() - 1).astype(np.int32)).long()
        else:
            g.edges["other"].data['a_e_fea'] = torch.zeros((0, 4))
            g.edges["other"].data['a_e_type'] = torch.zeros((0,)).long()

        # Process target agents
        now_pred_num = pred_num_lis[i]
        selected_pred_indices = list(range(0, now_pred_num))
        non_pred_indices = list(range(now_pred_num, now_agent_feature.shape[0]))

        now_full_agent_n_feature = now_agent_feature[selected_pred_indices].copy()
        ref_coor = now_full_agent_n_feature[:, -1, :2].copy()
        
        # Position labels (relative to last observation) - SCALED
        now_label_pos = label_pos_lis[i][selected_pred_indices].copy()
        now_label_pos = now_label_pos - ref_coor[:, np.newaxis, :]

        # Velocity labels - SCALED, per-frame
        now_label_vel = label_vel_lis[i][selected_pred_indices].copy()

        # Goal positions (final timestep) - SCALED
        now_goal_pos = goal_pos_lis[i][selected_pred_indices].copy()
        now_goal_pos = now_goal_pos - ref_coor  # Make relative to last observation

        ref_psi = now_full_agent_n_feature[:, -1, 4:5].copy()
        normal_val = np.concatenate([ref_coor, ref_psi], axis=-1)
        out_normal_lis.append(normal_val)

        sin_theta = np.sin(-ref_psi)
        cos_theta = np.cos(-ref_psi)

        # Rotate position labels
        rotate(now_label_pos, cos_theta, sin_theta)

        # Rotate velocity labels
        rotate(now_label_vel, cos_theta, sin_theta)

        # Rotate goal positions (add dimension for rotation, then remove)
        now_goal_pos = now_goal_pos[:, np.newaxis, :]  # (N, 2) -> (N, 1, 2)
        rotate(now_goal_pos, cos_theta, sin_theta)
        now_goal_pos = now_goal_pos[:, 0, :]  # (N, 1, 2) -> (N, 2)

        # Normalize agent features
        now_full_agent_n_feature = normal_agent_feature(
            now_full_agent_n_feature, ref_coor, ref_psi, cos_theta, sin_theta
        )

        # Process context agents
        now_all_agent_n_feature = now_full_agent_n_feature
        if now_pred_num < now_agent_feature.shape[0]:
            now_other_agent_n_feature = now_agent_feature[non_pred_indices].copy()
            other_ref_coor = now_other_agent_n_feature[:, -1, :2]
            other_ref_psi = now_other_agent_n_feature[:, -1, 4:5].copy()
            other_sin_theta = np.sin(-other_ref_psi)
            other_cos_theta = np.cos(-other_ref_psi)
            now_other_agent_n_feature = normal_agent_feature(
                now_other_agent_n_feature, other_ref_coor, other_ref_psi, other_cos_theta, other_sin_theta
            )
            now_all_agent_n_feature = np.concatenate([now_all_agent_n_feature, now_other_agent_n_feature], axis=0)

        # Assign to graph
        g.nodes["agent"].data["a_n_fea"] = torch.as_tensor(now_all_agent_n_feature.astype(np.float32))
        g.nodes["agent"].data["a_n_type"] = torch.as_tensor((now_agent_type - 1).astype(np.int32)).long()

        out_graph_lis.append(g)
        out_label_pos_lis.append(now_label_pos)
        out_label_vel_lis.append(now_label_vel)
        out_goal_pos_lis.append(now_goal_pos)
        out_label_mask_lis.append(label_mask_lis[i][selected_pred_indices])

    # Batch everything
    output_dic = {}
    output_dic["cuda_tensor_lis"] = [
        "graph_lis", "label_lis", "label_pos_lis", "goal_pos_lis", "label_mask_lis"
    ]

    # Velocity labels for diffusion training (SCALED, per-frame)
    output_dic["label_lis"] = torch.as_tensor(np.concatenate(out_label_vel_lis, axis=0).astype(np.float32))

    # Position labels for evaluation metrics (SCALED - will be de-scaled in train.py)
    output_dic["label_pos_lis"] = torch.as_tensor(np.concatenate(out_label_pos_lis, axis=0).astype(np.float32))

    # Goal positions for goal-conditioned prediction (SCALED, in ego-centric coordinates)
    output_dic["goal_pos_lis"] = torch.as_tensor(np.concatenate(out_goal_pos_lis, axis=0).astype(np.float32))

    output_dic["label_mask_lis"] = torch.as_tensor(np.concatenate(out_label_mask_lis, axis=0).astype(np.float32))

    # Batch graphs
    output_g = dgl.batch(out_graph_lis)

    # Create edge type dictionary
    a_e_type_dict = {}
    for out_etype in ["self", "other"]:
        a_e_type_dict[out_etype] = []
        for agent_type_index in range(6):
            a_e_type_dict[out_etype].append(
                torch.where(output_g.edges[out_etype].data["a_e_type"] == agent_type_index)[0]
            )

    # Create node type list
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

    del batch
    return output_dic
