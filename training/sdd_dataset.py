"""
SDD dataset loader: windows of tracks plus the scene's map elements, batched
into one heterogeneous graph per window for the HDGT-style model.

Conventions
- All velocities are PER-FRAME displacements, backward-looking:
  vel[t] = pos[t] - pos[t-1], so observed and future velocities meet without a
  discontinuity at the seam.
- Coordinates are divided by SCALE_FACTOR for numerical stability; every metric
  is reported back in original pixels.
- Each agent's features and labels are expressed in its OWN frame: the origin
  is its last observed position, the x axis its heading there.

Track data
- sdd/train/<video>.pickle (52 videos, 8 of them held out as validation) and
  sdd/test/<video>.pickle (8 videos, the final holdout).
- Each is a pandas DataFrame with columns
  ['trackId', 'frame', 'label', 'x', 'y', 'sceneId', 'metaId', 'xmin', 'ymin', 'xmax', 'ymax'].
- Sampled at 2.5 FPS (every 12th frame of 30 FPS = 0.4 s), 8 observed steps
  and 12 predicted ones.
- Each metaId is one 20-timestep trajectory. One sample per distinct trajectory
  start frame, whose targets are the trajectories starting there (the same
  windows as original_hdgt_on_sdd), so every trajectory is predicted once.
- Neighbours as in HDGT (and original_hdgt_on_sdd): every other agent seen
  during the observation, its observed steps without a sample filled (see
  fill_observation).

Map elements (hand-annotated with draw_shapes.py):
- sdd/shapes/<scene>_video<N>.pickle, one per video, same name as the track
  pickle. One row per rectangle / circle, in raw image pixels; see
  draw_shapes.py for the geometry convention.
- Classes: grass (pedestrians may walk on it, nobody else may) and obstacle
  (nobody may be inside it).
- Every sample's graph gets its video's shapes as "map" nodes, with map->agent
  and map->map edges. Map nodes are context only: nothing is predicted for them.
- A car, bus or cart whose every position in the video is inside the drawn
  obstacles is parked: that obstacle is how it is seen, and it is NOT an agent,
  so nothing is predicted for it either (see PARKED_TYPES).
- The collate fn also hands the model each target agent's forbidden shapes in
  that agent's own frame, for the optional map penalties on goals and
  trajectories (all off by default: the map reaches the model through the
  graph only).
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


# Agent type name mapping
AGENT_TYPE_NAMES = {
    1: "Pedestrian",
    2: "Biker",
    3: "Car",
    4: "Bus",
    5: "Cart",
    6: "Skater"
}
PEDESTRIAN_TYPE = 1


# =============================================================================
# MAP ELEMENTS
#
# Map node class (m_n_type) and what each class forbids:
#   0 grass     forbidden for every agent type except Pedestrian
#   1 obstacle  forbidden for every agent type
# Undrawn space is free for everyone.
#
# World-frame geometry of a scene, one row per shape, SCALED units:
#   [cx, cy, half_w, half_h, theta, is_circle]
# A circle's radius is its half_w (== half_h) and its theta is 0. Rectangles
# are stored with the long side on their own x axis (half_w >= half_h), which
# makes (size, theta mod pi) a unique description of each rectangle, so the
# orientation features below never see two encodings of one shape.
# =============================================================================
# Parked vehicles. A car, bus or cart the user drew obstacles on -- EVERY
# position of its trackId in the video lies inside the drawn obstacle shapes --
# is scene furniture, not traffic: that drawn obstacle is its map node, it is
# not an agent node, and nothing is predicted for it, so it also leaves the
# metrics. How much it moves does not matter (SDD's box around a parked car
# shifts 1-4 px between frames), but a vehicle outside the obstacles at even one
# position is an agent, so a moving one whose annotation strays into an
# obstacle (a camera jump, a misplaced box) is still predicted. Other agent
# types are never treated this way. The same rule in every version (keep them
# in sync with parked_track_ids in the others).
PARKED_TYPES = (3, 4, 5)        # Car, Bus, Cart

MAP_CLASS_NAMES = {0: "Grass", 1: "Obstacle"}
MAP_LABEL_TO_CLASS = {"grass": 0, "obstacle": 1}
MAP_GRASS, MAP_OBSTACLE = 0, 1
NUM_MAP_TYPES = 2
G_CX, G_CY, G_HW, G_HH, G_TH, G_CIRC = range(6)

# Map -> agent edges. An agent is linked to every shape whose NEAREST POINT
# (not its centre: a 350 px wall is relevant long before its centre is) lies
# within  base range for its type + the distance it covers over the prediction
# horizon at its current speed. From the train split, 4.8 s travel at p90 is
# ~200 px for pedestrians, ~500 px for bikers, ~420 px for skaters and ~350 px
# for carts; the speed term tracks that per agent, and the base covers turning
# and speeding up. Shapes the agent stands inside have distance < 0 and are
# always linked. SCALED units (x50 = pixels).
#
# Every base but the pedestrian's was raised by 50 px together with a widening of
# map_size_lis. map_size_lis has since been reverted to hdgt_goal's ranges; these
# map -> agent ranges were left as they are.
MAP_EDGE_BASE_RANGE = {
    1: 2.0,   # Pedestrian: 100 px (unchanged)
    2: 4.0,   # Biker:      200 px (was 150)
    3: 4.0,   # Car:        200 px (was 150)
    4: 5.0,   # Bus:        250 px (was 200)
    5: 4.0,   # Cart:       200 px (was 150)
    6: 4.0,   # Skater:     200 px (was 150)
}
MAX_MAP_EDGE_PER_AGENT = 32
# Map -> map edges join shapes whose gap is at most 25 px, so the pieces of one
# composite region (a building drawn as three rectangles, a lawn drawn as
# several circles) and tightly packed neighbours can exchange context.
M2M_GAP_THRESHOLD = 0.5
MAX_M2M_EDGE_PER_NODE = 16
# Boundary sampling step for the map->map gap, SCALED (2 px).
_BOUNDARY_STEP = 0.04

# Feature widths, mirrored in model_goal_multigoal.py.
MAP_NODE_FEA_DIM = 3    # [w, h, is_circle], own frame, SCALED
M2A_EDGE_FEA_DIM = 7    # see build_map_agent_edges
M2M_EDGE_FEA_DIM = 5    # see build_map_map_edges
MAP_LOSS_GEOM_DIM = 7   # [cx, cy, half_w, half_h, cos, sin, is_circle], agent frame


def _wrap_half_pi(theta):
    """Wrap to [-pi/2, pi/2): a rectangle turned half a circle is unchanged."""
    return (theta + np.pi / 2.0) % np.pi - np.pi / 2.0


def _canonicalise_rect(half_w, half_h, theta):
    """Put the long side on the rectangle's own x axis.

    Turning a rectangle a quarter turn and swapping its sides is the same
    rectangle, so containment is unchanged, and (size, theta) then describes
    each rectangle exactly once -- the orientation features never see two
    encodings of one shape.
    """
    swap = half_h > half_w
    return (np.where(swap, half_h, half_w), np.where(swap, half_w, half_h),
            _wrap_half_pi(np.where(swap, theta + np.pi / 2.0, theta)))


def shape_sdf_np(points, geom):
    """Signed distance from points to shapes, negative inside.

    points (..., 2) and geom (S, 6) in the same frame -> (..., S). A point is
    inside exactly when draw_shapes.contains() says so: sd <= 0 iff
    |u| <= half_w and |v| <= half_h (rectangle) or |p - c| <= r (circle).
    """
    p = np.asarray(points, dtype=np.float64)[..., np.newaxis, :]
    dx = p[..., 0] - geom[:, G_CX]
    dy = p[..., 1] - geom[:, G_CY]
    c, s = np.cos(geom[:, G_TH]), np.sin(geom[:, G_TH])
    u = dx * c + dy * s
    v = -dx * s + dy * c
    qx = np.abs(u) - geom[:, G_HW]
    qy = np.abs(v) - geom[:, G_HH]
    sd_rect = (np.hypot(np.maximum(qx, 0.0), np.maximum(qy, 0.0))
               + np.minimum(np.maximum(qx, qy), 0.0))
    sd_circ = np.hypot(dx, dy) - geom[:, G_HW]
    return np.where(geom[:, G_CIRC] > 0.5, sd_circ, sd_rect)


def shape_nearest_boundary_np(points, geom):
    """Nearest point ON THE BOUNDARY of each shape, (P, 2) x (S, 6) -> (P, S, 2).

    From outside that is the nearest point of the shape; from inside it is the
    nearest way out.
    """
    p = np.asarray(points, dtype=np.float64)[:, np.newaxis, :]
    dx = p[..., 0] - geom[:, G_CX]
    dy = p[..., 1] - geom[:, G_CY]
    c, s = np.cos(geom[:, G_TH]), np.sin(geom[:, G_TH])
    hw, hh = geom[:, G_HW], geom[:, G_HH]
    u = dx * c + dy * s
    v = -dx * s + dy * c
    inside = (np.abs(u) <= hw) & (np.abs(v) <= hh)
    su = np.where(u >= 0, 1.0, -1.0)
    sv = np.where(v >= 0, 1.0, -1.0)
    exit_u = (hw - np.abs(u)) <= (hh - np.abs(v))
    bu = np.where(inside, np.where(exit_u, su * hw, u), np.clip(u, -hw, hw))
    bv = np.where(inside, np.where(exit_u, v, sv * hh), np.clip(v, -hh, hh))
    rect = np.stack([geom[:, G_CX] + bu * c - bv * s,
                     geom[:, G_CY] + bu * s + bv * c], axis=-1)
    n = np.hypot(dx, dy)
    safe = np.where(n > 1e-9, n, 1.0)
    ux = np.where(n > 1e-9, dx / safe, 1.0)
    uy = np.where(n > 1e-9, dy / safe, 0.0)
    circ = np.stack([geom[:, G_CX] + hw * ux, geom[:, G_CY] + hw * uy], axis=-1)
    return np.where((geom[:, G_CIRC] > 0.5)[:, np.newaxis], circ, rect)


def _boundary_samples(g, step=_BOUNDARY_STEP):
    """Points along one shape's outline, at most `step` apart, corners included."""
    if g[G_CIRC] > 0.5:
        n = max(16, int(np.ceil(2.0 * np.pi * g[G_HW] / step)))
        a = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
        return np.stack([g[G_CX] + g[G_HW] * np.cos(a),
                         g[G_CY] + g[G_HW] * np.sin(a)], axis=-1)
    hw, hh = g[G_HW], g[G_HH]
    corners = np.array([[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh], [-hw, -hh]])
    pts = []
    for a, b in zip(corners[:-1], corners[1:]):
        n = max(2, int(np.ceil(np.linalg.norm(b - a) / step)) + 1)
        pts.append(a + np.linspace(0.0, 1.0, n)[:, np.newaxis] * (b - a))
    loc = np.concatenate(pts, axis=0)
    c, s = np.cos(g[G_TH]), np.sin(g[G_TH])
    return np.stack([g[G_CX] + loc[:, 0] * c - loc[:, 1] * s,
                     g[G_CY] + loc[:, 0] * s + loc[:, 1] * c], axis=-1)


def _orientation_feature(theta_rel, is_circle):
    """(cos 2d, sin 2d) of a rectangle's relative angle d: continuous across the
    theta wrap and blind to a half turn, which leaves a rectangle unchanged.
    A circle has no orientation and gets (0, 0)."""
    return np.where(is_circle, 0.0, np.cos(2.0 * theta_rel)), \
           np.where(is_circle, 0.0, np.sin(2.0 * theta_rel))


_EMPTY_M2M = (np.zeros(0, np.int64), np.zeros(0, np.int64),
              np.zeros((0, M2M_EDGE_FEA_DIM), np.float32))


def _m2m_features(geom, src, dst, gap):
    """Edge features in the DESTINATION shape's own frame (centre, theta):
        [rel_cx, rel_cy, gap, cos 2(th_src - th_dst), sin 2(...)]
    the orientation pair is (0, 0) when either end is a circle."""
    d = geom[src, :2] - geom[dst, :2]
    c, s = np.cos(-geom[dst, G_TH]), np.sin(-geom[dst, G_TH])
    either_circle = (geom[src, G_CIRC] > 0.5) | (geom[dst, G_CIRC] > 0.5)
    ocos, osin = _orientation_feature(geom[src, G_TH] - geom[dst, G_TH], either_circle)
    fea = np.stack([d[:, 0] * c - d[:, 1] * s,
                    d[:, 1] * c + d[:, 0] * s,
                    gap[src, dst], ocos, osin], axis=-1)
    return fea.astype(np.float32)


def _m2m_edges_from_gap(geom, gap, keep=None):
    """Edges under M2M_GAP_THRESHOLD, nearest first, capped per destination.
    keep(src, dst) -> bool filters candidate pairs."""
    src, dst = [], []
    for j in range(geom.shape[0]):
        near = np.where(gap[:, j] <= M2M_GAP_THRESHOLD)[0]
        if keep is not None and len(near):
            near = near[keep(near, j)]
        if len(near) > MAX_M2M_EDGE_PER_NODE:
            near = near[np.argsort(gap[near, j], kind="stable")][:MAX_M2M_EDGE_PER_NODE]
        src += near.tolist()
        dst += [j] * len(near)
    if not src:
        return _EMPTY_M2M
    src, dst = np.array(src, np.int64), np.array(dst, np.int64)
    return src, dst, _m2m_features(geom, src, dst, gap)


def build_map_map_edges(geom):
    """Map -> map edges between shapes at most M2M_GAP_THRESHOLD apart.

    The gap is the distance between the two outlines, 0 when the shapes touch,
    overlap or one contains the other. Measured on outline samples, so it is
    exact at corners and within ~1 px elsewhere.
    Returns (src, dst, fea) with fea (E, M2M_EDGE_FEA_DIM).
    """
    S = geom.shape[0]
    if S < 2:
        return _EMPTY_M2M
    gap = np.empty((S, S))
    for i in range(S):
        gap[i] = shape_sdf_np(_boundary_samples(geom[i]), geom).min(axis=0)
    gap = np.maximum(np.minimum(gap, gap.T), 0.0)
    np.fill_diagonal(gap, np.inf)
    return _m2m_edges_from_gap(geom, gap)


def _m2m_edges_touching_new(geom, n_static):
    """Map -> map edges where at least one end is a shape at index >= n_static.

    For the parked vehicles added to a window: the static shapes' own edges are
    already known, so only the pairs involving a new shape are computed.
    """
    S = geom.shape[0]
    if S < 2 or n_static >= S:
        return _EMPTY_M2M
    gap = np.full((S, S), np.inf)
    for i in range(n_static, S):
        g_i = shape_sdf_np(_boundary_samples(geom[i]), geom).min(axis=0)
        # A shape swallowed whole by i has no boundary point inside i, so its
        # outline distance misses the overlap; its centre inside i catches it.
        swallowed = shape_sdf_np(geom[:, :2], geom[i:i + 1]).ravel() < 0
        g_i = np.maximum(np.where(swallowed, 0.0, g_i), 0.0)
        gap[i] = np.minimum(gap[i], g_i)
        gap[:, i] = np.minimum(gap[:, i], g_i)      # the gap is symmetric
    np.fill_diagonal(gap, np.inf)
    return _m2m_edges_from_gap(
        geom, gap, keep=lambda near, j: (near >= n_static) | (j >= n_static))


def parked_track_ids(df, type_map, scene_map, scale_factor=SCALE_FACTOR):
    """trackIds of one video's parked vehicles: the Car, Bus and Cart tracks whose
    every row lies inside one of the video's hand-drawn OBSTACLE shapes. A
    vehicle driving past a building leaves the shape, and one waiting at a light
    is not inside a drawn shape at all.
    """
    obstacles = scene_map["geom"][scene_map["cls"] == MAP_OBSTACLE]
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


def scene_map_with_parked(scene_map, parked_geom):
    """The video's shapes plus this window's parked vehicles, as obstacles.

    "is_parked" marks which rows came from a vehicle rather than from
    draw_shapes.py — for reporting and for drawing them, not for the model.
    """
    if len(parked_geom) == 0:
        return scene_map
    parked_geom = np.asarray(parked_geom, dtype=np.float64).reshape(-1, 6)
    n_static = scene_map["geom"].shape[0]
    n_parked = parked_geom.shape[0]
    geom = np.concatenate([scene_map["geom"], parked_geom], axis=0)
    src, dst, fea = _m2m_edges_touching_new(geom, n_static)
    return {
        "scene_id": scene_map["scene_id"],
        "geom": geom,
        "cls": np.concatenate([scene_map["cls"],
                               np.full(n_parked, MAP_OBSTACLE, np.int64)]),
        "shape_id": np.concatenate([scene_map["shape_id"],
                                    -1 - np.arange(n_parked, dtype=np.int64)]),
        "node_fea": np.concatenate([
            scene_map["node_fea"],
            np.stack([2.0 * parked_geom[:, G_HW], 2.0 * parked_geom[:, G_HH],
                      parked_geom[:, G_CIRC]], axis=-1).astype(np.float32)]),
        "m2m_src": np.concatenate([scene_map["m2m_src"], src]),
        "m2m_dst": np.concatenate([scene_map["m2m_dst"], dst]),
        "m2m_fea": np.concatenate([scene_map["m2m_fea"], fea]),
        "is_parked": np.concatenate([scene_map["is_parked"],
                                     np.ones(n_parked, dtype=bool)]),
    }


def build_map_agent_edges(agent_pos, agent_psi, agent_range, geom):
    """Map -> agent edges: shape s -> agent a when sd(a, s) <= agent_range[a].

    agent_pos (N, 2) and agent_psi (N,) are each agent's last observed position
    and heading, the same reference its node features are normalised to.
    Edge features, in the agent's frame:
        [rel_cx, rel_cy, near_x, near_y, sd, cos 2(th - psi), sin 2(th - psi)]
    rel_c is the shape centre, near the nearest point on its outline, sd the
    signed distance (negative when the agent is inside); the orientation pair
    is (0, 0) for circles. Capped at MAX_MAP_EDGE_PER_AGENT, nearest first.
    Returns (src_map, dst_agent, fea).
    """
    N, S = agent_pos.shape[0], geom.shape[0]
    if S == 0 or N == 0:
        return (np.zeros(0, np.int64), np.zeros(0, np.int64),
                np.zeros((0, M2A_EDGE_FEA_DIM), np.float32))
    sd = shape_sdf_np(agent_pos, geom)                       # (N, S)
    src, dst = [], []
    for a in range(N):
        near = np.where(sd[a] <= agent_range[a])[0]
        if len(near) > MAX_MAP_EDGE_PER_AGENT:
            near = near[np.argsort(sd[a, near], kind="stable")][:MAX_MAP_EDGE_PER_AGENT]
        src += near.tolist()
        dst += [a] * len(near)
    if not src:
        return (np.zeros(0, np.int64), np.zeros(0, np.int64),
                np.zeros((0, M2A_EDGE_FEA_DIM), np.float32))
    src, dst = np.array(src, np.int64), np.array(dst, np.int64)

    nearest = shape_nearest_boundary_np(agent_pos, geom)[dst, src]   # (E, 2)
    c, s = np.cos(-agent_psi[dst]), np.sin(-agent_psi[dst])
    rc = geom[src, :2] - agent_pos[dst]
    rn = nearest - agent_pos[dst]
    ocos, osin = _orientation_feature(geom[src, G_TH] - agent_psi[dst],
                                      geom[src, G_CIRC] > 0.5)
    fea = np.stack([rc[:, 0] * c - rc[:, 1] * s, rc[:, 1] * c + rc[:, 0] * s,
                    rn[:, 0] * c - rn[:, 1] * s, rn[:, 1] * c + rn[:, 0] * s,
                    sd[dst, src], ocos, osin], axis=-1)
    return src, dst, fea.astype(np.float32)


def densify_path_np(pos, start):
    """(A, T, 2) positions + (A, 2) start -> (A, 2T, 2): each segment's midpoint
    followed by its end point, so a path cannot step over a thin wall between
    two samples. Same point set as model_goal_multigoal.densify_path."""
    prev = np.concatenate([start[:, np.newaxis, :], pos[:, :-1, :]], axis=1)
    mid = 0.5 * (prev + pos)
    return np.stack([mid, pos], axis=2).reshape(pos.shape[0], 2 * pos.shape[1], 2)


def forbidden_shapes(map_cls, agent_types):
    """(A, S) bool: obstacles for everyone, grass for everyone but pedestrians."""
    obstacle = (map_cls == MAP_OBSTACLE)[np.newaxis, :]
    grass = (map_cls == MAP_GRASS)[np.newaxis, :]
    wheeled = (np.asarray(agent_types) != PEDESTRIAN_TYPE)[:, np.newaxis]
    return obstacle | (grass & wheeled)


def load_scene_map(shapes_folder, scene_id, scale_factor=SCALE_FACTOR):
    """Load <shapes_folder>/<scene_id>.pickle into the arrays the loader uses.

    Refuses a file whose rows name a different video, an unknown class or shape
    kind, or a non-finite value, so a misplaced or stale file fails loudly
    instead of silently pairing one video's tracks with another's map.
    """
    path = os.path.join(shapes_folder, scene_id + ".pickle")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"No map for {scene_id}: expected {path}. Every track pickle needs a "
            f"shapes pickle of the same name (draw_shapes.py writes them).")
    df = pd.read_pickle(path)

    if len(df) and set(df["sceneId"].unique()) != {scene_id}:
        raise ValueError(f"{path} holds shapes of {sorted(df['sceneId'].unique())}, "
                         f"not {scene_id}")
    bad_label = set(df["label"].unique()) - set(MAP_LABEL_TO_CLASS)
    if bad_label:
        raise ValueError(f"{path}: unknown map classes {sorted(bad_label)}")
    bad_shape = set(df["shape"].unique()) - {"rectangle", "circle"}
    if bad_shape:
        raise ValueError(f"{path}: unknown shape kinds {sorted(bad_shape)}")

    is_circ = (df["shape"].values == "circle")
    radius = df["radius"].values.astype(np.float64)
    hw = np.where(is_circ, radius, df["half_w"].values.astype(np.float64))
    hh = np.where(is_circ, radius, df["half_h"].values.astype(np.float64))
    th = np.where(is_circ, 0.0, df["theta"].values.astype(np.float64))
    hw_c, hh_c, th_c = _canonicalise_rect(hw, hh, th)
    hw = np.where(is_circ, hw, hw_c)      # a circle has no long side
    hh = np.where(is_circ, hh, hh_c)
    th = np.where(is_circ, 0.0, th_c)

    geom = np.stack([df["cx"].values / scale_factor, df["cy"].values / scale_factor,
                     hw / scale_factor, hh / scale_factor, th,
                     is_circ.astype(np.float64)], axis=-1).reshape(-1, 6)
    if not np.all(np.isfinite(geom)) or np.any(geom[:, G_HW] <= 0) or np.any(geom[:, G_HH] <= 0):
        raise ValueError(f"{path}: non-finite or non-positive shape geometry")

    cls = np.array([MAP_LABEL_TO_CLASS[l] for l in df["label"].values], dtype=np.int64)
    m2m_src, m2m_dst, m2m_fea = build_map_map_edges(geom)
    return {
        "scene_id": scene_id,
        "geom": geom,                                   # (S, 6) world, SCALED
        "cls": cls,                                     # (S,) 0 grass / 1 obstacle
        "shape_id": df["shapeId"].values.astype(np.int64),
        "node_fea": np.stack([2.0 * geom[:, G_HW], 2.0 * geom[:, G_HH],
                              geom[:, G_CIRC]], axis=-1).astype(np.float32),
        "m2m_src": m2m_src, "m2m_dst": m2m_dst, "m2m_fea": m2m_fea,
        # All drawn here; scene_map_with_parked() appends this window's parked
        # vehicles with is_parked True.
        "is_parked": np.zeros(len(cls), dtype=bool),
    }


class SDDDataset(Dataset):
    def __init__(self, pickle_files, obs_len=8, pred_len=12, shapes_folder=None):
        """
        Args:
            pickle_files: List of pickle file paths
            obs_len: Number of observation frames (default 8)
            pred_len: Number of prediction frames (default 12)
            shapes_folder: folder of per-video map pickles. Default: "shapes"
                next to the folder holding each track pickle (../sdd/shapes).
        """
        self.shapes_folder = shapes_folder
        self.scene_maps = {}
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
            # The map is looked up by the track pickle's own video name, so a
            # video can only ever be paired with the shapes drawn on it.
            shapes_folder = self.shapes_folder or os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(pkl_file))), "shapes")
            scene_map = load_scene_map(shapes_folder, scene_id, self.scale_factor)
            self.scene_maps[scene_id] = scene_map
            n_before = len(self.samples)

            # Parked vehicles are obstacles (the shapes drawn on them), not agents:
            # their rows go before any window is cut
            parked = df['trackId'].isin(parked_track_ids(df, self.type_map, scene_map, self.scale_factor))
            self.parked_count += df.loc[parked, 'metaId'].nunique()
            df = df[~parked]
            tracks = self._tracks_of_scene(df)

            # All rows sorted by frame, to find the agents seen during an observation period
            order = np.argsort(df['frame'].values, kind='stable')
            row_frames = df['frame'].values[order]
            row_track_ids = df['trackId'].values[order]

            # Windows as in original_hdgt_on_sdd: each metaId is one 20-timestep
            # trajectory, sampled every 12 frames from its own first frame. One
            # sample per distinct trajectory start frame; its window is
            # start + 12*k, and the trajectories that start there are its targets,
            # so every trajectory is predicted exactly once. Windows overlap, so
            # split by video, never by window.
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
                                                        obs_frames, scene_id, scene_map,
                                                        set(starting_here['trackId']))
                if sample is not None:
                    self.samples.append(sample)

            n_obst = int((scene_map["cls"] == MAP_OBSTACLE).sum())
            n_grass = int((scene_map["cls"] == MAP_GRASS).sum())
            print(f"  [{pkl_idx+1:2d}/{len(pickle_files)}] {scene_id:28s} -> {len(self.samples):6d} total samples"
                  f"  (+{len(self.samples) - n_before:4d}) | map: {n_obst:2d} obstacle, {n_grass:2d} grass,"
                  f" {len(scene_map['m2m_src']):3d} map-map edges")

        print(f"\n{'='*80}")
        print(f"Dataset loading complete: {len(self.samples)} total samples")
        print(f"Parked cars/buses/carts: {self.parked_count} trajectories, every position of their "
              f"tracks inside the\nobstacles drawn in sdd/shapes. That obstacle is their map node; "
              f"they are NOT agents and NOTHING\nis predicted for them, so they are absent "
              f"from the counts below and from every metric.")
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
                                  scene_id, scene_map, starting_track_ids):
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
        # Always empty: a parked vehicle is an obstacle through the shape the user
        # drew on it, so no rectangle is generated here. Kept so
        # scene_map_with_parked stays a no-op rather than a second code path.
        parked_geoms = []

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
            # The video's shapes plus this window's parked vehicles. Without any
            # parked vehicle this is the shared map itself, not a copy.
            'scene_map': scene_map_with_parked(scene_map, parked_geoms),
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


def generate_heterogeneous_graph(agent_fea, agent_map_size_lis, agent_type=None,
                                 scene_map=None, pred_len=12):
    """Build the heterogeneous graph with proximity-based edges.

    Always: agent self-loops and agent -> agent ("other") edges.
    With scene_map: map -> agent ("m2a") and map -> map ("m2m") edges too, plus
    their features, returned under "m2a_fea" / "m2m_fea".
    """
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

    out = {}
    if scene_map is not None:
        # Map -> agent: range grows with the distance the agent covers over the
        # prediction horizon at its current (per-frame) speed.
        speed = np.hypot(agent_fea[:, -1, 2], agent_fea[:, -1, 3])
        base = np.array([MAP_EDGE_BASE_RANGE[int(t)] for t in agent_type], dtype=np.float64)
        m2a_src, m2a_dst, m2a_fea = build_map_agent_edges(
            agent_fea[:, -1, :2], agent_fea[:, -1, 4], base + pred_len * speed,
            scene_map["geom"])
        uv_dic[("map", "m2a", "agent")] = [m2a_src.tolist(), m2a_dst.tolist()]
        # Map -> map: static per video, precomputed at load time.
        uv_dic[("map", "m2m", "map")] = [scene_map["m2m_src"].tolist(),
                                         scene_map["m2m_dst"].tolist()]
        out["m2a_fea"] = m2a_fea
        out["m2m_fea"] = scene_map["m2m_fea"]

    for key in uv_dic:
        uv_dic[key] = (torch.LongTensor(uv_dic[key][0]), torch.LongTensor(uv_dic[key][1]))

    out["uv_dic"] = uv_dic
    return out


def build_map_loss_geometry(scene_map, target_types, ref_coor, ref_psi, gt_pos,
                            margin=0.0):
    """Per target agent: every shape of its video in the AGENT'S frame, plus
    which of them its goals and trajectories are penalised for entering.

    Args:
        target_types (A,), ref_coor (A, 2), ref_psi (A,): each target's type and
            the frame its labels and predictions live in (last observed
            position and heading, world SCALED).
        gt_pos (A, T, 2): GT future, world SCALED.
        margin: penalty margin, SCALED; must match the model's map_margin.
    Returns:
        geom   (A, S, MAP_LOSS_GEOM_DIM)  [cx, cy, half_w, half_h, cos, sin, is_circle]
        forbid (A, S) bool  the shape is off limits for this agent's type
        loss   (A, S) bool  forbid, minus shapes this agent's own GT path enters

    Why "loss" drops shapes the GT enters: parked cars sit inside the parking
    areas drawn as obstacles (in nexus_video11, 99.8% of stationary car points
    are inside an obstacle), and some paths clip an outline by a pixel or two.
    Penalising a prediction for going where the GT itself goes would fight the
    regression losses, so those shapes are exempt for that one agent. The test
    is run on exactly the point set the model penalises (densified path, same
    margin), which guarantees the GT itself has zero map penalty.
    """
    geom_w, cls = scene_map["geom"], scene_map["cls"]
    A, S = len(target_types), geom_w.shape[0]
    if S == 0:
        return (np.zeros((A, 0, MAP_LOSS_GEOM_DIM), np.float32),
                np.zeros((A, 0), bool), np.zeros((A, 0), bool))
    forbid = forbidden_shapes(cls, target_types)
    path = densify_path_np(gt_pos, ref_coor)                     # (A, 2T, 2)
    gt_touches = (shape_sdf_np(path, geom_w) < margin).any(axis=1)   # (A, S)

    psi = np.asarray(ref_psi, dtype=np.float64).reshape(-1)
    d = geom_w[np.newaxis, :, :2] - ref_coor[:, np.newaxis, :]   # (A, S, 2)
    c, s = np.cos(-psi)[:, np.newaxis], np.sin(-psi)[:, np.newaxis]
    th = geom_w[np.newaxis, :, G_TH] - psi[:, np.newaxis]
    geom = np.stack([
        d[..., 0] * c - d[..., 1] * s,
        d[..., 1] * c + d[..., 0] * s,
        np.broadcast_to(geom_w[:, G_HW], (A, S)),
        np.broadcast_to(geom_w[:, G_HH], (A, S)),
        np.cos(th), np.sin(th),
        np.broadcast_to(geom_w[:, G_CIRC], (A, S)),
    ], axis=-1)
    return geom.astype(np.float32), forbid, forbid & ~gt_touches


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
# How far an agent sees OTHER AGENTS, by its own type. SCALED (x50 = pixels),
# and the velocity term in the collate fn adds 2 x the current per-frame speed
# on top, so a moving agent always sees further than a standing one.
#
# Identical to hdgt_goal (and every other version), so the agent-agent
# interaction range is not a difference between versions. A +50 px widening of
# every type but the pedestrian (275/300/300/250/275 px) was reverted to match.
# =============================================================================
map_size_lis = {
    1: 3.0,   # Pedestrian: 150 px
    2: 4.5,   # Biker:      225 px
    3: 5.0,   # Car:        250 px
    4: 5.0,   # Bus:        250 px
    5: 4.0,   # Cart:       200 px
    6: 4.5    # Skater:     225 px
}


@torch.no_grad()
def SDD_collate_fn(batch, args, is_train):
    """Collate function to batch samples and build graphs for diffusion model"""
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None

    agent_drop = args.agent_drop if hasattr(args, 'agent_drop') else 0.0
    # Map nodes/edges in the graph. The map loss geometry below is built either
    # way (it is cheap), so the map penalties and metrics work without them too.
    use_map = bool(getattr(args, 'use_map', False))
    map_margin = 0.0   # a point outside every shape is never penalised

    scene_map_lis = [item["scene_map"] for item in batch]
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
    out_map_geom_lis = []
    out_map_forbid_lis = []
    out_map_loss_lis = []

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
        now_scene_map = scene_map_lis[i]
        graph_dic = generate_heterogeneous_graph(
            agent_feature_lis[i], all_agent_map_size, agent_type_lis[i],
            scene_map=now_scene_map if use_map else None,
            pred_len=args.pred_len)
        if use_map:
            # Explicit node counts: a shape with no edges is still a node.
            g = dgl.heterograph(data_dict=graph_dic["uv_dic"], num_nodes_dict={
                "agent": int(agent_feature_lis[i].shape[0]),
                "map": int(now_scene_map["geom"].shape[0])})
        else:
            g = dgl.heterograph(data_dict=graph_dic["uv_dic"])

        now_agent_feature = agent_feature_lis[i]
        now_agent_type = agent_type_lis[i]

        if use_map:
            # Map nodes: own-frame size and shape kind; the class is the node type.
            # Where a shape sits relative to an agent lives on the m2a edge,
            # the same way agent positions live on agent-agent edges.
            map_cls = now_scene_map["cls"]
            g.nodes["map"].data["m_n_fea"] = torch.as_tensor(now_scene_map["node_fea"])
            g.nodes["map"].data["m_n_type"] = torch.as_tensor(map_cls).long()
            # Edge order in g is the order the edges were given in, so the
            # features line up with g.edges(etype=...) by construction.
            m2a_u, _ = g.edges(etype="m2a")
            g.edges["m2a"].data["m2a_e_fea"] = torch.as_tensor(graph_dic["m2a_fea"]).reshape(-1, M2A_EDGE_FEA_DIM)
            g.edges["m2a"].data["m2a_e_type"] = torch.as_tensor(map_cls[m2a_u.numpy()]).long().reshape(-1)
            m2m_u, _ = g.edges(etype="m2m")
            g.edges["m2m"].data["m2m_e_fea"] = torch.as_tensor(graph_dic["m2m_fea"]).reshape(-1, M2M_EDGE_FEA_DIM)
            g.edges["m2m"].data["m2m_e_type"] = torch.as_tensor(map_cls[m2m_u.numpy()]).long().reshape(-1)

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

        # Map penalty geometry for the targets, in each target's own frame.
        # label_pos_lis[i] is still the untouched WORLD GT (only copies were
        # made relative above), which the exemption test needs.
        map_geom, map_forbid, map_loss = build_map_loss_geometry(
            now_scene_map, now_agent_type[selected_pred_indices], ref_coor,
            ref_psi[:, 0], label_pos_lis[i][selected_pred_indices], margin=map_margin)
        out_map_geom_lis.append(map_geom)
        out_map_forbid_lis.append(map_forbid)
        out_map_loss_lis.append(map_loss)

    # Batch everything
    output_dic = {}
    output_dic["cuda_tensor_lis"] = [
        "graph_lis", "label_lis", "label_pos_lis", "goal_pos_lis", "label_mask_lis",
        "map_geom_lis", "map_forbid_mask_lis", "map_loss_mask_lis",
    ]

    # Map penalty geometry, one row per target agent, padded to the largest
    # shape count in the batch. Padding has both masks False.
    s_max = max(m.shape[1] for m in out_map_geom_lis)
    geom_pad, forbid_pad, loss_pad = [], [], []
    for geom_i, forbid_i, loss_i in zip(out_map_geom_lis, out_map_forbid_lis, out_map_loss_lis):
        pad = s_max - geom_i.shape[1]
        geom_pad.append(np.pad(geom_i, ((0, 0), (0, pad), (0, 0))))
        forbid_pad.append(np.pad(forbid_i, ((0, 0), (0, pad)), constant_values=False))
        loss_pad.append(np.pad(loss_i, ((0, 0), (0, pad)), constant_values=False))
    # (N, S, 7) [cx, cy, half_w, half_h, cos, sin, is_circle], each agent's frame
    output_dic["map_geom_lis"] = torch.as_tensor(np.concatenate(geom_pad, axis=0).astype(np.float32))
    # (N, S) shape is off limits for the agent's type (for metrics)
    output_dic["map_forbid_mask_lis"] = torch.as_tensor(np.concatenate(forbid_pad, axis=0))
    # (N, S) off limits AND not entered by the agent's own GT (for the loss)
    output_dic["map_loss_mask_lis"] = torch.as_tensor(np.concatenate(loss_pad, axis=0))

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

    if use_map:
        # Map edges are grouped by the class of their SOURCE shape, the same way
        # agent edges are grouped by their source agent's type.
        for out_etype, fea_name in (("m2a", "m2a_e_type"), ("m2m", "m2m_e_type")):
            a_e_type_dict[out_etype] = [
                torch.where(output_g.edges[out_etype].data[fea_name] == map_type_index)[0]
                for map_type_index in range(NUM_MAP_TYPES)
            ]
        output_dic["m_n_type_lis"] = [
            torch.where(output_g.nodes["map"].data["m_n_type"] == _)[0]
            for _ in range(NUM_MAP_TYPES)
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
