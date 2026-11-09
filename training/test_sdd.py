#!/usr/bin/env python3
"""
Standalone test evaluation for the HDGT Goal-CVAE model.

Loads a checkpoint, runs inference on the chosen split, and saves a full
metrics report as <checkpoint>_<split>_report.txt next to the checkpoint.

Defaults to the VAL split (the 8 VAL_VIDEOS held out of the train folder).
The test folder is the final holdout: run it once, at the end, with --split test.

The report adds a GOAL HEAD block (the K goals in slot order: usage, collapse,
very-off goals), with the goal distribution (hdgt_goal_heatmap) a GOAL
DISTRIBUTION block (how much probability the true goal gets, peaks, spread,
best-of-k by rank, misses, tails), and a MAP COMPLIANCE block (paths entering
shapes their agent may not enter). --goal_pick and its settings override how a
goal-distribution checkpoint picks its K goals, to tune on val without
retraining. With --baseline_checkpoint (e.g. a WTA checkpoint of hdgt_goal or
hdgt_goal_sinkhorn; mtp and map nodes are read from its weights) both
checkpoints are evaluated on the same trajectories with the same sampling
noise, and a GOAL PREDICTOR vs BASELINE block says, metric by metric and per
agent type with a paired bootstrap, whether the new goal predictor is better.

Usage:
    python test_sdd.py --checkpoint logs/Hdgt_goal_DATE/snapshot/best_min_fde.pt
    python test_sdd.py --checkpoint logs/Hdgt_goal_DATE/snapshot/best_min_fde.pt \
        --baseline_checkpoint ../hdgt_goal/logs/Hdgt_goal_DATE/snapshot/best_min_fde.pt
    python test_sdd.py --checkpoint logs/Hdgt_goal_DATE/snapshot/best_min_fde.pt --split test

All architecture flags default to the same values used in train.py.
Pass any flag explicitly if you trained with a non-default value.
"""

import os
import sys
import copy
import time
import random
import argparse
import importlib
import numpy as np
import torch
from torch.utils.data import DataLoader
from functools import partial

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["DGLBACKEND"] = "pytorch"

from sdd_dataset import (SDDDataset, SDD_collate_fn, SCALE_FACTOR, AGENT_TYPE_NAMES,
                         resolve_split_files)
from sdd_dataset import BalancedBatchSampler


# ---------------------------------------------------------------------------
# Metric helpers (identical to train.py)
# ---------------------------------------------------------------------------

def euclid_np(label, pred):
    return np.sqrt((label[..., 0] - pred[..., 0]) ** 2 + (label[..., 1] - pred[..., 1]) ** 2)


def cal_ade_fde(labels, preds, masks, scale_factor=1.0):
    if labels.shape[0] == 0:
        return None, None
    l2 = euclid_np(labels, preds) * scale_factor
    masks_sum = masks.sum(1)
    ade_idx = masks_sum != 0
    ade = ((l2[ade_idx] * masks[ade_idx]).sum(1) / masks_sum[ade_idx]).mean()
    fde_idx = masks[:, -1] != 0
    fde = l2[fde_idx, -1].mean() if fde_idx.sum() != 0 else 0.0
    return [ade, fde], [ade_idx.sum(), fde_idx.sum()]


def cal_min_ade_fde(preds, labels, masks, scale_factor=1.0):
    if labels.shape[0] == 0:
        return None, None
    l2 = euclid_np(labels[:, np.newaxis, :, :], preds) * scale_factor
    masks_sum = masks.sum(1)
    ade_idx = masks_sum != 0
    ade_per = (l2[ade_idx] * masks[ade_idx, np.newaxis, :]).sum(-1) / masks_sum[ade_idx][:, np.newaxis]
    min_ade = ade_per.min(-1).mean()
    fde_idx = masks[:, -1] != 0
    min_fde = l2[fde_idx, :, -1].min(-1).mean() if fde_idx.sum() != 0 else 0.0
    return [min_ade, min_fde], [ade_idx.sum(), fde_idx.sum()]


def chain_errors(preds, labels, masks, scale_factor=1.0):
    """Every chain's ADE and FDE, (M, K) each, in pixels. ADE over the valid steps; FDE at the
    last step. NaN where a trajectory has no valid step (ADE) or no valid last step (FDE)."""
    l2 = euclid_np(labels[:, np.newaxis, :, :], preds) * scale_factor   # (M, K, T)
    masks_sum = masks.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        ade = (l2 * masks[:, np.newaxis, :]).sum(-1) / masks_sum[:, np.newaxis]
    ade[masks_sum == 0] = np.nan
    fde = l2[:, :, -1].copy()
    fde[masks[:, -1] == 0] = np.nan
    return ade, fde


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = self.avg = self.sum = self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0


def scorer_block(scorer_rows, agent_type_lis, num_prediction, pred_len, scorer_desc):
    """The SCORER RESULTS block of the report: how well the scorer orders the K chains"""
    K = num_prediction
    rows = {at: scorer_rows[at] for at in agent_type_lis if scorer_rows[at]}
    rows = {"ALL": [r for at in rows for r in rows[at]], **rows}

    def cat(rs, key):
        return np.concatenate([r[key] for r in rs], axis=0)

    lines = [
        f"SCORER RESULTS  ({scorer_desc}, {pred_len} steps = {pred_len * 0.4:.1f}s, pixels)",
        "-" * 80,
        "  The scorer sorts the K chains; its first chain gives ADE and FDE above.",
        "  hit@n    : % of trajectories whose best chain is among the scorer's first n",
        "  meanRank : mean position of the best chain in the scorer's order (0 = first)",
        "  picked   : error of the scorer's first chain      random: mean error over the K chains",
        "  goal     : error of the chain whose goal is closest to the true goal",
        "  best     : error of the best chain (minADE / minFDE)",
        "  closed   : share of the random-to-best gap the scorer closes, (random-picked)/(random-best)",
        f"  chance   : hit@1 {100 / K:.1f}%, hit@3 {min(300 / K, 100):.1f}%, hit@5 {min(500 / K, 100):.1f}%, "
        f"meanRank {(K - 1) / 2:.1f}, closed 0%",
    ]
    hdr = (f"  {'Type':<12}{'N':>6}{'hit@1':>8}{'hit@3':>8}{'hit@5':>8}{'meanRank':>10}"
           f"{'picked':>9}{'random':>9}{'goal':>9}{'best':>9}{'closed':>8}")
    for metric, name in (("ade", "ADE"), ("fde", "FDE")):
        lines += ["", f"  Best chain = lowest {name}", hdr, "  " + "-" * (len(hdr) - 2)]
        for at, rs in rows.items():
            err, goal_dist = cat(rs, metric), cat(rs, "goal_dist")
            valid = ~np.isnan(err).any(axis=1)
            if valid.sum() == 0:
                continue
            err = err[valid]
            rank = err.argmin(axis=1)            # chains come in the scorer's order
            picked, random_, best = err[:, 0].mean(), err.mean(axis=1).mean(), err.min(axis=1).mean()
            has_goal = valid & ~np.isnan(goal_dist).any(axis=1)
            goal_text = "--"
            if has_goal.sum() > 0:
                goal_err = cat(rs, metric)[has_goal]
                goal_text = f"{goal_err[np.arange(len(goal_err)), goal_dist[has_goal].argmin(axis=1)].mean():.3f}"
            closed = (random_ - picked) / (random_ - best) * 100 if random_ > best else float("nan")
            lines.append(
                f"  {at:<12}{len(rank):>6}{(rank < 1).mean() * 100:>7.1f}%{(rank < 3).mean() * 100:>7.1f}%"
                f"{(rank < 5).mean() * 100:>7.1f}%{rank.mean():>10.2f}{picked:>9.3f}{random_:>9.3f}"
                f"{goal_text:>9}{best:>9.3f}{closed:>7.1f}%")

    # Confidence, over every trajectory
    ade, probs = cat(rows["ALL"], "ade"), cat(rows["ALL"], "probs")
    valid = ~np.isnan(ade).any(axis=1)
    ade, probs = ade[valid], probs[valid]
    rank = ade.argmin(axis=1)
    entropy = -(probs * np.log(np.clip(probs, 1e-12, None))).sum(axis=1) / np.log(K)
    lines += [
        "",
        "  Confidence (the scorer's softmax over the K chains, every trajectory):",
        f"    picked chain : mean probability {probs[:, 0].mean():.3f}; it is the lowest-ADE chain "
        f"{(rank == 0).mean() * 100:.1f}% of the time",
        f"    best chain   : mean probability {probs[np.arange(len(rank)), rank].mean():.3f}  "
        f"(uniform {1 / K:.3f})",
        f"    spread       : normalised entropy {entropy.mean():.3f}  (1 = uniform over the K, 0 = all on one chain)",
    ]
    if rows["ALL"][0]["pred_err"] is not None:
        pred_err = cat(rows["ALL"], "pred_err")[valid][:, 0]
        actual = ade[:, 0]
        corr = float(np.corrcoef(pred_err, actual)[0, 1]) if len(actual) > 1 else float("nan")
        lines += [
            "  ModeScorer error head (its predicted ADE for the picked chain):",
            f"    predicted {pred_err.mean():.3f} px vs actual {actual.mean():.3f} px  "
            f"(bias {pred_err.mean() - actual.mean():+.3f} px, correlation {corr:.3f})",
        ]
    lines += ["=" * 80]
    return lines


# ---------------------------------------------------------------------------
# GOAL HEAD diagnostics and the comparison with a baseline goal predictor
# ---------------------------------------------------------------------------

# The GOAL PREDICTOR vs BASELINE metrics: key, label, the better direction (None: shown, not
# judged), group. The goal-head metrics come from model.goal_head_summary; the chain errors are at
# the full horizon, as in STANDARD METRICS. The top-1 group depends on how the chains are ranked,
# so it is judged only when both checkpoints rank them the same way.
COMPARE_METRICS = (
    ("entropy", "head-usage entropy (1 = balanced)", "higher", "dead heads"),
    ("eff_k", "heads in use (effK)", "higher", "dead heads"),
    ("win_min", "lowest head win rate (%)", "higher", "dead heads"),
    ("win_max", "highest head win rate (%)", "lower", "dead heads"),
    ("slot_worst", "worst head's mean goal error (px)", "lower", "dead heads"),
    ("slot_best", "best head's mean goal error (px)", None, "dead heads"),
    ("min_pair", "smallest head distance / s_n", "higher", "collapse"),
    ("close", "head pairs nearer than r_n (%)", "lower", "collapse"),
    ("off", "off goals (%)", "lower", "off goals"),
    ("mean_k", "mean-of-K goal error (px)", "lower", "off goals"),
    ("all_p95", "every head's goal error, p95 (px)", "lower", "off goals"),
    ("best", "best-of-K goal error (px)", "lower", "goal accuracy"),
    ("best_p95", "best-of-K goal error, p95 (px)", "lower", "goal accuracy"),
    ("min_ade", "minADE (px)", "lower", "coverage"),
    ("min_fde", "minFDE (px)", "lower", "coverage"),
    ("random_ade", "mean ADE over the K chains (px)", "lower", "candidates"),
    ("random_fde", "mean FDE over the K chains (px)", "lower", "candidates"),
    ("ade", "ADE, first chain (px)", "lower", "top-1"),
    ("fde", "FDE, first chain (px)", "lower", "top-1"),
    ("first", "first chain's goal error (px)", "lower", "top-1"),
    ("hit1", "hit@1, lowest-ADE chain first (%)", "higher", "top-1"),
    ("conf", "conf head's top goal error (px)", "lower", "top-1"),
)
COMPARE_GROUPS = ("dead heads", "collapse", "off goals", "goal accuracy", "coverage", "candidates", "top-1")
CHAIN_KEYS = ("min_ade", "min_fde", "random_ade", "random_fde", "ade", "fde", "hit1")
MIN_TRAJ_FOR_VERDICT = 10   # an agent type with fewer trajectories is shown, not judged
OVERALL_TEXT = {
    "BETTER": "the new goal predictor is better than the baseline",
    "WORSE": "the new goal predictor is worse than the baseline",
    "MIXED": "better in some groups, worse in others (see the groups and the tables below)",
    "NO CLEAR CHANGE": "no group changed beyond its 95% interval",
    "NOT JUDGED": f"fewer than {MIN_TRAJ_FOR_VERDICT} trajectories",
}


def map_block(map_rows, agent_type_lis):
    """MAP COMPLIANCE, a map component diagnostic: how often predicted paths enter a
    shape their agent may not enter (obstacles for everyone, grass for all but
    pedestrians), with the GT's own rate beside it, since annotated paths clip
    outlines and cross parking areas too. Every path is densified (each step's
    midpoint added), the same point set the training penalties use."""
    lines = [
        "MAP COMPLIANCE  (paths entering a shape their agent may not enter, 12 steps)",
        "-" * 80,
        "  GT     : % of ground-truth paths that enter one",
        "  top-1  : % of the first predicted paths that enter one",
        "  all K  : % of all K predicted paths that enter one",
        "  depth  : mean depth of the deepest point of the first paths that enter one, px",
        "",
        f"  {'Type':<12}{'N':>6}{'GT':>9}{'top-1':>9}{'all K':>9}{'depth':>9}",
        "  " + "-" * 54,
    ]
    rows = {"ALL": [r for at in agent_type_lis for r in map_rows[at]]}
    rows.update({at: map_rows[at] for at in agent_type_lis})
    inside_px = 0.01   # deeper than this counts as inside (ignores rounding on an outline)
    for name in ["ALL"] + agent_type_lis:
        if not rows[name]:
            continue
        chains = np.concatenate([r["chains"] for r in rows[name]], axis=0)   # (M, K) deepest point, px
        gt = np.concatenate([r["gt"] for r in rows[name]], axis=0)           # (M,)
        top1 = chains[:, 0]
        entering = top1 > inside_px
        depth = top1[entering].mean() if entering.any() else 0.0
        lines.append(f"  {name:<12}{len(gt):>6}{100 * (gt > inside_px).mean():>8.1f}%"
                     f"{100 * entering.mean():>8.1f}%{100 * (chains > inside_px).mean():>8.1f}%{depth:>9.2f}")
    lines.append("=" * 80)
    return lines


def goal_distribution_rows(model, gt, floors_unused=None):
    """Per trajectory, from the goal distribution of the last batch (model.last_goal_prob): the
    probability within 0.1 and 0.25 s_n of the true goal, and the number of separate peaks
    (local maxima holding >= 5% of the mass within 0.15 s_n, merged when closer than 0.3 s_n)"""
    prob, cv, s = model.last_goal_prob, model.last_goal_cv, model.last_goal_s
    grid, cells = model.goal_grid, model.goal_grid_cells
    step = 2.0 * model.goal_grid_range / (cells - 1)
    r = (gt - cv) / s.unsqueeze(-1)                                         # true goal, grid units
    d = (grid.unsqueeze(0) - r.unsqueeze(1)).norm(dim=-1)                   # (N, G*G)
    near01 = (prob * (d <= 0.1)).sum(dim=1)
    near025 = (prob * (d <= 0.25)).sum(dim=1)
    pg = prob.view(-1, 1, cells, cells)
    win = max(1, int(round(0.15 / step)))
    is_max = (torch.nn.functional.max_pool2d(pg, 5, 1, 2) == pg) & (pg > 0)
    mass = torch.nn.functional.avg_pool2d(pg, 2 * win + 1, 1, win) * (2 * win + 1) ** 2
    peaks = []
    for b in range(pg.shape[0]):
        idx = torch.where((is_max[b, 0] & (mass[b, 0] >= 0.05)).flatten())[0]
        pts, m = grid[idx], mass[b, 0].flatten()[idx]
        keep = []
        for j in torch.argsort(m, descending=True).tolist():
            if all(float((pts[j] - pts[k]).norm()) >= 0.3 for k in keep):
                keep.append(j)
        peaks.append(max(len(keep), 1))
    return {"near01": near01.cpu().numpy(), "near025": near025.cpu().numpy(), "peaks": np.array(peaks)}


def distribution_block(goal, dist, agent_type_lis, scale_factor):
    """GOAL DISTRIBUTION (goal-distribution checkpoints): how the predicted distribution and the
    K goals picked from it (slot order = most likely first) cover the true goal"""
    valid = goal["goal_valid"]
    goals, gt, s, cls = goal["goals"][valid], goal["gt"][valid], goal["s"][valid], goal["cls"][valid]
    err = np.linalg.norm(goals - gt[:, np.newaxis], axis=-1) * scale_factor          # (n, K) px
    best = err.min(axis=1)
    spread = (np.linalg.norm(goals - goals.mean(axis=1, keepdims=True), axis=-1) / s[:, np.newaxis]).mean(axis=1)
    miss = best > 0.5 * s * scale_factor
    k_list = [k for k in (1, 5, 10) if k < goals.shape[1]] + [goals.shape[1]]
    near01, near025, peaks = dist["near01"][valid], dist["near025"][valid], dist["peaks"][valid]
    lines = [
        f"GOAL DISTRIBUTION  (the predicted distribution over goals and the {goals.shape[1]} goals picked from it)",
        "-" * 80,
        "  P<0.1s / P<0.25s : probability the distribution puts within 0.1 / 0.25 s_n of the true goal",
        "  peaks            : separate peaks per agent (>= 5% of the mass each, >= 0.3 s_n apart)",
        "  spread           : mean distance of the goals from their centre, s_n units",
        "  best@k           : best of the k most likely goals, px",
        "  miss             : % whose best goal is farther than 0.5 s_n;  p90 / p95 / p99 of the best goal, px",
        "",
        f"  {'Type':<12}{'N':>6}{'P<0.1s':>8}{'P<0.25s':>9}{'peaks':>7}{'spread':>8}"
        + "".join(f"{'best@' + str(k):>9}" for k in k_list) + f"{'miss':>7}{'p90':>7}{'p95':>7}{'p99':>7}",
    ]
    for name in ["ALL"] + list(agent_type_lis):
        idx = np.ones(len(cls), bool) if name == "ALL" else (cls == agent_type_lis.index(name))
        if not idx.any():
            continue
        q = np.percentile(best[idx], [90, 95, 99])
        lines.append(f"  {name:<12}{idx.sum():>6}{near01[idx].mean():>8.3f}{near025[idx].mean():>9.3f}"
                     f"{peaks[idx].mean():>7.2f}{spread[idx].mean():>8.2f}"
                     + "".join(f"{err[idx, :k].min(axis=1).mean():>9.2f}" for k in k_list)
                     + f"{100 * miss[idx].mean():>6.2f}%{q[0]:>7.1f}{q[1]:>7.1f}{q[2]:>7.1f}")
    lines.append("=" * 80)
    return lines


def scorer_of(checkpoint):
    """The scorer a checkpoint was trained with, read from its weights: 'traj' (traj_scorer.*),
    'mode' (mode_scorer.*) or None"""
    return next((kind for kind, prefix in (("traj", "traj_scorer."), ("mode", "mode_scorer."))
                 if any(k.startswith(prefix) for k in checkpoint["model_state_dict"])), None)


GOAL_ARG_DEFAULTS = {  # train.py's defaults, for checkpoints whose args lack a flag
    "goal_map_crop": "True", "goal_reach_prior": "True", "goal_grid_cells": 61, "goal_grid_range": 1.5,
    "goal_heatmap_sigma": 0.075, "goal_crop_cells": 32, "goal_hidden": 128, "goal_pick": "kmeans",
    "goal_pick_iters": 25, "goal_pick_alpha": 0.5, "goal_pick_reserve": 4, "goal_pick_reserve_alpha": 0.25,
    "goal_pick_floor": 0.5, "goal_scale_min": 0.2, "goal_scale_class_frac": 0.5, "goal_detach_encoder": "True",
    "goal_cond_noise": 0.0}
PICK_FLAGS = ("goal_pick", "goal_pick_iters", "goal_pick_alpha", "goal_pick_reserve", "goal_pick_reserve_alpha",
              "goal_pick_floor")


def goal_predictor_of(checkpoint):
    """'heatmap' or 'mtp', from the checkpoint's args, else from its weights"""
    saved = checkpoint.get("args", {}).get("goal_predictor")
    if saved in ("heatmap", "mtp"):
        return saved
    return "mtp" if "goal_predictor.goal_head.weight" in checkpoint["model_state_dict"] else "heatmap"


def map_nodes_of(checkpoint):
    """Whether the checkpoint's encoder has map nodes (hdgt_map_gnn's encoder.map_emb weights)"""
    return any(k.startswith("encoder.map_emb") for k in checkpoint["model_state_dict"])


def build_model(model_module, checkpoint, args, device, use_scorer=None, scorer_type=None, pick_override=None):
    """The model rebuilt as the checkpoint was trained, in eval mode, and its scorer's name. The
    scorer is read from the checkpoint's weights unless use_scorer / scorer_type are given; the
    goal predictor, its settings and the map nodes from its args and weights; pick_override
    {flag: value} replaces how a goal distribution picks its K goals. The class statistics behind
    the s_n floors are loaded into the model (s_n scales the goal distribution's grid)"""
    model_args = copy.copy(args)
    in_weights = scorer_of(checkpoint)
    model_args.use_scorer = use_scorer if use_scorer is not None else ("True" if in_weights else "False")
    model_args.scorer_type = scorer_type if scorer_type is not None else (in_weights or "mode")
    saved = checkpoint.get("args", {})
    for key, default in GOAL_ARG_DEFAULTS.items():
        setattr(model_args, key, saved.get(key, default))
    for key, value in (pick_override or {}).items():
        setattr(model_args, key, value)
    model_args.goal_predictor = goal_predictor_of(checkpoint)
    model_args.use_map_nodes = map_nodes_of(checkpoint)
    model = model_module.HDGT_model(input_dim=8, args=model_args)
    model.load_state_dict(checkpoint["model_state_dict"])
    if "goal_class_scale" in checkpoint:
        model.goal_class_scale.load(checkpoint["goal_class_scale"], checkpoint.get("goal_class_count"))
    elif model_args.goal_predictor == "heatmap":
        raise RuntimeError("this goal-distribution checkpoint has no class statistics (goal_class_scale): "
                           "its s_n floors, and so its goal grid, cannot be rebuilt")
    model.to(device)
    model.eval()
    scorer_desc = ({"mode": "ModeScorer", "traj": "TrajScorer"}.get(getattr(model, "scorer_type", None), "?")
                   if getattr(model, "use_scorer", False) else "none (chains in goal order)")
    return model, scorer_desc


def goal_settings(checkpoint, pick_override=None):
    """The goal predictor a checkpoint was trained with, from its saved args and weights, for the report"""
    a = checkpoint.get("args", {})
    if goal_predictor_of(checkpoint) == "heatmap":
        g = {**GOAL_ARG_DEFAULTS, **{k: v for k, v in a.items() if k in GOAL_ARG_DEFAULTS}, **(pick_override or {})}
        pick = g["goal_pick"] + {"flat": f" (p^{g['goal_pick_alpha']})",
                                 "reserve": f" ({g['goal_pick_reserve']} for rare futures)",
                                 "sinkhorn": f" (floor {g['goal_pick_floor']})"}.get(g["goal_pick"], "")
        text = (f"goal distribution, {g['goal_grid_cells']}x{g['goal_grid_cells']} cells over +-{g['goal_grid_range']} s_n "
                f"(sigma {g['goal_heatmap_sigma']}), map crop {g['goal_map_crop']}, reachability prior "
                f"{g['goal_reach_prior']}; K goals picked by {pick}"
                + (" [picking set on the command line]" if pick_override else ""))
    elif "goal_assign" in a:
        text = f"K WTA heads, trained in hdgt_goal_sinkhorn with --goal_assign {a['goal_assign']}"
        if a["goal_assign"] == "hybrid":
            text += f" (wta + {a.get('goal_sinkhorn_weight')} x sinkhorn)"
    else:
        text = "K WTA heads (hdgt_goal's top-3 -> top-1)"
    if a.get("goal_detach_encoder", True) in (False, "False", "false", "0"):
        text += ", goal loss also trains the encoder"
    if float(a.get("goal_cond_noise", 0.0) or 0.0) > 0:
        text += f", decoder trained on noisy goals (RMS {a['goal_cond_noise']} s_n)"
    return text


def checkpoint_floors(model_module, checkpoint):
    """The class floors of s_n, (6,) scaled, from the checkpoint's class statistics (train.py's
    goal_scale_extra) and its --goal_scale_min / --goal_scale_class_frac, and where they come
    from. A checkpoint without the statistics (trained before this version) gets the global floor"""
    a = checkpoint.get("args", {})
    scale_min = float(a.get("goal_scale_min", 0.2))
    if "goal_class_scale" not in checkpoint:
        return (torch.full((6,), scale_min),
                f"--goal_scale_min {scale_min:g} for every class (no class statistics in the checkpoint)")
    stats = model_module.GoalClassScale(6)
    stats.load(checkpoint["goal_class_scale"], checkpoint.get("goal_class_count"))
    floors = model_module.class_floors(stats.scale, stats.count, scale_min,
                                       float(a.get("goal_scale_class_frac", 0.5)))
    return floors, "the checkpoint's class floors"


def evaluate(model, model_module, loader, args, device, agent_type_lis, length_lis, metric_type_lis, floors):
    """One pass over the split. It reseeds first, so every pass draws the same sampling noise (a
    baseline's chains get the new checkpoint's noise). Returns the standard-metric meters, the
    scorer rows (with a scorer) and, in batch order, the per-trajectory inputs of the GOAL HEAD
    diagnostics and of the comparison"""
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    recorder = {
        f"{at}_{ln}_{mt}": AverageMeter()
        for at in agent_type_lis
        for ln in length_lis
        for mt in metric_type_lis
    }

    # Scorer results (only with a scorer): every chain's error at the full horizon, in the
    # scorer's order, with the scorer's outputs for it. Per type, every trajectory counts once.
    use_scorer = getattr(model, "use_scorer", False)
    scorer_rows = {at: [] for at in agent_type_lis}
    # GOAL HEAD rows: every target of every batch; GOAL DISTRIBUTION rows (goal distribution only)
    goal_rows = []
    dist_rows = []
    # Map compliance: each chain's and the GT's deepest point inside a forbidden shape, px
    map_rows = {at: [] for at in agent_type_lis}
    floors = floors.to(device)

    print(f"\nRunning inference ...")
    t0 = time.time()

    with torch.no_grad():
        for batch_idx, data in enumerate(loader):
            if data is None:
                continue

            data["is_train"] = False
            data["gpu"] = max(args.gpu, 0)
            for key in data["cuda_tensor_lis"]:
                data[key] = data[key].to(device, non_blocking=True)

            trajectories, pred_bool_type_lis, agent_types = model(data)

            # trajectories: (N_agents, K, pred_len, 2)
            # refiner output is positions; raw diffusion output is velocities
            output_is_positions = getattr(model, "output_is_positions", False)
            if output_is_positions:
                pred_positions = trajectories.cpu().numpy()
            else:
                pred_positions = torch.cumsum(trajectories, dim=2).cpu().numpy()

            label_pos  = data["label_pos_lis"]
            label_mask = data["label_mask_lis"]

            pos_t = trajectories if output_is_positions else torch.cumsum(trajectories, dim=2)
            geom, forbid = data["map_geom_lis"], data["map_forbid_mask_lis"]
            depth_k = (model_module.map_depth(model_module.densify_path(pos_t), geom, forbid)
                       .amax(dim=-1) * SCALE_FACTOR).cpu().numpy()                  # (N, K)
            depth_gt = (model_module.map_depth(model_module.densify_path(label_pos), geom, forbid)
                        .amax(dim=-1) * SCALE_FACTOR).cpu().numpy()                 # (N,)
            if getattr(model, "last_goal_prob", None) is not None:
                dist_rows.append(goal_distribution_rows(model, label_pos[:, -1]))

            if use_scorer:
                # All three come in the same order as the returned chains
                goals_np = model.last_predicted_goals.cpu().numpy()                  # (N, K, 2)
                probs_np = model.last_mode_probs.cpu().numpy()                       # (N, K)
                perr = getattr(model, "last_pred_error", None)                       # ModeScorer only
                perr_np = perr.cpu().numpy() * SCALE_FACTOR if perr is not None else None

            # The goal heads in head order and every chain's error, for the GOAL HEAD block and the
            # comparison; s_n from the observed positions and the checkpoint's class floors
            labels_np, masks_np = label_pos.cpu().numpy(), label_mask.cpu().numpy()
            ade_all, fde_all = chain_errors(pred_positions, labels_np, masks_np, scale_factor=SCALE_FACTOR)
            goal_logits = model.last_goal_logits
            goal_rows.append({
                "goals": model.last_goals_head_order.cpu().numpy(),                 # (N, K, 2)
                "first": model.last_predicted_goals[:, 0].cpu().numpy(),            # (N, 2)
                "conf": goal_logits.cpu().numpy() if goal_logits is not None else None,
                "gt": labels_np[:, -1],
                "goal_valid": masks_np[:, -1] != 0,    # no true goal without the last step
                "s": model_module.motion_scale(model.last_target_obs, floors[agent_types],
                                               args.pred_len).cpu().numpy(),
                "cls": agent_types.cpu().numpy(),
                "ade": ade_all,                                                      # (N, K) px
                "fde": fde_all,
            })

            for type_idx, at in enumerate(agent_type_lis):
                mask = pred_bool_type_lis[type_idx].cpu().numpy()
                if mask.sum() == 0:
                    continue

                preds  = pred_positions[mask]          # (M, K, pred_len, 2)
                labels = label_pos[mask].cpu().numpy()  # (M, pred_len, 2)
                masks  = label_mask[mask].cpu().numpy() # (M, pred_len)

                # Chain 0: the scorer's pick with a scorer; the most likely goal's chain without one
                best_pred = preds[:, 0, :, :]
                map_rows[at].append({"chains": depth_k[mask], "gt": depth_gt[mask]})

                if use_scorer:
                    ade_k, fde_k = chain_errors(preds, labels, masks, scale_factor=SCALE_FACTOR)
                    goal_dist = np.linalg.norm(goals_np[mask] - labels[:, -1][:, np.newaxis], axis=-1)
                    goal_dist[masks[:, -1] == 0] = np.nan    # no true goal without the last step
                    scorer_rows[at].append({"ade": ade_k, "fde": fde_k, "goal_dist": goal_dist,
                                            "probs": probs_np[mask],
                                            "pred_err": perr_np[mask] if perr_np is not None else None})

                for ln in length_lis:
                    res, cnt = cal_ade_fde(
                        best_pred[:, :ln, :], labels[:, :ln, :], masks[:, :ln],
                        scale_factor=SCALE_FACTOR
                    )
                    if res is not None:
                        recorder[f"{at}_{ln}_ade"].update(res[0], cnt[0])
                        recorder[f"{at}_{ln}_fde"].update(res[1], cnt[1])

                    res, cnt = cal_min_ade_fde(
                        preds[:, :, :ln, :], labels[:, :ln, :], masks[:, :ln],
                        scale_factor=SCALE_FACTOR
                    )
                    if res is not None:
                        recorder[f"{at}_{ln}_min_ade"].update(res[0], cnt[0])
                        recorder[f"{at}_{ln}_min_fde"].update(res[1], cnt[1])

            if (batch_idx + 1) % 20 == 0:
                elapsed = time.time() - t0
                print(f"  [{batch_idx + 1}/{len(loader)}]  {elapsed:.1f}s")

    elapsed_total = time.time() - t0
    print(f"Inference done in {elapsed_total:.1f}s")
    goal = {key: (np.concatenate([row[key] for row in goal_rows], axis=0) if goal_rows[0][key] is not None
                  else None) for key in goal_rows[0]}
    dist = ({key: np.concatenate([row[key] for row in dist_rows], axis=0) for key in dist_rows[0]}
            if dist_rows else None)
    return {"recorder": recorder, "scorer_rows": scorer_rows, "goal": goal, "elapsed": elapsed_total,
            "use_scorer": use_scorer, "scorer_type": getattr(model, "scorer_type", None),
            "dist": dist, "map_rows": map_rows}


def comparison_arrays(model_module, goal, valid, use_scorer, rho):
    """One pass's per-trajectory arrays for the comparison, over the trajectories `valid`: the
    goal-head ones (model.goal_head_agent_arrays) and the chain errors at the full horizon"""
    arrays = model_module.goal_head_agent_arrays(
        goal["goals"][valid], goal["gt"][valid], goal["s"][valid], SCALE_FACTOR,
        first_goal=goal["first"][valid],
        conf_logits=goal["conf"][valid] if goal["conf"] is not None else None, rho=rho)
    ade, fde = goal["ade"][valid], goal["fde"][valid]
    arrays.update({
        "min_ade": ade.min(axis=1), "min_fde": fde.min(axis=1),
        "random_ade": ade.mean(axis=1), "random_fde": fde.mean(axis=1),
        "ade": ade[:, 0], "fde": fde[:, 0],
        # a ranking measure, so only with a scorer (without one, chain 0 is just head 0)
        "hit1": (ade.argmin(axis=1) == 0) * 100.0 if use_scorer else None,
    })
    return arrays


def comparison_summary(model_module, arrays, idx=None):
    """Every COMPARE_METRICS value of the trajectories idx (default: all)"""
    summary = model_module.goal_head_summary(arrays, idx)
    for key in CHAIN_KEYS:
        values = arrays[key]
        summary[key] = None if values is None else (values if idx is None else values[idx]).mean()
    return summary


def metric_verdict(interval, better):
    """better / worse when the 95% interval of new - baseline excludes 0, read in the metric's
    better direction; same when it contains 0; -- when the metric is not judged"""
    if interval is None or better is None or np.isnan(interval).any():
        return "--"
    if interval[0] > 0:
        return "better" if better == "higher" else "worse"
    if interval[1] < 0:
        return "better" if better == "lower" else "worse"
    return "same"


def group_verdict(verdicts):
    """better if a metric is better and none worse, worse the reverse, mixed if both, same if
    neither; -- without a judged metric"""
    verdicts = [v for v in verdicts if v != "--"]
    better, worse = "better" in verdicts, "worse" in verdicts
    if better and worse:
        return "mixed"
    return "better" if better else "worse" if worse else "same" if verdicts else "--"


def overall_verdict(group_verdicts):
    """BETTER if a group is better and none worse or mixed, WORSE the reverse, NO CLEAR CHANGE if
    every judged group is the same, MIXED otherwise"""
    judged = [v for v in group_verdicts if v != "--"]
    if not judged:
        return "NOT JUDGED"
    if "better" in judged and not any(v in ("worse", "mixed") for v in judged):
        return "BETTER"
    if "worse" in judged and not any(v in ("better", "mixed") for v in judged):
        return "WORSE"
    if all(v == "same" for v in judged):
        return "NO CLEAR CHANGE"
    return "MIXED"


def compare_lines(arrays_new, arrays_base, cls, model_module, agent_type_lis, split, pred_len,
                  n_bootstrap, seed, new_info, base_info, same_ranking):
    """The GOAL PREDICTOR vs BASELINE block. arrays_new / arrays_base: comparison_arrays of the two
    passes over the same trajectories, cls (n,) their agent types; new_info / base_info: (path,
    description lines); same_ranking: both rank their chains the same way (the top-1 group)"""
    rng = np.random.default_rng(seed)
    tables = []
    for name, members in [("ALL", np.arange(len(cls)))] + [
            (at, np.flatnonzero(cls == c)) for c, at in enumerate(agent_type_lis)]:
        if len(members) == 0:
            continue
        s_new = comparison_summary(model_module, arrays_new, members)
        s_base = comparison_summary(model_module, arrays_base, members)
        judged = [key for key, _, better, group in COMPARE_METRICS
                  if better is not None and (group != "top-1" or same_ranking)
                  and s_new[key] is not None and s_base[key] is not None]
        # Paired bootstrap: the same resample of the trajectories for both checkpoints
        intervals = {}
        if len(members) >= MIN_TRAJ_FOR_VERDICT and judged and n_bootstrap > 0:
            diffs = {key: np.empty(n_bootstrap) for key in judged}
            for b in range(n_bootstrap):
                idx = members[rng.integers(0, len(members), len(members))]
                b_new = comparison_summary(model_module, arrays_new, idx)
                b_base = comparison_summary(model_module, arrays_base, idx)
                for key in judged:
                    diffs[key][b] = b_new[key] - b_base[key]
            intervals = {key: np.percentile(diffs[key], [2.5, 97.5]) for key in judged}
        rows = [(key, label, better, group, s_base[key], s_new[key], intervals.get(key),
                 metric_verdict(intervals.get(key), better))
                for key, label, better, group in COMPARE_METRICS]
        verdicts = {group: group_verdict([row[7] for row in rows if row[3] == group]) for group in COMPARE_GROUPS}
        tables.append((name, len(members), rows, verdicts, s_new, s_base))

    def text(value, signed=False):
        """3 decimals, or scientific notation for a value too small for them"""
        if value == 0 or abs(value) >= 0.01:
            return f"{value:+.3f}" if signed else f"{value:.3f}"
        return f"{value:+.2e}" if signed else f"{value:.2e}"

    def num(value, signed=False):
        return f"{'--':>10}" if value is None else f"{text(value, signed):>10}"

    def rel(change, v_base):
        return f"{'--':>8}" if change is None or not v_base else f"{100 * change / abs(v_base):>+7.1f}%"

    def interval_text(interval):
        return "--" if interval is None else f"[{text(interval[0], True)}, {text(interval[1], True)}]"

    lines = [
        f"GOAL PREDICTOR vs BASELINE  ({split.upper()} split, {pred_len} steps = {pred_len * 0.4:.1f}s, pixels)",
        "-" * 80,
    ]
    for tag, (path, desc) in (("new", new_info), ("baseline", base_info)):
        lines.append(f"  {tag:<9}: {path}")
        lines += [f"  {'':<9}  {line}" for line in desc]
    lines += [
        "  Both are run here on the same trajectories with the same code, seed and sampling noise;",
        "  s_n, r_n and the floors are the new checkpoint's, for both.",
        "  change   : new - baseline (rel: % of the baseline); 95% interval: paired bootstrap over the",
        f"             trajectories, {n_bootstrap} resamples (seed {seed})",
        "  verdict  : better / worse when the interval excludes 0, read in the metric's better direction;",
        f"             same when it contains 0; -- not judged (no better direction, or fewer than "
        f"{MIN_TRAJ_FOR_VERDICT} trajectories)",
        "  group    : better if a metric is better and none worse, worse the reverse, mixed if both,",
        "             same if neither",
        "  overall  : BETTER if a group is better and none worse or mixed, WORSE the reverse,",
        "             NO CLEAR CHANGE if every group is the same, MIXED otherwise",
    ]
    if not same_ranking:
        lines += ["  top-1    : not judged -- the two rank their chains differently (see their scorers), so",
                  "             their first chains differ by construction, whatever the goal heads"]

    widths = [max(len(group), 6) + 2 for group in COMPARE_GROUPS]
    lines += ["", "  Diagnosis (group verdicts; each group's metrics are in the tables below)",
              (f"  {'Type':<12}{'N':>6}  " + "".join(f"{group:<{w}}" for group, w in zip(COMPARE_GROUPS, widths))).rstrip()]
    for name, n, _, verdicts, _, _ in tables:
        lines.append((f"  {name:<12}{n:>6}  " + "".join(f"{verdicts[group]:<{w}}"
                                                      for group, w in zip(COMPARE_GROUPS, widths))).rstrip())
    overall = overall_verdict(tables[0][3].values())
    lines += ["", f"  OVERALL, every trajectory: {overall} -- {OVERALL_TEXT[overall]}"]

    for name, n, rows, verdicts, s_new, s_base in tables:
        note = "" if n >= MIN_TRAJ_FOR_VERDICT else f"; fewer than {MIN_TRAJ_FOR_VERDICT}, not judged"
        lines += ["", f"  {name}  ({n} trajectories{note})",
                  f"    {'metric':<36}{'better':>7}{'baseline':>10}{'new':>10}{'change':>10}{'rel':>8}  "
                  f"{'95% interval':<24}verdict"]
        for group in COMPARE_GROUPS:
            shown = [row for row in rows if row[3] == group and not (row[4] is None and row[5] is None)]
            if not shown:
                continue
            lines.append(f"    {group}: {verdicts[group]}")
            for _, label, better, _, v_base, v_new, interval, verdict in shown:
                change = None if v_base is None or v_new is None else v_new - v_base
                lines.append(f"      {label:<34}{better or '--':>7}{num(v_base)}{num(v_new)}{num(change, True)}"
                             f"{rel(change, v_base)}  {interval_text(interval):<24}{verdict}")
        if name == "ALL":
            K = len(s_new["win_rates"])
            lines.append(f"    win % per head (chance {100 / K:.1f}%):")
            for start in range(0, K, 10):
                heads = range(start, min(start + 10, K))
                lines += ["      head    " + "".join(f"{k:>6d}" for k in heads),
                          "      baseline" + "".join(f"{s_base['win_rates'][k]:>6.1f}" for k in heads),
                          "      new     " + "".join(f"{s_new['win_rates'][k]:>6.1f}" for k in heads)]
    lines.append("=" * 80)
    return lines


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def comparison_block(recorder, agent_type_lis, length_lis, metric_keys, num_prediction, split):
    """The trajectory metrics of the report -- the same in every version (keep them in sync)

    recorder holds one AverageMeter per (agent type, horizon, metric), keyed
    "<TYPE>_<steps>_<metric>"; metric_keys maps ADE, FDE, minADE and minFDE to this version's
    metric names. All numbers are in pixels.

    STANDARD METRICS are the ones published SDD results report: at the full horizon (the last of
    length_lis), ADE averages the error over every predicted step and FDE takes the last step.
    ADE/FDE: the version's first trajectory (its top-ranked one, or its first sample when it
    ranks none); minADE/minFDE: the best of the K, each chosen on its own. Sample-weighted: every
    trajectory counts once, whatever its agent type (on val, its minADE / minFDE are what
    best_min_ade.pt / best_min_fde.pt were chosen on). Type-averaged: every agent type with
    trajectories weighs the same. INTERMEDIATE HORIZONS: the same metrics at the shorter
    horizons, kept apart because they are not the standard metric.
    """
    names = ["ADE", "FDE", "minADE", "minFDE"]
    full = length_lis[-1]

    def meter(agent_type, length, name):
        return recorder[f"{agent_type}_{length}_{metric_keys[name]}"]

    counts = {agent_type: int(meter(agent_type, full, "ADE").count) for agent_type in agent_type_lis}
    total = sum(counts.values())
    present = [agent_type for agent_type in agent_type_lis if counts[agent_type] > 0]

    def sample_weighted(length):
        values = []
        for name in names:
            meters = [meter(agent_type, length, name) for agent_type in agent_type_lis]
            count = sum(m.count for m in meters)
            values.append(sum(m.sum for m in meters) / count if count > 0 else float('nan'))
        return values

    def type_averaged(length):
        return [sum(meter(agent_type, length, name).avg for agent_type in present) / len(present)
                if present else float('nan') for name in names]

    def per_type(length, agent_type):
        return [meter(agent_type, length, name).avg if counts[agent_type] > 0 else float('nan')
                for name in names]

    def row(label, n, values):
        return f"    {label:<22}{n:>6}" + "".join(f"{value:>10.4f}" for value in values)

    header = f"    {'':<22}{'N':>6}" + "".join(f"{name:>10}" for name in names)
    lines = [
        "STANDARD METRICS  (as in published SDD results; the same in all 9 versions)",
        "-" * 80,
        f"  {split.upper()} split, {total} trajectories, K = {num_prediction} (--num_prediction), "
        "errors in pixels",
        f"  Horizon: all {full} predicted steps ({full * 0.4:.1f} s). ADE: mean error over the "
        f"{full} steps;",
        f"  FDE: error at step {full}.",
        "  ADE, FDE      : the version's first trajectory (top-ranked, or its first",
        "                  sample when it ranks none)",
        f"  minADE, minFDE: the best of the {num_prediction} trajectories, picked separately for each",
        "",
        "  Sample-weighted (every trajectory counts once):",
        header,
        row("ALL", total, sample_weighted(full)),
        "",
        f"  Type-averaged (each of the {len(present)} agent types weighs 1/{len(present)}):",
        header,
        row("ALL", total, type_averaged(full)),
        "",
        "  Per agent type:",
        header,
    ]
    lines += [row(agent_type, counts[agent_type], per_type(full, agent_type)) for agent_type in agent_type_lis]
    lines += [
        "=" * 80,
        "INTERMEDIATE HORIZONS  (the same metrics at shorter horizons; not standard)",
        "-" * 80,
        header,
    ]
    for length in length_lis[:-1]:
        lines.append(f"  {length} steps ({length * 0.4:.1f} s):")
        lines.append(row("ALL, sample-weighted", total, sample_weighted(length)))
        lines.append(row("ALL, type-averaged", total, type_averaged(length)))
        lines += [row(agent_type, counts[agent_type], per_type(length, agent_type)) for agent_type in agent_type_lis]
    lines.append("=" * 80)
    return lines


def main():
    parser = argparse.ArgumentParser("HDGT Goal-CVAE — Test Evaluation")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to the checkpoint, e.g. best_min_fde.pt or best_min_ade.pt")
    parser.add_argument("--data_folder", type=str, default="../sdd/",
                        help="Root SDD folder (contains train/ and test/)")
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"],
                        help="'val' (default) = the held-out validation videos, safe to "
                             "run as often as you like. 'test' = the final holdout; "
                             "run it once, at the end.")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_worker", type=int, default=4)
    parser.add_argument("--gpu", type=int, default=0, help="-1 for CPU")
    parser.add_argument("--seed", type=int, default=42, help="random seed for reproducibility")

    # Architecture — must match what was used during training
    parser.add_argument("--obs_len", type=int, default=8)
    parser.add_argument("--pred_len", type=int, default=12)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--head_dim", type=int, default=32)
    parser.add_argument("--num_of_gnn_layer", type=int, default=6)
    parser.add_argument("--diffusion_steps", type=int, default=100)
    parser.add_argument("--diffusion_layers", type=int, default=6)
    parser.add_argument("--ddim_steps", type=int, default=20)
    parser.add_argument("--num_prediction", type=int, default=20)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--agent_drop", type=float, default=0.0)
    parser.add_argument("--use_refiner", type=str, default="True")
    parser.add_argument("--model_module", type=str, default="model",
                        help="model file to load (model = model.py) — must match the checkpoint")
    # Unused at test time but needed to instantiate the model
    parser.add_argument("--goal_loss_weight", type=float, default=0.5)
    parser.add_argument("--goal_curriculum_epochs", type=int, default=5)
    # Scorer: read from the checkpoint's weights by default, so the model is rebuilt as trained
    parser.add_argument("--use_scorer", type=str, default=None)
    parser.add_argument("--scorer_type", type=str, default=None, choices=["mode", "traj"])
    # The previous goal predictor, compared with --checkpoint (GOAL PREDICTOR vs BASELINE)
    # How a goal-distribution checkpoint picks its K goals: the checkpoint's own unless given
    parser.add_argument("--goal_pick", type=str, default=None, choices=["kmeans", "flat", "reserve", "sinkhorn"])
    parser.add_argument("--goal_pick_iters", type=int, default=None)
    parser.add_argument("--goal_pick_alpha", type=float, default=None)
    parser.add_argument("--goal_pick_reserve", type=int, default=None)
    parser.add_argument("--goal_pick_reserve_alpha", type=float, default=None)
    parser.add_argument("--goal_pick_floor", type=float, default=None)
    parser.add_argument("--baseline_checkpoint", type=str, default=None,
                        help="checkpoint of the previous goal predictor, e.g. one of hdgt_goal's best_min_fde.pt "
                             "or a --goal_predictor mtp run (mtp and map nodes are read from its weights): evaluated on the same trajectories with "
                             "the same noise and compared with --checkpoint. Its scorer is read from its weights; "
                             "--use_scorer / --scorer_type apply to --checkpoint only")
    parser.add_argument("--n_bootstrap", type=int, default=1000,
                        help="paired bootstrap resamples behind the comparison's 95%% intervals")

    args = parser.parse_args()
    args.use_refiner = (args.use_refiner == "True")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(f"cuda:{args.gpu}" if args.gpu >= 0 and torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------
    # Load checkpoint
    # ------------------------------------------------------------------
    for path in (args.checkpoint, args.baseline_checkpoint):
        if path is not None and not os.path.isfile(path):
            print(f"ERROR: checkpoint not found: {path}")
            sys.exit(1)

    print("=" * 80)
    print("HDGT Goal-CVAE — Test Evaluation")
    print("=" * 80)
    print(f"Checkpoint : {args.checkpoint}")
    print(f"Device     : {device}")

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    saved_epoch = checkpoint.get("epoch", "?")
    saved_val   = checkpoint.get("val_loss", float("nan"))
    saved_metric = checkpoint.get("select_metric", "min_fde")
    # Checkpoints written before the val split existed were selected on the test
    # set. Absent stamp therefore means "test", not "unknown" — assume the worse
    # of the two so a stale checkpoint cannot pass itself off as clean.
    select_split = checkpoint.get("select_split", "test")
    print(f"Saved epoch: {saved_epoch}   |   selected on {saved_metric} = "
          f"{saved_val:.4f} px  (on the {select_split.upper()} split)")
    if select_split == "test":
        print("WARNING: this checkpoint was selected on the TEST set. Test numbers "
              "from it are\n         optimistic. Retrain with --use_val_split True "
              "for a clean final result.")

    # The scorer the checkpoint was trained with, read from its weights (traj_scorer.* =
    # TrajScorer, mode_scorer.* = ModeScorer, neither = no scorer) unless given on the command line
    model_module = importlib.import_module(args.model_module)
    pick_override = {key: getattr(args, key) for key in PICK_FLAGS if getattr(args, key) is not None}
    model, scorer_desc = build_model(model_module, checkpoint, args, device, args.use_scorer, args.scorer_type,
                                     pick_override)
    map_nodes = map_nodes_of(checkpoint)
    if args.baseline_checkpoint is not None:
        map_nodes = map_nodes or map_nodes_of(torch.load(args.baseline_checkpoint, map_location="cpu"))
    # The loader adds the map nodes to the graph when either model has them; a model without
    # them reads only the agent nodes and edges
    args.use_map_nodes = map_nodes
    print(f"Model loaded — refiner={'ON' if args.use_refiner else 'OFF'}, scorer={scorer_desc}, K={args.num_prediction}")

    # The goal-head loss it was trained with, and the class floors of s_n (GOAL HEAD block)
    goal_desc = goal_settings(checkpoint, pick_override)
    floors, floors_source = checkpoint_floors(model_module, checkpoint)
    rho = model_module.GOAL_CLOSE_RHO
    model_map_nodes = map_nodes_of(checkpoint)
    print(f"Goal predictor: {goal_desc}")
    del checkpoint

    # ------------------------------------------------------------------
    # Load evaluation dataset (--split: val or test)
    # ------------------------------------------------------------------
    test_files = resolve_split_files(args.data_folder, args.split)
    if len(test_files) == 0:
        print(f"ERROR: no .pickle files found for split '{args.split}' "
              f"under {args.data_folder}")
        sys.exit(1)

    if args.split == "test":
        print("\n" + "!" * 80)
        print("Evaluating on the TEST set — the final holdout. Every look at it "
              "costs a little\nof its value as an unbiased estimate. Use --split val "
              "while iterating.")
        print("!" * 80)

    print(f"\n{args.split.capitalize()} files ({len(test_files)}):")
    for f in test_files:
        print(f"  {os.path.basename(f)}")

    test_dataset = SDDDataset(test_files, obs_len=args.obs_len, pred_len=args.pred_len)
    # Balanced batches as in the original HDGT, the same in every version: no shuffling, and the
    # left-over samples as a last batch, so every trajectory is evaluated
    test_sizes = np.array([sample['agent_feature'].shape[0] for sample in test_dataset.samples])
    test_sampler = BalancedBatchSampler(test_sizes, seed_num=args.seed, gpu=0, gpu_cnt=1,
                                        batch_size=args.batch_size, is_train=False, keep_remainder=True)
    test_loader = DataLoader(
        test_dataset,
        batch_sampler=test_sampler,
        pin_memory=(device.type == "cuda"),
        collate_fn=partial(SDD_collate_fn, args=args, is_train=False),
        num_workers=args.num_worker,
    )
    print(f"Samples: {len(test_dataset)}   |   Batches: {len(test_loader)}")

    # ------------------------------------------------------------------
    # Evaluation: the checkpoint, then the baseline on the same batches and noise
    # ------------------------------------------------------------------
    length_lis     = [int(args.pred_len * 0.4), int(args.pred_len * 0.65), args.pred_len]
    agent_type_lis = ["PEDESTRIAN", "BIKER", "CAR", "BUS", "CART", "SKATER"]
    metric_type_lis = ["ade", "fde", "min_ade", "min_fde"]

    new = evaluate(model, model_module, test_loader, args, device, agent_type_lis, length_lis,
                   metric_type_lis, floors)
    del model

    base = None
    if args.baseline_checkpoint is not None:
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"\nBaseline   : {args.baseline_checkpoint}")
        base_checkpoint = torch.load(args.baseline_checkpoint, map_location="cpu")
        base_epoch = base_checkpoint.get("epoch", "?")
        base_goal_desc = goal_settings(base_checkpoint)
        # Its own scorer, from its weights
        base_model, base_scorer_desc = build_model(model_module, base_checkpoint, args, device)
        del base_checkpoint
        print(f"Baseline loaded — scorer={base_scorer_desc}; goal predictor: {base_goal_desc}")
        base = evaluate(base_model, model_module, test_loader, args, device, agent_type_lis, length_lis,
                        metric_type_lis, floors)
        del base_model

    # ------------------------------------------------------------------
    # Build report string
    # ------------------------------------------------------------------
    lines = []
    sep  = "=" * 80

    lines += [
        sep,
        f"HDGT Goal-CVAE — Evaluation Report ({args.split.upper()} split)",
        sep,
        f"Checkpoint  : {os.path.abspath(args.checkpoint)}",
        f"Saved epoch : {saved_epoch}",
        f"Selected on : {saved_metric} = {saved_val:.4f} px  "
        f"(on the {select_split.upper()} split)",
        f"Leakage     : {'NONE — selected on val, evaluated on ' + args.split.upper() if select_split == 'val' else 'YES — checkpoint was selected on TEST; these numbers are optimistic'}",
        f"Eval split  : {args.split}  ({len(test_files)} videos)",
        f"Eval samples: {len(test_dataset)}",
        f"Scale factor: {SCALE_FACTOR}  (metrics in original pixel coordinates)",
        f"K samples   : {args.num_prediction}",
        f"Refiner     : {'ON' if args.use_refiner else 'OFF'}",
        f"Scorer      : {scorer_desc}",
        f"Goal pred.  : {goal_desc}",
        f"Map nodes   : {'ON (shapes in the encoder graph)' if model_map_nodes else 'OFF'}",
        f"s_n floors  : {floors_source}",
    ]
    if base is not None:
        lines.append(f"Baseline    : {os.path.abspath(args.baseline_checkpoint)}  "
                     f"(see GOAL PREDICTOR vs BASELINE)")
    lines += [
        f"Random seed : {args.seed}",
        f"Eval time   : {new['elapsed']:.1f}s",
        sep,
    ]

    lines += comparison_block(new["recorder"], agent_type_lis, length_lis,
                              {"ADE": "ade", "FDE": "fde", "minADE": "min_ade", "minFDE": "min_fde"},
                              args.num_prediction, args.split)

    if new["use_scorer"]:
        lines += scorer_block(new["scorer_rows"], agent_type_lis, args.num_prediction, args.pred_len, scorer_desc)
    lines += map_block(new["map_rows"], agent_type_lis)

    # GOAL HEAD: the checkpoint's K goal heads (every trajectory with a true goal)
    goal = new["goal"]
    valid = goal["goal_valid"]
    first_desc = ("the scorer's pick" if new["use_scorer"]
                  else "slot 0: without a scorer the chains keep the goal order")
    if valid.any():
        diagnostics = model_module.goal_head_diagnostics(
            goal["goals"][valid], goal["gt"][valid], goal["s"][valid], goal["cls"][valid], agent_type_lis,
            SCALE_FACTOR, first_goal=goal["first"][valid],
            conf_logits=goal["conf"][valid] if goal["conf"] is not None else None, rho=rho)
        lines += model_module.goal_head_lines(diagnostics, agent_type_lis, (floors * SCALE_FACTOR).tolist(),
                                              rho, args.pred_len, first_desc)
    if new["dist"] is not None:
        lines += distribution_block(goal, new["dist"], agent_type_lis, SCALE_FACTOR)

    if base is not None:
        print(f"\nComparing with the baseline (paired bootstrap, {args.n_bootstrap} resamples) ...")
        base_goal = base["goal"]
        # Both passes must have seen the same trajectories in the same order
        for key in ("gt", "cls", "s", "goal_valid"):
            if not np.array_equal(goal[key], base_goal[key]):
                raise RuntimeError(f"The checkpoint and the baseline passes saw different trajectories "
                                   f"('{key}' differs); the comparison would be meaningless.")
        valid = (goal["goal_valid"] & ~np.isnan(goal["ade"]).any(axis=1) & ~np.isnan(goal["fde"]).any(axis=1)
                 & ~np.isnan(base_goal["ade"]).any(axis=1) & ~np.isnan(base_goal["fde"]).any(axis=1))
        same_ranking = (new["use_scorer"] == base["use_scorer"]
                        and (not new["use_scorer"] or new["scorer_type"] == base["scorer_type"]))
        lines += compare_lines(
            comparison_arrays(model_module, goal, valid, new["use_scorer"], rho),
            comparison_arrays(model_module, base_goal, valid, base["use_scorer"], rho),
            goal["cls"][valid], model_module, agent_type_lis, args.split, args.pred_len,
            args.n_bootstrap, args.seed,
            (os.path.abspath(args.checkpoint), [f"epoch {saved_epoch}; scorer: {scorer_desc}",
                                                f"goal predictor: {goal_desc}"]),
            (os.path.abspath(args.baseline_checkpoint), [f"epoch {base_epoch}; scorer: {base_scorer_desc}",
                                                         f"goal predictor: {base_goal_desc}",
                                                         f"eval time {base['elapsed']:.1f}s"]),
            same_ranking)

    report = "\n".join(lines)

    # ------------------------------------------------------------------
    # Print and save
    # ------------------------------------------------------------------
    print("\n" + report)

    save_dir  = os.path.dirname(os.path.abspath(args.checkpoint))
    # One file per split, so a val run never overwrites the final test report
    save_path = os.path.join(save_dir, f"{os.path.splitext(os.path.basename(args.checkpoint))[0]}_{args.split}_report.txt")
    with open(save_path, "w") as f:
        f.write(report + "\n")

    print(f"\nReport saved to: {save_path}")


if __name__ == "__main__":
    main()
