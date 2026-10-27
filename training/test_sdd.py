#!/usr/bin/env python3
"""
Standalone test evaluation for the HDGT Goal-CVAE model.

Loads a checkpoint, runs inference on the chosen split, and saves a full
metrics report as <checkpoint>_<split>_report.txt next to the checkpoint.

Defaults to the VAL split (the 8 VAL_VIDEOS held out of the train folder).
The test folder is the final holdout: run it once, at the end, with --split test.

The ranker is part of the model and trained with it, so it comes with the
checkpoint: ADE/FDE are its first chain, minADE/minFDE the best of the K.

Usage:
    python test_sdd.py --checkpoint logs/Hdgt_goal_DATE/snapshot/best_min_fde.pt
    python test_sdd.py --checkpoint logs/Hdgt_goal_DATE/snapshot/best_min_fde.pt --split test

All architecture flags default to the same values used in train.py.
Pass any flag explicitly if you trained with a non-default value.
"""

import os
import sys
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


def ranker_block(ranker_rows, agent_type_lis, num_prediction, pred_len, ranker_desc):
    """The RANKER RESULTS block of the report: how well the ranker orders the K chains"""
    K = num_prediction
    rows = {at: ranker_rows[at] for at in agent_type_lis if ranker_rows[at]}
    rows = {"ALL": [r for at in rows for r in rows[at]], **rows}

    def cat(rs, key):
        return np.concatenate([r[key] for r in rs], axis=0)

    lines = [
        f"RANKER RESULTS  ({ranker_desc}, {pred_len} steps = {pred_len * 0.4:.1f}s, pixels)",
        "-" * 80,
        "  The ranker sorts the K chains; its first chain gives ADE and FDE above.",
        "  hit@n    : % of trajectories whose best chain is among the ranker's first n",
        "  meanRank : mean position of the best chain in the ranker's order (0 = first)",
        "  picked   : error of the ranker's first chain      random: mean error over the K chains",
        "  goal     : error of the chain whose goal is closest to the true goal",
        "  best     : error of the best chain (minADE / minFDE)",
        "  closed   : share of the random-to-best gap the ranker closes, (random-picked)/(random-best)",
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
            rank = err.argmin(axis=1)            # chains come in the ranker's order
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

    # The ranker's absolute error estimate for its picked chain, over every trajectory
    ade = cat(rows["ALL"], "ade")
    valid = ~np.isnan(ade).any(axis=1)
    pred_err = cat(rows["ALL"], "pred_err")[valid][:, 0]
    actual = ade[valid][:, 0]
    corr = float(np.corrcoef(pred_err, actual)[0, 1]) if len(actual) > 1 else float("nan")
    lines += [
        "",
        "  Ranker error head (its predicted ADE for the picked chain, every trajectory):",
        f"    predicted {pred_err.mean():.3f} px vs actual {actual.mean():.3f} px  "
        f"(bias {pred_err.mean() - actual.mean():+.3f} px, correlation {corr:.3f})",
    ]
    lines += ["=" * 80]
    return lines


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
    # Map nodes in the graph: read from the checkpoint by default
    parser.add_argument("--use_map", type=str, default=None)

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
    if not os.path.isfile(args.checkpoint):
        print(f"ERROR: checkpoint not found: {args.checkpoint}")
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

    # Without ranker weights, ADE/FDE would be an untrained ranker's pick: refuse such a checkpoint
    if not any(k.startswith("ranker.") for k in checkpoint["model_state_dict"]):
        print("ERROR: the checkpoint has no ranker weights, so it was not trained by this "
              "version's train.py.\n       ADE/FDE would come from an untrained ranker.")
        sys.exit(1)

    # Map nodes in the graph, as trained (the checkpoint's args, else its weights), unless given
    if args.use_map is None:
        saved = checkpoint.get("args", {}).get("use_map")
        args.use_map = (saved if isinstance(saved, bool) else
                        any(k.startswith("encoder.map_emb") for k in checkpoint["model_state_dict"]))
    else:
        args.use_map = (args.use_map == "True")

    model_module = importlib.import_module(args.model_module)
    model = model_module.HDGT_model(input_dim=8, args=args)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    ranker_desc = "TrajRankNN, trained with the model"
    print(f"Model loaded — refiner={'ON' if args.use_refiner else 'OFF'}, ranker={ranker_desc}, "
          f"map={'ON' if args.use_map else 'OFF'}, K={args.num_prediction}")

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
    # Evaluation loop
    # ------------------------------------------------------------------
    length_lis     = [int(args.pred_len * 0.4), int(args.pred_len * 0.65), args.pred_len]
    agent_type_lis = ["PEDESTRIAN", "BIKER", "CAR", "BUS", "CART", "SKATER"]
    metric_type_lis = ["ade", "fde", "min_ade", "min_fde"]

    recorder = {
        f"{at}_{ln}_{mt}": AverageMeter()
        for at in agent_type_lis
        for ln in length_lis
        for mt in metric_type_lis
    }

    # Ranker results: every chain's error at the full horizon, in the ranker's order, with the
    # ranker's outputs for it. Per type, every trajectory counts once.
    ranker_rows = {at: [] for at in agent_type_lis}
    # Map compliance: each chain's and the GT's deepest point inside a forbidden shape, px.
    # With the map off too: the shapes come with every batch, so the two runs compare
    map_rows = {at: [] for at in agent_type_lis}

    print(f"\nRunning inference ...")
    t0 = time.time()

    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            if data is None:
                continue

            data["is_train"] = False
            data["gpu"] = max(args.gpu, 0)
            for key in data["cuda_tensor_lis"]:
                data[key] = data[key].to(device, non_blocking=True)

            trajectories, pred_bool_type_lis, _ = model(data)

            # trajectories: (N_agents, K, pred_len, 2)
            # refiner output is positions; raw diffusion output is velocities
            output_is_positions = getattr(model, "output_is_positions", False)
            if output_is_positions:
                pred_positions = trajectories.cpu().numpy()
            else:
                pred_positions = torch.cumsum(trajectories, dim=2).cpu().numpy()

            label_pos  = data["label_pos_lis"]
            label_mask = data["label_mask_lis"]

            # Both come in the same order as the returned chains
            goals_np = model.last_predicted_goals.cpu().numpy()                      # (N, K, 2)
            perr_np = model.last_pred_error.cpu().numpy()   # (N, K), already pixels: no SCALE_FACTOR

            pos_t = trajectories if output_is_positions else torch.cumsum(trajectories, dim=2)
            geom, forbid = data["map_geom_lis"], data["map_forbid_mask_lis"]
            depth_k = (model_module.map_depth(model_module.densify_path(pos_t), geom, forbid)
                       .amax(dim=-1) * SCALE_FACTOR).cpu().numpy()                  # (N, K)
            depth_gt = (model_module.map_depth(model_module.densify_path(label_pos), geom, forbid)
                        .amax(dim=-1) * SCALE_FACTOR).cpu().numpy()                 # (N,)

            for type_idx, at in enumerate(agent_type_lis):
                mask = pred_bool_type_lis[type_idx].cpu().numpy()
                if mask.sum() == 0:
                    continue

                preds  = pred_positions[mask]          # (M, K, pred_len, 2)
                labels = label_pos[mask].cpu().numpy()  # (M, pred_len, 2)
                masks  = label_mask[mask].cpu().numpy() # (M, pred_len)

                # Chain 0: the ranker's pick
                best_pred = preds[:, 0, :, :]

                ade_k, fde_k = chain_errors(preds, labels, masks, scale_factor=SCALE_FACTOR)
                goal_dist = np.linalg.norm(goals_np[mask] - labels[:, -1][:, np.newaxis], axis=-1)
                goal_dist[masks[:, -1] == 0] = np.nan    # no true goal without the last step
                ranker_rows[at].append({"ade": ade_k, "fde": fde_k, "goal_dist": goal_dist,
                                        "pred_err": perr_np[mask]})
                map_rows[at].append({"chains": depth_k[mask], "gt": depth_gt[mask]})

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
                print(f"  [{batch_idx + 1}/{len(test_loader)}]  {elapsed:.1f}s")

    elapsed_total = time.time() - t0
    print(f"Inference done in {elapsed_total:.1f}s")

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
        f"Ranker      : {ranker_desc}",
        f"Map         : {'ON (shapes in the graph; agents attend to them in every GNN layer)' if args.use_map else 'OFF'}",
        f"Random seed : {args.seed}",
        f"Eval time   : {elapsed_total:.1f}s",
        sep,
    ]

    lines += comparison_block(recorder, agent_type_lis, length_lis,
                              {"ADE": "ade", "FDE": "fde", "minADE": "min_ade", "minFDE": "min_fde"},
                              args.num_prediction, args.split)

    lines += ranker_block(ranker_rows, agent_type_lis, args.num_prediction, args.pred_len, ranker_desc)
    lines += map_block(map_rows, agent_type_lis)

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
