#!/usr/bin/env python3
"""
Standalone test evaluation for the HDGT GT-goal model (model.py).

Loads a checkpoint, runs inference on the chosen split, and saves a full
metrics report as <checkpoint>_<split>_report.txt next to the checkpoint.

Defaults to the VAL split (the 8 VAL_VIDEOS held out of the train folder).
The test folder is the final holdout: run it once, at the end, with --split test.

This model is conditioned on the GROUND-TRUTH goal (the last GT position) at
inference as well, so its numbers are an oracle upper bound, not comparable to
models that predict their goals.

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
    parser = argparse.ArgumentParser("HDGT GT-goal — Test Evaluation")
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
                        help="model file to load (the one train.py trains) — must match the checkpoint")

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
    print("HDGT GT-goal — Test Evaluation")
    print("=" * 80)
    print(f"Checkpoint : {args.checkpoint}")
    print(f"Device     : {device}")

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    saved_epoch = checkpoint.get("epoch", "?")
    saved_val   = checkpoint.get("val_loss", float("nan"))
    # Checkpoints written before the val split existed were selected on the test
    # set. Absent stamp therefore means "test", not "unknown" — assume the worse
    # of the two so a stale checkpoint cannot pass itself off as clean.
    select_split = checkpoint.get("select_split", "test")
    print(f"Saved epoch: {saved_epoch}   |   Saved val minFDE: {saved_val:.4f} px  "
          f"(on the {select_split.upper()} split)")
    if select_split == "test":
        print("WARNING: this checkpoint was selected on the TEST set. Test numbers "
              "from it are\n         optimistic. Retrain with --use_val_split True "
              "for a clean final result.")

    model_module = importlib.import_module(args.model_module)
    model = model_module.HDGT_model(input_dim=8, args=args)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    print(f"Model loaded — refiner={'ON' if args.use_refiner else 'OFF'}, K={args.num_prediction}")
    print("Goal input : GROUND TRUTH (oracle) — an upper bound, not a prediction result")

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

            for type_idx, at in enumerate(agent_type_lis):
                mask = pred_bool_type_lis[type_idx].cpu().numpy()
                if mask.sum() == 0:
                    continue

                preds  = pred_positions[mask]          # (M, K, pred_len, 2)
                labels = label_pos[mask].cpu().numpy()  # (M, pred_len, 2)
                masks  = label_mask[mask].cpu().numpy() # (M, pred_len)

                best_pred = preds[:, 0, :, :]           # best-of-1 (first sample)

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
        f"HDGT GT-goal — Evaluation Report ({args.split.upper()} split)",
        sep,
        f"Checkpoint  : {os.path.abspath(args.checkpoint)}",
        f"Saved epoch : {saved_epoch}",
        f"Val minFDE  : {saved_val:.4f} px  (on the {select_split.upper()} split)",
        f"Leakage     : {'NONE — selected on val, evaluated on ' + args.split.upper() if select_split == 'val' else 'YES — checkpoint was selected on TEST; these numbers are optimistic'}",
        f"Goal input  : GROUND TRUTH (oracle upper bound, not a prediction result)",
        f"Eval split  : {args.split}  ({len(test_files)} videos)",
        f"Eval samples: {len(test_dataset)}",
        f"Scale factor: {SCALE_FACTOR}  (metrics in original pixel coordinates)",
        f"K samples   : {args.num_prediction}",
        f"Refiner     : {'ON' if args.use_refiner else 'OFF'}",
        f"Random seed : {args.seed}",
        f"Eval time   : {elapsed_total:.1f}s",
        sep,
    ]

    lines += comparison_block(recorder, agent_type_lis, length_lis,
                              {"ADE": "ade", "FDE": "fde", "minADE": "min_ade", "minFDE": "min_fde"},
                              args.num_prediction, args.split)

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
