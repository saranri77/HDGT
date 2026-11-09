#!/usr/bin/env python3
"""
Visualize predicted trajectories on the SDD scene reference images.

Runs a checkpoint (default best_min_fde.pt) on the chosen split (--split, VAL by default; the test
folder is the final holdout), converts predictions from the model's
ego-centric scaled frame back to absolute pixel coordinates, and draws
observation / ground-truth / K predicted trajectories over
../sdd/images/<scene>/reference_video<N>.jpg.

Usage:
    python visualize_goal.py --checkpoint best_min_fde.pt

Outputs PNGs to ./visualizations/ (one per selected sample window).
Per scene, the --samples_per_scene windows with the largest agent motion
are selected (static windows make uninformative pictures).
"""

import os
import sys
import argparse
import importlib
import numpy as np
import torch
from torch.utils.data import DataLoader
from functools import partial

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["DGLBACKEND"] = "pytorch"

from sdd_dataset import (SDDDataset, SDD_collate_fn, SCALE_FACTOR,
                         resolve_split_files)

# Colors by role (photo background -> saturated hues + white halo under-strokes;
# CVD safety comes from line style, not color alone)
COL_OBS = "#00B8D9"    # observation: solid cyan
COL_GT = "#36B37E"     # ground truth: dashed green
COL_PRED = "#FFC400"   # other predictions: thin translucent amber
COL_TOP1 = "#FF5630"   # most-confident prediction: bold red-orange, solid
COL_ORACLE = "#6554C0" # lowest-ADE prediction (minADE chain): purple, dash-dot
COL_BESTGOAL = "#FF8B00"  # WTA winner goal (predicted goal closest to GT goal)


def rotate_inv(xy, psi):
    """Inverse of the collate-fn rotation: rotate by +psi (forward used -psi)."""
    c, s = np.cos(psi), np.sin(psi)
    x, y = xy[..., 0], xy[..., 1]
    return np.stack([x * c - y * s, y * c + x * s], axis=-1)


def main():
    parser = argparse.ArgumentParser("HDGT goal model — trajectory visualization")
    parser.add_argument("--checkpoint", type=str, default="best_min_fde.pt")
    parser.add_argument("--data_folder", type=str, default="../sdd/")
    parser.add_argument("--out_dir", type=str, default="visualizations")
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"],
                        help="'val' (default) = held-out validation videos. "
                             "'test' = the final holdout; leave it alone while iterating.")
    parser.add_argument("--samples_per_scene", type=int, default=3)
    parser.add_argument("--max_agents_per_plot", type=int, default=8)
    parser.add_argument("--focus_agent", action="store_true",
                        help="one figure per agent (single agent per plot, zoomed) — "
                             "the readable way to inspect the 20 goal markers")
    parser.add_argument("--zoom_margin", type=float, default=120.0,
                        help="pixels of padding around the agent in --focus_agent mode")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_worker", type=int, default=4)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)

    # Architecture — must match training (same defaults as test_sdd.py)
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
    parser.add_argument("--goal_loss_weight", type=float, default=0.5)
    parser.add_argument("--goal_curriculum_epochs", type=int, default=10)
    # Scorer: read from the checkpoint's weights by default, so the model is rebuilt as trained
    parser.add_argument("--use_scorer", type=str, default=None)
    parser.add_argument("--scorer_type", type=str, default=None, choices=["mode", "traj"])
    args = parser.parse_args()
    args.use_refiner = (args.use_refiner == "True")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    # The scorer the checkpoint was trained with, read from its weights (traj_scorer.* =
    # TrajScorer, mode_scorer.* = ModeScorer, neither = no scorer) unless given on the command line
    scorer_in_weights = next((kind for kind, prefix in (("traj", "traj_scorer."), ("mode", "mode_scorer."))
                              if any(k.startswith(prefix) for k in checkpoint["model_state_dict"])), None)
    if args.use_scorer is None:
        args.use_scorer = "True" if scorer_in_weights else "False"
    if args.scorer_type is None:
        args.scorer_type = scorer_in_weights or "mode"
    model_module = importlib.import_module("model")
    # Rebuilt as trained (goal predictor and its settings, map nodes, class statistics), as in
    # test_sdd.py; the loader adds the map nodes when the checkpoint has them
    from test_sdd import build_model, map_nodes_of
    model, _ = build_model(model_module, checkpoint, args, device, args.use_scorer, args.scorer_type)
    args.use_map_nodes = map_nodes_of(checkpoint)
    print(f"Loaded {args.checkpoint} (epoch {checkpoint.get('epoch', '?')}, "
          f"val minFDE {checkpoint.get('val_loss', float('nan')):.4f} px)")

    # ------------------------------------------------------------------
    # Eval data (shuffle=False so batch order == dataset.samples order)
    # ------------------------------------------------------------------
    test_files = resolve_split_files(args.data_folder, args.split)
    print(f"Visualizing the {args.split.upper()} split ({len(test_files)} videos)")
    dataset = SDDDataset(test_files, obs_len=args.obs_len, pred_len=args.pred_len)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        pin_memory=(device.type == "cuda"),
                        collate_fn=partial(SDD_collate_fn, args=args, is_train=False),
                        num_workers=args.num_worker)

    # ------------------------------------------------------------------
    # Inference: collect absolute-pixel trajectories per sample window
    # ------------------------------------------------------------------
    records = []       # one dict per sample window
    global_idx = 0     # index into dataset.samples
    denorm_checked = False

    with torch.no_grad():
        for batch_idx, data in enumerate(loader):
            if data is None:
                continue
            data["is_train"] = False
            data["gpu"] = args.gpu if device.type == "cuda" else 0
            for key in data["cuda_tensor_lis"]:
                data[key] = data[key].to(device, non_blocking=True)

            trajectories, _, _ = model(data)  # (N, K, 12, 2) confidence-sorted
            if not getattr(model, "output_is_positions", False):
                trajectories = torch.cumsum(trajectories, dim=2)
            preds_rel = trajectories.cpu().numpy()
            goals_rel = model.last_predicted_goals.cpu().numpy()  # (N, K, 2), same order
            gt_rel = data["label_pos_lis"].cpu().numpy()
            normal = data["normal_lis"]                     # (N, 3): ref_x, ref_y, ref_psi
            pred_nums = data["pred_num_lis"]

            offset = 0
            for i, pred_num in enumerate(pred_nums):
                sample = dataset.samples[global_idx + i]
                assert sample["pred_num"] == pred_num, \
                    f"sample alignment broken at dataset index {global_idx + i}"

                sl = slice(offset, offset + pred_num)
                ref = normal[sl, :2]                        # (n, 2)
                psi = normal[sl, 2]                         # (n,)

                # predictions: ego frame -> absolute pixels; preds_rel[sl] is
                # (n, K, 12, 2) so psi must broadcast as (n, 1, 1)
                preds_abs = rotate_inv(preds_rel[sl], psi[:, None, None])
                preds_abs = (preds_abs + ref[:, None, None, :]) * SCALE_FACTOR

                # predicted goals: (n, K, 2) -> absolute pixels
                goals_abs = rotate_inv(goals_rel[sl], psi[:, None])
                goals_abs = (goals_abs + ref[:, None, :]) * SCALE_FACTOR

                # sanity check the de-normalization once, against the dataset's
                # absolute GT: inverse(collate GT) must equal raw label_pos
                if not denorm_checked:
                    gt_abs_check = (rotate_inv(gt_rel[sl], psi[:, None]) +
                                    ref[:, None, :]) * SCALE_FACTOR
                    gt_abs_direct = sample["label_pos"] * SCALE_FACTOR
                    err = np.abs(gt_abs_check - gt_abs_direct).max()
                    assert err < 1e-2, f"de-normalization mismatch: {err:.4f} px"
                    print(f"De-normalization verified (max GT round-trip error {err:.2e} px)")
                    denorm_checked = True

                obs_abs = sample["agent_feature"][:pred_num, :, :2] * SCALE_FACTOR
                gt_abs = sample["label_pos"] * SCALE_FACTOR

                move = np.linalg.norm(gt_abs[:, -1] - obs_abs[:, -1], axis=-1)
                records.append({
                    "scene_id": sample["scene_id"],
                    "sample_idx": global_idx + i,
                    "obs": obs_abs, "gt": gt_abs, "preds": preds_abs,
                    "goals": goals_abs,
                    "move": move,
                })
                offset += pred_num
            global_idx += len(pred_nums)

            if (batch_idx + 1) % 10 == 0:
                print(f"  inference [{batch_idx + 1}/{len(loader)}]")

    print(f"Inference done: {len(records)} sample windows")

    # ------------------------------------------------------------------
    # Goal-predictor accuracy over the whole evaluated split (pixels).
    # "top-conf" = the goal the model itself ranks first (no GT used).
    # "best of K" = the goal closest to the GT goal — what the WTA loss
    #               optimizes; measures the goal predictor independent of
    #               how well the confidence head ranks the modes.
    # ------------------------------------------------------------------
    e_top, e_best, e_mean = [], [], []
    for rec in records:
        errs = np.linalg.norm(rec["goals"] - rec["gt"][:, -1][:, None, :], axis=-1)
        e_top.append(errs[:, 0])
        e_best.append(errs.min(axis=1))
        e_mean.append(errs.mean(axis=1))
    e_top = np.concatenate(e_top)
    e_best = np.concatenate(e_best)
    e_mean = np.concatenate(e_mean)
    print(f"\n{'='*68}\nGOAL PREDICTOR ACCURACY  ({len(e_top)} agents, pixels)\n{'-'*68}")
    print(f"  Top-confidence goal error : mean {e_top.mean():7.2f}   median {np.median(e_top):7.2f}")
    print(f"  Best-of-K goal error (WTA): mean {e_best.mean():7.2f}   median {np.median(e_best):7.2f}")
    print(f"  Mean over all K goals     : mean {e_mean.mean():7.2f}   median {np.median(e_mean):7.2f}")
    print(f"  Ranking gap (top - best)  : mean {(e_top - e_best).mean():7.2f}"
          f"   [how much a perfect ranker would recover]")
    print(f"{'='*68}\n")

    # ------------------------------------------------------------------
    # Goal-conditioning fidelity: does chain k actually END at goal k?
    # The goal is a soft conditioning signal (one addend in cond_emb) and
    # the refiner is supervised against GT with no goal penalty, so the
    # endpoint is free to drift. This measures how much it does.
    # Compared against the goal's own distance to GT, to say whether
    # drift helps (toward GT) or hurts.
    # ------------------------------------------------------------------
    d_follow, d_goal_gt, d_end_gt = [], [], []
    for rec in records:
        endpoints = rec["preds"][:, :, -1, :]          # (n, K, 2)
        goals = rec["goals"]                            # (n, K, 2)
        gt_goal = rec["gt"][:, -1][:, None, :]          # (n, 1, 2)
        d_follow.append(np.linalg.norm(endpoints - goals, axis=-1).ravel())
        d_goal_gt.append(np.linalg.norm(goals - gt_goal, axis=-1).ravel())
        d_end_gt.append(np.linalg.norm(endpoints - gt_goal, axis=-1).ravel())
    d_follow = np.concatenate(d_follow)
    d_goal_gt = np.concatenate(d_goal_gt)
    d_end_gt = np.concatenate(d_end_gt)

    print(f"{'='*68}\nGOAL-CONDITIONING FIDELITY  ({len(d_follow)} chains, pixels)\n{'-'*68}")
    print(f"  |chain endpoint - its own goal| : mean {d_follow.mean():7.2f}   "
          f"median {np.median(d_follow):7.2f}")
    print(f"  |goal - GT goal|                : mean {d_goal_gt.mean():7.2f}   "
          f"median {np.median(d_goal_gt):7.2f}")
    print(f"  |chain endpoint - GT goal|      : mean {d_end_gt.mean():7.2f}   "
          f"median {np.median(d_end_gt):7.2f}")
    ratio = d_follow.mean() / max(d_goal_gt.mean(), 1e-6)
    print(f"{'-'*68}")
    print(f"  Drift / goal-error ratio: {ratio:.2f}  "
          f"(<<1 = decoder follows its goal closely; ~1 or more = goal largely ignored)")
    better = (d_end_gt < d_goal_gt).mean() * 100
    print(f"  Endpoint closer to GT than its goal was: {better:.1f}% of chains  "
          f"(>50% = drift tends to help)")
    print(f"{'='*68}\n")

    # ------------------------------------------------------------------
    # Per scene, plot the windows with the most agent motion
    # ------------------------------------------------------------------
    by_scene = {}
    for rec in records:
        by_scene.setdefault(rec["scene_id"], []).append(rec)

    legend_handles = [
        Line2D([], [], color=COL_OBS, lw=2.5, label="Observed (3.2s)"),
        Line2D([], [], color=COL_GT, lw=2.5, ls="--", label="Ground truth (4.8s)"),
        Line2D([], [], color=COL_TOP1, lw=2.5, label="Predicted traj. (top confidence)"),
        Line2D([], [], color=COL_ORACLE, lw=2.5, ls="-.", label="Predicted traj. (minADE = best path)"),
        Line2D([], [], color=COL_ORACLE, marker="D", ls="none", markerfacecolor="none",
               markeredgewidth=1.8, label="minFDE endpoint (when a different traj.)"),
        Line2D([], [], color=COL_PRED, lw=1.5, alpha=0.6, label="Other predicted traj."),
        Line2D([], [], color=COL_GT, marker="s", ls="none", label="GT goal (true endpoint)"),
        Line2D([], [], color=COL_TOP1, marker="x", ls="none", markeredgewidth=2.5,
               label="Predicted goal, top confidence"),
        Line2D([], [], color=COL_BESTGOAL, marker="x", ls="none", markeredgewidth=2.5,
               label="Predicted goal, closest to GT (WTA winner)"),
        Line2D([], [], color=COL_PRED, marker="x", ls="none",
               label="Predicted goals, other 18"),
    ]

    def draw_agent(ax, rec, a):
        """Draw one agent's observation, GT, K predictions and K goals.
        Returns a stats dict for the caption."""
        obs, gt, preds = rec["obs"][a], rec["gt"][a], rec["preds"][a]
        goals = rec["goals"][a]                      # (K, 2), confidence-sorted
        last_obs = obs[-1:]
        gt_line = np.concatenate([last_obs, gt], axis=0)

        # Oracle selections — these use GT, so they are diagnostics, not
        # model choices. minADE = best average path; minFDE = best endpoint;
        # they can be different trajectories.
        dists = np.linalg.norm(preds - gt[None], axis=-1)  # (K, 12)
        min_ade_k = int(dists.mean(-1).argmin())
        min_fde_k = int(dists[:, -1].argmin())

        # WTA winner GOAL: the predicted goal closest to the GT goal. This is
        # what the training goal-loss optimizes; it measures the goal
        # predictor itself, independent of trajectory ranking.
        goal_errs = np.linalg.norm(goals - gt[-1][None], axis=-1)  # (K,)
        best_goal_k = int(goal_errs.argmin())

        # white halos first, so every colored line stays readable on the photo
        ax.plot(obs[:, 0], obs[:, 1], color="white", lw=4.5, alpha=0.9,
                solid_capstyle="round", zorder=2)
        ax.plot(gt_line[:, 0], gt_line[:, 1], color="white", lw=4.5,
                alpha=0.9, solid_capstyle="round", zorder=2)

        for k in range(preds.shape[0] - 1, 0, -1):  # non-top modes, faint
            p = np.concatenate([last_obs, preds[k]], axis=0)
            ax.plot(p[:, 0], p[:, 1], color=COL_PRED, lw=1.3, alpha=0.35, zorder=3)
        ax.plot(obs[:, 0], obs[:, 1], color=COL_OBS, lw=2.5,
                solid_capstyle="round", zorder=4)
        ax.plot(gt_line[:, 0], gt_line[:, 1], color=COL_GT, lw=2.5, ls="--", zorder=5)

        # minADE trajectory (purple dash-dot)
        oracle = np.concatenate([last_obs, preds[min_ade_k]], axis=0)
        ax.plot(oracle[:, 0], oracle[:, 1], color="white", lw=4.0,
                alpha=0.9, zorder=5, solid_capstyle="round")
        ax.plot(oracle[:, 0], oracle[:, 1], color=COL_ORACLE, lw=2.5,
                ls="-.", zorder=6, solid_capstyle="round")

        # top-confidence trajectory (red, drawn last so the model's own choice
        # stays visible when it overlaps the oracle)
        top1 = np.concatenate([last_obs, preds[0]], axis=0)
        ax.plot(top1[:, 0], top1[:, 1], color="white", lw=4.0, alpha=0.9,
                zorder=6, solid_capstyle="round")
        ax.plot(top1[:, 0], top1[:, 1], color=COL_TOP1, lw=2.5, zorder=7,
                solid_capstyle="round")

        # GOALS: faint amber for the rest, orange for the WTA winner,
        # red for the model's top-confidence pick
        other = [k for k in range(goals.shape[0]) if k not in (0, best_goal_k)]
        ax.scatter(goals[other, 0], goals[other, 1], s=30, color=COL_PRED,
                   marker="x", alpha=0.55, linewidths=1.5, zorder=8)
        if best_goal_k != 0:
            ax.scatter(goals[best_goal_k, 0], goals[best_goal_k, 1], s=90,
                       color=COL_BESTGOAL, marker="x", linewidths=2.8, zorder=9)
        ax.scatter(goals[0, 0], goals[0, 1], s=90, color=COL_TOP1,
                   marker="x", linewidths=2.8, zorder=10)

        ax.scatter(*last_obs[0], s=45, color=COL_OBS, edgecolor="white",
                   lw=1.2, zorder=10)
        ax.scatter(*gt[-1], s=70, color=COL_GT, edgecolor="white",
                   marker="s", lw=1.2, zorder=10)
        ax.scatter(*preds[0, -1], s=55, color=COL_TOP1, edgecolor="white",
                   lw=1.2, zorder=10)
        ax.scatter(*preds[min_ade_k, -1], s=55, color=COL_ORACLE,
                   edgecolor="white", lw=1.2, zorder=10)
        if min_fde_k != min_ade_k:
            ax.scatter(*preds[min_fde_k, -1], s=75, facecolor="none",
                       edgecolor=COL_ORACLE, marker="D", lw=1.8, zorder=10)

        return {
            "ade_top1": float(dists[0].mean()),
            "fde_top1": float(dists[0, -1]),
            "min_ade": float(dists[min_ade_k].mean()),
            "min_fde": float(dists[:, -1].min()),
            "goal_err_top1": float(goal_errs[0]),
            "goal_err_best": float(goal_errs[best_goal_k]),
            "best_goal_k": best_goal_k,
            "min_ade_k": min_ade_k,
            "min_fde_k": min_fde_k,
            "xy": np.concatenate([obs, gt, preds.reshape(-1, 2), goals], axis=0),
        }

    n_saved = 0
    for scene_id, recs in sorted(by_scene.items()):
        scene, video = scene_id.split("_video")
        img_path = os.path.join(args.data_folder, "images", scene,
                                f"reference_video{video}.jpg")
        if not os.path.isfile(img_path):
            # tolerate filename typos (e.g. coupa's reference_vidoe3.jpg):
            # accept any reference_*<N>.jpg for this video number
            import glob
            candidates = glob.glob(os.path.join(
                args.data_folder, "images", scene, f"reference_*{video}.jpg"))
            if candidates:
                img_path = candidates[0]
                print(f"NOTE: using {os.path.basename(img_path)} for {scene_id}")
            else:
                print(f"WARNING: no reference image for {scene_id} ({img_path}); skipping")
                continue
        img = plt.imread(img_path)

        h, w = img.shape[:2]
        recs = sorted(recs, key=lambda r: r["move"].max(), reverse=True)

        for rank, rec in enumerate(recs[:args.samples_per_scene]):
            if args.focus_agent:
                # One figure per agent, zoomed — the readable way to inspect
                # 20 goal markers without 8 agents' worth of clutter.
                agents = np.argsort(-rec["move"])[:args.max_agents_per_plot]
                for slot, a in enumerate(agents):
                    fig, ax = plt.subplots(figsize=(11, 8), dpi=110)
                    ax.imshow(img)
                    st = draw_agent(ax, rec, a)

                    xy = st["xy"]
                    m = args.zoom_margin
                    ax.set_xlim(max(0, xy[:, 0].min() - m), min(w, xy[:, 0].max() + m))
                    ax.set_ylim(min(h, xy[:, 1].max() + m), max(0, xy[:, 1].min() - m))

                    ax.set_title(
                        f"{scene_id} — window #{rec['sample_idx']}, agent {a}",
                        fontsize=12)
                    caption = (
                        f"Trajectory:  top-conf ADE {st['ade_top1']:.1f} / FDE {st['fde_top1']:.1f} px"
                        f"   |   best-of-20  minADE {st['min_ade']:.1f} / minFDE {st['min_fde']:.1f} px\n"
                        f"Goal:  top-conf error {st['goal_err_top1']:.1f} px"
                        f"   |   closest of 20 (WTA winner, head {st['best_goal_k']}) {st['goal_err_best']:.1f} px"
                    )
                    ax.set_xlabel(caption, fontsize=9.5, labelpad=8)
                    ax.legend(handles=legend_handles, loc="upper right", fontsize=7.5,
                              framealpha=0.88)
                    ax.set_xticks([]); ax.set_yticks([])

                    out_path = os.path.join(
                        args.out_dir,
                        f"{scene_id}_w{rec['sample_idx']}_agent{a}.png")
                    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.08)
                    plt.close(fig)
                    n_saved += 1
                    print(f"  saved {out_path}")
            else:
                # Scene overview: several agents on the full frame.
                order = np.argsort(-rec["move"])[:args.max_agents_per_plot]
                fig, ax = plt.subplots(figsize=(w / 100, h / 100), dpi=100)
                ax.imshow(img)
                ax.set_xlim(0, w)
                ax.set_ylim(h, 0)

                for a in order:
                    draw_agent(ax, rec, a)

                shown = len(order)
                total = rec["obs"].shape[0]
                agents_note = (f"{shown}/{total} agents" if shown < total
                               else f"{total} agents")
                ax.set_title(f"{scene_id} — window #{rec['sample_idx']} ({agents_note})",
                             fontsize=13)
                ax.legend(handles=legend_handles, loc="upper right", fontsize=9,
                          framealpha=0.85)
                ax.set_axis_off()

                out_path = os.path.join(args.out_dir, f"{scene_id}_top{rank + 1}.png")
                fig.savefig(out_path, bbox_inches="tight", pad_inches=0.05)
                plt.close(fig)
                n_saved += 1
                print(f"  saved {out_path}")

    print(f"\nDone: {n_saved} figures in {os.path.abspath(args.out_dir)}/")


if __name__ == "__main__":
    main()
