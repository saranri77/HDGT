"""
Test script for HDGT on SDD dataset
Evaluates a checkpoint on one split (--split: val by default, or test / train) and computes all metrics
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")
import argparse
import time
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from functools import partial
import importlib

# Import dataset
from sdd_dataset import SDDDataset, SDD_collate_fn, SCALE_FACTOR, split_files, BalancedBatchSampler

# Set seed for reproducibility
SEED = 42

def set_seed(seed=42):
    """Set seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Full float32 instead of TF32 (PyTorch's default on Ampere GPUs, ~3 significant digits):
    # with TF32 a trajectory's prediction depends on which samples share its batch, so results
    # would change with --batch_size. train.py keeps the default, as the original HDGT
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    print(f"Random seed set to: {seed}")

# Argument parser
parser = argparse.ArgumentParser('HDGT Test Script for SDD')
parser.add_argument('--checkpoint', type=str, required=True, help='Path to the model checkpoint to evaluate (e.g. best_min_fde.pt)')
# Split to evaluate, defined in sdd_dataset.split_files. val by default: the test split is the
# final holdout and should be evaluated once, at the end
parser.add_argument('--split', type=str, default='val', choices=['val', 'test', 'train'], help='Split to evaluate (default: %(default)s)')
# The sdd dataset folder is in the parent directory of these scripts (../sdd), found from this
# file's location, so it does not depend on the directory the script is started from
DEFAULT_DATA_FOLDER = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "sdd"))
parser.add_argument('--data_folder', type=str, default=DEFAULT_DATA_FOLDER, help='SDD folder with train/, test/ and shapes/ (default: %(default)s)')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size for evaluation')
parser.add_argument('--num_worker', type=int, default=8, help='Number of workers for dataloader')
parser.add_argument('--gpu', type=int, default=0, help='GPU device ID')

# Model hyperparameters (should match training configuration)
parser.add_argument('--obs_len', type=int, default=8, help='Observation length')
parser.add_argument('--pred_len', type=int, default=12, help='Prediction length')
parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension')
parser.add_argument('--head_dim', type=int, default=32, help='Attention head dimension')
parser.add_argument('--num_of_gnn_layer', type=int, default=6, help='Number of GNN layers')
parser.add_argument('--num_prediction', type=int, default=6, help='Number of trajectory modes')
parser.add_argument('--refine_num', type=int, default=5, help='Number of refinement iterations')
parser.add_argument('--output_vel', type=str, default="True", help='Output velocity')
parser.add_argument('--cumsum_vel', type=str, default="True", help='Cumulative sum velocity')
parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate')
parser.add_argument('--agent_drop', type=float, default=0.0, help='Agent dropout (disabled for testing)')


def euclid_np(label, pred):
    """Euclidean distance between 2D points"""
    return np.sqrt((label[..., 0] - pred[..., 0])**2 + (label[..., 1] - pred[..., 1])**2)


def cal_ade_fde_mr(labels, preds, masks):
    """
    Calculate ADE, FDE, and MR for single mode predictions
    
    Args:
        labels: Ground truth trajectories (N, T, 2)
        preds: Predicted trajectories (N, T, 2)
        masks: Valid timestep masks (N, T)
    
    Returns:
        metrics: [ade, fde, mr]
        counts: [ade_count, fde_count, mr_count]
    """
    if labels.shape[0] == 0:
        return None, None
    
    l2_norm = euclid_np(labels, preds) * SCALE_FACTOR  # convert scaled units -> pixels

    # ADE: Average over valid timesteps
    masks_sum = masks.sum(1)
    ade_indices = masks_sum != 0
    ade_cnt = ade_indices.sum()
    ade = 0.0
    if ade_cnt > 0:
        ade = ((l2_norm[ade_indices] * masks[ade_indices]).sum(1) / masks_sum[ade_indices]).mean()

    # FDE: Final displacement error
    fde_indices = masks[:, -1] != 0
    fde_cnt = fde_indices.sum()
    fde = 0.0
    mr = 0.0
    if fde_cnt > 0:
        fde = l2_norm[fde_indices, -1]
        mr = (fde > 2.0 * SCALE_FACTOR).mean()  # threshold in pixels
        fde = fde.mean()
    
    return [ade, fde, mr], [ade_cnt, fde_cnt, fde_cnt]


def cal_min6_ade_fde_mr(preds, labels, masks):
    """
    Calculate minADE, minFDE and MR over the K predicted modes (named ade6/fde6/mr6 from K = 6)
    
    Args:
        preds: Predicted trajectories (N, K, T, 2)
        labels: Ground truth trajectories (N, T, 2)
        masks: Valid timestep masks (N, T)
    
    Returns:
        metrics: [ade6, fde6, mr6]
        counts: [ade_count, fde_count, mr_count]
    """
    if labels.shape[0] == 0:
        return None, None
    
    l2_norm = euclid_np(labels[:, np.newaxis, :, :], preds) * SCALE_FACTOR  # convert scaled units -> pixels

    # minADE: Minimum ADE across the K modes
    masks_sum = masks.sum(1)
    ade_indices = masks_sum != 0
    ade_cnt = ade_indices.sum()
    ade6 = 0.0
    if ade_cnt > 0:
        ade6 = ((l2_norm[ade_indices] * masks[ade_indices, np.newaxis, :]).sum(-1) / masks_sum[ade_indices][:, np.newaxis]).min(-1).mean()

    # minFDE: Minimum FDE across the K modes
    fde_indices = masks[:, -1] != 0
    fde_cnt = fde_indices.sum()
    fde6 = 0.0
    mr6 = 0.0
    if fde_cnt > 0:
        fde6 = l2_norm[fde_indices, :, -1].min(-1)
        mr6 = (fde6 > 2.0 * SCALE_FACTOR).mean()  # threshold in pixels
        fde6 = fde6.mean()
    
    return [ade6, fde6, mr6], [ade_cnt, fde_cnt, fde_cnt]


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def test_model(model, dataloader, args, device):
    """
    Evaluate the model on the split chosen with --split

    Args:
        model: HDGT model
        dataloader: Dataloader of that split
        args: Arguments
        device: CUDA device
    
    Returns:
        Dictionary of all metrics
    """
    model.eval()
    
    # SDD: Evaluate at 1.6s, 2.8s, and full 4.8s (4, 7, 12 timesteps at 2.5 FPS)
    length_lis = [int(args.pred_len * 0.4), int(args.pred_len * 0.65), args.pred_len]
    
    # SDD has 6 agent types
    agent_type_lis = ["PEDESTRIAN", "BIKER", "CAR", "BUS", "CART", "SKATER"]
    metric_type_lis = ["ade", "fde", "mr", "ade6", "fde6", "mr6"]
    acc_metric_type_lis = ["accs", "acc6s"]
    
    # Initialize recorders for all metrics
    recorder = {}
    for agent_type in agent_type_lis:
        for acc_metric_type in acc_metric_type_lis:
            recorder[agent_type + "_" + acc_metric_type] = AverageMeter()
        for length in length_lis:
            for metric_type in metric_type_lis:
                recorder[agent_type + "_" + str(length) + "_" + metric_type] = AverageMeter()
    
    # Loss recorders
    recorder["loss"] = AverageMeter()
    recorder["reg_loss"] = AverageMeter()
    recorder["cls_loss"] = AverageMeter()
    
    # Loss criteria
    reg_criteria = torch.nn.SmoothL1Loss(reduction="none").to(device)
    cls_weight = 0.1
    reg_weight = 50.0
    refine_num = args.refine_num
    num_prediction = args.num_prediction
    
    start_time = time.time()
    total_samples = 0
    
    print("\n" + "="*80)
    print(f"Starting evaluation on the {args.split} split...")
    print("="*80)
    
    with torch.no_grad():
        for batch_index, data in enumerate(dataloader):
            data["is_train"] = False
            data["gpu"] = args.gpu
            
            # Move tensors to GPU
            for tensor_name in data["cuda_tensor_lis"]:
                data[tensor_name] = data[tensor_name].to(device, non_blocking=True)
            
            num_of_sample = len(data["pred_num_lis"])
            total_samples += num_of_sample
            
            # Forward pass
            agent_reg_res, agent_cls_res, pred_indice_bool_type_lis = model(data)
            
            # Get labels for each agent type (6 types)
            reg_labels = [data["label_lis"][pred_indice_bool_type_lis[_]] for _ in range(6)]
            auxiliary_labels = [data["auxiliary_label_lis"][pred_indice_bool_type_lis[_]] for _ in range(6)]
            auxiliary_labels_future = [data["auxiliary_label_future_lis"][pred_indice_bool_type_lis[_]] for _ in range(6)]
            label_masks = [data["label_mask_lis"][pred_indice_bool_type_lis[_]] for _ in range(6)]
            
            # Compute loss
            agent_closest_index_lis = [[] for _ in range(6)]
            loss = 0.0
            reg_loss = 0.0
            cls_loss = 0.0
            total_num_of_mask = 0.0
            total_num_of_agent = 0.0
            
            for agent_type_index in range(6):
                if agent_reg_res[agent_type_index].shape[0] == 0:
                    continue
                
                num_of_mask_per_agent = label_masks[agent_type_index].sum(dim=-1)
                mask_sum = num_of_mask_per_agent.sum()
                
                if mask_sum != 0:
                    # Compute distance between predictions and labels
                    dist_between_pred_label = reg_criteria(
                        agent_reg_res[agent_type_index],
                        reg_labels[agent_type_index].unsqueeze(1).unsqueeze(1).repeat(1, refine_num+1, num_prediction, 1, 1)
                    ).mean(-1)
                    
                    dist_between_pred_label = (dist_between_pred_label * label_masks[agent_type_index].unsqueeze(1).unsqueeze(1)).sum(-1) / (num_of_mask_per_agent.unsqueeze(-1).unsqueeze(-1) + 1)
                    
                    # Find closest prediction mode
                    agent_closest_index = dist_between_pred_label[:, -1, :].argmin(dim=-1)
                    
                    reg_loss += (dist_between_pred_label[torch.arange(agent_closest_index.shape[0]), :, agent_closest_index]).sum() / (refine_num + 1)
                    
                    # Classification loss
                    log_pis = agent_cls_res[agent_type_index]
                    log_pis = log_pis - torch.logsumexp(log_pis, dim=-1, keepdim=True)
                    log_pi = log_pis[torch.arange(agent_closest_index.shape[0]), agent_closest_index].sum()
                    cls_loss += (-log_pi)
                    
                    agent_closest_index_lis[agent_type_index] = agent_closest_index
                    total_num_of_agent += len(agent_closest_index)
                    total_num_of_mask += mask_sum
            
            if total_num_of_agent > 0:
                loss = cls_loss / total_num_of_agent * cls_weight + reg_loss / total_num_of_mask * reg_weight
                reg_loss_cnt = total_num_of_mask
                cls_loss_cnt = total_num_of_agent
                
                recorder["loss"].update(loss.item(), num_of_sample)
                recorder["cls_loss"].update(cls_loss.item() / cls_loss_cnt * cls_weight, cls_loss_cnt)
                recorder["reg_loss"].update(reg_loss.item() / reg_loss_cnt * reg_weight, reg_loss_cnt)
            
            # Compute metrics for each agent type
            for agent_type_index in range(6):
                now_agent_cls_res = agent_cls_res[agent_type_index]
                if now_agent_cls_res.shape[0] == 0:
                    continue
                
                now_agent_reg_res = agent_reg_res[agent_type_index][:, -1, ...].detach().cpu().numpy()
                now_labels = reg_labels[agent_type_index].detach().cpu().numpy()
                now_label_masks = label_masks[agent_type_index].detach().cpu().numpy()
                now_agent_closest_index = agent_closest_index_lis[agent_type_index].detach().cpu().numpy()
                now_cls_sorted_index = now_agent_cls_res.argsort(dim=-1, descending=True).detach().cpu().numpy()
                now_agent_cls_res = now_agent_cls_res.detach().cpu().numpy()
                
                # Classification accuracy
                cls_acc = 0.0
                cls_acc6 = 0.0
                best_preds = [0] * len(now_labels)
                best_6preds = [0] * len(now_labels)
                
                for item_index in range(len(now_labels)):
                    # Check if best prediction matches ground truth closest mode
                    if now_agent_closest_index[item_index] == now_cls_sorted_index[item_index][0]:
                        cls_acc += 1.0
                    if now_agent_closest_index[item_index] in now_cls_sorted_index[item_index][:6].tolist():
                        cls_acc6 += 1.0
                    
                    # Get best single prediction and all K = --num_prediction predictions, most
                    # confident first (top-6 of 6 by default)
                    best_preds[item_index] = now_agent_reg_res[item_index, ...][now_cls_sorted_index[item_index][0], :, :]
                    best_6preds[item_index] = now_agent_reg_res[item_index][now_cls_sorted_index[item_index], :, :]
                
                cls_acc /= now_agent_reg_res.shape[0]
                cls_acc6 /= now_agent_reg_res.shape[0]
                recorder[agent_type_lis[agent_type_index] + "_accs"].update(cls_acc, now_agent_reg_res.shape[0])
                recorder[agent_type_lis[agent_type_index] + "_acc6s"].update(cls_acc6, now_agent_reg_res.shape[0])
                
                best_preds = np.stack(best_preds, axis=0)
                best_6preds = np.stack(best_6preds, axis=0)
                
                # Compute metrics for different horizons
                for length_indices in range(3):
                    # Single-mode evaluation
                    res_lis, res_cnt_lis = cal_ade_fde_mr(
                        best_preds[:, :length_lis[length_indices], :],
                        now_labels[:, :length_lis[length_indices], :],
                        now_label_masks[:, :length_lis[length_indices]]
                    )
                    if res_lis:
                        for metric_indices, metric_type in enumerate(["ade", "fde", "mr"]):
                            if res_cnt_lis[metric_indices] > 0:
                                recorder[agent_type_lis[agent_type_index] + "_" + str(length_lis[length_indices]) + "_" + metric_type].update(
                                    res_lis[metric_indices], res_cnt_lis[metric_indices]
                                )
                    
                    # Multi-mode evaluation (best of the K modes; the ade6/fde6 names date from K = 6)
                    res_lis, res_cnt_lis = cal_min6_ade_fde_mr(
                        best_6preds[:, :, :length_lis[length_indices], :],
                        now_labels[:, :length_lis[length_indices], :],
                        now_label_masks[:, :length_lis[length_indices]]
                    )
                    if res_lis:
                        for metric_indices, metric_type in enumerate(["ade6", "fde6", "mr6"]):
                            if res_cnt_lis[metric_indices] > 0:
                                recorder[agent_type_lis[agent_type_index] + "_" + str(length_lis[length_indices]) + "_" + metric_type].update(
                                    res_lis[metric_indices], res_cnt_lis[metric_indices]
                                )
            
            # Print progress
            if (batch_index + 1) % 10 == 0:
                print(f"Processed {batch_index + 1}/{len(dataloader)} batches ({total_samples} samples)", flush=True)
    
    elapsed_time = time.time() - start_time
    print(f"\nEvaluation completed in {elapsed_time:.2f} seconds")
    print(f"Total samples evaluated: {total_samples}")
    
    return recorder, length_lis, agent_type_lis


def classifier_block(recorder, agent_type_lis):
    """The original HDGT's own heads -- component diagnostics, not trajectory metrics: its losses
    on the split, and per agent type how often the mode classifier ranks the mode closest to the
    truth first (accs) or among its first 6 (acc6s)"""
    lines = [
        "MODE CLASSIFIER  (component diagnostics of the original HDGT)",
        "-" * 80,
        f"  Loss: {recorder['loss'].avg:.8f}",
        f"  Classification Loss: {recorder['cls_loss'].avg:.8f}",
        f"  Regression Loss: {recorder['reg_loss'].avg:.8f}",
        "",
        "  accs : share of trajectories whose most confident mode is the closest one",
        "  acc6s: share whose closest mode is among the 6 most confident",
    ]
    for agent_type in agent_type_lis:
        if recorder[agent_type + "_accs"].count > 0:
            lines.append(f"    {agent_type:<12} accs {recorder[agent_type + '_accs'].avg:.4f}   "
                         f"acc6s {recorder[agent_type + '_acc6s'].avg:.4f}")
    lines.append("=" * 80)
    return lines


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
    args = parser.parse_args()
    
    # Set seed for reproducibility
    set_seed(SEED)
    
    # Set environment
    os.environ["DGLBACKEND"] = "pytorch"
    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")
    
    print("\n" + "="*80)
    print("HDGT Test Script for SDD Dataset")
    print("="*80)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Split: {args.split}")
    print(f"Data folder: {args.data_folder}")
    print(f"Batch size: {args.batch_size}")
    print(f"GPU: {args.gpu}")
    print(f"Random seed: {SEED}")
    print("="*80 + "\n")

    # Check if checkpoint file exists
    if not os.path.exists(args.checkpoint):
        print(f"ERROR: Checkpoint file not found at {args.checkpoint}")
        sys.exit(1)

    # Load the chosen split (same split definition as train.py)
    print(f"Loading {args.split} split...")
    try:
        eval_files = split_files(args.data_folder, args.split)
    except (FileNotFoundError, ValueError) as error:
        print(f"ERROR: {error}")
        sys.exit(1)

    if len(eval_files) == 0:
        print(f"ERROR: No pickle files found for the {args.split} split in {args.data_folder}")
        sys.exit(1)

    print(f"Found {len(eval_files)} {args.split} files")

    # Create dataset
    eval_dataset = SDDDataset(eval_files, obs_len=args.obs_len, pred_len=args.pred_len)

    # Create dataloader with the balanced batches of train.py's validation (no shuffling, the
    # left-over samples as a last batch): in file order, crowded stretches of a video land in the
    # same batch and can need several times the memory of an average one. With TF32 off (see
    # set_seed) the ADE/FDE metrics do not depend on how samples are batched; the averaged loss
    # does a little, since it is averaged per batch
    eval_sizes = np.array([sample['agent_feature'].shape[0] for sample in eval_dataset.samples])
    eval_sampler = BalancedBatchSampler(eval_sizes, seed_num=SEED, gpu=0, gpu_cnt=1,
                                        batch_size=args.batch_size, is_train=False, keep_remainder=True)
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_sampler=eval_sampler,
        pin_memory=True,
        collate_fn=partial(SDD_collate_fn, args=args, is_train=False),
        num_workers=args.num_worker
    )

    print(f"{args.split.capitalize()} samples: {len(eval_dataset)}")
    print()
    
    # Load model
    print("Loading model...")
    model_module = importlib.import_module("model")
    
    # SDD features: [x, y, vx, vy, cos_yaw, sin_yaw, width, length] = 8 dimensions
    model = model_module.HDGT_model(input_dim=8, args=args)
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Model loaded from checkpoint")
        if "epoch" in checkpoint:
            print(f"  Epoch: {checkpoint['epoch']}")
        if "val_loss" in checkpoint:
            print(f"  Validation loss: {checkpoint['val_loss']:.4f}")
        if "val_minADE" in checkpoint:
            print(f"  Validation minADE: {checkpoint['val_minADE']:.4f}, minFDE: {checkpoint['val_minFDE']:.4f} (pixels)")
    else:
        model.load_state_dict(checkpoint)
        print("Model loaded (direct state dict)")
    
    model = model.to(device)
    print(f"Model moved to {device}")
    print()
    
    # Run evaluation
    recorder, length_lis, agent_type_lis = test_model(model, eval_dataloader, args, device)

    # The report: the trajectory metrics, the same in every version, then the mode classifier's
    # own diagnostics. Printed, and saved as one file per split (e.g. best_min_fde_val_report.txt)
    lines = [
        "=" * 80,
        f"HDGT Results on the SDD {args.split} split",
        "=" * 80,
        f"Checkpoint: {args.checkpoint}",
        f"Split: {args.split}",
        f"Random Seed: {SEED}",
        f"{args.split.capitalize()} samples: {len(eval_dataset)}",
        "",
    ]
    lines += comparison_block(recorder, agent_type_lis, length_lis,
                              {"ADE": "ade", "FDE": "fde", "minADE": "ade6", "minFDE": "fde6"},
                              args.num_prediction, args.split)
    lines += classifier_block(recorder, agent_type_lis)
    report = "\n".join(lines)
    print("\n" + report)

    # splitext also handles .pth paths (str.replace('.pt', ...) would produce "..._results.txth")
    results_file = os.path.splitext(args.checkpoint)[0] + f'_{args.split}_report.txt'
    with open(results_file, 'w') as f:
        f.write(report + "\n")
    
    print(f"\nResults saved to: {results_file}")
    print(f"\nEvaluation on the {args.split} split completed successfully!")


if __name__ == '__main__':
    main()
