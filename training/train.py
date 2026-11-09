import os
os.environ["NCCL_P2P_DISABLE"] = "1"
import random
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import argparse
import time
import sys
import shutil
import importlib
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from sdd_dataset import *
from sdd_dataset import SCALE_FACTOR, AGENT_TYPE_NAMES
from torch.utils.tensorboard import SummaryWriter

# Checkpoint selection -- the same rule in every version (keep them in sync): best_min_ade.pt and
# best_min_fde.pt are chosen on minADE / minFDE over every validation trajectory (each counts once,
# whatever its agent type) at the full prediction horizon, in pixels. VAL_SEED fixes the noise of
# every validation pass, so two epochs are compared on the same draw; the training random stream
# is restored afterwards.
VAL_SEED = 42


def full_horizon_sample_weighted(recorder, agent_type_lis, pred_len, metric_type_lis):
    """{metric: its mean over every trajectory at the full horizon, whatever its agent type}"""
    selection = {}
    for metric_type in metric_type_lis:
        meters = [recorder[f"{agent_type}_{pred_len}_{metric_type}"] for agent_type in agent_type_lis]
        count = sum(meter.count for meter in meters)
        selection[metric_type] = sum(meter.sum for meter in meters) / count if count > 0 else float('inf')
    return selection


# Checkpoints -- the same rule in every version (keep them in sync): best_min_ade.pt and
# best_min_fde.pt, chosen as described above, and last.pt, the resume point, written after every
# validation. Each file is overwritten in place, so a run keeps exactly these three files. To
# resume a stopped run, pass its last.pt to --checkpoint: weights, optimizer, scheduler, AMP
# scaler, best-so-far scores and the batch order go on from the next epoch.
def save_checkpoint(path, model, optimizer, scheduler, scaler, args, epoch, epoch_complete,
                    best_val_loss, val_metrics, select_metric, val_videos, extra=None):
    """Write one checkpoint atomically: a crash while writing cannot destroy the previous file"""
    module = model.module if hasattr(model, "module") else model
    checkpoint = {
        "model_state_dict": module.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "epoch": epoch,
        "epoch_complete": epoch_complete,
        "best_val_loss": dict(best_val_loss),
        "select_metric": select_metric,
        "val_loss": val_metrics.get(select_metric, float("nan")),
        "val_metrics": dict(val_metrics),
        "args": {k: v for k, v in vars(args).items() if isinstance(v, (bool, int, float, str, type(None)))},
        # Which split chose this checkpoint, so a report can never claim a clean holdout for
        # weights that were selected on the test set
        "select_split": "val" if str(getattr(args, "use_val_split", True)) == "True" else "test",
        "val_videos": list(val_videos),
    }
    checkpoint.update(extra or {})
    torch.save(checkpoint, path + ".tmp")
    os.replace(path + ".tmp", path)


def restore_training_state(checkpoint, args, optimizer, scheduler, scaler, best_val_loss, train_sampler):
    """Resume from a checkpoint written by save_checkpoint (normally last.pt): optimizer,
    scheduler, AMP scaler and best-so-far scores. Returns the epoch to start at: --start_epoch if
    given, else the one after the checkpoint's, or 1 without a checkpoint"""
    start_epoch = args.start_epoch
    if checkpoint is not None:
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scaler is not None and checkpoint.get("scaler_state_dict") is not None:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        # Without the best-so-far scores the saver would start from inf and overwrite good
        # checkpoints with worse ones
        if isinstance(checkpoint.get("best_val_loss"), dict):
            best_val_loss.update(checkpoint["best_val_loss"])
        if start_epoch is None and checkpoint.get("epoch") is not None:
            start_epoch = checkpoint["epoch"] + (1 if checkpoint.get("epoch_complete", True) else 0)
    if start_epoch is None:
        start_epoch = 1
    if checkpoint is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    elif start_epoch > 1:
        scheduler.warmup_steps = 0      # no saved scheduler state: restart without the warm-up
    # The balanced sampler reshuffles its groups in place every epoch: replay the epochs already
    # done, so a resumed run gets the same batches as one that never stopped
    for _ in range(start_epoch - 1):
        next(iter(train_sampler), None)
    return start_epoch


parser = argparse.ArgumentParser('HDGT Goal-CVAE Training - 3-Loss Architecture')
##### Optimizer - Scheduler
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=1e-6, help='weight decay')
parser.add_argument('--batch_size', type=int, default=16, help='batch size')
parser.add_argument('--val_batch_size', type=int, default=64, help='validation batch size')
parser.add_argument('--n_epoch', type=int, default=50, help='number of epochs')
parser.add_argument('--warmup', type=float, default=2.0, help='warmup epochs')
parser.add_argument('--lr_decay_epoch', type=str, default="30-42-48", help='LR decay schedule')
parser.add_argument('--num_prediction', type=int, default=20, help='number of trajectory samples (K)')

#### Diffusion Parameters
parser.add_argument('--diffusion_steps', type=int, default=100, help='diffusion timesteps')
parser.add_argument('--diffusion_layers', type=int, default=6, help='transformer layers')
parser.add_argument('--ddim_steps', type=int, default=20, help='DDIM steps for inference')

#### Refiner Settings
parser.add_argument('--use_refiner', type=str, default="True", help='use refiner')

#### Model Architecture
parser.add_argument('--num_of_gnn_layer', type=int, default=6, help='HDGT encoder layers')
parser.add_argument('--hidden_dim', type=int, default=256, help='hidden dimension')
parser.add_argument('--head_dim', type=int, default=32, help='attention head dimension')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
parser.add_argument('--num_worker', type=int, default=8, help='dataloader workers')

#### Dataset Settings
parser.add_argument('--agent_drop', type=float, default=0.0, help='agent dropout ratio')
parser.add_argument('--data_folder', type=str, default="../sdd/", help='dataset folder')
parser.add_argument('--obs_len', type=int, default=8, help='observation length')
parser.add_argument('--pred_len', type=int, default=12, help='prediction length')

#### Initialize
parser.add_argument('--checkpoint', type=str, default="none", help='checkpoint path')
parser.add_argument('--start_epoch', type=int, default=None, help='epoch to start at (default: 1, or the epoch after the one in --checkpoint, to resume)')
parser.add_argument('--dev_mode', type=str, default="False", help='development mode')
parser.add_argument('--ddp_mode', type=str, default="False", help='DDP mode')
parser.add_argument('--port', type=str, default="31243", help='DDP port')
parser.add_argument('--amp', type=str, default="none", help='mixed precision: none, fp16, bf16')

#### Goal Predictor (multimodal K-head; GOAL_HEAD_PLAN.md: Sinkhorn-balanced, repulsive, leashed heads)
parser.add_argument('--goal_loss_weight', type=float, default=0.5, help='weight on the goal loss (the goal distribution\'s cross-entropy; mtp: the WTA regression, + confidence CE without a scorer)')
parser.add_argument('--goal_detach_encoder', type=str, default="True", help='True = the goal predictor reads detached encoder features, so the goal loss trains only the goal predictor (as in hdgt_goal); False = the goal loss also trains the encoder, beside the diffusion and refiner losses')
parser.add_argument('--goal_curriculum_epochs', type=int, default=10, help='epochs to ramp oracle→predicted goal')
parser.add_argument('--goal_predictor', type=str, default="heatmap", choices=["heatmap", "mtp"], help='heatmap = a goal distribution over a grid around each agent (map crop + reachability prior), the K goals picked from it (GOAL_HEATMAP_PLAN.md); mtp = hdgt_goal\'s K WTA heads (top-3 -> top-1), as older checkpoints')
parser.add_argument('--goal_map_crop', type=str, default="True", help='heatmap: give the goal predictor a crop of the drawn shapes around the agent (grass, obstacle, forbidden for it, inside the image) (True/False)')
parser.add_argument('--goal_reach_prior', type=str, default="True", help='heatmap: add a learned prior from each goal cell\'s travel distance around forbidden shapes (True/False)')
parser.add_argument('--goal_grid_cells', type=int, default=61, help='heatmap: cells per side of the goal grid')
parser.add_argument('--goal_grid_range', type=float, default=1.5, help='heatmap: the goal grid spans +- this x s_n around the constant-velocity endpoint')
parser.add_argument('--goal_heatmap_sigma', type=float, default=0.075, help='heatmap: std of the Gaussian training target around the true goal, s_n units')
parser.add_argument('--goal_crop_cells', type=int, default=32, help='heatmap: cells per side of the map crop')
parser.add_argument('--goal_hidden', type=int, default=128, help='heatmap: width of the goal predictor')
parser.add_argument('--goal_pick', type=str, default="kmeans", choices=["kmeans", "flat", "reserve", "sinkhorn"], help='heatmap: how the K goals are picked from the distribution: kmeans (best mean), flat (k-means on p^alpha: more spread, fewer misses), reserve (K - n by k-means + n for rare futures), sinkhorn (balanced optimal transport)')
parser.add_argument('--goal_pick_iters', type=int, default=25, help='heatmap: k-means refinement steps')
parser.add_argument('--goal_pick_alpha', type=float, default=0.5, help='heatmap, --goal_pick flat: the exponent alpha in (0, 1]')
parser.add_argument('--goal_pick_reserve', type=int, default=4, help='heatmap, --goal_pick reserve: goals kept for rare futures')
parser.add_argument('--goal_pick_reserve_alpha', type=float, default=0.25, help='heatmap, --goal_pick reserve: they maximise p^this x squared distance to the goals so far')
parser.add_argument('--goal_pick_floor', type=float, default=0.5, help='heatmap, --goal_pick sinkhorn: each goal carries at least this / K of the probability')
parser.add_argument('--goal_scale_min', type=float, default=0.2, help='global floor of the motion scale s_n (it scales the goal grid and the map crop), scaled units (0.2 = 10 px); a class\'s floor until the class has seen 50 agents')
parser.add_argument('--goal_scale_class_frac', type=float, default=0.5, help='class floor of s_n = max(--goal_scale_min, this x the class\'s running mean GT displacement); 0 = no class floors')
parser.add_argument('--goal_cond_noise', type=float, default=0.0, help='> 0: in training only, the goals the denoiser and the refiner are conditioned on get Gaussian noise of RMS this x s_n (against over-trusting the goals; not used for now); the denoiser endpoint penalty keeps the clean goal; inference unchanged; 0 = off')

#### Map nodes in the encoder graph (hdgt_map_gnn): off by default
parser.add_argument('--use_map_nodes', type=str, default="False", help='add the grass/obstacle shapes to the encoder graph as map nodes: agents attend to them in every GNN layer, as in hdgt_map_gnn (True/False)')

#### Scorer: ranks the K refined chains (index 0 = the single-trajectory prediction)
parser.add_argument('--use_scorer', type=str, default="False", help='rank the K chains with a scorer (off by default in this version; --use_scorer True --scorer_type mode or traj turns one on); False = just goal-aware, as hdgt_goal_WTA_regressor (confidence head trained, chains left in the goal heads\' order)')
parser.add_argument('--scorer_type', type=str, default="mode", choices=["mode", "traj"], help='mode = ModeScorer (hand-made features of each chain), traj = TrajScorer (raw chains, attention across them and to the neighbours)')
parser.add_argument('--scorer_loss_weight', type=float, default=0.5, help='weight on the scorer loss, applied once the goal curriculum is done (isolated: cannot affect the generator); 0.5 = hdgt_bridge\'s effective weight')
parser.add_argument('--scorer_err_weight', type=float, default=1.0, help='ModeScorer only: weight on the per-mode log-ADE regression inside its loss')
parser.add_argument('--clip_scorer_separately', type=str, default="True", help='clip the scorer\'s gradients on their own, max norm 1.0 (True), or leave them unclipped (False); the rest of the model is always clipped on its own, never in one global clip')
parser.add_argument('--scorer_dropout', type=str, default="False", help='use --dropout inside the scorer (True) or no dropout in it (False)')
parser.add_argument('--conf_tau_min', type=float, default=0.02, help='TrajScorer only: floor for the adaptive soft-target temperature, scaled units (0.02 = 1 px)')
parser.add_argument('--conf_goal_blend', type=float, default=0.5, help='TrajScorer only: weight of goal-space distance in the soft target')
parser.add_argument('--conf_sharpen', type=float, default=1.0, help='TrajScorer only: sharpness of the soft target')

#### Log
parser.add_argument('--use_val_split', type=str, default="True", help='select checkpoints on the held-out VAL videos (True) instead of the test set (False, leaks)')
parser.add_argument('--val_every_train_step', type=int, default=-1, help='validation frequency')
parser.add_argument('--name', type=str, default="Hdgt_goal", help='experiment name')

args = parser.parse_args()
os.environ["DGLBACKEND"] = "pytorch"


class Logger():
    def __init__(self, lognames):
        self.terminal = sys.stdout
        self.logs = []
        for log_name in lognames:
            self.logs.append(open(log_name, 'w'))
    def write(self, message):
        self.terminal.write(message)
        for log in self.logs:
            log.write(message)
            log.flush()
    def flush(self):
        pass


def euclid_np(label, pred):
    return np.sqrt((label[..., 0] - pred[..., 0])**2 + (label[..., 1] - pred[..., 1])**2)


def cal_ade_fde(labels, preds, masks, scale_factor=1.0):
    """Calculate ADE and FDE in original pixel coordinates"""
    if labels.shape[0] == 0:
        return None, None
    
    l2_norm = euclid_np(labels, preds) * scale_factor
    
    masks_sum = masks.sum(1)
    ade_indices = masks_sum != 0
    ade_cnt = ade_indices.sum()
    ade = ((l2_norm[ade_indices] * masks[ade_indices]).sum(1) / masks_sum[ade_indices]).mean()

    fde_indices = masks[:, -1] != 0
    fde_cnt = fde_indices.sum()
    fde = 0.0
    if fde_cnt != 0:
        fde = l2_norm[fde_indices, -1].mean()
    return [ade, fde], [ade_cnt, fde_cnt]


def flag_on(value):
    """A flag that may arrive as a bool or as argparse's "True"/"False" string"""
    return value in (True, 'True', 'true', '1')


def clip_grads(model, max_norm=1.0, clip_scorer=True):
    """Clip the rest of the model, and the scorer on its own (--clip_scorer_separately).

    A single clip_grad_norm_ over all parameters couples modules that share
    nothing: the scorer's gradient enters the GLOBAL norm, so a large scorer
    gradient scales DOWN the encoder / decoder / refiner updates. Measured at
    3.5% shrinkage on an untrained model, and the scorer's gradient is largest
    exactly when it starts training (curriculum alpha reaches 1) — mid-run for
    the generator. Clipping the two groups separately restores the isolation
    that detaching the scorer's inputs was meant to provide.

    There is never one global clip. clip_scorer=False leaves the scorer's
    gradients unclipped; the rest of the model is always clipped on its own.
    Without a scorer, the rest is the whole model (minus the goal distribution).
    """
    m = model.module if hasattr(model, "module") else model
    # The goal distribution (--goal_predictor heatmap) is clipped on its own too: its
    # cross-entropy gradient must not scale down the decoder's updates. The mtp heads stay in
    # the rest, as before.
    own_goal = getattr(m, "goal_predictor_type", "mtp") == "heatmap"
    scorer, goal, rest = [], [], []
    for name, p in m.named_parameters():
        if name.startswith(("mode_scorer", "traj_scorer")):
            scorer.append(p)
        elif own_goal and name.startswith("goal_predictor."):
            goal.append(p)
        else:
            rest.append(p)
    torch.nn.utils.clip_grad_norm_(rest, max_norm)
    if goal:
        torch.nn.utils.clip_grad_norm_(goal, max_norm)
    if clip_scorer and scorer:
        torch.nn.utils.clip_grad_norm_(scorer, max_norm)


def cal_min_ade_fde(preds, labels, masks, scale_factor=1.0):
    """Calculate minADE and minFDE over K samples in original pixel coordinates"""
    if labels.shape[0] == 0:
        return None, None
    
    l2_norm = euclid_np(labels[:, np.newaxis, :, :], preds) * scale_factor
    
    masks_sum = masks.sum(1)
    ade_indices = masks_sum != 0
    ade_cnt = ade_indices.sum()
    ade_per_sample = (l2_norm[ade_indices] * masks[ade_indices, np.newaxis, :]).sum(-1) / masks_sum[ade_indices][:, np.newaxis]
    min_ade = ade_per_sample.min(-1).mean()
    
    fde_indices = masks[:, -1] != 0
    fde_cnt = fde_indices.sum()
    min_fde = 0.0
    if fde_cnt != 0:
        min_fde = l2_norm[fde_indices, :, -1].min(-1).mean()
    
    return [min_ade, min_fde], [ade_cnt, fde_cnt]


class AverageMeter(object):
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
        self.avg = self.sum / self.count if self.count > 0 else 0


# The goal statistics the model stashes at every training step (HDGT_model.forward, LOSS 3): the
# goal loss itself (the distribution's cross-entropy, or mtp's WTA regression)
GOAL_LOG_KEYS = ("assign",)


def make_recorder(agent_type_lis, length_lis, metric_type_lis):
    """Fresh meters, for an epoch or for the training batches since the last validation"""
    recorder = {f"{agent_type}_{length}_{metric_type}": AverageMeter()
                for agent_type in agent_type_lis for length in length_lis for metric_type in metric_type_lis}
    for name in ("loss", "diffusion_loss", "refiner_loss", "goal_loss", "goal_recon_loss",
                 "scorer_cls", "scorer_err", "hit1", "meanrank") + tuple("goal_" + key for key in GOAL_LOG_KEYS):
        recorder[name] = AverageMeter()
    return recorder


def goal_log_text(recorder, model, agent_type_lis):
    """The goal line of the training log: the goal loss's running average, the predictor and how
    its goals are picked, and the class floors of s_n as they are now"""
    meter = recorder["goal_assign"]
    loss = format(meter.avg, ".4f") if meter.count > 0 else "--"
    floors = ", ".join(f"{name} {floor * SCALE_FACTOR:.1f}"
                       for name, floor in zip(agent_type_lis, model.goal_class_floors().tolist()))
    if model.goal_predictor_type == "heatmap":
        return f"GoalDist[pick {model.goal_pick}]: cross-entropy {loss}, floors(px) {floors}"
    return f"GoalHeads[mtp]: wta {loss}, floors(px) {floors}"


def goal_scale_extra(model):
    """The running statistics behind the class floors of s_n (non-persistent buffers), saved beside
    the weights: test_sdd.py computes s_n with them, and a resumed run goes on from them"""
    m = model.module if hasattr(model, "module") else model
    return {"goal_class_scale": m.goal_class_scale.scale.tolist(),
            "goal_class_count": m.goal_class_scale.count.tolist()}


def main():
    args = parser.parse_args()
    gpu_count = torch.cuda.device_count()
    global_seed = int(args.port)
    
    print("=" * 80)
    print("HDGT DIFFUSION MODEL - 3-LOSS (Multimodal K-Goal + Diffusion + Refiner)")
    print("=" * 80)
    print(f"GPU Count: {gpu_count}, Seed: {global_seed}")
    print(f"Coordinate Scale: {SCALE_FACTOR} (metrics in original pixels)")
    print(f"Refiner: {'ENABLED' if args.use_refiner == 'True' else 'DISABLED'}")
    print("-" * 80)
    print("ARCHITECTURE:")
    print(" Encoder: HDGT heterogeneous graph network")
    print(" Decoder: Diffusion model (generates K diverse samples)")
    print(" Refiner: 5-step CNN refinement (like original HDGT)")
    print("-" * 80)
    print("TRAINING LOSSES (3 LOSSES):")
    print("  1. Diffusion Loss (weight=1.0): noise prediction + alpha_t-weighted goal penalty")
    print("  2. Refiner Loss (weight=1.0): best-of-K distance to GT over 6 steps")
    if args.goal_predictor == "heatmap":
        goal_desc = (f"goal distribution, {args.goal_grid_cells}x{args.goal_grid_cells} cells over +-{args.goal_grid_range} s_n, "
                     f"cross-entropy (sigma {args.goal_heatmap_sigma} s_n); map crop {args.goal_map_crop}, "
                     f"reachability prior {args.goal_reach_prior}; K goals picked by {args.goal_pick}")
    else:
        goal_desc = ("K WTA heads (top-3 -> top-1)"
                     + ("" if flag_on(args.use_scorer) and args.use_refiner == 'True' else " + confidence CE"))
    print(f"  3. Goal Loss (weight={args.goal_loss_weight}): {goal_desc}"
          f"{'' if flag_on(args.goal_detach_encoder) else '; also trains the encoder (--goal_detach_encoder False)'}")
    print(f"  Map nodes in the encoder graph: {'ON' if flag_on(args.use_map_nodes) else 'OFF'}")
    print("-" * 80)
    print("KEY FEATURES:")
    print(" Multi-step refiner loss (equal gradient to all layers)")
    print(" Joint training (both components trained together)")
    print(" Stable gradients (no backprop through sampling)")
    print(" Follows original HDGT design philosophy")
    print("=" * 80)
    print(args)
    print("=" * 80)
    
    if gpu_count > 1:
        if args.ddp_mode == "multi_node":
            main_worker(int(os.environ["LOCAL_RANK"]), int(os.environ["WORLD_SIZE"]), global_seed, args)
        else:
            mp.spawn(main_worker, nprocs=gpu_count, args=(gpu_count, global_seed, args))
    else:
        main_worker(0, gpu_count, global_seed, args)


def main_worker(gpu, gpu_count, global_seed, args):
    if args.ddp_mode == "multi_node":
        global_rank = int(os.environ["RANK"])
        init_port = "tcp://" + os.environ["MASTER_ADDR"] + ":" + os.environ["MASTER_PORT"]
    else:
        global_rank = gpu
        init_port = "tcp://127.0.0.1:" + args.port
    
    print(f"GPU: {gpu}, Global Rank: {global_rank}, World Size: {gpu_count}")
    
    seed_num = random.randint(0, 1000000)
    torch.manual_seed(seed_num + global_rank)
    random.seed(seed_num + global_rank)
    np.random.seed(seed_num + global_rank)

    if gpu_count > 1:
        dist.init_process_group(backend="nccl", world_size=gpu_count, init_method=init_port, rank=global_rank)
    torch.cuda.set_device(gpu)
    device = torch.device("cuda:" + str(gpu))

    snapshot_dir = None
    writer = None
    # One "best" per selection rule — see the saver in run_model for what each
    # rule is for. Threaded through run_model in the old best_val_loss slot.
    best_val_loss = {"min_ade": float('inf'), "min_fde": float('inf')}
    
    if global_rank == 0:
        setting_name = args.name
        log_dir = "logs/" + str(setting_name + "_" + time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time())))

        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)
        sys.stdout = Logger([f"{setting_name}.log", os.path.join(log_dir, f"{setting_name}.log")])
        snapshot_dir = os.path.join(log_dir, "snapshot")
        if not os.path.isdir(snapshot_dir):
            os.makedirs(snapshot_dir)

        tensorboard_dir = os.path.join(log_dir, "tensorboard")
        writer = SummaryWriter(tensorboard_dir)
        print("TensorBoard:", tensorboard_dir)
        print("Log Directory:", log_dir)
        shutil.copyfile(__file__, os.path.join(log_dir, "train.py"))
        shutil.copyfile("model.py", os.path.join(log_dir, "model.py"))

    model_module = importlib.import_module("model")
    
    args.use_refiner = (args.use_refiner == "True")
    args.use_val_split = (args.use_val_split == "True")
    args.use_map_nodes = flag_on(args.use_map_nodes)

    model = model_module.HDGT_model(input_dim=8, args=args)
    model.apply(model_module.weights_init)
    # weights_init re-randomizes every Linear, including the diffusion decoder's output layer,
    # which the model itself starts at zero (the denoiser first predicts zero noise): put it back
    nn.init.zeros_(model.diffusion_decoder.output_proj.weight)
    nn.init.zeros_(model.diffusion_decoder.output_proj.bias)

    print("Loading Dataset...")
    train_dataloader, val_dataloader, train_sample_num, val_sample_num = obtain_dataset(global_rank, gpu_count, global_seed, args)

    checkpoint = None
    if args.checkpoint != "none":
        print("Loading checkpoint:", args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        ckpt_has_map = any(k.startswith("encoder.map_emb") for k in checkpoint["model_state_dict"])
        if ckpt_has_map != args.use_map_nodes:
            raise RuntimeError(f"--use_map_nodes {args.use_map_nodes}, but the checkpoint {'has' if ckpt_has_map else 'has no'} "
                               f"map-node weights: resume it with --use_map_nodes {ckpt_has_map}")
        model.load_state_dict(checkpoint["model_state_dict"])
        # The class floors' running statistics are not in the state_dict (see goal_scale_extra)
        if "goal_class_scale" in checkpoint:
            model.goal_class_scale.load(checkpoint["goal_class_scale"], checkpoint.get("goal_class_count"))

    model = model.to(device, non_blocking=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    
    step_per_epoch = train_sample_num // args.batch_size // max(gpu_count, 1) + 1
    total_steps = step_per_epoch * args.n_epoch

    # Built with the full warm-up; restore_training_state() moves it to where a resumed run stopped
    scheduler = model_module.WarmupLinearSchedule(optimizer, step_per_epoch * args.warmup, total_steps)
    
    
    if gpu_count > 1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu], output_device=gpu, find_unused_parameters=True)
    
    if global_rank == 0:
        print("Starting Training...")
    
    # Mixed precision
    if args.amp == "fp16":
        amp_data_type = torch.float16
        scaler = torch.cuda.amp.GradScaler()
    elif args.amp == "bf16":
        amp_data_type = torch.bfloat16
        scaler = torch.cuda.amp.GradScaler()
    else:
        amp_data_type = torch.float32
        scaler = None

    # Resume (see save_checkpoint): optimizer, scheduler, AMP scaler, best-so-far scores and the
    # batch order of the epochs already done, then go on from the next epoch
    args.start_epoch = restore_training_state(checkpoint, args, optimizer, scheduler, scaler,
                                              best_val_loss, train_dataloader.batch_sampler)
    if checkpoint is not None and global_rank == 0:
        print(f"Resuming at epoch {args.start_epoch} of {args.n_epoch}; best so far: " + ", ".join(
            f"{k} {v:.4f}" for k, v in best_val_loss.items() if v != float('inf')))

    for epoch in range(args.start_epoch, args.n_epoch + 1):
        best_val_loss = run_model(
            dataloader=train_dataloader,
            num_sample=train_sample_num,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            gpu=gpu,
            global_rank=global_rank,
            gpu_count=gpu_count,
            is_train=True,
            args=args,
            val_dataloader=val_dataloader,
            val_sample_num=val_sample_num,
            snapshot_dir=snapshot_dir,
            scaler=scaler,
            amp_data_type=amp_data_type,
            writer=writer,
            best_val_loss=best_val_loss
        )


def run_model(dataloader, num_sample, model, optimizer, scheduler, epoch, gpu, global_rank, gpu_count, is_train, args, val_dataloader=None, val_sample_num=None, snapshot_dir=None, scaler=None, amp_data_type=None, writer=None, best_val_loss=None):
    
    length_lis = [int(args.pred_len * 0.4), int(args.pred_len * 0.65), args.pred_len]
    agent_type_lis = ["PEDESTRIAN", "BIKER", "CAR", "BUS", "CART", "SKATER"]
    metric_type_lis = ["ade", "fde", "min_ade", "min_fde"]
    
    # Initialize recorders (3-LOSS: goal_cvae + diffusion + refiner)
    recorder = make_recorder(agent_type_lis, length_lis, metric_type_lis)
    # Validation: the GOAL HEAD inputs of every batch (see the validation summary)
    goal_rows = []
    model_module = importlib.import_module("model")

    # Goal curriculum (only meaningful during training)
    if is_train:
        model_ref = model.module if hasattr(model, 'module') else model
        model_ref.goal_curriculum_alpha = min(1.0, (epoch - 1) / max(1, args.goal_curriculum_epochs))

    start_time = time.time()
    if is_train:
        model.train()
        batch_size = args.batch_size
    else:
        model.eval()
        batch_size = args.val_batch_size
        gpu_count = 1
    
    is_dev = (args.dev_mode == "True")
    
    if is_dev:
        val_every_train_step = 1
        print_freq = 1
    else:
        val_every_train_step = args.val_every_train_step
        print_freq = 100
    
    device = torch.device("cuda:" + str(gpu))
    
    # Learning rate decay
    lr_decay_epoch = [int(e) for e in args.lr_decay_epoch.split("-")]
    decay_coefficient = 1.0
    for decay_epoch in lr_decay_epoch:
        if epoch >= decay_epoch:
            decay_coefficient *= 0.5

    if is_train and scheduler is not None and optimizer is not None:
        scheduler.base_lrs = [args.lr * decay_coefficient]
        for param_group in optimizer.param_groups:
            param_group["lr"] = args.lr * decay_coefficient

    use_amp = (scaler is not None)
    global_step = (epoch - 1) * (num_sample // batch_size // max(gpu_count, 1))
    
    # Fixed noise for every validation pass (see VAL_SEED); the training stream is put back at
    # the end of the validation summary
    rng_state_cpu = rng_state_cuda = None
    if not is_train:
        rng_state_cpu = torch.get_rng_state()
        if torch.cuda.is_available():
            rng_state_cuda = torch.cuda.get_rng_state_all()
        torch.manual_seed(VAL_SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(VAL_SEED)

    with torch.set_grad_enabled(is_train):
        for batch_index, data in enumerate(dataloader, 0):
            if data is None:
                continue
                
            data["is_train"] = is_train
            data["gpu"] = gpu
            for tensor_name in data["cuda_tensor_lis"]:
                data[tensor_name] = data[tensor_name].to(device, non_blocking=True)
            
            if is_train:
                optimizer.zero_grad()
            num_of_sample = len(data["pred_num_lis"])

            with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_data_type):
                if is_train:
                    # Training: returns (total_loss, diffusion_loss, refiner_loss, goal_loss, goal_best_head_loss, type_masks, types)
                    total_loss, diffusion_loss, refiner_loss, goal_loss, goal_recon_loss, pred_indice_bool_type_lis, agent_types = model(data)
                    loss = total_loss
                else:
                    # Inference: returns (trajectories, type_masks, types)
                    with torch.no_grad():
                        trajectories, pred_indice_bool_type_lis, agent_types = model(data)
                    loss = torch.tensor(0.0, device=device)
                    diffusion_loss = torch.tensor(0.0, device=device)
                    refiner_loss = torch.tensor(0.0, device=device)
                    goal_loss = torch.tensor(0.0, device=device)
                    goal_recon_loss = torch.tensor(0.0, device=device)

            if is_train:
                if not (torch.isnan(loss) or torch.isinf(loss)):
                    if scaler is not None:
                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                        clip_grads(model, 1.0, flag_on(args.clip_scorer_separately))
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        clip_grads(model, 1.0, flag_on(args.clip_scorer_separately))
                        optimizer.step()
                    if scheduler is not None:
                        scheduler.step()
                else:
                    print(f"Bad loss at epoch {epoch}, batch {batch_index}: "
                          f"total={loss.item():.4f}, diff={diffusion_loss.item():.4f}, "
                          f"refiner={refiner_loss.item():.4f}, goal={goal_loss.item():.4f}, "
                          f"goal_best={goal_recon_loss.item():.4f}")
                    optimizer.zero_grad()
                    continue

            # Logging
            if global_rank == 0:
                with torch.no_grad():
                    recorder["loss"].update(loss.item(), num_of_sample)
                    recorder["diffusion_loss"].update(diffusion_loss.item(), num_of_sample)
                    recorder["refiner_loss"].update(refiner_loss.item(), num_of_sample)
                    recorder["goal_loss"].update(goal_loss.item(), num_of_sample)
                    recorder["goal_recon_loss"].update(goal_recon_loss.item(), num_of_sample)

                    # Scorer: stashed on the model rather than returned, so
                    # forward()'s signature stays as train/test scripts expect.
                    # Only set while training with the scorer enabled.
                    if is_train:
                        _m = model.module if hasattr(model, 'module') else model
                        if getattr(_m, "last_scorer_cls", None) is not None:
                            recorder["scorer_cls"].update(_m.last_scorer_cls.item(), num_of_sample)
                            if getattr(_m, "last_scorer_err", None) is not None:    # ModeScorer only
                                recorder["scorer_err"].update(_m.last_scorer_err.item(), num_of_sample)
                        # Goal-head terms, stashed the same way (see GOAL_LOG_KEYS)
                        goal_stats = {key: float(value) for key, value in (_m.last_goal_stats or {}).items()}
                        for key in GOAL_LOG_KEYS:
                            if key in goal_stats:
                                recorder["goal_" + key].update(goal_stats[key], num_of_sample)

                    if writer is not None and is_train:
                        current_step = global_step + batch_index
                        writer.add_scalar('Loss/total', loss.item(), current_step)
                        writer.add_scalar('Loss/diffusion', diffusion_loss.item(), current_step)
                        writer.add_scalar('Loss/refiner', refiner_loss.item(), current_step)
                        writer.add_scalar('Loss/goal', goal_loss.item(), current_step)
                        writer.add_scalar('Loss/goal_best_head', goal_recon_loss.item(), current_step)
                        writer.add_scalar('Learning_Rate', scheduler.get_last_lr()[0], current_step)
                        for key in GOAL_LOG_KEYS:
                            if key in goal_stats:
                                writer.add_scalar(f'Goal/{key}', goal_stats[key], current_step)
                        for agent_type, floor in zip(agent_type_lis, _m.goal_class_floors().tolist()):
                            writer.add_scalar(f'Goal/floor_px_{agent_type}', floor * SCALE_FACTOR, current_step)

                    # Validation metrics
                    if not is_train:
                        label_pos_lis = data["label_pos_lis"]
                        label_mask_lis = data["label_mask_lis"]
                        
                        model_ref = model.module if hasattr(model, 'module') else model
                        output_is_positions = getattr(model_ref, 'output_is_positions', False)
                        
                        if output_is_positions:
                            pred_positions = trajectories.cpu().numpy()
                        else:
                            pred_positions = torch.cumsum(trajectories, dim=2).cpu().numpy()
                        
                        for agent_type_index in range(6):
                            type_mask = pred_indice_bool_type_lis[agent_type_index].cpu().numpy()
                            if type_mask.sum() == 0:
                                continue
                            
                            now_preds = pred_positions[type_mask]
                            now_labels = label_pos_lis[type_mask].cpu().numpy()
                            now_masks = label_mask_lis[type_mask].cpu().numpy()
                            
                            # Chains are confidence-sorted by the model; index 0 = most-confident mode
                            best_pred = now_preds[:, 0, :, :]

                            # Ranking quality at the full horizon. Chains arrive
                            # already sorted, so a chain's rank IS its index and
                            # the truly-best chain's index is its rank.
                            if not is_train:
                                ade_k = np.linalg.norm(
                                    now_preds - now_labels[:, None], axis=-1).mean(-1)
                                rank_of_best = ade_k.argmin(1)
                                n_ag = ade_k.shape[0]
                                recorder["hit1"].update(
                                    float((rank_of_best == 0).mean()) * 100.0, n_ag)
                                recorder["meanrank"].update(
                                    float(rank_of_best.mean()), n_ag)

                            for length_idx, length in enumerate(length_lis):
                                res, cnt = cal_ade_fde(
                                    best_pred[:, :length, :],
                                    now_labels[:, :length, :],
                                    now_masks[:, :length],
                                    scale_factor=SCALE_FACTOR
                                )
                                if res is not None:
                                    recorder[f"{agent_type_lis[agent_type_index]}_{length}_ade"].update(res[0], cnt[0])
                                    recorder[f"{agent_type_lis[agent_type_index]}_{length}_fde"].update(res[1], cnt[1])
                                
                                res, cnt = cal_min_ade_fde(
                                    now_preds[:, :, :length, :],
                                    now_labels[:, :length, :],
                                    now_masks[:, :length],
                                    scale_factor=SCALE_FACTOR
                                )
                                if res is not None:
                                    recorder[f"{agent_type_lis[agent_type_index]}_{length}_min_ade"].update(res[0], cnt[0])
                                    recorder[f"{agent_type_lis[agent_type_index]}_{length}_min_fde"].update(res[1], cnt[1])

                        # GOAL HEAD inputs: every target of the batch, the heads in head order (the
                        # model's inference stash), s_n with the model's current class floors
                        goal_logits = model_ref.last_goal_logits
                        goal_rows.append({
                            "goals": model_ref.last_goals_head_order.cpu().numpy(),
                            "first": model_ref.last_predicted_goals[:, 0].cpu().numpy(),
                            "conf": goal_logits.cpu().numpy() if goal_logits is not None else None,
                            "gt": label_pos_lis[:, -1].cpu().numpy(),
                            "valid": (label_mask_lis[:, -1] != 0).cpu().numpy(),
                            "s": model_module.motion_scale(model_ref.last_target_obs,
                                                           model_ref.goal_class_floors()[agent_types],
                                                           args.pred_len).cpu().numpy(),
                            "cls": agent_types.cpu().numpy(),
                        })

                    # Print during training
                    if is_train and ((batch_index + 1) % print_freq) == 0:
                        model_ref_log = model.module if hasattr(model, 'module') else model
                        print_text = f' Epoch: [{epoch}][{(batch_index+1)*batch_size*max(gpu_count,1)}/{num_sample}], '
                        print_text += f"Loss: {recorder['loss'].avg:.6f}, "
                        print_text += f"Diffusion: {recorder['diffusion_loss'].avg:.6f}, "
                        print_text += f"Refiner: {recorder['refiner_loss'].avg:.6f}, "
                        print_text += f"Goal: {recorder['goal_loss'].avg:.6f}, "
                        print_text += f"GoalBest: {recorder['goal_recon_loss'].avg:.6f}, "
                        if recorder["scorer_cls"].count > 0:
                            gated = model_ref_log.goal_curriculum_alpha >= 1.0
                            if recorder["scorer_err"].count > 0:      # ModeScorer: CE / error regression
                                scorer_text = (f"Scorer(ce/err): {recorder['scorer_cls'].avg:.3f}/"
                                               f"{recorder['scorer_err'].avg:.3f}")
                            else:                                       # TrajScorer: KL to the soft target
                                scorer_text = f"Scorer(kl): {recorder['scorer_cls'].avg:.3f}"
                            print_text += scorer_text + f"{'' if gated else ' [not applied yet]'}, "
                        print_text += f"Curriculum: {model_ref_log.goal_curriculum_alpha:.3f}, "
                        print_text += f"Time: {time.time()-start_time:.2f}s, "
                        print_text += f"LR: {scheduler.get_last_lr()[0]:.2e}"
                        print(print_text, flush=True)
                        print("   " + goal_log_text(recorder, model_ref_log, agent_type_lis), flush=True)

                    # Validation: every val_every_train_step batches if > 0,
                    # otherwise once at the end of the epoch. (Python quirk:
                    # x % -1 == 0 for any x, so the old unguarded modulo made
                    # this fire on every print step with the default -1.)
                    if is_train and ((val_every_train_step > 0 and (batch_index + 1) % val_every_train_step == 0) or (val_every_train_step <= 0 and batch_index == len(dataloader) - 1)) and not is_dev:

                        if writer is not None:
                            writer.add_scalar('Train_Epoch/loss', recorder["loss"].avg, epoch)
                            writer.add_scalar('Train_Epoch/diffusion_loss', recorder["diffusion_loss"].avg, epoch)
                            writer.add_scalar('Train_Epoch/refiner_loss', recorder["refiner_loss"].avg, epoch)
                            writer.add_scalar('Train_Epoch/goal_loss', recorder["goal_loss"].avg, epoch)
                            writer.add_scalar('Train_Epoch/goal_recon_loss', recorder["goal_recon_loss"].avg, epoch)
                            if recorder["scorer_cls"].count > 0:
                                writer.add_scalar('Train_Epoch/scorer_cls', recorder["scorer_cls"].avg, epoch)
                                if recorder["scorer_err"].count > 0:
                                    writer.add_scalar('Train_Epoch/scorer_err', recorder["scorer_err"].avg, epoch)
                            for key in GOAL_LOG_KEYS:
                                if recorder["goal_" + key].count > 0:
                                    writer.add_scalar(f'Train_Epoch/goal_{key}', recorder["goal_" + key].avg, epoch)

                        # Reset recorder
                        recorder = make_recorder(agent_type_lis, length_lis, metric_type_lis)

                        # Run validation
                        if gpu_count > 1:
                            val_model = model.module
                        else:
                            val_model = model
                        val_model.eval()
                        
                        val_metrics = run_model(
                            val_dataloader, val_sample_num, val_model, optimizer, scheduler,
                            epoch, gpu, global_rank, gpu_count, is_train=False, args=args,
                            scaler=scaler, amp_data_type=amp_data_type, writer=writer,
                            best_val_loss=best_val_loss
                        )
                        model.train()

                        # Checkpoints, the same three in every version (see save_checkpoint)
                        epoch_complete = batch_index == len(dataloader) - 1
                        for metric_name, fname in (("min_ade", "best_min_ade.pt"), ("min_fde", "best_min_fde.pt")):
                            if val_metrics[metric_name] < best_val_loss[metric_name]:
                                best_val_loss[metric_name] = val_metrics[metric_name]
                                save_checkpoint(os.path.join(snapshot_dir, fname), model, optimizer, scheduler, scaler, args,
                                                epoch, epoch_complete, best_val_loss, val_metrics, metric_name, VAL_VIDEOS,
                                                extra=goal_scale_extra(model))
                                print(f"[save] {fname:<17} epoch {epoch}  {metric_name} = {val_metrics[metric_name]:.4f} px "
                                      f"(every val trajectory, {args.pred_len * 0.4:.1f}s)")
                        save_checkpoint(os.path.join(snapshot_dir, "last.pt"), model, optimizer, scheduler, scaler, args,
                                        epoch, epoch_complete, best_val_loss, val_metrics, "last", VAL_VIDEOS,
                                        extra=goal_scale_extra(model))

            del data

    # Validation summary
    if not is_train:
        print_dic = {metric_type: 0.0 for metric_type in metric_type_lis}
        sub_print_dic = {}
        
        for agent_type in agent_type_lis:
            for metric_type in metric_type_lis:
                sub_print_dic[f"{agent_type}_{metric_type}"] = 0
                for length in length_lis:
                    sub_print_dic[f"{agent_type}_{metric_type}"] += recorder[f"{agent_type}_{length}_{metric_type}"].avg

        detail_text = ""
        for agent_type in agent_type_lis:
            for length in length_lis:
                for metric_type in metric_type_lis:
                    print_dic[metric_type] += recorder[f"{agent_type}_{length}_{metric_type}"].avg
                    detail_text += f", {agent_type}_{length}_{metric_type}: {recorder[f'{agent_type}_{length}_{metric_type}'].avg:.4f}"

        print_dic = {k: v / 18.0 for k, v in print_dic.items()}
        sub_print_dic = {k: v / 3.0 for k, v in sub_print_dic.items()}

        model_ref = model.module if hasattr(model, 'module') else model
        refiner_status = "WITH REFINER" if getattr(model_ref, 'use_refiner', False) else "DIFFUSION ONLY"

        print(f"\n{'='*80}")
        print(f"VALIDATION RESULTS - Epoch {epoch} [{refiner_status}]")
        print(f"{'='*80}")
        print(f"Loss: {recorder['loss'].avg:.6f}")
        print(f"  Diffusion Loss: {recorder['diffusion_loss'].avg:.6f}")
        print(f"  Refiner Loss:   {recorder['refiner_loss'].avg:.6f}")
        print(f"  Goal Loss: {recorder['goal_loss'].avg:.6f}")
        # goal_recon_loss is the per-coordinate mean of squared errors, so the
        # Euclidean pixel RMSE is sqrt(2 * loss) * SCALE_FACTOR
        print(f"  Best-Head Goal Error (RMSE={(2*recorder['goal_recon_loss'].avg)**0.5*50:.2f}px): {recorder['goal_recon_loss'].avg:.6f}")
        print(f"{'-'*80}")
        # Scorer ranking quality. K chains -> chance hit@1 = 100/K, chance
        # meanRank = (K-1)/2. This is the number that says whether the scorer
        # is doing anything; `ade` vs `min_ade` below is the same story in px.
        if recorder["hit1"].count > 0:
            n_modes = getattr(model_ref, 'num_samples', 20)
            print(f"Scorer ranking:  hit@1 = {recorder['hit1'].avg:.1f}%  "
                  f"(chance {100.0/n_modes:.1f}%)   "
                  f"meanRank = {recorder['meanrank'].avg:.2f}  "
                  f"(chance {(n_modes-1)/2:.1f})")
            print(f"{'-'*80}")
        # GOAL HEAD diagnostics (dead heads, collapse, off goals), the same block as test_sdd.py's
        goal_diag = None
        if goal_rows:
            rows = {key: (np.concatenate([row[key] for row in goal_rows], axis=0)
                          if goal_rows[0][key] is not None else None) for key in goal_rows[0]}
            valid = rows["valid"]
            if valid.any():
                goal_diag = model_module.goal_head_diagnostics(
                    rows["goals"][valid], rows["gt"][valid], rows["s"][valid], rows["cls"][valid],
                    agent_type_lis, SCALE_FACTOR, first_goal=rows["first"][valid],
                    conf_logits=rows["conf"][valid] if rows["conf"] is not None else None,
                    rho=model_module.GOAL_CLOSE_RHO)
                first_desc = ("the scorer's pick" if model_ref.use_scorer
                              else "head 0: without a scorer the chains keep the head order")
                for line in model_module.goal_head_lines(
                        goal_diag, agent_type_lis, (model_ref.goal_class_floors() * SCALE_FACTOR).tolist(),
                        model_module.GOAL_CLOSE_RHO, args.pred_len, first_desc):
                    print(line)
        print(f"Averaged Metrics (pixels, scale={SCALE_FACTOR}):")
        for k, v in print_dic.items():
            print(f"  {k}: {v:.4f}")
        print(f"{'-'*80}")
        print(f"Per-Agent Metrics:")
        for agent_type in agent_type_lis:
            for metric_type in metric_type_lis:
                print(f"  {agent_type}_{metric_type}: {sub_print_dic[f'{agent_type}_{metric_type}']:.4f}")
        print(f"{'='*80}\n")

        if writer is not None:
            writer.add_scalar('Val/loss', recorder["loss"].avg, epoch)
            if recorder["hit1"].count > 0:
                writer.add_scalar('Val/scorer_hit1', recorder["hit1"].avg, epoch)
                writer.add_scalar('Val/scorer_meanrank', recorder["meanrank"].avg, epoch)
            for metric_type in metric_type_lis:
                writer.add_scalar(f'Val/{metric_type}_pixels', print_dic[metric_type], epoch)
            for agent_type in agent_type_lis:
                for metric_type in metric_type_lis:
                    writer.add_scalar(f'Val/{agent_type}_{metric_type}', sub_print_dic[f"{agent_type}_{metric_type}"], epoch)
            if goal_diag is not None:
                for key in ("entropy", "eff_k", "win_min", "win_max", "slot_worst", "best", "mean_k",
                            "all_p95", "min_pair", "close", "off"):
                    writer.add_scalar(f'Val/goal_{key}', goal_diag["ALL"][key], epoch)

        # Return every metric: the saver applies three different selection
        # rules to them, so a single scalar is not enough.
        # Checkpoint selection numbers, the same in every version (see VAL_SEED)
        selection = full_horizon_sample_weighted(recorder, agent_type_lis, args.pred_len, metric_type_lis)
        print(f"Selection metrics, every val trajectory counted once at {args.pred_len * 0.4:.1f}s "
              f"(best_min_ade.pt / best_min_fde.pt are chosen on these):")
        for metric_type in metric_type_lis:
            print(f"  {metric_type}: {selection[metric_type]:.4f}")
        if writer is not None:
            for metric_type in metric_type_lis:
                writer.add_scalar(f'Val/{metric_type}_selection_pixels', selection[metric_type], epoch)

        # Hand the training stream back exactly where it was
        if rng_state_cpu is not None:
            torch.set_rng_state(rng_state_cpu)
            if rng_state_cuda is not None:
                torch.cuda.set_rng_state_all(rng_state_cuda)

        # The selection numbers under the metric names, the type-averaged ones beside them
        val_metrics = {f"type_averaged_{k}": v for k, v in print_dic.items()}
        val_metrics.update(selection)
        return val_metrics

    return best_val_loss


if __name__ == '__main__':
    main()
