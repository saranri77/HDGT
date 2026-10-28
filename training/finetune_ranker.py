#!/usr/bin/env python3
"""
Fine-tune the ranker of a trained checkpoint on the chains it ranks at test time.

Online, the ranker learns from the training chains, which are drawn with the model in
train mode: dropout live in the encoder and the denoiser, BatchNorm on batch statistics.
At test time every chain comes from the model in eval mode. This script freezes the
generator (encoder, goal predictor, diffusion decoder, refiner, and the map modules when
the checkpoint has them) and trains only the ranker, on chains drawn exactly through the
inference path of HDGT_model.forward, with fresh DDIM noise every step.

minADE / minFDE cannot change: the generator receives no update, and the saved checkpoint
carries the original generator tensors byte for byte. Only the order of the chains, and so
ADE / FDE, can move.

    python finetune_ranker.py --checkpoint logs/<run>/snapshot/best_min_fde.pt

Writes <checkpoint>_ranker_ft.pt next to the checkpoint, with the ranker of the epoch whose
pick has the lowest val ADE (epoch 0, the ranker as trained, if no epoch beats it).
Evaluate it with test_sdd.py like any other checkpoint.

--train_mode_chains True is the control: the same fine-tuning on chains drawn the way
online training draws them (dropout live, BatchNorm on batch statistics with its running
statistics left untouched). If the default run improves the ranker and the control does
not, the train/test mismatch of the chains was what held it back. If both improve alike,
it was the generator moving under it during online training.

All architecture flags default to the values train.py uses. Pass any flag explicitly if
you trained with a non-default value, as for test_sdd.py.
"""

import os
import sys
import time
import random
import argparse
import importlib
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["DGLBACKEND"] = "pytorch"

from sdd_dataset import obtain_dataset, SCALE_FACTOR

VAL_SEED = 42   # as train.py: every validation pass draws the same chains
agent_type_lis = ["PEDESTRIAN", "BIKER", "CAR", "BUS", "CART", "SKATER"]   # target_agent_types 0..5


def to_device(data, device, is_train):
    data["is_train"] = is_train
    data["gpu"] = device.index or 0
    for key in data["cuda_tensor_lis"]:
        data[key] = data[key].to(device, non_blocking=True)
    return data


def ranker_inputs(model, data, train_mode_chains):
    """One batch's ranker inputs, from chains drawn through the inference path of forward().

    The ranker itself stays in eval mode for this pass, so its running normalisation
    statistics are updated once per step, by the training pass that follows.
    """
    model.eval()
    if train_mode_chains:
        for name, child in model.named_children():
            if name != "ranker":
                child.train()      # dropout live, BatchNorm on batch statistics
    model.keep_rank_inputs = True
    with torch.no_grad():
        model(data)
    model.keep_rank_inputs = False
    return model.last_rank_inputs


@torch.no_grad()
def evaluate(model, loader, device):
    """The val split with the chains drawn as at test time, from fixed noise.

    Returns every chain's ADE and FDE in pixels, (N, K) each in the ranker's order, and the
    agent types (N,).
    """
    model.eval()
    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all()
    torch.manual_seed(VAL_SEED)
    torch.cuda.manual_seed_all(VAL_SEED)
    ade, fde, types = [], [], []
    for data in loader:
        if data is None:
            continue
        data = to_device(data, device, is_train=False)
        trajectories, _, agent_types = model(data)
        if not getattr(model, "output_is_positions", False):
            trajectories = torch.cumsum(trajectories, dim=2)
        # Every target has all its future steps (label_mask_lis is all ones)
        err = (trajectories - data["label_pos_lis"].unsqueeze(1)).norm(dim=-1) * SCALE_FACTOR   # (N, K, T)
        ade.append(err.mean(dim=-1).cpu())
        fde.append(err[..., -1].cpu())
        types.append(agent_types.cpu())
    torch.set_rng_state(cpu_state)
    torch.cuda.set_rng_state_all(cuda_state)
    return torch.cat(ade).numpy(), torch.cat(fde).numpy(), torch.cat(types).numpy()


def summarise(ade, fde):
    """The ranker's quality from (N, K) chain errors in its order, as test_sdd.py defines it"""
    pick, best, rand = ade[:, 0].mean(), ade.min(axis=1).mean(), ade.mean(axis=1).mean()
    return {
        "pick_ade": pick, "pick_fde": fde[:, 0].mean(),
        "min_ade": best, "min_fde": fde.min(axis=1).mean(),
        "rand_ade": rand, "regret": pick - best,
        "closed": 100.0 * (rand - pick) / (rand - best) if rand > best else float("nan"),
        "hit1": 100.0 * (ade.argmin(axis=1) == 0).mean(),
    }


def by_type(ade, fde, types):
    return {name: summarise(ade[types == i], fde[types == i])
            for i, name in enumerate(agent_type_lis) if (types == i).any()}


def main():
    parser = argparse.ArgumentParser("Fine-tune the ranker on test-time chains, generator frozen")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="a checkpoint trained by this version's train.py, e.g. best_min_fde.pt")
    parser.add_argument("--out", type=str, default=None,
                        help="where to write the fine-tuned checkpoint (default: <checkpoint>_ranker_ft.pt)")
    parser.add_argument("--data_folder", type=str, default="../sdd/")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--steps_per_epoch", type=int, default=0,
                        help="batches per epoch; 0 = the whole train split")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="ranker learning rate, cosine-decayed over the epochs")
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="ranker AdamW weight decay")
    parser.add_argument("--tau_pick", type=float, default=0.3,
                        help="soft-pick temperature of rank_loss; 0.3 is where online training ends")
    parser.add_argument("--train_mode_chains", type=str, default="False",
                        help="control: draw the chains as online training does (True) instead of as at test time")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--val_batch_size", type=int, default=64)
    parser.add_argument("--num_worker", type=int, default=4)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)

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
    # Unused here but needed to instantiate the model
    parser.add_argument("--goal_loss_weight", type=float, default=0.5)
    parser.add_argument("--goal_curriculum_epochs", type=int, default=10)
    # Map nodes in the graph: read from the checkpoint by default
    parser.add_argument("--use_map", type=str, default=None)

    args = parser.parse_args()
    args.use_refiner = (args.use_refiner == "True")
    args.use_val_split = True
    args.scale_factor = SCALE_FACTOR
    train_mode_chains = (args.train_mode_chains == "True")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)

    # ------------------------------------------------------------------
    # Model: the checkpoint as trained, everything but the ranker frozen
    # ------------------------------------------------------------------
    if not os.path.isfile(args.checkpoint):
        sys.exit(f"ERROR: checkpoint not found: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    if not any(k.startswith("ranker.") for k in checkpoint["model_state_dict"]):
        sys.exit("ERROR: the checkpoint has no ranker weights, so it was not trained by this "
                 "version's train.py")

    # Map nodes in the graph, as trained (the checkpoint's args, else its weights), unless given
    if args.use_map is None:
        saved = checkpoint.get("args", {}).get("use_map")
        args.use_map = (saved if isinstance(saved, bool) else
                        any(k.startswith("encoder.map_emb") for k in checkpoint["model_state_dict"]))
    else:
        args.use_map = (args.use_map == "True")

    model_module = importlib.import_module("model")
    model = model_module.HDGT_model(input_dim=8, args=args)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    for name, param in model.named_parameters():
        param.requires_grad_(name.startswith("ranker."))
    ranker_params = list(model.ranker.parameters())
    if train_mode_chains:
        # In train mode BatchNorm normalises with batch statistics; momentum 0 keeps the
        # running ones (what eval mode, and so test_sdd.py, uses) exactly as trained
        for name, child in model.named_children():
            if name != "ranker":
                for module in child.modules():
                    if isinstance(module, nn.BatchNorm1d):
                        module.momentum = 0.0

    out_path = args.out or os.path.splitext(args.checkpoint)[0] + "_ranker_ft.pt"
    n_ranker = sum(param.numel() for param in ranker_params)
    n_frozen = sum(param.numel() for param in model.parameters()) - n_ranker
    print("=" * 80)
    print("Fine-tune the ranker on test-time chains (generator frozen)")
    print("=" * 80)
    print(f"Checkpoint : {args.checkpoint}  (epoch {checkpoint.get('epoch', '?')})")
    print(f"Chains     : {'TRAIN mode, as online training draws them (control)' if train_mode_chains else 'EVAL mode, as at test time'}")
    print(f"Map        : {'ON' if args.use_map else 'OFF'}   |   refiner: {'ON' if args.use_refiner else 'OFF'}")
    print(f"Ranker     : {n_ranker:,} parameters trained, {n_frozen:,} frozen")
    print(f"Optimiser  : AdamW lr {args.lr:g} (cosine), weight decay {args.weight_decay:g}, tau_pick {args.tau_pick:g}")
    print(f"Output     : {out_path}")
    print("=" * 80)

    train_loader, val_loader, _, _ = obtain_dataset(0, 1, args.seed, args)
    steps = len(train_loader) if args.steps_per_epoch <= 0 else min(args.steps_per_epoch, len(train_loader))
    optimizer = torch.optim.AdamW(ranker_params, lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))

    def ranker_state():
        return {"ranker." + k: v.detach().cpu().clone() for k, v in model.ranker.state_dict().items()}

    def row(epoch, loss, s):
        return (f"{epoch:>5} {loss:>11} | {s['pick_ade']:>8.3f} {s['pick_fde']:>8.3f} {s['regret']:>8.3f} "
                f"{s['closed']:>6.1f}% {s['hit1']:>5.1f}% | {s['min_ade']:>7.3f} {s['min_fde']:>7.3f}")

    # ------------------------------------------------------------------
    # Epoch 0: the ranker as trained, on the same val chains every epoch is scored on
    # ------------------------------------------------------------------
    print(f"\nVal split, chains from fixed noise (seed {VAL_SEED}); errors in pixels. regret = ADE - minADE")
    header = (f"{'epoch':>5} {'train loss':>11} | {'ADE':>8} {'FDE':>8} {'regret':>8} {'closed':>7} "
              f"{'hit@1':>6} | {'minADE':>7} {'minFDE':>7}")
    print(header)
    print("-" * len(header))
    ade, fde, types = evaluate(model, val_loader, device)
    before, before_types = summarise(ade, fde), by_type(ade, fde, types)
    best, best_types, best_state = dict(before, epoch=0), before_types, ranker_state()
    print(row(0, "as trained", before), flush=True)

    # ------------------------------------------------------------------
    # Fine-tuning
    # ------------------------------------------------------------------
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        loss_sum, n_seen = 0.0, 0
        for step, data in enumerate(train_loader):
            if step >= steps:
                break
            if data is None:
                continue
            data = to_device(data, device, is_train=True)

            inp = ranker_inputs(model, data, train_mode_chains)
            model.ranker.train()
            err_score, log_ade = model.ranker(inp["state"], inp["geom"], inp["agent_types"],
                                              inp["ctx"], inp["ctx_mask"])
            with torch.no_grad():
                ade_k = ((inp["positions"] - data["label_pos_lis"].unsqueeze(1)).norm(dim=-1).mean(dim=-1)
                         * SCALE_FACTOR)                                    # (N, K), pixels
            loss = model_module.rank_loss(err_score.float(), log_ade.float(), ade_k.float(),
                                          inp["agent_types"], tau_pick=args.tau_pick)
            optimizer.zero_grad()
            if torch.isfinite(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(ranker_params, 1.0)
                optimizer.step()
                n = int(inp["agent_types"].shape[0])
                loss_sum += loss.item() * n
                n_seen += n
        scheduler.step()

        ade, fde, types = evaluate(model, val_loader, device)
        s = summarise(ade, fde)
        mark = ""
        if s["pick_ade"] < best["pick_ade"]:
            best, best_types, best_state = dict(s, epoch=epoch), by_type(ade, fde, types), ranker_state()
            mark = "  <- best"
        print(row(epoch, f"{loss_sum / max(n_seen, 1):.4f}", s) + mark +
              f"   ({(time.time() - t0) / 60:.1f} min)", flush=True)

    # ------------------------------------------------------------------
    # Save: the original generator tensors with the best epoch's ranker
    # ------------------------------------------------------------------
    state = dict(checkpoint["model_state_dict"])
    assert set(best_state) == {k for k in state if k.startswith("ranker.")}, \
        "the ranker's tensors differ from the checkpoint's"
    state.update(best_state)
    # Not a resume point: the optimizer, scheduler and scaler states were the whole model's
    out = {k: v for k, v in checkpoint.items()
           if k not in ("optimizer_state_dict", "scheduler_state_dict", "scaler_state_dict")}
    out["model_state_dict"] = state
    out["ranker_finetune"] = {
        "source_checkpoint": os.path.abspath(args.checkpoint), "best_epoch": best["epoch"],
        "epochs": args.epochs, "steps_per_epoch": steps, "lr": args.lr,
        "weight_decay": args.weight_decay, "tau_pick": args.tau_pick,
        "train_mode_chains": train_mode_chains,
        "val_before": {k: float(v) for k, v in before.items()},
        "val_after": {k: float(v) for k, v in best.items() if k != "epoch"},
    }
    torch.save(out, out_path + ".tmp")
    os.replace(out_path + ".tmp", out_path)

    print("=" * 80)
    print(f"Best epoch {best['epoch']} (of {args.epochs}), val, before -> after:")
    print(f"  ADE    {before['pick_ade']:8.3f} -> {best['pick_ade']:8.3f} px")
    print(f"  FDE    {before['pick_fde']:8.3f} -> {best['pick_fde']:8.3f} px")
    print(f"  regret {before['regret']:8.3f} -> {best['regret']:8.3f} px   (ADE - minADE)")
    print(f"  minADE / minFDE {before['min_ade']:.3f} / {before['min_fde']:.3f} px, unchanged: the generator is frozen")
    print("  Regret by agent type:")
    for name, b in before_types.items():
        print(f"    {name:<11} {b['regret']:8.3f} -> {best_types[name]['regret']:8.3f} px")
    print(f"Saved {out_path}")
    print(f"Evaluate it with: python test_sdd.py --checkpoint {out_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
