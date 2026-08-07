from __future__ import annotations
import argparse
import json
import logging
import os
import random  
import time
from pathlib import Path
import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from β_VAE_256 import VoxelVAE256, vae_loss          # DENSE model (defined in β_VAE_256.py)
from clDice import SoftClDiceLoss, hard_cldice_batch
import math

def _nz(seq):
    """Replace non-finite values (e.g. NaN during clDice warmup) with None so the
    native chart shows a gap instead of erroring."""
    return [None if (v is None or not math.isfinite(v)) else float(v) for v in seq]

def _line_series(title, tr_x, tr_y, va_x, va_y):
    """One native, interactive W&B line chart with two lines: training + validation."""
    import wandb
    return wandb.plot.line_series(
        xs=[list(tr_x), list(va_x)],
        ys=[_nz(tr_y), _nz(va_y)],
        keys=["training", "validation"],
        title=title,
        xname="epoch",)

LR_SCHEDULE  = {0: 1e-5, 1: 5e-4}
CFG_DEFAULTS = {
                "batch_size": 2,            # 128^3 dense is memory-heavy; must be >=2 (enc_fc/dec_fc use BatchNorm1d, which needs batch>1 in train mode)
                "max_epochs": 150,          # original: cfg['max_epochs'] = 150
                "reg": 2e-3,                # original: cfg['reg'] = 0.001 (L2 weight decay)
                "checkpoint_every_nth": 5,  # original: cfg['checkpoint_every_nth'] = 5
                "num_latents": 100,         # our scale-up choice (paper used 100)
                "gamma": 0.99,              # weighted-BCE positive weight (released code)
                "use_kl": True,             # paper text says KL is part of the loss
                "num_workers": 4,
                "seed": 0,
                "beta": 4.0,                   # β-VAE constant; paper uses β > 1 (e.g. 4–250 depending on dataset)
                "viz_every": 10,               # ADDED: dump a reconstruction preview every N epochs (0 disables)
                "viz_threshold": 0.9,          # ADDED: prob cutoff for the saved binary recon — keep only voxels the model scores > 0.9
                "cldice_weight": 0.6,          # ADD: weight on the soft clDice loss term once active
                "cldice_warmup_epochs": 10,    # ADD: epochs of BCE-only before clDice switches on
                "cldice_iters": 8,             # ADD: thinning iterations for soft_skel (cover thickest tube radius)
                "cldice_eval_threshold": 0.5,  # ADD: prob cutoff for the HARD eval metric (0.5 == logits>=0, matches tp/tn)
                "cldice_eval_every": 5,        # FIX(4): run the exact eval-mode topology metric every N epochs (skimage skeletonize is slow)
                "cldice_eval_samples": 16,}    # FIX(4): fixed #samples (clean, no-aug, eval mode) the topology metric is averaged over


def split_train_val(data_dirs, val_counts, seed=0):
    """Per-directory holdout for a clean validation set.
    data_dirs  : list of dirs, SAME order as --data-dirs
    val_counts : files to hold out from each dir, e.g. [6, 4]
    Returns (train_paths, val_paths). Deterministic for a given seed,
    so a resumed run reuses the exact same validation files.
    """
    train_paths, val_paths = [], []
    for i, d in enumerate(data_dirs):
        files = sorted(Path(d).glob("*.nii.gz"))
        k = val_counts[i]
        if len(files) < k:
            raise ValueError(f"{d} has only {len(files)} files, can't hold out {k}")
        rng = random.Random(seed + i)   # deterministic, different per dir
        rng.shuffle(files)
        val_paths.extend(files[:k])
        train_paths.extend(files[k:])
    return train_paths, val_paths

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class NiftiMaskDataset(Dataset):
    """
    Loads binary 3D masks from one or more directories of .nii.gz files.
    Files are expected to be already at the target resolution (128^3) with
    binary {0, 1} values. Anything > 0.5 is treated as foreground; values
    are cast to float32. DENSE pipeline: both `input` and `target` are the same dense occupancy
    grid of shape (1, 128, 128, 128) — the network reconstructs its input.
    """

    def __init__(self, data_dirs=None, *, paths=None,):

        if paths is not None:
            self.paths = list(paths)
        else:
            self.paths = []
            for d in data_dirs:
                d = Path(d)
                if not d.is_dir():
                    raise FileNotFoundError(f"Not a directory: {d}")
                self.paths.extend(sorted(d.glob("*.nii.gz")))
        if not self.paths:
            raise ValueError("No .nii.gz files found.")
        self._n = len(self.paths)
        logging.info("Dataset: %d files", self._n)

    def __len__(self) -> int:
        return self._n  

    def _load(self, idx: int) -> np.ndarray:
        img = nib.load(str(self.paths[idx]))
        arr = np.asarray(img.dataobj)
        if arr.ndim != 3:
            raise ValueError(f"Expected 3D array at {self.paths[idx]}, got {arr.shape}")
        # The dense VoxelVAE is hardwired for 128^3 (encoder 128->...->7, decoder
        # 7->...->128). A wrong resolution otherwise fails deep inside a conv with
        # a cryptic shape error, so we check here with a clear message.
        if arr.shape != (256, 256, 256):
            raise ValueError(
                f"Dense VoxelVAE requires 256x256x256 volumes; {self.paths[idx]} "
                f"has shape {arr.shape}. Resample/crop to 256^3 first.")
        return (arr > 0.5).astype(np.float32)        # (256,256,256) {0,1}

    def __getitem__(self, idx: int) -> dict:
        mask = self._load(idx)
        # DENSE: the network input IS the occupancy grid; target is the same grid.
        vol = torch.from_numpy(np.ascontiguousarray(mask)).unsqueeze(0).float()  # (1,256,256,256)
        return {"input": vol, "target": vol}

def set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for g in optimizer.param_groups:
        g["lr"] = lr

def lr_for_epoch(schedule: dict, epoch: int, current_lr: float) -> float:
    """If `epoch` is in the schedule dict, return that LR; else keep current.
       Matches the original's per-epoch dict lookup."""
    if epoch in schedule:
        return float(schedule[epoch])
    return current_lr

# ---------------------------------------------------------------------------
# Metrics (faithful to the original)
# ---------------------------------------------------------------------------

def reconstruction_accuracy(logits: torch.Tensor, target_binary: torch.Tensor) -> tuple[float, float, float]:

    """
    Faithfully reproduces the original's three reconstruction metrics:
      error_rate     = mean( (X_hat >= 0) != (X >= 0) )   [in rescaled space, X >= 0 is X >= 0.5, in binary space]
      true_positives = mean( (X_hat >= 0 == X >= 0.5) & X >= 0.5 ) / mean(X >= 0.5)
      true_negatives = mean( (X_hat >= 0 == X >= 0.5) & X < 0.5 )  / mean(X < 0.5)
    Returns (accuracy, tp_rate, tn_rate).
    """

    with torch.no_grad():
        pred_pos = logits >= 0           # predicted-positive mask
        true_pos = target_binary >= 0.5  # true-positive mask
        true_neg = ~true_pos
        correct  = (pred_pos == true_pos)
        acc      = correct.float().mean().item()
        n_pos    = true_pos.float().mean().item()
        n_neg    = true_neg.float().mean().item()
        # Guard against degenerate batches (all-zero or all-one targets).
        tp = (correct & true_pos).float().mean().item() / n_pos if n_pos > 0 else 0.0
        tn = (correct & true_neg).float().mean().item() / n_neg if n_neg > 0 else 0.0

    return acc, tp, tn

# ---------------------------------------------------------------------------
# Reconstruction visualization (ADDED)
# ---------------------------------------------------------------------------
# Why these particular outputs for airway trees:
#
#   * .nii.gz  -> the ground truth. Airways are thin, sparse, branching tubes;
#                 the only faithful way to judge a reconstruction is to scroll
#                 and 3D-render it in ITK-SNAP or 3D Slicer. We write the input,
#                 the binarized recon, AND the soft probability map so you can
#                 inspect where the model is uncertain (the prob map is great for
#                 picking a better threshold than 0.5 if recall on distal
#                 branches is poor).

def logits_to_binary(logits: torch.Tensor, threshold: float = 0.5):
    """Decoder emits raw logits; convert to probabilities and a binary mask.

    Thresholding the probability at 0.5 is identical to the `logits >= 0` rule
    used by the training metrics, so the preview matches the logged tp/tn.
    Lower the threshold if distal airways are being dropped (favours recall).
    """
    probs  = torch.sigmoid(logits)
    binary = (probs >= threshold)
    return probs, binary


def _save_nifti(volume_np: np.ndarray, path: Path, affine: np.ndarray | None = None) -> None:
    """Write a single 3D volume to .nii.gz for ITK-SNAP / 3D Slicer.

    NOTE: the dataset discards the source affine, so we write an identity
    affine here — geometry displays correctly but voxel spacing is nominal
    (1 mm iso). If you need true spacing, thread the source img.affine through
    NiftiMaskDataset and pass it in.
    """
    if affine is None:
        affine = np.eye(4, dtype=np.float32)
    nib.save(nib.Nifti1Image(volume_np.astype(np.float32), affine), str(path))


def save_reconstruction_views(logits_cpu: torch.Tensor,
                              x_bin_cpu: torch.Tensor,
                              out_dir: Path,
                              epoch: int,
                              threshold: float = 0.5,
                              sample_idx: int = 0) -> None:
    """Orchestrator: turn one captured (logits, target) batch into the full set
    of preview artifacts for a single sample. Tensors are expected on CPU,
    shape (B, 1, D, H, W)."""
    out_dir.mkdir(parents=True, exist_ok=True)

    _, binary = logits_to_binary(logits_cpu, threshold)
    rec_prob  = torch.sigmoid(logits_cpu)[sample_idx, 0].numpy()
    rec_bin   = binary[sample_idx, 0].numpy().astype(np.uint8)
    inp_bin   = (x_bin_cpu[sample_idx, 0].numpy() >= 0.5).astype(np.uint8)
    stem      = f"epoch{epoch:04d}"

    _save_nifti(inp_bin,  out_dir / f"{stem}_input.nii.gz")
    _save_nifti(rec_bin,  out_dir / f"{stem}_recon.nii.gz")
    _save_nifti(rec_prob, out_dir / f"{stem}_recon_prob.nii.gz")
    logging.info("Saved reconstruction previews for epoch %d -> %s", epoch, out_dir)

@torch.no_grad()
def reconstruct_set(model, loader, device, viz_dir, epoch, cfg, like_training: bool = True):
    """Save reconstruction previews for the FIRST batch of `loader`, reusing the
    EXACT same routine, threshold and artifacts as the in-training previews.

    like_training=True (default): match training — model.train() so z is sampled
        stochastically and BatchNorm uses batch stats. BN running stats are
        snapshotted/restored so a validation batch never updates the model.
    like_training=False: deterministic — model.eval(), z = mu, running-stat BN.
    """
    was_training = model.training

    bn_snapshot = []
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            bn_snapshot.append((
                m,
                None if m.running_mean        is None else m.running_mean.clone(),
                None if m.running_var         is None else m.running_var.clone(),
                None if m.num_batches_tracked is None else m.num_batches_tracked.clone(),
            ))

    model.train(like_training)
    try:
        batch = next(iter(loader))
        x_in  = batch["input"].to(device, non_blocking=True)
        x_bin = batch["target"].to(device, non_blocking=True)
        logits, _, _ = model(x_in)
        save_reconstruction_views(
            logits.detach().float().cpu(),
            x_bin.detach().float().cpu(),
            viz_dir, epoch,
            threshold=cfg.get("viz_threshold", 0.5),
        )
    finally:
        for m, rm, rv, nbt in bn_snapshot:
            if rm  is not None: m.running_mean.copy_(rm)
            if rv  is not None: m.running_var.copy_(rv)
            if nbt is not None: m.num_batches_tracked.copy_(nbt)
        model.train(was_training)

# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    cfg: dict,
    start_itr: int,                 # global iteration count entering this epoch
    total_iterations: int,          # T for the β schedule
    epoch: int = 0,                 # ADDED: current epoch, used for viz gating + filenames
    viz_every: int = 0,             # ADDED: capture a preview every N epochs (0 disables)
    viz_dir: "Path | None" = None,  # ADDED: where previews are written
    cldice_fn = None,               # ADD: optional SoftClDiceLoss instance for the epoch
) -> tuple[dict, int]:
    
    model.train()   
    do_viz = (viz_every > 0) and (viz_dir is not None) and (epoch % viz_every == 0)
    viz_capture = None
    running = {"vloss": 0.0, "kl": 0.0, "beta_mean": 0.0, "n": 0, "cldice_soft": 0.0, "cldice_soft_n": 0, "cldice_w": 0.0, "dice": 0.0, "cldice_hard_train": 0.0}
    pooled  = {"tp_correct": 0, "pos_total": 0, "tn_correct": 0, "neg_total": 0}
    # FIX(4): hard clDice is no longer computed here (it was train-mode, 1 batch).
    # It now comes from evaluate_topology() under model.eval() over a fixed subset.

    itr = start_itr

    for batch in loader:
        itr   += 1
        beta  = cfg["beta"]
        x_in  = batch["input"].to(device, non_blocking=True)    # (B,1,128,128,128) dense -> encoder
        x_bin = batch["target"].to(device, non_blocking=True)   # (B,1,128,128,128) -> loss/metrics/viz

        logits, mu, logsigma = model(x_in)
        if do_viz and viz_capture is None:
            viz_capture = (logits.detach().float().cpu(), x_bin.detach().float().cpu())

        out = vae_loss(logits, x_bin, mu, logsigma, gamma=cfg["gamma"], beta=beta, use_kl=cfg["use_kl"],)

        # --- clDice topology term (soft, differentiable), warmed up -----------
        w_cl = cfg["cldice_weight"] if epoch >= cfg["cldice_warmup_epochs"] else 0.0
        loss = out.total
        cldice_soft_score = float("nan")   # FIX(4): NaN (= "not measured") during warmup, not a misleading 0
        if cldice_fn is not None and w_cl > 0.0:
            cldice_term = cldice_fn(logits, x_bin)      # this is a LOSS in [0,1] (0 = perfect topology)
            scale = out.recon.detach()                  # current BCE magnitude, constant (no grad)
            loss = loss + w_cl * scale * cldice_term    # clDice now lives on the BCE scale
            # FIX(3): log the SCORE (1 - loss) so clD_soft and clD_hard both mean
            # "higher is better" and can be read side by side.
            cldice_soft_score = (1.0 - cldice_term).item()
        # ----------------------------------------------------------------------

        optimizer.zero_grad(set_to_none=True)
        loss.backward()          # was out.total.backward()
        optimizer.step()

        # acc, _, _ = reconstruction_accuracy(logits, x_bin)
        bs        = x_bin.shape[0]

        # globally-pooled tp/tn: accumulate raw voxel counts, divide once at epoch end
        with torch.no_grad():
            pred_pos = logits >= 0  # predicted-positive mask as predicted by the model (logits >= 0 corresponds to predicted probability >= 0.5 after sigmoid)
            true_pos = x_bin >= 0.5 # true-positive mask (ground-truth positive voxels in the binary target) >= 0.5 corresponds to target value of 1 in the binary mask
            true_neg = ~true_pos    # ~ is ~ logical NOT, so true_neg is the mask of true-negative voxels (where the target binary mask is 0)
            correct  = (pred_pos == true_pos)
            pooled["tp_correct"] += (correct & true_pos).sum().item()
            pooled["pos_total"]  += true_pos.sum().item()
            pooled["tn_correct"] += (correct & true_neg).sum().item()
            pooled["neg_total"]  += true_neg.sum().item()
            # Per-sample (macro) foreground Dice, same threshold as tp/tn above
            # (logits >= 0  <=>  prob >= 0.5). Summed now; divided by n at epoch end
            # so the epoch value is the mean Dice per case, not per voxel.
            inter = (pred_pos & true_pos).flatten(1).sum(dim=1).float()          # TP per sample
            sizes = pred_pos.flatten(1).sum(dim=1).float() \
                + true_pos.flatten(1).sum(dim=1).float()                        # |pred| + |gt| per sample
            dice_b = (2.0 * inter + 1e-7) / (sizes + 1e-7)                        # (B,)  both-empty -> ~1
            running["dice"] += dice_b.sum().item()
            # Train-mode HARD clDice score (higher = better). NOTE: measured under
            # model.train() (stochastic z, batch-stat BN), so noisier than the eval
            # cldice_hard, and skimage skeletonize makes this slow on every batch.
            running["cldice_hard_train"] += hard_cldice_batch(logits, x_bin, threshold=cfg["cldice_eval_threshold"]) * bs

        running["vloss"]       += out.recon.item() * bs
        if cldice_soft_score == cldice_soft_score:       # FIX(4): True only when not NaN
            running["cldice_soft"]   += cldice_soft_score * bs
            running["cldice_soft_n"] += bs
        running["cldice_w"]    += w_cl * bs              # ADD (so you can see when it switched on)
        running["kl"]          += out.kl.item()    * bs
        # running["acc"]       += acc * bs
        running["beta_mean"]   += beta * bs
        running["n"]           += bs

    # --- ADDED: write previews once the epoch's compute is done -------------
    # We deferred the disk I/O to here (rather than mid-loop) so file writes
    # never sit between a forward and its backward/step. The capture timing
    # above is still "right after the forward pass"; only the saving is later.
    if viz_capture is not None:
        cap_logits, cap_xbin = viz_capture
        save_reconstruction_views(cap_logits, cap_xbin, viz_dir, epoch,
                                  threshold=cfg.get("viz_threshold", 0.5))
    # ------------------------------------------------------------------------

    n = max(running["n"], 1)
    # Average most running sums over all samples; the soft-clDice score is averaged
    # only over the batches where it was actually measured (NaN during warmup).
    soft_n = running.pop("cldice_soft_n")
    soft_sum = running.pop("cldice_soft")
    metrics = {k: v / n for k, v in running.items() if k != "n"}
    metrics["cldice_soft"] = (soft_sum / soft_n) if soft_n > 0 else float("nan")  # FIX(3): train-mode SCORE, NaN in warmup
    metrics["tp"] = pooled["tp_correct"] / pooled["pos_total"] if pooled["pos_total"] > 0 else 0.0
    metrics["tn"] = pooled["tn_correct"] / pooled["neg_total"] if pooled["neg_total"] > 0 else 0.0
    metrics["acc"] = (pooled["tp_correct"] + pooled["tn_correct"]) / (pooled["pos_total"] + pooled["neg_total"]) if (pooled["pos_total"] + pooled["neg_total"]) > 0 else 0.0
    return metrics, itr

# ---------------------------------------------------------------------------
# Topology evaluation (FIX 3 & 4): exact hard clDice + soft clDice, computed
# under model.eval() (deterministic z = mu, running-stat BatchNorm) over a
# FIXED subset of clean, un-augmented samples. Both are returned as SCORES in
# [0, 1] where higher = better, so they read consistently against each other
# and against the train-mode soft score.
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_topology(model: nn.Module, loader: DataLoader, device: torch.device, cfg: dict, cldice_fn=None, max_samples: int | None = None) -> dict:
    was_training = model.training
    model.eval()                                 # deterministic decode (z=mu) + running-stat BN
    hard_sum, soft_sum, seen = 0.0, 0.0, 0
    vloss_sum, kl_sum, dice_sum = 0.0, 0.0, 0.0          # ADD: val recon / KL / Dice
    tp_c, pos_t, tn_c, neg_t = 0, 0, 0, 0                # ADD: pooled tp/tn voxel counts
    try:
        for batch in loader:
            x_in  = batch["input"].to(device, non_blocking=True)
            x_bin = batch["target"].to(device, non_blocking=True)
            logits, mu, logsigma = model(x_in)          # CHANGED: keep mu/logsigma (were discarded)
            bs = x_bin.shape[0]

            # ADD: same loss terms as training, measured on the val set
            out = vae_loss(logits, x_bin, mu, logsigma, gamma=cfg["gamma"], beta=cfg["beta"], use_kl=cfg["use_kl"])
            vloss_sum += out.recon.item() * bs
            kl_sum    += out.kl.item()    * bs

            # ADD: pooled tp/tn + per-sample Dice, identical rule to train (logits >= 0)
            pred_pos = logits >= 0
            true_pos = x_bin >= 0.5
            correct  = (pred_pos == true_pos)
            tp_c  += (correct &  true_pos).sum().item()
            pos_t += true_pos.sum().item()
            tn_c  += (correct & ~true_pos).sum().item()
            neg_t += (~true_pos).sum().item()
            inter = (pred_pos & true_pos).flatten(1).sum(dim=1).float()
            sizes = pred_pos.flatten(1).sum(dim=1).float() + true_pos.flatten(1).sum(dim=1).float()
            dice_sum += ((2.0 * inter + 1e-7) / (sizes + 1e-7)).sum().item()

            # exact, non-differentiable clDice SCORE in [0,1] (higher = better)
            hard_sum += hard_cldice_batch(logits, x_bin, threshold=cfg["cldice_eval_threshold"]) * bs
            if cldice_fn is not None:
                soft_sum += (1.0 - cldice_fn(logits, x_bin)).item() * bs

            seen += bs
            if max_samples is not None and seen >= max_samples:
                break
    finally:
        if was_training:
            model.train()                        # always restore the caller's mode

    if seen == 0:
        return {"cldice_hard": float("nan"), "cldice_soft_eval": float("nan"), "eval_n": 0,
                "val_vloss": float("nan"), "val_kl": float("nan"), "val_dice": float("nan"),
                "val_tp": float("nan"), "val_tn": float("nan")}
    return {"cldice_hard": hard_sum / seen,
            "cldice_soft_eval": soft_sum / seen,
            "eval_n": seen,
            "val_vloss": vloss_sum / seen,                    # ADD
            "val_kl":    kl_sum   / seen,                     # ADD
            "val_dice":  dice_sum / seen,                     # ADD
            "val_tp":    tp_c / pos_t if pos_t > 0 else 0.0,  # ADD
            "val_tn":    tn_c / neg_t if neg_t > 0 else 0.0}  # ADD

# ---------------------------------------------------------------------------
# Checkpointing & logging
# ---------------------------------------------------------------------------

def save_checkpoint(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, itr: int, hist: dict | None = None) -> None:

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")                    # latest.pt.tmp
    torch.save({"epoch": epoch, "itr": itr, "ts": time.time(),
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "hist": hist}, tmp)                                 # W&B combined-chart history
    os.replace(tmp, path)          # atomic rename on the same filesystem

class JsonlLogger:
    """Append-only JSONL logger, equivalent in spirit to utils.metrics_logging."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.f = open(path, "w", buffering=1)  # line-buffered

    def log(self, **kv) -> None:
        self.f.write(json.dumps(kv) + "\n")

    def close(self) -> None:
        self.f.close()

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dirs", type=Path, nargs="+", required=True,help="One or more directories containing .nii.gz mask files. " "All files in all listed directories are pooled into ""a single training set.")
    parser.add_argument("--out-dir", type=Path, required=True, help="Where checkpoints + metrics.jsonl are written.")
    parser.add_argument("--batch-size", type=int, default=CFG_DEFAULTS["batch_size"])
    parser.add_argument("--max-epochs", type=int, default=CFG_DEFAULTS["max_epochs"])
    parser.add_argument("--num-workers", type=int, default=CFG_DEFAULTS["num_workers"])
    parser.add_argument("--num-latents", type=int, default=CFG_DEFAULTS["num_latents"])
    parser.add_argument("--seed", type=int, default=CFG_DEFAULTS["seed"])
    parser.add_argument("--resume", type=Path, default=None, help="Path to a checkpoint to resume from.")
    parser.add_argument("--continue", dest="cont", action="store_true", help="Resume from <out-dir>/latest.pt if it exists; else start fresh.")
    parser.add_argument("--beta", type=float, default=CFG_DEFAULTS["beta"], help="Static beta value for beta-VAE formulation.")
    parser.add_argument("--viz-every", type=int, default=CFG_DEFAULTS["viz_every"], help="ADDED: save a reconstruction preview every N epochs (0 disables).")
    parser.add_argument("--wandb", action="store_true", help="Stream metrics live to Weights & Biases.")
    parser.add_argument("--wandb-project", type=str, default="airway-bvae")
    args = parser.parse_args()

    # Compose final cfg
    cfg = dict(CFG_DEFAULTS)
    cfg.update({"batch_size":  args.batch_size,
                "max_epochs":  args.max_epochs,
                "num_workers": args.num_workers,
                "num_latents": args.num_latents,
                "seed":        args.seed,
                "beta":        args.beta,
                "viz_every":   args.viz_every})  # ADDED

    args.out_dir.mkdir(parents=True, exist_ok=True)
    use_wandb = args.wandb
    if use_wandb:
        import wandb
        wandb.init(project=args.wandb_project,
                   name=args.out_dir.name,
                   id=args.out_dir.name,      # stable id -> --continue resumes the SAME run
                   resume="allow",
                   config=cfg)  

    logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=[logging.FileHandler(args.out_dir / "train.log"),logging.StreamHandler(),],)
    
    mlog = JsonlLogger(args.out_dir / "metrics.jsonl")

    # ADDED: previews go in a dedicated subdirectory next to the checkpoints.
    # Tip for a deterministic preview instead of the stochastic training one:
    # build a small helper that does `model.eval(); decode(mu)` on a fixed
    # batch and call it from here on the same epoch cadence.
    viz_dir = args.out_dir / "reconstructions"

    logging.info("Config: %s", cfg)

    # Reproducibility
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        logging.warning("No CUDA GPU detected — running on CPU. A dense 128^3 VAE ""will be extremely slow/memory-heavy on CPU.")
    logging.info("Device: %s", device)

    train_paths, val_paths = split_train_val(args.data_dirs, val_counts=[6, 4], seed=cfg["seed"])
    logging.info("Split: %d train / %d val", len(train_paths), len(val_paths))
    for p in val_paths:
        logging.info("  held-out val: %s", p.name)

    train_ds     = NiftiMaskDataset(paths=train_paths)
    train_loader = DataLoader(train_ds,
                              batch_size=cfg["batch_size"],
                              shuffle=True,
                              num_workers=cfg["num_workers"],
                              pin_memory=False,
                              drop_last=True,)   # default collate stacks (B,1,128,128,128)
    
    logging.info("Train set: %d files (len=%d)", train_ds._n, len(train_ds))

    # Evaluation loader: clean copies only (augment=False), SHUFFLED — so each
    # topology eval draws a fresh random subset of cldice_eval_samples trees.
    # Note: the metric is no longer measured on the same samples every time, so
    # epoch-to-epoch values are noisier / less directly comparable.
    eval_ds     = NiftiMaskDataset(paths=val_paths)
    eval_loader = DataLoader(eval_ds,
                             batch_size=max(2, cfg["batch_size"] // 2),  # eval is heavier; BatchNorm1d needs >=2
                             shuffle=True,  # shuffle is not strictly necessary for eval, but it doesn't hurt   
                             num_workers=cfg["num_workers"],
                             pin_memory=False,
                             drop_last=False,)   # default collate stacks (B,1,128,128,128)

    # ---- Model ----
    model     = VoxelVAE256(num_latents=cfg["num_latents"]).to(device)
    cldice_fn = SoftClDiceLoss(iters=cfg["cldice_iters"]).to(device)   # ADD
    logging.info("clDice: weight=%.3f after %d warmup epochs, soft iters=%d, eval thr=%.2f", cfg["cldice_weight"], cfg["cldice_warmup_epochs"], cfg["cldice_iters"], cfg["cldice_eval_threshold"])     # ADD
    n_params = sum(p.numel() for p in model.parameters())
    logging.info("Model: %s, %d params (%.2f M)", type(model).__name__, n_params, n_params / 1e6)
    optimizer = torch.optim.Adam( model.parameters(), lr=LR_SCHEDULE[0], weight_decay=cfg["reg"],)

   # ---- Resume ----
    start_epoch, itr = 0, 0
    resumed_hist     = None        # restored from checkpoint on resume (for W&B charts)
    resume_path      = args.resume
    if args.cont and resume_path is None:      # --continue -> newest checkpoint_N.pt in out-dir
        ckpts = list(args.out_dir.glob("checkpoint_*.pt"))
        if ckpts:
            resume_path = max(ckpts, key=lambda p: int(p.stem.split("_")[1]))  # largest epoch N
            logging.info("--continue: newest checkpoint is %s", resume_path.name)
        else:
            logging.info("--continue set but no checkpoint_*.pt in %s; starting fresh.", args.out_dir)

    if resume_path is not None:
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optim_state"])
        start_epoch = int(ckpt["epoch"]) + 1
        itr = int(ckpt.get("itr", 0))
        resumed_hist = ckpt.get("hist", None)   # None for older checkpoints saved before this field existed
        logging.info("Resumed from %s @ epoch %d, itr %d", resume_path, start_epoch, itr)

    # ---- Training loop ----
    current_lr = LR_SCHEDULE[max(k for k in LR_SCHEDULE if k <= start_epoch)]
    set_lr(optimizer, current_lr) # 
    logging.info("Initial learning rate: %g", current_lr)
    total_iterations = cfg["max_epochs"] * len(train_loader)

    # FIX(4): holds the most recent eval-mode topology scores, carried forward on
    # epochs where the (slow) exact eval doesn't run.
    last_eval = {"cldice_hard": float("nan"), "cldice_soft_eval": float("nan"), "eval_n": 0,
                 "val_vloss": float("nan"), "val_kl": float("nan"), "val_dice": float("nan"),
                 "val_tp": float("nan"), "val_tn": float("nan")}

    # History feeding the three combined native line charts. Training points are
    # appended every epoch; validation points only on epochs where eval runs.
    hist = resumed_hist if resumed_hist is not None else {"epoch": [], "vloss": [], "cldice_soft": [], "cldice_hard_train": [], "dice": [], "val_epoch": [], "val_vloss": [], "cldice_hard": [], "val_dice": []}

    for epoch in range(start_epoch, cfg["max_epochs"]):

        ran_eval = False   # set True below when the eval pass runs this epoch
        new_lr = lr_for_epoch(LR_SCHEDULE, epoch, current_lr) # LR schedule check (matches the original's per-epoch dict lookup)

        if new_lr != current_lr:
            logging.info("Changing learning rate from %g to %g", current_lr, new_lr)
            set_lr(optimizer, new_lr)
            current_lr = new_lr

        t0 = time.time()
        train_metrics, itr = train_one_epoch( model, train_loader, optimizer, device, cfg, start_itr=itr, total_iterations=total_iterations, epoch=epoch, viz_every=cfg["viz_every"], viz_dir=viz_dir / "train", cldice_fn=cldice_fn,)

        # FIX(4): exact, eval-mode topology metric on a cadence (skimage skeletonize
        # is slow). Runs on the first epoch and every cldice_eval_every epochs; the
        # last measured values are carried forward on the epochs in between.
        if (epoch == start_epoch) or ((epoch + 1) % cfg["cldice_eval_every"] == 0) or (epoch == cfg["max_epochs"] - 1):
            eval_metrics = evaluate_topology(model, eval_loader, device, cfg, cldice_fn=cldice_fn, max_samples=cfg["cldice_eval_samples"])
            last_eval    = eval_metrics
            ran_eval     = True
            logging.info("Val eval @ epoch %d over %d samples: clD_hard=%.4f  clD_soft=%.4f  "
                         "val_loss=%.4f  val_kl=%.4f  val_dice=%.4f",
                         epoch, eval_metrics["eval_n"], eval_metrics["cldice_hard"], eval_metrics["cldice_soft_eval"],
                         eval_metrics["val_vloss"], eval_metrics["val_kl"], eval_metrics["val_dice"])
            
        # Validation-set reconstruction previews — same routine/threshold as the
        # training previews, same viz cadence. Train -> reconstructions/train,
        # validation -> reconstructions/val.
        if cfg["viz_every"] > 0 and (epoch % cfg["viz_every"] == 0):
            reconstruct_set(model, eval_loader, device, viz_dir / "val", epoch, cfg)
        # `last_eval` persists across epochs; initialised once before the loop.
        # The per-epoch line reports only the train-mode soft score (fresh every
        # epoch). The eval-mode topology scores (clD_hard / clD_soft(ev)) are NOT
        # echoed here — they're printed on their own "Topology eval @ epoch ..."
        # line above, only on the epochs where they are actually recomputed, so
        # stale (carried-forward) values never appear in the logs.
        #   clD_soft(tr) = train-mode soft score (NaN during warmup)

        logging.info( "Epoch %d/%d  lr=%g  β=%.3f  v_loss=%.4f  D_kl=%.4f  acc=%.4f  tp=%.4f  tn=%.4f  "
                      "clD_soft(tr)=%.4f  clD_hard(tr)=%.4f  dice=%.4f  (%.1fs)",
                      epoch, cfg["max_epochs"] - 1, current_lr, train_metrics["beta_mean"],
                      train_metrics["vloss"], train_metrics["kl"],
                      train_metrics["acc"], train_metrics["tp"], train_metrics["tn"],
                      train_metrics["cldice_soft"], train_metrics["cldice_hard_train"],
                      train_metrics["dice"], dt := time.time() - t0)

        record = dict(phase="train", epoch=epoch, itr=itr, lr=current_lr, dt=dt, **train_metrics, **last_eval)
        mlog.log(**record)   # all previous metrics still saved to metrics.jsonl

        # ---- append history, then log ONLY the three combined native charts ----
        hist["epoch"].append(epoch)
        hist["vloss"].append(train_metrics["vloss"])
        hist["cldice_soft"].append(train_metrics["cldice_soft"])          # train clDice (soft)
        hist["cldice_hard_train"].append(train_metrics["cldice_hard_train"])    # train clDice (hard)
        hist["dice"].append(train_metrics["dice"])
        if ran_eval:   # validation values are only fresh on eval epochs
            hist["val_epoch"].append(epoch)
            hist["val_vloss"].append(last_eval["val_vloss"])
            hist["cldice_hard"].append(last_eval["cldice_hard"])   # val clDice (hard)
            hist["val_dice"].append(last_eval["val_dice"])

        if use_wandb:
            wandb.log({
                "loss_combined":   _line_series("Loss (train vs val)", hist["epoch"], hist["vloss"], hist["val_epoch"], hist["val_vloss"]),
                "cldice_combined": _line_series("clDice hard (train vs val)", hist["epoch"], hist["cldice_hard_train"], hist["val_epoch"], hist["cldice_hard"]),
                "dice_combined":   _line_series("Dice (train vs val)", hist["epoch"], hist["dice"],  hist["val_epoch"], hist["val_dice"]),
            }, step=epoch)

        if (epoch + 1) % 10 == 0:                                   # your cadence
            ckpt_name = f"checkpoint_{epoch + 1}.pt"                # global epoch in the name
            save_checkpoint(args.out_dir / ckpt_name, model, optimizer, epoch, itr, hist=hist)
            for old in args.out_dir.glob("checkpoint_*.pt"):        # keep ONLY this one: remove any earlier checkpoint_*.pt
                if old.name != ckpt_name:
                    old.unlink(missing_ok=True)
            logging.info("Checkpoint @ epoch %d -> %s (older ones removed)", epoch, ckpt_name)

    # Final checkpoint
    save_checkpoint(args.out_dir / "final.pt", model, optimizer, cfg["max_epochs"] - 1, itr, hist=hist)
    logging.info("Training done.")
    mlog.close()
    if use_wandb:
        wandb.finish()

if __name__ == "__main__":
    main()