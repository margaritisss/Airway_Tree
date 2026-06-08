"""
PyTorch training script for VoxelVAE128, faithful to:

    Brock, Lim, Ritchie, Weston (2016).
    "Generative and Discriminative Voxel Modeling with Convolutional Neural Networks."
    arXiv:1608.04236

Original Theano/Lasagne training script:
    https://github.com/ajbrock/Generative-and-Discriminative-Voxel-Modeling
    (VAE_OL.py)

Faithful reproductions of the original
--------------------------------------
- Nesterov-momentum SGD with momentum=0.9.
- L2 weight regularization (cfg['reg'] = 0.001) applied to *all* trainable
  parameters via the optimizer's weight_decay.
- Two-tier learning-rate schedule: first epoch warmup at 1e-4, jump to 5e-3
  at the start of epoch 1, then constant. Matches the released code's
  `lr_schedule = {0: 0.0001, 1: 0.005}`.
- Input rescaling: binary {0,1} voxel grids are mapped to {-1, 2} via
  `3*x - 1` before being fed to the encoder. The reconstruction target stays
  in {0,1}, as in the released code.
- Data augmentation per epoch:
    * a clean copy of each example, and
    * a jittered copy with random flips along the first two spatial axes
      (each with probability 0.2),
  shuffled together. Random translations from the original `jitter_chunk`
  are intentionally omitted: our airway masks are pre-registered, so
  translating breaks the spatial correspondence the model can exploit.
- Reconstruction accuracy is measured as (logits >= 0) == (target >= 0.5),
  with true-positive and true-negative rates broken out separately, matching
  the original.
- Checkpointing every N epochs.

Adjustments vs. the original (made on purpose for our 128^3 + ATM22 setup)
-------------------------------------------------------------------------
- Grid size is 128^3 (vs the paper's 32^3).
- batch_size defaults to 4 (vs 64). Our scaled-up model is ~21M params and
  a single 128^3 fp32 sample is 8 MB; 64 won't fit. Override on the CLI.
- Data is loaded from .nii.gz files in one or more directories. All folders
  are pooled into a single training set (no train/val split, no CSV).
- No introspective loss, no discriminative loss, no class-conditional
  decoder, no validation pass. These existed in the original under
  cfg['introspect'], cfg['discriminative'], cfg['cc'] flags; we keep the
  pure-VAE path only.

"""

from __future__ import annotations
import argparse
from html import parser
import json
import logging
import time
from pathlib import Path
import nibabel as nib
import numpy as np
import torch
import torch._dynamo  # noqa: F401
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# Model + loss live in VAE.py next to this file
from airway_project.Encoding.other.VAE_cbdice_DICE import VoxelVAE128, vae_loss, cyclical_beta
from loss.cbdice_loss import SoftcbDiceLoss

# ---------------------------------------------------------------------------
# Config (mirrors the structure of the original cfg dict)
# ---------------------------------------------------------------------------

# Learning-rate schedule, keyed by epoch index. Matches the original
# `lr_schedule = {0: 0.0001, 1: 0.005}`: warmup for one epoch, then jump.
LR_SCHEDULE = {0: 1e-5, 1: 1e-4}

CFG_DEFAULTS = {
    "batch_size": 4,            # original: 64 at 32^3; we drop for 128^3
    "max_epochs": 150,          # original: cfg['max_epochs'] = 150
    "reg":        2e-3,         # original: cfg['reg'] = 0.001 (L2 weight decay)
    "momentum":   0.9,          # original: cfg['momentum'] = 0.9, Nesterov
    # "max_jitter": 16,         # scaled from 4 (32^3) to 16 (128^3)
    "flip_prob": 0.2,           # original jitter_chunk used binomial(1, 0.2)
    "checkpoint_every_nth": 5,  # original: cfg['checkpoint_every_nth'] = 5
    "num_latents": 100,         # our scale-up choice (paper used 100)
    "gamma": 0.99,              # weighted-BCE positive weight (released code)
    "use_kl": True,             # paper text says KL is part of the loss
    "num_workers": 4,
    "seed": 0,
    "beta_n_cycles": 4,          # M in Fu et al.
    "beta_ratio":    0.5,        # R in Fu et al.
    "beta_max":      0.001,        # peak β within each cycle
    "beta_shape":    "linear",   # "linear" | "sigmoid" | "cosine"
    # ---- Reconstruction blend: alpha_dice * Dice + (1 - alpha_dice) * cbDice ----
    # During warmup (epoch < alpha_dice_warmup) we use pure Dice (alpha=1.0) so
    # the decoder learns to produce tree-shaped outputs before the noisy
    # cbDice gradient is introduced. Then we linearly ramp alpha from 1.0 down
    # to alpha_dice_min over alpha_dice_ramp epochs.
    "alpha_dice_warmup":  15,    # epochs of pure Dice before bringing cbDice in
    "alpha_dice_ramp":    10,    # epochs over which to ramp alpha 1.0 -> alpha_dice_min
    "alpha_dice_min":     0.5,   # final Dice weight (cbDice weight = 1 - this)
    "dice_smooth":        1.0,
}


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class NiftiMaskDataset(Dataset):
    """
    Loads binary 3D masks from one or more directories of .nii.gz files.

    Files are expected to be already at the target resolution (128^3) with
    binary {0, 1} values.Anything > 0.5 is treated as foreground; values
    are cast to float32.

    Per-epoch augmentation (mirrors the original `data_loader` + `jitter_chunk`):

      - emits BOTH the clean sample and a jittered copy in the same epoch,
        shuffled together by the DataLoader's `shuffle=True`,
      - independent probability `flip_prob` of flipping along the first two
        spatial axes, matching the original's binomial(1, 0.2) flips on
        `dst[:, :, ::-1, :, :]` and `dst[:, :, :, ::-1, :]`.

    Index range:
      indices [0, N)        -> clean copies
      indices [N, 2N)       -> jittered copies (same underlying file as i-N)
    so a single epoch sees one clean + one noisy version of every sample,
    shuffled together by the DataLoader. This is the original's
    "training on one noisy and one uncorrupted copy" behaviour.
    """

    def __init__(
        self,
        data_dirs: list[Path],
        # max_jitter: int = 16,
        flip_prob: float = 0.2,
        augment: bool = True,
    ):
        # self.max_jitter = max_jitter
        self.flip_prob = flip_prob
        self.augment = augment

        self.paths: list[Path] = []
        for d in data_dirs:
            d = Path(d)
            if not d.is_dir():
                raise FileNotFoundError(f"Not a directory: {d}")
            found = sorted(d.glob("*.nii.gz"))
            if not found:
                logging.warning("No .nii.gz files in %s", d)
            self.paths.extend(found)

        if not self.paths:
            raise ValueError(
                f"No .nii.gz files found in any of: {[str(d) for d in data_dirs]}"
            )

        self._n = len(self.paths)
        logging.info("Found %d .nii.gz files across %d directories",
                     self._n, len(data_dirs))

    def __len__(self) -> int:
        # 2x because each epoch emits a clean and a jittered copy of every
        # sample, matching the original.
        return 2 * self._n if self.augment else self._n

    def _load(self, idx: int) -> np.ndarray:
        # nibabel returns the array in storage order. For a binary mask the
        # axis convention doesn't matter to the network — augmentation is
        # symmetric, and the VAE has no notion of anatomical orientation.
        img = nib.load(str(self.paths[idx]))
        arr = np.asarray(img.dataobj)  # avoids forcing float64
        if arr.ndim != 3:
            raise ValueError(
                f"Expected 3D array at {self.paths[idx]}, got shape {arr.shape}"
            )
        # Binarize defensively in case the mask has stray values
        # (uint8 0/1, int16, float32 ~1.0, etc.)
        arr = (arr > 0.5).astype(np.float32)
        # Add channel dim -> (1, D, H, W)
        return arr[None, ...]

    def _jitter(self, x: np.ndarray) -> np.ndarray:
        # Mirrors jitter_chunk: flips on axes corresponding to the first
        # two spatial axes (the original applied them to a chunk tensor of
        # shape (N, C, D, H, W); after dropping the batch dim here, the
        # spatial axes of `x` are at positions 1, 2, 3).
        dst = x.copy()
        if np.random.binomial(1, self.flip_prob):
            dst = dst[:, ::-1, :, :]
        if np.random.binomial(1, self.flip_prob):
            dst = dst[:, :, ::-1, :]
        
        # Negative-stride slices from the flips above produce non-contiguous
        # arrays; ensure contiguity so PyTorch is happy.
        return np.ascontiguousarray(dst)

    def __getitem__(self, idx: int) -> torch.Tensor:
        if self.augment:
            base = idx % self._n
            do_jitter = idx >= self._n
        else:
            base = idx
            do_jitter = False

        x = self._load(base)
        if do_jitter:
            x = self._jitter(x)

        return torch.from_numpy(x)  # (1, D, H, W) float32 in {0., 1.}

# ---------------------------------------------------------------------------
# LR schedule (matches the original's epoch-keyed dict behaviour)
# ---------------------------------------------------------------------------

def set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for g in optimizer.param_groups:
        g["lr"] = lr

def lr_for_epoch(schedule: dict, epoch: int, current_lr: float) -> float:
    """If `epoch` is in the schedule dict, return that LR; else keep current."""
    if epoch in schedule:
        return float(schedule[epoch])
    return current_lr

def alpha_dice_for_epoch(epoch: int, cfg: dict) -> float:
    """
    Alpha schedule for the Dice/cbDice blend.
      - epochs [0, warmup):              alpha = 1.0  (pure Dice)
      - epochs [warmup, warmup+ramp):    linear ramp 1.0 -> alpha_dice_min
      - epochs [warmup+ramp, end):       alpha = alpha_dice_min  (balanced)
    """
    w   = cfg["alpha_dice_warmup"]
    r   = cfg["alpha_dice_ramp"]
    lo  = cfg["alpha_dice_min"]
    if epoch < w:
        return 1.0
    if epoch < w + r:
        # linear interpolation 1.0 -> lo over `r` epochs
        frac = (epoch - w) / r
        return 1.0 + (lo - 1.0) * frac
    return lo


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def reconstruction_accuracy(logits: torch.Tensor, target_binary: torch.Tensor) -> tuple[float, float, float]:
    """
    Faithfully reproduces the original's three reconstruction metrics:

      error_rate     = mean( (X_hat >= 0) != (X >= 0) )   [in rescaled space, X >= 0 is X >= 0.5, in binary space]
      true_positives = mean( (X_hat >= 0 == X >= 0.5) & X >= 0.5 ) / mean(X >= 0.5)
      true_negatives = mean( (X_hat >= 0 == X >= 0.5) & X < 0.5 )  / mean(X < 0.5)

    Returns (accuracy, tp_rate, tn_rate).
    """
    with torch.no_grad(): # with torch.no_grad() to avoid accidentally backpropagating through these metrics
        pred_pos = logits >= 0           # predicted-positive mask
        true_pos = target_binary >= 0.5  # true-positive mask
        true_neg = ~true_pos

        correct = (pred_pos == true_pos) # overall correctness mask
        acc     = correct.float().mean().item()     # 

        n_pos = true_pos.float().mean().item()
        n_neg = true_neg.float().mean().item()
        # Guard against degenerate batches (all-zero or all-one targets).
        tp = (correct & true_pos).float().mean().item() / n_pos if n_pos > 0 else 0.0
        tn = (correct & true_neg).float().mean().item() / n_neg if n_neg > 0 else 0.0

    return acc, tp, tn

# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

def train_one_epoch(model: nn.Module,
                    loader: DataLoader,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device,
                    cfg: dict,
                    cbdice_fn: SoftcbDiceLoss,
                    start_itr: int,           # global iteration count entering this epoch
                    total_iterations: int,) -> tuple[dict, int]:
                    
    model.train()

    running = {"vloss":           0.0, 
               "kl":              0.0, 
               "dice":            0.0, 
               "cbdice":          0.0,
               "acc":             0.0, 
               "tp":              0.0, 
               "tn":              0.0,
               "beta_mean":       0.0, 
               "alpha_dice_mean": 0.0, 
               "n":                0}

    itr = start_itr 

    for x_bin in loader:
        itr += 1
        beta = cyclical_beta(
                            iteration=itr,
                            total_iterations=total_iterations,
                            n_cycles=cfg["beta_n_cycles"],
                            ratio=cfg["beta_ratio"],
                            beta_max=cfg["beta_max"],
                            shape=cfg["beta_shape"],)

        x_bin = x_bin.to(device, non_blocking=True)
        x_in  = 3.0 * x_bin - 1.0

        logits, mu, logsigma = model(x_in)
        
        out = vae_loss( 
            logits, x_bin, mu, logsigma,
            cbdice_module = cbdice_fn,
            beta          = beta,
            alpha_dice    = cfg["alpha_dice_current"],
            use_kl        = cfg["use_kl"],
            dice_smooth   = cfg["dice_smooth"],)

        optimizer.zero_grad(set_to_none=True) # Clear gradients before backward pass
        out.total.backward()                  # Compute gradients
        optimizer.step()                      # Update model parameters

        acc, tp, tn = reconstruction_accuracy(logits, x_bin)
        bs = x_bin.shape[0]  

        running["vloss"]           += out.recon.item()  * bs   # combined recon
        running["kl"]              += out.kl.item()     * bs
        running["dice"]            += out.dice.item()   * bs   # in [0, 1], lower better
        running["cbdice"]          += out.cbdice.item() * bs   # in [0, 1], lower better
        running["acc"]             += acc * bs
        running["tp"]              += tp  * bs
        running["tn"]              += tn  * bs
        running["beta_mean"]       += beta * bs
        running["alpha_dice_mean"] += cfg["alpha_dice_current"] * bs
        running["n"]               += bs
 
    n       = max(running["n"], 1) 
    metrics = {k: v / n for k, v in running.items() if k != "n"}

    return metrics, itr

# ---------------------------------------------------------------------------
# Checkpointing & logging
# ---------------------------------------------------------------------------

def save_checkpoint(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, itr: int) -> None:
    
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
                {   "epoch": epoch,
                    "itr": itr,
                    "ts": time.time(),
                    "model_state": model.state_dict(),
                    "optim_state": optimizer.state_dict(),
                },path,)

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
    parser.add_argument("--data-dirs", type=Path, nargs="+", required=True, help="One or more directories containing .nii.gz mask files. " "All files in all listed directories are pooled into " "a single training set.")
    parser.add_argument("--out-dir", type=Path, required=True, help="Where checkpoints + metrics.jsonl are written.")
    parser.add_argument("--batch-size", type=int, default=CFG_DEFAULTS["batch_size"])
    parser.add_argument("--max-epochs", type=int, default=CFG_DEFAULTS["max_epochs"])
    parser.add_argument("--num-workers", type=int, default=CFG_DEFAULTS["num_workers"])
    parser.add_argument("--num-latents", type=int, default=CFG_DEFAULTS["num_latents"])
    parser.add_argument("--seed", type=int, default=CFG_DEFAULTS["seed"])
    parser.add_argument("--resume", type=Path, default=None, help="Path to a checkpoint to resume from.")
    parser.add_argument("--data-augmentation", type=str, choices=['True', 'False','true', 'false'], default='false', help="Enable or disable data augmentation (true or false).")
    args = parser.parse_args()

    # Compose final cfg
    cfg = dict(CFG_DEFAULTS)
    cfg.update({
                "batch_size":  args.batch_size,
                "max_epochs":  args.max_epochs,
                "num_workers": args.num_workers,
                "num_latents": args.num_latents,
                "seed":        args.seed,
                "augment":     args.data_augmentation.lower() == 'true'})

    args.out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
                        level=logging.INFO,
                        format="%(asctime)s %(levelname)s| %(message)s",
                        handlers=[
                            logging.FileHandler(args.out_dir / "train.log"),
                            logging.StreamHandler(),],)
    
    mlog = JsonlLogger(args.out_dir / "metrics.jsonl")
    logging.info("Config: %s", cfg)

    # Reproducibility
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Device: %s", device)

    # ---- Data ----
    train_ds     = NiftiMaskDataset(args.data_dirs, flip_prob=cfg["flip_prob"], augment=cfg["augment"],)
    train_loader = DataLoader(
                            train_ds,
                            batch_size=cfg["batch_size"],
                            shuffle=True,                 # critical: mixes clean and jittered
                            num_workers=cfg["num_workers"],
                            pin_memory=(device.type == "cuda"),
                            drop_last=True,)               # keeps BN happy with consistent batch sizes
                        
    logging.info("Train set: %d files (x2 with augmentation) -> %d samples per epoch", train_ds._n, len(train_ds))

    # ---- Model ----
    model     = VoxelVAE128(num_latents=cfg["num_latents"]).to(device)
    cbdice_fn = SoftcbDiceLoss(iter_=10, smooth=1.0).to(device)
    n_params  = sum(p.numel() for p in model.parameters())
    logging.info("Model: %s, %d params (%.2f M)", type(model).__name__, n_params, n_params / 1e6)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR_SCHEDULE[0], weight_decay=cfg["reg"],)

    # ---- Resume ----
    start_epoch = 0
    itr         = 0
    if args.resume is not None:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optim_state"])
        start_epoch = int(ckpt["epoch"]) + 1
        itr = int(ckpt.get("itr", 0))
        logging.info("Resumed from %s @ epoch %d, itr %d", args.resume, start_epoch, itr)

    # ---- Training loop ----
    current_lr = LR_SCHEDULE[0] 
    set_lr(optimizer, current_lr)
    logging.info("Initial learning rate: %g", current_lr)
    total_iterations = cfg["max_epochs"] * len(train_loader)

    for epoch in range(start_epoch, cfg["max_epochs"]):
        
        new_lr = lr_for_epoch(LR_SCHEDULE, epoch, current_lr) # LR schedule check (matches the original's per-epoch dict lookup)
        if new_lr != current_lr:
            logging.info("Changing learning rate from %g to %g", current_lr, new_lr)
            set_lr(optimizer, new_lr)
            current_lr = new_lr
        
        # ---- Dice/cbDice blend schedule (warmup pure Dice, then ramp) ----
        cfg["alpha_dice_current"] = alpha_dice_for_epoch(epoch, cfg)
        logging.info("epoch %d  alpha_dice=%.3f", epoch, cfg["alpha_dice_current"])

        t0                 = time.time()
        train_metrics, itr = train_one_epoch( model, train_loader, optimizer, device, cfg, cbdice_fn, start_itr=itr, total_iterations=total_iterations,)
        dt                 = time.time() - t0

        logging.info(
                    "Epoch %d/%d  lr=%g  β=%.3f  α_dice=%.3f  recon=%.4f  D_kl=%.4f  "
                    "Dice=%.4f  cbDice=%.4f  acc=%.4f  tp=%.4f  tn=%.4f  (%.1fs)",
                    epoch, cfg["max_epochs"] - 1, current_lr,
                    train_metrics["beta_mean"], 
                    train_metrics["alpha_dice_mean"],
                    train_metrics["vloss"],
                    train_metrics["kl"],
                    train_metrics["dice"],
                    train_metrics["cbdice"],
                    train_metrics["acc"],
                    train_metrics["tp"],
                    train_metrics["tn"], dt,)

        mlog.log(phase="train", epoch=epoch, itr=itr, lr=current_lr, dt=dt, **train_metrics) # 

    # Final checkpoint
    save_checkpoint(args.out_dir / "final.pt", model, optimizer, cfg["max_epochs"] - 1, itr)
    logging.info("Training done.")
    mlog.close()

if __name__ == "__main__":
    main()