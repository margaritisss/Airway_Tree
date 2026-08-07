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
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# --- ADDED: headless plotting backend ---------------------------------------
# We render reconstruction previews to PNG files on the training machine,
# which usually has no display. Selecting the "Agg" backend BEFORE importing
# pyplot makes matplotlib write to files instead of trying to open a window.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
# ----------------------------------------------------------------------------

# Model + loss live in VAE.py next to this file
from β_VAE import VoxelVAE128, vae_loss
# ---------------------------------------------------------------------------
# Config (mirrors the structure of the original cfg dict)
# ---------------------------------------------------------------------------

# Learning-rate schedule, keyed by epoch index. Matches the original
# `lr_schedule = {0: 0.0001, 1: 0.005}`: warmup for one epoch, then jump.
LR_SCHEDULE  = {0: 1e-5, 1: 5e-4}
CFG_DEFAULTS = {
                "batch_size": 4,            # original: 64 at 32^3; we drop for 128^3
                "max_epochs": 150,          # original: cfg['max_epochs'] = 150
                "reg": 2e-3,                # original: cfg['reg'] = 0.001 (L2 weight decay)
                "flip_prob": 0.2,           # original jitter_chunk used binomial(1, 0.2)
                "checkpoint_every_nth": 5,  # original: cfg['checkpoint_every_nth'] = 5
                "num_latents": 100,         # our scale-up choice (paper used 100)
                "gamma": 0.99,              # weighted-BCE positive weight (released code)
                "use_kl": True,             # paper text says KL is part of the loss
                "num_workers": 4,
                "seed": 0,
                "beta": 4.0,   # β-VAE constant; paper uses β > 1 (e.g. 4–250 depending on dataset)
                "viz_every": 10,  # ADDED: dump a reconstruction preview every N epochs (0 disables)
                "viz_threshold": 0.9,  # ADDED: prob cutoff for the saved binary recon — keep only voxels the model scores > 0.9
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

        correct = (pred_pos == true_pos)
        acc     = correct.float().mean().item()

        n_pos = true_pos.float().mean().item()
        n_neg = true_neg.float().mean().item()
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
#
#   * MIP PNG  -> the quick-look. A montage of raw 2D slices is a poor monitor
#                 for sparse tubular structures: most slices are nearly empty,
#                 so you see a few dots and learn nothing. A Maximum Intensity
#                 Projection collapses the whole volume onto a plane while
#                 preserving the branching pattern, so a single image shows
#                 whether the tree topology is being recovered. We project along
#                 all three axes for input vs. recon side by side. This is the
#                 image to glance at in your log directory between checkpoints.
#
#   * mesh HTML-> optional, interactive surface render via marching cubes +
#                 plotly. Headless-friendly (writes a standalone .html) and
#                 import-guarded so training never breaks if the libs are absent.


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


def _save_mip_triptych(input_bin: np.ndarray, recon_bin: np.ndarray, path: Path, title: str = "") -> None:
    """2x3 grid of Maximum Intensity Projections: input (top) vs recon (bottom),
    projected along each of the three spatial axes. Orientation labels are
    nominal (the masks are not in a fixed anatomical frame here)."""
    axis_names = ["proj. axis 0", "proj. axis 1", "proj. axis 2"]
    fig, axs = plt.subplots(2, 3, figsize=(9, 6))
    for col, axis in enumerate((0, 1, 2)):
        in_mip  = input_bin.max(axis=axis)
        rec_mip = recon_bin.max(axis=axis)
        axs[0, col].imshow(in_mip.T,  origin="lower", cmap="gray")
        axs[0, col].set_title(f"input  {axis_names[col]}")
        axs[1, col].imshow(rec_mip.T, origin="lower", cmap="gray")
        axs[1, col].set_title(f"recon  {axis_names[col]}")
        axs[0, col].axis("off")
        axs[1, col].axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _save_mesh_html(recon_bin: np.ndarray, path: Path) -> None:
    """Optional interactive 3D surface (marching cubes -> plotly HTML).
    Import-guarded: if scikit-image / plotly aren't installed, we skip it
    rather than crash the training run."""
    try:
        from skimage import measure
        import plotly.graph_objects as go
    except Exception as e:  # pragma: no cover - optional dependency path
        logging.warning("Mesh export skipped (need scikit-image + plotly): %s", e)
        return
    if recon_bin.sum() == 0:
        logging.warning("Mesh export skipped: reconstruction is empty at this epoch.")
        return
    verts, faces, _, _ = measure.marching_cubes(recon_bin.astype(np.float32), level=0.5)
    fig = go.Figure(data=[go.Mesh3d(
        x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        color="lightpink", opacity=0.5,
    )])
    fig.write_html(str(path))


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
    rec_prob = torch.sigmoid(logits_cpu)[sample_idx, 0].numpy()
    rec_bin  = binary[sample_idx, 0].numpy().astype(np.uint8)
    inp_bin  = (x_bin_cpu[sample_idx, 0].numpy() >= 0.5).astype(np.uint8)

    stem = f"epoch{epoch:04d}"
    _save_nifti(inp_bin,  out_dir / f"{stem}_input.nii.gz")
    _save_nifti(rec_bin,  out_dir / f"{stem}_recon.nii.gz")
    _save_nifti(rec_prob, out_dir / f"{stem}_recon_prob.nii.gz")
    _save_mip_triptych(
        inp_bin, rec_bin, out_dir / f"{stem}_mip.png",
        title=f"epoch {epoch}  |  foreground voxels: input={int(inp_bin.sum())}, "
              f"recon={int(rec_bin.sum())}",
    )
    _save_mesh_html(rec_bin, out_dir / f"{stem}_recon_mesh.html")
    logging.info("Saved reconstruction previews for epoch %d -> %s", epoch, out_dir)


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    cfg: dict,
    start_itr: int,           # global iteration count entering this epoch
    total_iterations: int,    # T for the β schedule
    epoch: int = 0,                    # ADDED: current epoch, used for viz gating + filenames
    viz_every: int = 0,                # ADDED: capture a preview every N epochs (0 disables)
    viz_dir: "Path | None" = None,     # ADDED: where previews are written
) -> tuple[dict, int]:
    model.train()

    # ADDED: decide once whether this epoch produces a preview, and prepare a
    # slot to stash the captured batch. We capture only the FIRST batch of the
    # epoch to keep overhead negligible.
    do_viz = (viz_every > 0) and (viz_dir is not None) and (epoch % viz_every == 0)
    viz_capture = None

    running = {"vloss": 0.0, "kl": 0.0, "beta_mean": 0.0, "n": 0}
    pooled  = {"tp_correct": 0, "pos_total": 0, "tn_correct": 0, "neg_total": 0}

    itr = start_itr

    for x_bin in loader:
        itr += 1
        beta = cfg["beta"]

        x_bin = x_bin.to(device, non_blocking=True)
        x_in  = 3.0 * x_bin - 1.0

        logits, mu, logsigma = model(x_in)

        # --- ADDED: reconstruction capture ----------------------------------
        # Grab the decoder output for this batch HERE: immediately after the
        # forward pass and before the loss, backward, or optimizer step. This
        # is exactly the volume the model produced this step. We capture only
        # the first batch of a viz-epoch, and `.detach()` + move to CPU right
        # away so it carries no grad and frees GPU memory immediately.
        #
        # This runs in train() mode, so `z` is sampled (reparameterize) and
        # BatchNorm uses batch statistics — i.e. it reflects the true
        # stochastic training forward, as requested. If you'd rather preview a
        # deterministic reconstruction, run a separate model.eval() pass that
        # decodes `mu` instead of a sampled `z`; see the note in main().
        if do_viz and viz_capture is None:
            viz_capture = (logits.detach().float().cpu(),
                           x_bin.detach().float().cpu())
        # --------------------------------------------------------------------

        out                  = vae_loss(logits, x_bin, mu, logsigma, gamma=cfg["gamma"], beta=beta, use_kl=cfg["use_kl"],)

        optimizer.zero_grad(set_to_none=True)
        out.total.backward()
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

        running["vloss"]     += out.recon.item() * bs
        running["kl"]        += out.kl.item()    * bs
        # running["acc"]       += acc * bs
        running["beta_mean"] += beta * bs
        running["n"]         += bs

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
    metrics = {k: v / n for k, v in running.items() if k != "n"}
    metrics["tp"] = pooled["tp_correct"] / pooled["pos_total"] if pooled["pos_total"] > 0 else 0.0
    metrics["tn"] = pooled["tn_correct"] / pooled["neg_total"] if pooled["neg_total"] > 0 else 0.0
    metrics["acc"] = (pooled["tp_correct"] + pooled["tn_correct"]) / (pooled["pos_total"] + pooled["neg_total"]) if (pooled["pos_total"] + pooled["neg_total"]) > 0 else 0.0
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
            "optim_state": optimizer.state_dict(),},
        path,)

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
    parser.add_argument("--data-augmentation", type=str, choices=['True', 'False','true', 'false'], default='false', help="Enable or disable data augmentation (true or false).")
    parser.add_argument("--beta", type=float, default=CFG_DEFAULTS["beta"], help="Static beta value for beta-VAE formulation.")
    parser.add_argument("--viz-every", type=int, default=CFG_DEFAULTS["viz_every"],
                        help="ADDED: save a reconstruction preview every N epochs (0 disables).")
    args = parser.parse_args()

    # Compose final cfg
    cfg = dict(CFG_DEFAULTS)
    cfg.update({"batch_size":  args.batch_size,
                "max_epochs":  args.max_epochs,
                "num_workers": args.num_workers,
                "num_latents": args.num_latents,
                "seed":        args.seed,
                "augment":     args.data_augmentation.lower() == 'true',
                "beta":        args.beta,
                "viz_every":   args.viz_every})  # ADDED

    args.out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s| %(message)s", handlers=[logging.FileHandler(args.out_dir / "train.log"),logging.StreamHandler(),],)
    
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
    logging.info("Device: %s", device)

    # ---- Data ----
    train_ds = NiftiMaskDataset( args.data_dirs,
                                flip_prob=cfg["flip_prob"],
                                augment=cfg["augment"],)
    
    train_loader = DataLoader( train_ds,
                                batch_size=cfg["batch_size"],
                                shuffle=True,                 # critical: mixes clean and jittered
                                num_workers=cfg["num_workers"],
                                pin_memory=(device.type == "cuda"),
                                drop_last=True,)               # keeps BN happy with consistent batch sizes
    
    logging.info("Train set: %d files (x2 with augmentation) -> %d samples per epoch", train_ds._n, len(train_ds))

    # ---- Model ----
    model    = VoxelVAE128(num_latents=cfg["num_latents"]).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logging.info("Model: %s, %d params (%.2f M)", type(model).__name__, n_params, n_params / 1e6)
    optimizer = torch.optim.Adam( model.parameters(),
                                  lr=LR_SCHEDULE[0],
                                  weight_decay=cfg["reg"],)

    # ---- Resume ----
    start_epoch = 0
    itr         = 0
    if args.resume is not None:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optim_state"])
        start_epoch = int(ckpt["epoch"]) + 1
        itr = int(ckpt.get("itr", 0))
        logging.info("Resumed from %s @ epoch %d, itr %d",
                     args.resume, start_epoch, itr)

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

        t0 = time.time()
        train_metrics, itr = train_one_epoch( model, train_loader, optimizer, device, cfg, start_itr=itr, total_iterations=total_iterations, epoch=epoch, viz_every=cfg["viz_every"], viz_dir=viz_dir,)
        dt = time.time() - t0

        logging.info( "Epoch %d/%d  lr=%g  β=%.3f  v_loss=%.4f  D_kl=%.4f  acc=%.4f  tp=%.4f  tn=%.4f  (%.1fs)",
                     epoch, cfg["max_epochs"] - 1, current_lr, train_metrics["beta_mean"],
                     train_metrics["vloss"], train_metrics["kl"],
                     train_metrics["acc"], train_metrics["tp"], train_metrics["tn"], dt,)
        
        mlog.log(phase="train", epoch=epoch, itr=itr, lr=current_lr, dt=dt, **train_metrics)

    # Final checkpoint
    save_checkpoint(args.out_dir / "final.pt", model, optimizer, cfg["max_epochs"] - 1, itr)
    logging.info("Training done.")
    mlog.close()

if __name__ == "__main__":
    main()