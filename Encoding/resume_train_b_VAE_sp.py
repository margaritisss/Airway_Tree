from __future__ import annotations
"""
Resume training of the sparse β-VAE from a checkpoint.

This is a thin wrapper around the original training script. It imports the
EXACT same model, dataset, loss, training loop, metrics and visualisation code
from ``train_β_VAE_with_rec_sp.py`` (so nothing about how a step is computed can
drift), and only re-implements ``main()`` to:

  1. load a checkpoint (model + optimizer state, epoch, itr),
  2. continue the loop from the next epoch, and
  3. write NEW output files into the SAME run directory, with a numeric suffix
     so your originals are never touched:

         run_<JOBID>/metrics.jsonl        <- original (run 0)
         run_<JOBID>/train.log            <- original (run 0)
         run_<JOBID>/metrics_1.jsonl      <- first  resume
         run_<JOBID>/train_1.log          <- first  resume
         run_<JOBID>/metrics_2.jsonl      <- second resume
         run_<JOBID>/train_2.log          <- second resume
         ...

The suffix is chosen automatically: it scans the output directory for existing
``metrics_<k>.jsonl`` files and uses (max k) + 1, so each new run increments.
You can force a specific number with ``--run-suffix``.

This script keeps exactly ONE checkpoint, overwritten every 100 epochs by
default (configurable with --checkpoint-every; the final epoch is always saved
too). It is written atomically, so a kill mid-write can't corrupt it:

    run_<JOBID>/checkpoint.pt        (single rolling checkpoint, default name)

You resume from it again next time by pointing --resume at this same file.
The metrics / log files are still suffixed per run (so your history is kept),
but the checkpoint is a single updated file as requested.

Reconstruction previews (if --viz-every > 0) go into a suffixed subdir
``reconstructions_<suffix>/`` so resumed runs don't collide on epoch filenames.

USAGE (typical)
---------------
    python -u resume_train_b_VAE_sp.py \
        --resume     /.../run_876062/final.pt \
        --out-dir    /.../run_876062 \
        --data-dirs  /.../AIIB23_128 /.../ATM22_128 \
        --batch-size 6 --num-workers 7 --num-latents 32 \
        --beta 16.0 --data-augmentation False \
        --max-epochs 2000

NOTE: if you resume from a *finished* run (e.g. final.pt at epoch 999) you MUST
raise --max-epochs beyond the saved epoch, otherwise there are no epochs left to
run and the script exits with a clear message.

IMPORTANT: --num-latents (and the data dirs / model-affecting args) MUST match
the run that produced the checkpoint, or the weights won't load.
"""

import argparse
import importlib
import logging
import os
import re
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchsparse.utils.collate import sparse_collate_fn

# Import the original training module *by its real (Greek-β) name*. We go
# through importlib because the module name contains a non-ASCII character.
# Importing it only defines functions/constants (its own main() is guarded by
# `if __name__ == "__main__"`), so nothing runs at import time.
train = importlib.import_module("train_β_VAE_with_rec_sp")

VoxelVAE128       = train.VoxelVAE128
NiftiMaskDataset  = train.NiftiMaskDataset
train_one_epoch   = train.train_one_epoch
save_checkpoint   = train.save_checkpoint
JsonlLogger       = train.JsonlLogger
set_lr            = train.set_lr
lr_for_epoch      = train.lr_for_epoch
LR_SCHEDULE       = train.LR_SCHEDULE
CFG_DEFAULTS      = train.CFG_DEFAULTS


# ---------------------------------------------------------------------------
# Helpers specific to resuming
# ---------------------------------------------------------------------------

def next_run_suffix(out_dir: Path) -> int:
    """Return (max existing metrics_<k>.jsonl suffix) + 1, or 1 if none exist.

    The original (un-suffixed) ``metrics.jsonl`` counts as run 0, so the first
    resume gets suffix 1, the second gets 2, and so on.
    """
    pat   = re.compile(r"^metrics_(\d+)\.jsonl$")
    found = []
    for p in out_dir.glob("metrics_*.jsonl"): 
        m = pat.match(p.name)
        if m:
            found.append(int(m.group(1)))
    return (max(found) + 1) if found else 1


def effective_lr_at(schedule: dict, epoch: int, fallback: float) -> float:
    """LR dictated by the schedule for a given epoch.

    The schedule is keyed by absolute epoch (e.g. {0: 1e-5, 1: 5e-4}: a one-epoch
    warmup, then a jump). When we resume mid-training, the resume epoch is not a
    key in the dict, so we take the value of the most recent key <= epoch. For
    {0: 1e-5, 1: 5e-4} that means any resume at epoch >= 1 correctly continues at
    5e-4 instead of snapping back to the 1e-5 warmup value.
    """
    keys = sorted(k for k in schedule if k <= epoch)
    return float(schedule[keys[-1]]) if keys else float(fallback)


def save_checkpoint_atomic(path: Path, model, optimizer, epoch: int, itr: int) -> None:
    """Overwrite the single rolling checkpoint safely.

    We write to a temporary file in the same directory and then os.replace() it
    onto the target name. os.replace is atomic on the same filesystem, so if the
    job is killed mid-write the existing checkpoint.pt is never left corrupted.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    save_checkpoint(tmp, model, optimizer, epoch, itr)
    os.replace(tmp, path)


def setup_logging(log_path: Path) -> None:
    """Send logging to a fresh suffixed file + stdout. We configure the root
    logger directly (the imported module only calls basicConfig inside its own
    main(), which we never invoke), clearing any pre-existing handlers first."""
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for h in list(root.handlers):
        root.removeHandler(h)
    fmt = logging.Formatter("%(asctime)s %(levelname)s| %(message)s")
    fh = logging.FileHandler(log_path, mode="w")   # fresh file, never appends
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    root.addHandler(fh)
    root.addHandler(sh)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    # --- required: where to resume from and where to write ---
    parser.add_argument("--resume", type=Path, required=True,
                        help="Path to the checkpoint .pt to resume from.")
    parser.add_argument("--out-dir", type=Path, required=True,
                        help="Run directory (the SAME folder that holds the "
                             "original metrics.jsonl / train.log / checkpoint).")
    parser.add_argument("--data-dirs", type=Path, nargs="+", required=True,
                        help="One or more directories of .nii.gz mask files "
                             "(must match the original run).")

    # --- training args (mirror the original; must be consistent with ckpt) ---
    parser.add_argument("--batch-size",  type=int, default=CFG_DEFAULTS["batch_size"])
    parser.add_argument("--max-epochs",  type=int, default=CFG_DEFAULTS["max_epochs"],
                        help="Train UP TO this absolute epoch. Must be greater "
                             "than the checkpoint's epoch, or there's nothing to do.")
    parser.add_argument("--num-workers", type=int, default=CFG_DEFAULTS["num_workers"])
    parser.add_argument("--num-latents", type=int, default=CFG_DEFAULTS["num_latents"],
                        help="MUST match the checkpoint, or weights won't load.")
    parser.add_argument("--seed",        type=int, default=CFG_DEFAULTS["seed"])
    parser.add_argument("--data-augmentation", type=str,
                        choices=['True', 'False', 'true', 'false'], default='false',
                        help="Enable or disable data augmentation (true or false).")
    parser.add_argument("--beta", type=float, default=CFG_DEFAULTS["beta"],
                        help="Static beta value for the beta-VAE formulation.")
    parser.add_argument("--viz-every", type=int, default=CFG_DEFAULTS["viz_every"],
                        help="Save a reconstruction preview every N epochs (0 disables).")

    # --- resume-specific knobs ---
    parser.add_argument("--run-suffix", type=int, default=None,
                        help="Force the output-file suffix (default: auto-increment).")
    parser.add_argument("--lr", type=float, default=None,
                        help="Override the learning rate. Default: the value the "
                             "LR schedule dictates for the resume epoch.")
    parser.add_argument("--checkpoint-name", type=str, default="checkpoint.pt",
                        help="Filename for the single rolling checkpoint, written "
                             "into --out-dir and overwritten on each save.")
    parser.add_argument("--checkpoint-every", type=int, default=100,
                        help="Overwrite the single checkpoint every N epochs "
                             "(the final epoch is always saved too).")

    args = parser.parse_args()

    # ---- compose cfg exactly like the original ----
    cfg = dict(CFG_DEFAULTS)
    cfg.update({"batch_size":  args.batch_size,
                "max_epochs":  args.max_epochs,
                "num_workers": args.num_workers,
                "num_latents": args.num_latents,
                "seed":        args.seed,
                "augment":     args.data_augmentation.lower() == 'true',
                "beta":        args.beta,
                "viz_every":   args.viz_every})

    out_dir = args.out_dir
    if not out_dir.is_dir():
        raise FileNotFoundError(
            f"--out-dir does not exist: {out_dir}. Point it at the existing run "
            f"folder that holds your checkpoint and original metrics.jsonl.")
    if not args.resume.is_file():
        raise FileNotFoundError(f"--resume checkpoint not found: {args.resume}")

    # ---- pick the suffix for this run's output files ----
    suffix = args.run_suffix if args.run_suffix is not None else next_run_suffix(out_dir)

    metrics_path = out_dir / f"metrics_{suffix}.jsonl"
    log_path     = out_dir / f"train_{suffix}.log"
    viz_dir      = out_dir / f"reconstructions_{suffix}"
    ckpt_path    = out_dir / args.checkpoint_name   # single rolling checkpoint

    setup_logging(log_path)
    mlog = JsonlLogger(metrics_path)

    logging.info("=== RESUME run (suffix=%d) ===", suffix)
    logging.info("Resuming from checkpoint: %s", args.resume)
    logging.info("Writing metrics -> %s", metrics_path)
    logging.info("Writing log     -> %s", log_path)
    logging.info("Checkpoint (single file) -> %s  (saved every %d epochs)",
                 ckpt_path, args.checkpoint_every)
    logging.info("Config: %s", cfg)

    # ---- reproducibility (offset by suffix so each resume reshuffles differently) ----
    torch.manual_seed(cfg["seed"] + suffix)
    np.random.seed(cfg["seed"] + suffix)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("torchsparse encoder requires CUDA; no GPU detected.")
    logging.info("Device: %s", device)

    # ---- data (identical to the original) ----
    train_ds = NiftiMaskDataset(args.data_dirs,
                                flip_prob=cfg["flip_prob"],
                                augment=cfg["augment"])
    train_loader = DataLoader(train_ds,
                              batch_size=cfg["batch_size"],
                              shuffle=True,
                              num_workers=cfg["num_workers"],
                              pin_memory=False,
                              drop_last=True,
                              collate_fn=sparse_collate_fn)
    logging.info("Train set: %d files (x2 with augmentation) -> %d samples per epoch",
                 train_ds._n, len(train_ds))

    # ---- model + optimizer (identical construction to the original) ----
    model = VoxelVAE128(num_latents=cfg["num_latents"]).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logging.info("Model: %s, %d params (%.2f M)", type(model).__name__, n_params, n_params / 1e6)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR_SCHEDULE[0], weight_decay=cfg["reg"])

    # ---- load the checkpoint ----
    ckpt = torch.load(args.resume, map_location=device)
    try:
        model.load_state_dict(ckpt["model_state"])
    except RuntimeError as e:
        raise RuntimeError(
            "Failed to load model weights. The most common cause is a mismatch "
            "between --num-latents (or other architecture args) and the run that "
            f"produced this checkpoint.\nOriginal error:\n{e}")
    optimizer.load_state_dict(ckpt["optim_state"])

    start_epoch = int(ckpt["epoch"]) + 1
    itr         = int(ckpt.get("itr", 0))
    logging.info("Loaded checkpoint @ saved epoch %d, itr %d -> continuing from epoch %d",
                 int(ckpt["epoch"]), itr, start_epoch)

    if start_epoch >= cfg["max_epochs"]:
        logging.error(
            "Nothing to do: checkpoint is at epoch %d, so training would resume at "
            "epoch %d, but --max-epochs is %d. Raise --max-epochs above %d to "
            "continue training.", int(ckpt["epoch"]), start_epoch,
            cfg["max_epochs"], start_epoch)
        mlog.close()
        return

    # ---- learning rate: continue at the schedule-correct value ----
    # (The original resume path snaps the LR back to the 1e-5 warmup; here we
    #  resume at the schedule value for `start_epoch`, or your --lr override.)
    current_lr = args.lr if args.lr is not None else effective_lr_at(
        LR_SCHEDULE, start_epoch, LR_SCHEDULE[0])
    set_lr(optimizer, current_lr)
    logging.info("Resuming learning rate at: %g%s", current_lr,
                 " (user override)" if args.lr is not None else " (from schedule)")

    total_iterations = cfg["max_epochs"] * len(train_loader)

    # ---- training loop (same body as the original) ----
    for epoch in range(start_epoch, cfg["max_epochs"]):

        new_lr = lr_for_epoch(LR_SCHEDULE, epoch, current_lr)
        if new_lr != current_lr:
            logging.info("Changing learning rate from %g to %g", current_lr, new_lr)
            set_lr(optimizer, new_lr)
            current_lr = new_lr

        t0 = time.time()
        train_metrics, itr = train_one_epoch(
            model, train_loader, optimizer, device, cfg,
            start_itr=itr, total_iterations=total_iterations,
            epoch=epoch, viz_every=cfg["viz_every"], viz_dir=viz_dir)
        dt = time.time() - t0

        logging.info(
            "Epoch %d/%d  lr=%g  β=%.3f  v_loss=%.4f  D_kl=%.4f  acc=%.4f  tp=%.4f  tn=%.4f  (%.1fs)",
            epoch, cfg["max_epochs"] - 1, current_lr, train_metrics["beta_mean"],
            train_metrics["vloss"], train_metrics["kl"],
            train_metrics["acc"], train_metrics["tp"], train_metrics["tn"], dt)

        mlog.log(phase="train", epoch=epoch, itr=itr, lr=current_lr, dt=dt, **train_metrics)

        # Single rolling checkpoint: overwrite the same file every N epochs
        # (and always on the very last epoch so the finished model is saved).
        is_last = (epoch == cfg["max_epochs"] - 1)
        if (args.checkpoint_every > 0 and epoch % args.checkpoint_every == 0) or is_last:
            save_checkpoint_atomic(ckpt_path, model, optimizer, epoch, itr)
            logging.info("Checkpoint updated @ epoch %d -> %s", epoch, ckpt_path)

    logging.info("Training done. Checkpoint: %s", ckpt_path)
    mlog.close()


if __name__ == "__main__":
    main()
