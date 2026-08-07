from __future__ import annotations
import argparse
import json
import logging
import os
import time
from pathlib import Path
import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchsparse import SparseTensor
from torchsparse.utils.collate import sparse_collate_fn
from β_VAE_sp_clD_128 import VoxelVAE128, vae_loss
from clDice import SoftClDiceLoss, hard_cldice_batch   

# ---------------------------------------------------------------------------
# Config (mirrors the structure of the original cfg dict)
# ---------------------------------------------------------------------------

# Learning-rate schedule, keyed by epoch index. Matches the original
# `lr_schedule = {0: 0.0001, 1: 0.005}`: warmup for one epoch, then jump.
LR_SCHEDULE = {0: 1e-5, 1: 5e-4}

CFG_DEFAULTS = {
                "batch_size": 4,            # original: 64 at 32^3; we drop for 128^3
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
                "viz_threshold": 0.95,          # ADDED: prob cutoff for the saved binary recon — keep only voxels the model scores > 0.9
                "cldice_weight": 0.6,          # ADD: weight on the soft clDice loss term once active
                "cldice_warmup_epochs": 10,    # ADD: epochs of BCE-only before clDice switches on
                "cldice_iters": 8,             # ADD: thinning iterations for soft_skel (cover thickest tube radius)
                "cldice_eval_threshold": 0.5,} # ADD: prob cutoff for the HARD eval metric (0.5 == logits>=0, matches tp/tn)

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class NiftiMaskDataset(Dataset):
    """
    Loads binary 3D masks from one or more directories of .nii.gz files.
    Files are expected to be already at the target resolution (128^3) with
    binary {0, 1} values.Anything > 0.5 is treated as foreground; values
    are cast to float32.
    """

    def __init__(self, data_dirs: list[Path]):

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
                f"No .nii.gz files found in any of: {[str(d) for d in data_dirs]}")

        self._n = len(self.paths)
        logging.info("Found %d .nii.gz files across %d directories", self._n, len(data_dirs))

    def _load(self, idx: int) -> np.ndarray:
        img = nib.load(str(self.paths[idx]))
        arr = np.asarray(img.dataobj)
        if arr.ndim != 3:
            raise ValueError(f"Expected 3D array at {self.paths[idx]}, got {arr.shape}")
        return (arr > 0.5).astype(np.float32)        # (128,128,128) {0,1}
    
    def __len__(self) -> int:
        return self._n  

    def __getitem__(self, idx: int) -> dict:
        mask = self._load(idx)
        mask_t = torch.from_numpy(np.ascontiguousarray(mask))   # (128,128,128)
        coords = torch.nonzero(mask_t > 0.5).to(torch.int32)     # (N,3)
        feats  = torch.ones((coords.shape[0], 1), dtype=torch.float32)
        inp    = SparseTensor(coords=coords, feats=feats)        # batch col added in collate
        target = mask_t.unsqueeze(0).float()                     # (1,128,128,128)
        return {"input": inp, "target": target}

# ---------------------------------------------------------------------------
# LR schedule 
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
    rec_prob = torch.sigmoid(logits_cpu)[sample_idx, 0].numpy()
    rec_bin  = binary[sample_idx, 0].numpy().astype(np.uint8)
    inp_bin  = (x_bin_cpu[sample_idx, 0].numpy() >= 0.5).astype(np.uint8)

    stem = f"epoch{epoch:04d}"
    _save_nifti(inp_bin,  out_dir / f"{stem}_input.nii.gz")
    _save_nifti(rec_bin,  out_dir / f"{stem}_recon.nii.gz")
    _save_nifti(rec_prob, out_dir / f"{stem}_recon_prob.nii.gz")
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

    # running = {"vloss": 0.0, "kl": 0.0, "beta_mean": 0.0, "n": 0, "cldice_soft": 0.0, "cldice_w": 0.0, "dice": 0.0}  
    running = {"vloss": 0.0, "kl": 0.0, "beta_mean": 0.0, "n": 0, "cldice_soft": 0.0, "cldice_w": 0.0, "dice": 0.0, "cldice_hard": 0.0} # ADDED
    pooled  = {"tp_correct": 0, "pos_total": 0, "tn_correct": 0, "neg_total": 0}
    # epoch_cldice_hard = None   # ADD: filled once per epoch from the first batch

    itr = start_itr

    for batch in loader:
        itr += 1
        beta = cfg["beta"]

        x_in  = batch["input"].to(device, non_blocking=True)    # SparseTensor -> encoder
        x_bin = batch["target"].to(device, non_blocking=True)   # (B,1,128,128,128) -> loss/metrics/viz

        logits, mu, logsigma = model(x_in)
        if do_viz and viz_capture is None:
            viz_capture = (logits.detach().float().cpu(),
                           x_bin.detach().float().cpu())

        out = vae_loss(logits, x_bin, mu, logsigma, gamma=cfg["gamma"], beta=beta, use_kl=cfg["use_kl"],)

        # --- clDice topology term (soft, differentiable), warmed up -----------
        w_cl = cfg["cldice_weight"] if epoch >= cfg["cldice_warmup_epochs"] else 0.0
        loss = out.total
        cldice_soft_val = 0.0
        if cldice_fn is not None and w_cl > 0.0:
            cldice_term = cldice_fn(logits, x_bin)
            scale = out.recon.detach()                 # current BCE magnitude, constant (no grad)
            loss = loss + w_cl * scale * cldice_term    # clDice now lives on the BCE scale
            cldice_soft_val = cldice_term.item()
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
            running["dice"]        += dice_b.sum().item()
            running["cldice_hard"] += hard_cldice_batch(logits, x_bin, threshold=cfg["cldice_eval_threshold"]) * bs #ADDED

            # if epoch_cldice_hard is None:          # ADD: first batch of the epoch only
            #     epoch_cldice_hard = hard_cldice_batch(
            #         logits, x_bin, threshold=cfg["cldice_eval_threshold"])


        running["vloss"]       += out.recon.item() * bs
        running["cldice_soft"] += cldice_soft_val * bs   # ADD
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
        save_reconstruction_views(cap_logits, cap_xbin, viz_dir, epoch,threshold=cfg.get("viz_threshold", 0.5))
    # ------------------------------------------------------------------------

    n = max(running["n"], 1)
    metrics = {k: v / n for k, v in running.items() if k != "n"}
    # metrics["cldice_hard"] = epoch_cldice_hard if epoch_cldice_hard is not None else 0.0   # ADD
    metrics["tp"]  = pooled["tp_correct"] / pooled["pos_total"] if pooled["pos_total"] > 0 else 0.0
    metrics["tn"]  = pooled["tn_correct"] / pooled["neg_total"] if pooled["neg_total"] > 0 else 0.0
    metrics["acc"] = (pooled["tp_correct"] + pooled["tn_correct"]) / (pooled["pos_total"] + pooled["neg_total"]) if (pooled["pos_total"] + pooled["neg_total"]) > 0 else 0.0
    return metrics, itr

# ---------------------------------------------------------------------------
# Checkpointing & logging
# ---------------------------------------------------------------------------

def save_checkpoint(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, itr: int) -> None:

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")                    # latest.pt.tmp
    torch.save({"epoch": epoch, "itr": itr, "ts": time.time(),
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict()}, tmp)
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
    if device.type != "cuda":
        raise RuntimeError("torchsparse encoder requires CUDA; no GPU detected.")
    logging.info("Device: %s", device)

    # ---- Data ----
    train_ds = NiftiMaskDataset( args.data_dirs)
    
    train_loader = DataLoader(train_ds,
                              batch_size=cfg["batch_size"],
                              shuffle=True,
                              num_workers=cfg["num_workers"],
                              pin_memory=False,
                              drop_last=True,
                              collate_fn=sparse_collate_fn,)
    
    logging.info("Train set: %d files -> %d samples per epoch", train_ds._n, len(train_ds))

    # ---- Model ----
    model     = VoxelVAE128(num_latents=cfg["num_latents"]).to(device)
    cldice_fn = SoftClDiceLoss(iters=cfg["cldice_iters"]).to(device)   # ADD
    logging.info("clDice: weight=%.3f after %d warmup epochs, soft iters=%d, eval thr=%.2f", cfg["cldice_weight"], cfg["cldice_warmup_epochs"], cfg["cldice_iters"], cfg["cldice_eval_threshold"])     # ADD
    n_params = sum(p.numel() for p in model.parameters())
    logging.info("Model: %s, %d params (%.2f M)", type(model).__name__, n_params, n_params / 1e6)
    optimizer = torch.optim.Adam( model.parameters(), lr=LR_SCHEDULE[0], weight_decay=cfg["reg"],)

   # ---- Resume ----
    start_epoch, itr = 0, 0
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
        logging.info("Resumed from %s @ epoch %d, itr %d", resume_path, start_epoch, itr)


    # ---- Training loop ----
    current_lr = LR_SCHEDULE[max(k for k in LR_SCHEDULE if k <= start_epoch)]
    set_lr(optimizer, current_lr) # 
    logging.info("Initial learning rate: %g", current_lr)
    total_iterations = cfg["max_epochs"] * len(train_loader)

    for epoch in range(start_epoch, cfg["max_epochs"]):
        
        new_lr = lr_for_epoch(LR_SCHEDULE, epoch, current_lr) # LR schedule check (matches the original's per-epoch dict lookup)

        if new_lr != current_lr:
            logging.info("Changing learning rate from %g to %g", current_lr, new_lr)
            set_lr(optimizer, new_lr)
            current_lr = new_lr

        t0 = time.time()
        train_metrics, itr = train_one_epoch( model, train_loader, optimizer, device, cfg, start_itr=itr, total_iterations=total_iterations, epoch=epoch, viz_every=cfg["viz_every"], viz_dir=viz_dir, cldice_fn=cldice_fn,)
        dt = time.time() - t0         
        logging.info( "Epoch %d/%d  lr=%g  β=%.3f  v_loss=%.4f  D_kl=%.4f  acc=%.4f  tp=%.4f  tn=%.4f  "
                      "clD_soft=%.4f  clD_hard=%.4f  dice=%.4f  (%.1fs)",                    # ADD the two clD fields
                      epoch, cfg["max_epochs"] - 1, current_lr, train_metrics["beta_mean"],
                      train_metrics["vloss"], train_metrics["kl"],
                      train_metrics["acc"], train_metrics["tp"], train_metrics["tn"],
                      train_metrics["dice"],
                      train_metrics["cldice_soft"], train_metrics["cldice_hard"], dt,)   # ADD
             
        mlog.log(phase="train", epoch=epoch, itr=itr, lr=current_lr, dt=dt, **train_metrics)

        if (epoch + 1) % 10 == 0:                                   # your cadence
            ckpt_name = f"checkpoint_{epoch + 1}.pt"                # global epoch in the name
            save_checkpoint(args.out_dir / ckpt_name, model, optimizer, epoch, itr)
            for old in args.out_dir.glob("checkpoint_*.pt"): # keep ONLY this one: remove any earlier checkpoint_*.pt
                if old.name != ckpt_name:
                    old.unlink(missing_ok=True)
            logging.info("Checkpoint @ epoch %d -> %s (older ones removed)", epoch, ckpt_name)

    # Final checkpoint
    save_checkpoint(args.out_dir / "final.pt", model, optimizer, cfg["max_epochs"] - 1, itr)
    logging.info("Training done.")
    mlog.close()

if __name__ == "__main__":
    main()