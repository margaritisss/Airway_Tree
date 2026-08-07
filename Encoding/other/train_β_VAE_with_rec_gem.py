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
import matplotlib.pyplot as plt  # <--- ADDED for rapid 2D slice visualization

# Model + loss live in VAE.py next to this file
from β_VAE import VoxelVAE128, vae_loss

# ---------------------------------------------------------------------------
# Config (mirrors the structure of the original cfg dict)
# ---------------------------------------------------------------------------
LR_SCHEDULE = {0: 1e-5, 1: 5e-4}

CFG_DEFAULTS = {
                "batch_size": 4,
                "max_epochs": 150,
                "reg": 2e-3,
                "flip_prob": 0.2,
                "checkpoint_every_nth": 5,
                "num_latents": 100,
                "gamma": 0.99,
                "use_kl": True,
                "num_workers": 4,
                "seed": 0,
                "beta": 4.0,
}

# ---------------------------------------------------------------------------
# Dataset (Unchanged)
# ---------------------------------------------------------------------------
class NiftiMaskDataset(Dataset):
    # ... [Keep your exact Dataset implementation here] ...
    def __init__(self, data_dirs: list[Path], flip_prob: float = 0.2, augment: bool = True):
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
            raise ValueError(f"No .nii.gz files found in any of: {[str(d) for d in data_dirs]}")
        self._n = len(self.paths)
        logging.info("Found %d .nii.gz files across %d directories", self._n, len(data_dirs))

    def __len__(self) -> int:
        return 2 * self._n if self.augment else self._n

    def _load(self, idx: int) -> np.ndarray:
        img = nib.load(str(self.paths[idx]))
        arr = np.asarray(img.dataobj)
        if arr.ndim != 3:
            raise ValueError(f"Expected 3D array at {self.paths[idx]}, got shape {arr.shape}")
        arr = (arr > 0.5).astype(np.float32)
        return arr[None, ...]

    def _jitter(self, x: np.ndarray) -> np.ndarray:
        dst = x.copy()
        if np.random.binomial(1, self.flip_prob):
            dst = dst[:, ::-1, :, :]
        if np.random.binomial(1, self.flip_prob):
            dst = dst[:, :, ::-1, :]
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
        return torch.from_numpy(x)

# ---------------------------------------------------------------------------
# LR schedule & Metrics (Unchanged)
# ---------------------------------------------------------------------------
def set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for g in optimizer.param_groups:
        g["lr"] = lr

def lr_for_epoch(schedule: dict, epoch: int, current_lr: float) -> float:
    if epoch in schedule:
        return float(schedule[epoch])
    return current_lr

def reconstruction_accuracy(logits: torch.Tensor, target_binary: torch.Tensor) -> tuple[float, float, float]:
    with torch.no_grad():
        pred_pos = logits >= 0
        true_pos = target_binary >= 0.5
        true_neg = ~true_pos
        correct = (pred_pos == true_pos)
        acc     = correct.float().mean().item()
        n_pos = true_pos.float().mean().item()
        n_neg = true_neg.float().mean().item()
        tp = (correct & true_pos).float().mean().item() / n_pos if n_pos > 0 else 0.0
        tn = (correct & true_neg).float().mean().item() / n_neg if n_neg > 0 else 0.0
    return acc, tp, tn

# ---------------------------------------------------------------------------
# Visualization Helper
# ---------------------------------------------------------------------------
def _save_visualizations(logits: torch.Tensor, target: torch.Tensor, epoch: int, out_dir: Path, prefix: str = "fixed_tree"):
    """
    Saves a 3D NIfTI and a 2D mid-slice montage for the fixed validation sample.
    """
    vis_dir = out_dir / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)

    # Convert logits to binary mask (logits >= 0 is equivalent to sigmoid >= 0.5)
    pred_mask = (logits[0, 0].detach() >= 0).cpu().numpy().astype(np.uint8)
    target_mask = target[0, 0].cpu().numpy().astype(np.uint8)

    # Save 3D NIfTI
    nib.save(nib.Nifti1Image(pred_mask, np.eye(4)), vis_dir / f"{prefix}_epoch_{epoch:03d}_pred.nii.gz")
    
    # Save target NIfTI (redundant after epoch 0, but useful for side-by-side comparison)
    if epoch == 0:
        nib.save(nib.Nifti1Image(target_mask, np.eye(4)), vis_dir / f"{prefix}_target.nii.gz")

    # Save 2D Mid-slice montage
    fig, axes = plt.subplots(2, 3, figsize=(10, 6))
    fig.suptitle(f"Epoch {epoch} - Fixed Airway Tree Reconstruction", fontsize=14)

    d, h, w = target_mask.shape
    md, mh, mw = d//2, h//2, w//2

    # Target
    axes[0, 0].imshow(target_mask[md, :, :], cmap='gray')
    axes[0, 0].set_title("Target - Axial")
    axes[0, 1].imshow(target_mask[:, mh, :], cmap='gray')
    axes[0, 1].set_title("Target - Coronal")
    axes[0, 2].imshow(target_mask[:, :, mw], cmap='gray')
    axes[0, 2].set_title("Target - Sagittal")

    # Prediction
    axes[1, 0].imshow(pred_mask[md, :, :], cmap='gray')
    axes[1, 0].set_title("Prediction - Axial")
    axes[1, 1].imshow(pred_mask[:, mh, :], cmap='gray')
    axes[1, 1].set_title("Prediction - Coronal")
    axes[1, 2].imshow(pred_mask[:, :, mw], cmap='gray')
    axes[1, 2].set_title("Prediction - Sagittal")

    for ax in axes.flatten():
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(vis_dir / f"{prefix}_epoch_{epoch:03d}_slices.png", bbox_inches='tight', dpi=150)
    plt.close()

# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    cfg: dict,
    start_itr: int,           
    total_iterations: int,    
) -> tuple[dict, int]:
    model.train()

    running = {"vloss": 0.0, "kl": 0.0, "beta_mean": 0.0, "n": 0}
    pooled  = {"tp_correct": 0, "pos_total": 0, "tn_correct": 0, "neg_total": 0}

    itr = start_itr

    # Reverted back to a standard loop without 'enumerate'
    for x_bin in loader: 
        itr += 1
        beta = cfg["beta"]

        x_bin = x_bin.to(device, non_blocking=True)
        x_in  = 3.0 * x_bin - 1.0

        logits, mu, logsigma = model(x_in)
            
        out = vae_loss(logits, x_bin, mu, logsigma, gamma=cfg["gamma"], beta=beta, use_kl=cfg["use_kl"])

        optimizer.zero_grad(set_to_none=True)
        out.total.backward()
        optimizer.step()

        bs = x_bin.shape[0]

        with torch.no_grad():
            pred_pos = logits >= 0  
            true_pos = x_bin >= 0.5 
            true_neg = ~true_pos    
            correct  = (pred_pos == true_pos)
            pooled["tp_correct"] += (correct & true_pos).sum().item()
            pooled["pos_total"]  += true_pos.sum().item()
            pooled["tn_correct"] += (correct & true_neg).sum().item()
            pooled["neg_total"]  += true_neg.sum().item()

        running["vloss"]     += out.recon.item() * bs
        running["kl"]        += out.kl.item()    * bs
        running["beta_mean"] += beta * bs
        running["n"]         += bs

    n = max(running["n"], 1)
    metrics = {k: v / n for k, v in running.items() if k != "n"}
    metrics["tp"] = pooled["tp_correct"] / pooled["pos_total"] if pooled["pos_total"] > 0 else 0.0
    metrics["tn"] = pooled["tn_correct"] / pooled["neg_total"] if pooled["neg_total"] > 0 else 0.0
    metrics["acc"] = (pooled["tp_correct"] + pooled["tn_correct"]) / (pooled["pos_total"] + pooled["neg_total"]) if (pooled["pos_total"] + pooled["neg_total"]) > 0 else 0.0
    return metrics, itr




# ---------------------------------------------------------------------------
# Checkpointing, logging, and Main logic below...
# ---------------------------------------------------------------------------
# ... [Keep your JsonlLogger and save_checkpoint identical] ...

class JsonlLogger:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.f = open(path, "w", buffering=1)
    def log(self, **kv) -> None:
        self.f.write(json.dumps(kv) + "\n")
    def close(self) -> None:
        self.f.close()

def save_checkpoint(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, itr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"epoch": epoch, "itr": itr, "ts": time.time(), "model_state": model.state_dict(), "optim_state": optimizer.state_dict()}, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=CFG_DEFAULTS["batch_size"])
    parser.add_argument("--max-epochs", type=int, default=CFG_DEFAULTS["max_epochs"])
    parser.add_argument("--num-workers", type=int, default=CFG_DEFAULTS["num_workers"])
    parser.add_argument("--num-latents", type=int, default=CFG_DEFAULTS["num_latents"])
    parser.add_argument("--seed", type=int, default=CFG_DEFAULTS["seed"])
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--data-augmentation", type=str, choices=['True', 'False','true', 'false'], default='false')
    parser.add_argument("--beta", type=float, default=CFG_DEFAULTS["beta"])
    args = parser.parse_args()

    cfg = dict(CFG_DEFAULTS)
    cfg.update({"batch_size":  args.batch_size, "max_epochs":  args.max_epochs, "num_workers": args.num_workers, "num_latents": args.num_latents, "seed": args.seed, "augment": args.data_augmentation.lower() == 'true', "beta": args.beta})

    args.out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s| %(message)s", handlers=[logging.FileHandler(args.out_dir / "train.log"),logging.StreamHandler(),])
    mlog = JsonlLogger(args.out_dir / "metrics.jsonl")

    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = NiftiMaskDataset(args.data_dirs, flip_prob=cfg["flip_prob"], augment=cfg["augment"])
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=cfg["num_workers"], pin_memory=(device.type == "cuda"), drop_last=True)
    
    # ---- Model ----
    model = VoxelVAE128(num_latents=cfg["num_latents"]).to(device)
    optimizer = torch.optim.Adam( model.parameters(), lr=LR_SCHEDULE[0], weight_decay=cfg["reg"])

    start_epoch, itr = 0, 0
    if args.resume is not None:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optim_state"])
        start_epoch = int(ckpt["epoch"]) + 1
        itr = int(ckpt.get("itr", 0))

    current_lr = LR_SCHEDULE[0]
    set_lr(optimizer, current_lr)
    total_iterations = cfg["max_epochs"] * len(train_loader)

    # ---------------------------------------------------------
    # NEW: Extract Fixed Validation Sample
    # train_ds[0] returns (1, D, H, W). unsqueeze(0) makes it (1, 1, D, H, W)
    # ---------------------------------------------------------
    fixed_val_sample = train_ds[0].unsqueeze(0).to(device)
    logging.info("Extracted fixed airway tree sample for consistent visualization.")

    for epoch in range(start_epoch, cfg["max_epochs"]):
        
        new_lr = lr_for_epoch(LR_SCHEDULE, epoch, current_lr)

        if new_lr != current_lr:
            logging.info("Changing learning rate from %g to %g", current_lr, new_lr)
            set_lr(optimizer, new_lr)
            current_lr = new_lr

        # ---------------------------------------------------------
        # NEW: Evaluate and visualize fixed sample every 10 epochs
        # ---------------------------------------------------------
        if epoch % 10 == 0:
            model.eval()  # Freeze BatchNorm moving averages
            with torch.no_grad(): # Save memory/compute by turning off autograd
                # Apply the exact same scaling as the training loop
                fixed_in = 3.0 * fixed_val_sample - 1.0
                logits, _, _ = model(fixed_in)
                _save_visualizations(logits, fixed_val_sample, epoch, args.out_dir)
            model.train() # Turn BatchNorm/Dropout back to training mode

        t0 = time.time()
        train_metrics, itr = train_one_epoch( model, train_loader, optimizer, device, cfg, start_itr=itr, total_iterations=total_iterations)
        dt = time.time() - t0

        logging.info( "Epoch %d/%d  lr=%g  β=%.3f  v_loss=%.4f  D_kl=%.4f  acc=%.4f  tp=%.4f  tn=%.4f  (%.1fs)",
                     epoch, cfg["max_epochs"] - 1, current_lr, train_metrics["beta_mean"],
                     train_metrics["vloss"], train_metrics["kl"],
                     train_metrics["acc"], train_metrics["tp"], train_metrics["tn"], dt)
        
        mlog.log(phase="train", epoch=epoch, itr=itr, lr=current_lr, dt=dt, **train_metrics)

    # Final checkpoint
    save_checkpoint(args.out_dir / "final.pt", model, optimizer, cfg["max_epochs"] - 1, itr)
    logging.info("Training done.")
    mlog.close()

if __name__ == "__main__":
    main()