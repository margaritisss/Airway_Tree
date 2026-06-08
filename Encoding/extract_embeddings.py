"""
extract_embeddings.py

Step 1 of latent phenotyping: freeze the trained VoxelVAE128 encoder and
turn every patient mask into a 100-D `mu` vector.

CRITICAL preprocessing parity with training (see train_VAE_CA_b_wBN.py):
  - binarize mask at > 0.5
  - add channel dim -> (1, D, H, W)
  - rescale to {-1, 2} via  x_in = 3*x - 1   BEFORE the encoder
  - run model in .eval() so:
      * reparameterize() returns mu directly (no sampling)
      * BatchNorm uses running statistics, not batch statistics
  - NO augmentation (no flips)

Output: an .npz with
  mu       (N, 100)  float32   <- your feature matrix for clustering
  logsigma (N, 100)  float32   <- kept for diagnostics (posterior collapse)
  paths    (N,)      str       <- file path per row, so you can trace back
"""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import nibabel as nib
import torch
from VAE_CA_b_wBN import VoxelVAE128


def load_mask(path: Path) -> np.ndarray:
    """Load a .nii.gz mask exactly as the training Dataset did."""
    img = nib.load(str(path))
    arr = np.asarray(img.dataobj)
    if arr.ndim != 3: #
        raise ValueError(f"Expected 3D array at {path}, got shape {arr.shape}")
    arr = (arr > 0.5).astype(np.float32)   # defensive binarize, matches training
    return arr[None, ...]                  # (1, D, H, W)


@torch.no_grad() 
def main():
    ap = argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dirs", type=Path, nargs="+", required=True,  help="Directories of .nii.gz masks (pooled, same as training).")
    ap.add_argument("--checkpoint", type=Path, required=True, help="Path to final.pt / epoch_XXXX.pt from training.")
    ap.add_argument("--out", type=Path, default=Path("embeddings.npz"))
    ap.add_argument("--num-latents", type=int, default=100, help="Must match the trained model (default 100).")
    ap.add_argument("--batch-size", type=int, default=4, help="Inference batch size. Keep >1 so BatchNorm is stable, ""though in eval mode it uses running stats anyway.")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] device: {device}")

    # ---- gather files in a stable, reproducible order ----
    paths: list[Path] = []
    for d in args.data_dirs:
        d = Path(d)
        if not d.is_dir():
            raise FileNotFoundError(f"Not a directory: {d}")
        paths.extend(sorted(d.glob("*.nii.gz")))
    if not paths:
        raise ValueError(f"No .nii.gz files in {args.data_dirs}")
    print(f"[info] found {len(paths)} masks")

    # ---- build model + load weights ----
    model = VoxelVAE128(num_latents=args.num_latents).to(device)
    ckpt  = torch.load(args.checkpoint, map_location=device)
    state = ckpt["model_state"] if "model_state" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()  # <- essential: mu is returned, BN uses running stats
    print(f"[info] loaded checkpoint: {args.checkpoint}")

    mus, logsigmas, kept_paths = [], [], []

    batch, batch_paths = [], []

    def flush():
        if not batch:
            return
        x = torch.from_numpy(np.stack(batch, axis=0)).to(device)  # (B,1,D,H,W)
        x_in = 3.0 * x - 1.0                                      # training parity
        mu, logsigma = model.encode(x_in)
        mus.append(mu.cpu().numpy())
        logsigmas.append(logsigma.cpu().numpy())
        kept_paths.extend(batch_paths)
        batch.clear()
        batch_paths.clear()

    for i, p in enumerate(paths):
        try:
            batch.append(load_mask(p))
            batch_paths.append(str(p))
        except Exception as e:
            print(f"[warn] skipping {p}: {e}")
            continue
        if len(batch) == args.batch_size:
            flush()
        if (i + 1) % 25 == 0:
            print(f"[info] encoded {i + 1}/{len(paths)}")
    flush()

    mu = np.concatenate(mus, axis=0).astype(np.float32)
    logsigma = np.concatenate(logsigmas, axis=0).astype(np.float32)
    paths_arr = np.array(kept_paths)

    np.savez(args.out, mu=mu, logsigma=logsigma, paths=paths_arr)
    print(f"[done] wrote {args.out}  mu shape {mu.shape}")

    # ---- quick posterior-collapse diagnostic ----
    var = mu.var(axis=0)
    order = np.argsort(var)[::-1]
    active = int((var > 0.05 * var.max()).sum())
    print(f"[diag] per-dim variance of mu: max={var.max():.4f} min={var.min():.4f}")
    print(f"[diag] ~{active}/{args.num_latents} dims look active "
          f"(var > 5% of max). Top dims: {order[:10].tolist()}")


if __name__ == "__main__":
    main()


