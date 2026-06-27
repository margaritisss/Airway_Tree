"""
Generate / visualize samples from a trained VoxelVAE128 (β-VAE).

Run this file next to β_VAE.py (it imports the model from there).

Three modes
-----------
  sample      Draw z ~ N(0, I) from the prior, decode, save occupancy grids.
              "What can the model dream up from scratch?"
  reconstruct Encode a real .nii.gz mask -> mu -> decode it back.
              "How faithfully does the model reproduce real inputs?"
  traverse    Walk one latent dimension while holding the rest fixed.
              "What did each latent dimension learn?" (the point of a β-VAE)

Each mode writes .nii.gz volumes you can open in 3D Slicer / ITK-SNAP / FSLeyes,
and (if matplotlib + scikit-image are available) PNG surface renders so you can
eyeball results immediately without a NIfTI viewer.

Examples
--------
  # 8 fresh samples from the prior, slightly sharpened with temperature 0.8
  python generate_β_VAE.py sample \
      --checkpoint runs/final.pt --out-dir gen_out --n 8 --temperature 0.8

  # reconstruct a few real masks
  python generate_β_VAE.py reconstruct \
      --checkpoint runs/final.pt --out-dir recon_out \
      --inputs data/case_0001.nii.gz data/case_0002.nii.gz

  # traverse latent dim 12 from -3 to +3 in 9 steps
  python generate_β_VAE.py traverse \
      --checkpoint runs/final.pt --out-dir trav_out --dim 12 --steps 9
"""

from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import torch

# Model lives in β_VAE.py next to this script.
from β_VAE import VoxelVAE128

# Optional viz deps — script still works (saves .nii.gz) if these are missing.
try:
    import nibabel as nib
    _HAVE_NIB = True
except Exception:
    _HAVE_NIB = False

try:
    import matplotlib
    matplotlib.use("Agg")  # headless / no display needed
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from skimage import measure
    _HAVE_VIZ = True
except Exception:
    _HAVE_VIZ = False


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_model(checkpoint: Path, device: torch.device) -> VoxelVAE128:
    """Load a trained VoxelVAE128 from a checkpoint saved by train_β_VAE.py."""
    ckpt = torch.load(checkpoint, map_location=device)
    state = ckpt["model_state"] if "model_state" in ckpt else ckpt

    # Infer num_latents straight from the weights so we never mismatch the
    # value used at training time: enc_mu is Linear(fc_dim, num_latents).
    num_latents = state["enc_mu.weight"].shape[0]

    model = VoxelVAE128(num_latents=num_latents).to(device)
    model.load_state_dict(state)
    model.eval()  # CRITICAL: BatchNorm must use running stats, not batch stats,
                  # and reparameterize() becomes deterministic in eval mode.
    print(f"Loaded {checkpoint}  (num_latents={num_latents}, "
          f"epoch={ckpt.get('epoch', '?')})")
    return model


# ---------------------------------------------------------------------------
# Core decode helpers
# ---------------------------------------------------------------------------
@torch.no_grad()
def logits_to_occupancy(logits: torch.Tensor, threshold: float = 0.5) -> np.ndarray:
    """(B,1,D,H,W) logits -> (B,D,H,W) uint8 occupancy at the given prob threshold."""
    probs = torch.sigmoid(logits)
    occ = (probs > threshold).squeeze(1).to(torch.uint8)
    return occ.cpu().numpy()


@torch.no_grad()
def sample_from_prior(model, n, device, temperature=1.0, seed=None):
    """z ~ N(0, temperature^2 * I) -> decode -> occupancy grids + raw probs."""
    if seed is not None:
        torch.manual_seed(seed)
    z = torch.randn(n, model.num_latents, device=device) * temperature
    logits = model.decode(z)
    return logits_to_occupancy(logits), torch.sigmoid(logits).squeeze(1).cpu().numpy()


@torch.no_grad()
def reconstruct(model, volume_bin, device):
    """Encode a binary {0,1} volume (D,H,W) -> mu -> decode. Mirrors training preprocessing."""
    x = torch.from_numpy(volume_bin.astype(np.float32))[None, None]  # (1,1,D,H,W)
    x = x.to(device)
    x_in = 3.0 * x - 1.0           # SAME rescaling the encoder saw during training
    mu, _ = model.encode(x_in)     # use the mean, not a sample, for a clean recon
    logits = model.decode(mu)
    return logits_to_occupancy(logits)[0]


@torch.no_grad()
def traverse_dim(model, dim, steps, span, device, base_z=None):
    """Hold all latents at base_z, sweep one dim across [-span, +span]."""
    if base_z is None:
        base_z = torch.zeros(1, model.num_latents, device=device)
    values = torch.linspace(-span, span, steps, device=device)
    z = base_z.repeat(steps, 1)
    z[:, dim] = values
    logits = model.decode(z)
    return logits_to_occupancy(logits), values.cpu().numpy()


# ---------------------------------------------------------------------------
# I/O + visualization
# ---------------------------------------------------------------------------
def save_nifti(volume_uint8: np.ndarray, path: Path) -> None:
    if not _HAVE_NIB:
        return
    img = nib.Nifti1Image(volume_uint8.astype(np.uint8), affine=np.eye(4))
    nib.save(img, str(path))


def render_voxels(volume_uint8: np.ndarray, path: Path, title: str = "") -> None:
    """Marching-cubes surface render -> PNG. Skips silently if a volume is empty."""
    if not _HAVE_VIZ:
        return
    vol = volume_uint8.astype(np.float32)
    if vol.sum() == 0:
        # Nothing to surface; still drop a labelled blank so the grid stays aligned.
        fig = plt.figure(figsize=(3, 3))
        ax = fig.add_subplot(111, projection="3d")
        ax.set_title(title + "\n(empty)", fontsize=8)
        ax.set_axis_off()
        fig.savefig(path, dpi=110, bbox_inches="tight")
        plt.close(fig)
        return
    verts, faces, _, _ = measure.marching_cubes(vol, level=0.5)
    fig = plt.figure(figsize=(3, 3))
    ax = fig.add_subplot(111, projection="3d")
    mesh = Poly3DCollection(verts[faces], alpha=1.0)
    mesh.set_edgecolor("none")
    mesh.set_facecolor((0.32, 0.55, 0.85))
    ax.add_collection3d(mesh)
    d = vol.shape
    ax.set_xlim(0, d[0]); ax.set_ylim(0, d[1]); ax.set_zlim(0, d[2])
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=20, azim=45)
    ax.set_axis_off()
    if title:
        ax.set_title(title, fontsize=8)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def contact_sheet(pngs: list[Path], out_path: Path, cols: int = 4) -> None:
    """Stitch individual PNG renders into one overview image."""
    if not _HAVE_VIZ or not pngs:
        return
    import matplotlib.image as mpimg
    n = len(pngs)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.6, rows * 2.6))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.set_axis_off()
    for ax, p in zip(axes, pngs):
        ax.imshow(mpimg.imread(p))
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  contact sheet -> {out_path}")


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------
def mode_sample(model, args, device):
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    occ, _ = sample_from_prior(model, args.n, device,
                               temperature=args.temperature, seed=args.seed)
    pngs = []
    for i in range(args.n):
        vox = occ[i]
        save_nifti(vox, out / f"sample_{i:03d}.nii.gz")
        png = out / f"sample_{i:03d}.png"
        render_voxels(vox, png, title=f"sample {i}  (vox={int(vox.sum())})")
        if png.exists():
            pngs.append(png)
        print(f"  sample {i}: {int(vox.sum())} occupied voxels")
    contact_sheet(pngs, out / "samples_overview.png", cols=args.cols)


def mode_reconstruct(model, args, device):
    if not _HAVE_NIB:
        raise SystemExit("reconstruct mode needs nibabel to read the input .nii.gz files.")
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    pngs = []
    for k, inp in enumerate(args.inputs):
        arr = np.asarray(nib.load(str(inp)).dataobj)
        gt = (arr > 0.5).astype(np.uint8)
        rec = reconstruct(model, gt, device)
        save_nifti(gt,  out / f"recon_{k:03d}_input.nii.gz")
        save_nifti(rec, out / f"recon_{k:03d}_output.nii.gz")
        p_in  = out / f"recon_{k:03d}_input.png"
        p_out = out / f"recon_{k:03d}_output.png"
        render_voxels(gt,  p_in,  title=f"input {k}")
        render_voxels(rec, p_out, title=f"recon {k}")
        # Simple overlap stats so you get a number, not just a picture.
        inter = np.logical_and(gt, rec).sum()
        denom = gt.sum() + rec.sum()
        dice = (2 * inter / denom) if denom else 1.0
        print(f"  {Path(inp).name}: Dice(input, recon) = {dice:.3f}")
        for p in (p_in, p_out):
            if p.exists():
                pngs.append(p)
    contact_sheet(pngs, out / "recon_overview.png", cols=2)


def mode_traverse(model, args, device):
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    occ, vals = traverse_dim(model, args.dim, args.steps, args.span, device)
    pngs = []
    for i in range(args.steps):
        vox = occ[i]
        save_nifti(vox, out / f"dim{args.dim:03d}_step{i:02d}.nii.gz")
        png = out / f"dim{args.dim:03d}_step{i:02d}.png"
        render_voxels(vox, png, title=f"z[{args.dim}]={vals[i]:+.2f}")
        if png.exists():
            pngs.append(png)
    contact_sheet(pngs, out / f"dim{args.dim:03d}_traverse.png", cols=args.steps)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="mode", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--checkpoint", type=Path, required=True)
    common.add_argument("--out-dir", type=Path, required=True)
    common.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is present.")

    ps = sub.add_parser("sample", parents=[common], help="Sample from the prior.")
    ps.add_argument("--n", type=int, default=8)
    ps.add_argument("--temperature", type=float, default=1.0,
                    help="Scale on z. <1 gives cleaner/more typical samples, >1 more diverse.")
    ps.add_argument("--seed", type=int, default=0)
    ps.add_argument("--cols", type=int, default=4)

    pr = sub.add_parser("reconstruct", parents=[common], help="Reconstruct real masks.")
    pr.add_argument("--inputs", type=Path, nargs="+", required=True)

    pt = sub.add_parser("traverse", parents=[common], help="Latent traversal.")
    pt.add_argument("--dim", type=int, required=True)
    pt.add_argument("--steps", type=int, default=9)
    pt.add_argument("--span", type=float, default=3.0)
    return p


def main():
    args = build_parser().parse_args()
    device = torch.device("cuda" if (torch.cuda.is_available() and not args.cpu) else "cpu")
    print(f"Device: {device}")
    if not _HAVE_VIZ:
        print("[note] matplotlib/scikit-image not found -> writing .nii.gz only, no PNG renders.")

    model = load_model(args.checkpoint, device)

    {"sample": mode_sample,
     "reconstruct": mode_reconstruct,
     "traverse": mode_traverse}[args.mode](model, args, device)

    print(f"Done. Outputs in {args.out_dir}")


if __name__ == "__main__":
    main()
