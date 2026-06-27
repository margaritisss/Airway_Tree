"""
Round-trip sanity check for VAEsselSparse on a GPU node.

Run BEFORE any real training. It verifies the parts that are custom and
unverified-by-the-paper: (1) the torchsparse->sp wrap, (2) the generative
transposed-conv decoder, and (3) spatial_range propagation (the leash that
keeps the generative upsampler from clamping/collapsing coordinates).

What it checks, per the reasoning we worked through:
  [runs]    encode->decode executes (no assert from the generative conv)
  [extent]  decoded coords reach ~full resolution (NOT clamped to a coarse box)
  [growth]  active voxel count increases stage by stage (upsampling really happens)
  [grad]    autograd flows back through the generative conv

Usage:
    python sanity_check.py                 # small fast grid (64^3)
    python sanity_check.py --grid 640 640 832   # real padded size (slow, big mem)

Decoder upsampling uses sp.SparseSubdivide (the generative transposed conv
clamps/shrinks on this torchsparse build), so [extent] should now climb
79->159->319->639 across the three up-stages.
"""
import os
# windowed attention backend must be chosen before importing the sparse module.
os.environ.setdefault("ATTN_BACKEND", "flash_attn")   # or "xformers"

import argparse
import torch
from torchsparse import SparseTensor as TSTensor
from torchsparse.utils.collate import sparse_collate

from vaessel_2 import Encoder, Decoder, to_sp


def extent(h):
    """max (x,y,z) coordinate present in an sp.SparseTensor."""
    return h.coords[:, 1:4].max(0).values.tolist()


def make_fake_batch(grid, n_active, batch_size, device):
    """A few synthetic sparse volumes (random active voxels) -> batched, wrapped."""
    samples = []
    for _ in range(batch_size):
        c = torch.randint(0, min(grid), (n_active, 3), dtype=torch.int32)
        c = torch.unique(c, dim=0)                                   # coords (N,3) xyz
        f = torch.ones(c.shape[0], 1, dtype=torch.float32)          # occupancy feature
        samples.append(TSTensor(coords=c, feats=f))
    ts = sparse_collate(samples).to(device)                         # adds batch col -> (N,4)
    B = int(ts.coords[:, 0].max()) + 1
    return to_sp(ts, spatial_range=(B, *grid)), B                   # set spatial_range here


@torch.enable_grad()
def decode_verbose(dec, z, grid):
    """Replicates Decoder.forward but prints extent/voxels/scale/range per stage."""
    h = dec.from_latent(z)
    coords = h.coords[:, 1:4].float()
    h = h.replace(h.feats + dec.pos_embedder(coords))
    for blk in dec.attn_blocks:
        h = blk(h)
    print(f"  after attn      : extent={extent(h)}  voxels={h.coords.shape[0]}  "
          f"scale={h._scale}  range={getattr(h.data, 'spatial_range', None)}")
    for i, stage in enumerate(dec.stages):
        h = stage(h)
        print(f"  after up-stage{i}: extent={extent(h)}  voxels={h.coords.shape[0]}  "
              f"scale={h._scale}  range={getattr(h.data, 'spatial_range', None)}")
    return dec.head(h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=int, nargs=3, default=[64, 64, 64], help="H W D")
    ap.add_argument("--n-active", type=int, default=3000)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    grid = tuple(args.grid)
    assert max(grid) <= 1024, "torchsparse z-order coords must be <1024 per dim"
    assert all(g % 8 == 0 for g in grid), "grid must be divisible by 8 (rs=8)"
    device = args.device

    torch.manual_seed(0)
    enc = Encoder(in_channels=1).to(device).train()
    dec = Decoder(out_channels=1).to(device).train()

    x, B = make_fake_batch(grid, args.n_active, args.batch_size, device)
    print(f"input  : grid={grid}  batch={B}  voxels={x.coords.shape[0]}  "
          f"extent={extent(x)}  range={getattr(x.data, 'spatial_range', None)}")

    # ---- encode + decode under bf16 autocast (FlashAttention needs fp16/bf16) ----
    amp = torch.autocast("cuda", dtype=torch.bfloat16)
    # ---- encode ----
    with amp:
        z, mu, logvar = enc(x)
    print(f"latent : extent={extent(z)}  voxels={z.coords.shape[0]}  "
          f"scale={z._scale}  range={getattr(z.data, 'spatial_range', None)}  "
          f"feat_dim={z.feats.shape[1]}")
    assert z.feats.shape[1] == mu.shape[1], "latent channel mismatch"

    # ---- decode (instrumented) ----
    print("decode :")
    with amp:
        logits = decode_verbose(dec, z, grid)
    out_ext = extent(logits)
    print(f"output : extent={out_ext}  voxels={logits.coords.shape[0]}  "
          f"logit_dim={logits.feats.shape[1]}")

    # ---- assertions ----
    # [extent] decoder reached ~full resolution (not clamped to a coarse corner)
    assert all(e >= 0.5 * g for e, g in zip(out_ext, grid)), (
        f"[extent] FAILED: decoded coords {out_ext} are far below grid {grid} -> "
        "spatial_range mis-propagated (clamped). Fix range per stage or use SparseSubdivide.")
    # [growth] upsampling actually added voxels
    assert logits.coords.shape[0] > z.coords.shape[0], (
        f"[growth] FAILED: output voxels {logits.coords.shape[0]} <= latent {z.coords.shape[0]}")

    # ---- [grad] autograd through the generative conv ----
    loss = logits.feats.float().pow(2).mean()        # dummy differentiable objective
    loss.backward()
    has_grad = any(p.grad is not None and torch.isfinite(p.grad).all() for p in dec.parameters())
    assert has_grad, "[grad] FAILED: no finite gradients reached decoder parameters"

    print("\nALL CHECKS PASSED ✓  (runs, extent reaches full res, voxels grow, grads flow)")


if __name__ == "__main__":
    main()