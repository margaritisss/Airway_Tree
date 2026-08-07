#!/usr/bin/env python3
"""Sanity-check the .npy files the C++ tools produced, before converting to .mat.
    python inspect_npy.py out/subject_surface.npy out/subject_sdf.npy [out/subject_norm.npz]
"""
import sys
import numpy as np

def check_surface(path):
    a = np.load(path)
    print(f"\n=== {path} (surface points -> 'p') ===")
    print(f"shape          {a.shape}   (want N x 6)")
    if a.shape[1] != 6:
        print("  !! expected 6 columns [x y z nx ny nz]")
        return
    xyz, nrm = a[:, :3], a[:, 3:]
    print(f"xyz min/max    {xyz.min(0).round(4)} / {xyz.max(0).round(4)}")
    r = np.linalg.norm(xyz, axis=1)
    print(f"radius max     {r.max():.4f}   (want <= ~0.98, inside the unit sphere)")
    n = np.linalg.norm(nrm, axis=1)
    print(f"normal length  mean {n.mean():.4f}  min {n.min():.4f}  max {n.max():.4f}   (want ~1)")
    print(f"NaNs           {np.isnan(a).sum()}")
    if r.max() > 1.0:
        print("  !! points outside the unit sphere - normalization did not run")

def check_sdf(path):
    a = np.load(path)
    print(f"\n=== {path} (free-space samples -> 'p_sdf') ===")
    print(f"shape          {a.shape}   (want M x 4)")
    if a.shape[1] != 4:
        print("  !! expected 4 columns [x y z sdf]")
        return
    xyz, sdf = a[:, :3], a[:, 3]
    print(f"xyz min/max    {xyz.min(0).round(4)} / {xyz.max(0).round(4)}")
    print(f"sdf min/max    {sdf.min():.4f} / {sdf.max():.4f}")
    print(f"sdf negative   {(sdf < 0).sum():>9d}  ({100 * (sdf < 0).mean():.1f}%)  = inside")
    print(f"sdf positive   {(sdf > 0).sum():>9d}  ({100 * (sdf > 0).mean():.1f}%)  = outside")
    print(f"NaNs           {np.isnan(a).sum()}")
    exact_minus_one = int((sdf == -1.0).sum())
    print(f"sdf == -1.0    {exact_minus_one}   (DGCI treats this value as 'no supervision')")
    if (sdf < 0).sum() == 0:
        print("  !! no interior samples - check that the mesh is watertight")

def check_norm(path):
    z = np.load(path)
    print(f"\n=== {path} (normalization) ===")
    off, scale = z["offset"], z["scale"]
    print(f"offset         {off}")
    print(f"scale          {scale}")
    print("transinfo.json would be:  s = scale,  t = offset * scale,  R = identity")
    if np.allclose(off, 0, atol=1e-6) and np.allclose(scale, 1.0, atol=1e-6):
        print("  !! offset~0 and scale~1 - this was computed AFTER normalization,")
        print("     meaning the -n block was not hoisted above BoundingCubeNormalization")

if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        sys.exit(1)
    for p in args:
        if p.endswith(".npz"):
            check_norm(p)
        elif "surface" in p:
            check_surface(p)
        else:
            check_sdf(p)
