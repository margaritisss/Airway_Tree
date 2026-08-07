#!/usr/bin/env python3
"""Convert the DeepSDF .npy output to the .mat files DGCI's dataset.py reads.

    python npy2mat.py IN_DIR OUT_ROOT

Copies the arrays verbatim - same values, same dtype. Nothing is filtered,
clipped or rescaled.

IN_DIR contains <subject>_surface.npy and <subject>_sdf.npy.

OUT_ROOT is filled in exactly the layout dataset.py expects:

    OUT_ROOT/surface_pts_n_normal/<subject>.mat    variable 'p'      N x 6
    OUT_ROOT/free_space_pts/<subject>.mat          variable 'p_sdf'  M x 4

Then set point_cloud_path in your config to OUT_ROOT/surface_pts_n_normal.
Both directory names are load-bearing: dataset.py locates the free-space file
by replacing 'surface_pts_n_normal' with 'free_space_pts' in the path.
"""

import os
import sys

import numpy as np
from scipy.io import savemat


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    in_dir, out_root = sys.argv[1], sys.argv[2]

    surf_dir = os.path.join(out_root, "surface_pts_n_normal")
    free_dir = os.path.join(out_root, "free_space_pts")
    os.makedirs(surf_dir, exist_ok=True)
    os.makedirs(free_dir, exist_ok=True)

    names = sorted(f[: -len("_surface.npy")]
                   for f in os.listdir(in_dir) if f.endswith("_surface.npy"))
    if not names:
        sys.exit(f"no *_surface.npy files in {in_dir}")

    done = 0
    for name in names:
        surf_npy = os.path.join(in_dir, name + "_surface.npy")
        sdf_npy = os.path.join(in_dir, name + "_sdf.npy")

        if not os.path.isfile(sdf_npy):
            print(f"SKIP {name}: no {name}_sdf.npy")
            continue

        p = np.load(surf_npy)
        p_sdf = np.load(sdf_npy)

        if p.ndim != 2 or p.shape[1] != 6:
            print(f"SKIP {name}: surface is {p.shape}, expected N x 6")
            continue
        if p_sdf.ndim != 2 or p_sdf.shape[1] != 4:
            print(f"SKIP {name}: sdf is {p_sdf.shape}, expected M x 4")
            continue

        savemat(os.path.join(surf_dir, name + ".mat"), {"p": p})
        savemat(os.path.join(free_dir, name + ".mat"), {"p_sdf": p_sdf})

        done += 1
        print(f"{name}: p{p.shape} p_sdf{p_sdf.shape} {p.dtype}")

    print(f"\n{done} subjects written to {out_root}")
    print(f"point_cloud_path: {os.path.abspath(surf_dir)}")
    print(f"num_instances:    {len(os.listdir(surf_dir))}")


if __name__ == "__main__":
    main()
