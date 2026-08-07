#!/usr/bin/env python3
"""Export the DeepSDF .npy point clouds as PLY files for CloudCompare / MeshLab.

    python npy_to_ply.py subject_surface.npy            -> subject_surface.ply
    python npy_to_ply.py subject_sdf.npy                -> subject_sdf.ply
    python npy_to_ply.py ~/dgci_npy/AIIB23_30_R_mesh_*.npy --outdir ~/ply

Surface files (N x 6) become point clouds carrying normals, so CloudCompare can
shade them and you can check the normals point outward.

SDF files (M x 4) become point clouds carrying the signed distance as both a
scalar field named 'sdf' and an RGB colour (red = inside, blue = outside), so
the sign pattern is visible the moment the file opens.
"""

import argparse
import os
import sys

import numpy as np


def _write_ply(path, arr, header_props):
    header = ["ply", "format binary_little_endian 1.0",
              f"element vertex {len(arr)}"]
    header += header_props
    header.append("end_header")
    with open(path, "wb") as f:
        f.write(("\n".join(header) + "\n").encode("ascii"))
        arr.tofile(f)


def export_surface(npy, out, stride):
    a = np.load(npy)
    if a.shape[1] != 6:
        raise ValueError(f"expected N x 6, got {a.shape}")
    a = a[::stride]

    dt = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                   ("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4")])
    rec = np.empty(len(a), dtype=dt)
    for i, k in enumerate(("x", "y", "z", "nx", "ny", "nz")):
        rec[k] = a[:, i]

    _write_ply(out, rec, [
        "property float x", "property float y", "property float z",
        "property float nx", "property float ny", "property float nz",
    ])
    print(f"  {os.path.basename(out)}  {len(a)} points with normals")


def export_sdf(npy, out, stride, clip):
    a = np.load(npy)
    if a.shape[1] != 4:
        raise ValueError(f"expected M x 4, got {a.shape}")
    a = a[::stride]
    xyz, sdf = a[:, :3], a[:, 3]

    # Colour on a symmetric scale so inside/outside read at a glance. Most
    # samples sit very close to zero, so clip the scale rather than the data -
    # otherwise a handful of far-field points wash everything else out.
    lim = clip if clip > 0 else float(np.percentile(np.abs(sdf), 95)) or 1.0
    t = np.clip(sdf / lim, -1.0, 1.0)
    # Diverging red-white-blue: saturated red deep inside, white at the zero
    # level set, saturated blue far outside.
    red = np.where(t < 0, 255.0, 255.0 * (1.0 - t)).astype(np.uint8)
    blue = np.where(t < 0, 255.0 * (1.0 + t), 255.0).astype(np.uint8)
    green = (255.0 * (1.0 - np.abs(t))).astype(np.uint8)

    dt = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("sdf", "<f4"),
                   ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    rec = np.empty(len(a), dtype=dt)
    rec["x"], rec["y"], rec["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    rec["sdf"] = sdf
    rec["red"], rec["green"], rec["blue"] = red, green, blue

    _write_ply(out, rec, [
        "property float x", "property float y", "property float z",
        "property float sdf",
        "property uchar red", "property uchar green", "property uchar blue",
    ])
    inside = int((sdf < 0).sum())
    print(f"  {os.path.basename(out)}  {len(a)} points  "
          f"({inside} inside / {len(a) - inside} outside, colour scale +-{lim:.3f})")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("npy", nargs="+", help="one or more *_surface.npy / *_sdf.npy")
    ap.add_argument("--outdir", default=None, help="destination (default: next to input)")
    ap.add_argument("--stride", type=int, default=1,
                    help="keep every Nth point (default 1; use 4 or 10 if your "
                         "viewer struggles)")
    ap.add_argument("--clip", type=float, default=0.0,
                    help="colour scale limit for sdf files (default: 95th "
                         "percentile of |sdf|)")
    args = ap.parse_args()

    for npy in args.npy:
        if not os.path.isfile(npy):
            print(f"{npy}: not found", file=sys.stderr)
            continue
        base = os.path.basename(npy)[: -len(".npy")]
        outdir = args.outdir or os.path.dirname(os.path.abspath(npy))
        os.makedirs(outdir, exist_ok=True)
        out = os.path.join(outdir, base + ".ply")
        print(base + ":")
        try:
            if base.endswith("_surface"):
                export_surface(npy, out, args.stride)
            elif base.endswith("_sdf"):
                export_sdf(npy, out, args.stride, args.clip)
            else:  # fall back on the column count
                a = np.load(npy)
                (export_surface if a.shape[1] == 6 else export_sdf)(
                    npy, out, args.stride, *([] if a.shape[1] == 6 else [args.clip]))
        except Exception as exc:
            print(f"  FAILED - {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()
