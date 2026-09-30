#!/usr/bin/env python3
"""Convert DeepSDF SdfSamples .npz files (pos / neg arrays) into PLY point clouds.

Each point keeps its SDF value as an extra "sdf" property and gets a color:
  red  = outside (pos, sdf > 0)
  blue = inside  (neg, sdf <= 0)
Same color convention as writeSDFToPLY in DeepSDF's PreprocessMesh.cpp.

Examples:
  python npz_to_ply.py data/SdfSamples/Airways/airway/AIIB23_31_mesh.npz
  python npz_to_ply.py data/SdfSamples/Airways/airway --out_dir ply --which neg
"""

import argparse  # command-line options
from pathlib import Path  # path handling

import numpy as np  # read .npz and write binary PLY data


def collect_inputs(paths):
    """Expand the given files/folders into a sorted list of .npz files."""
    npz_files = []  # every file we will convert
    for path in paths:  # user may mix files and folders
        if path.is_dir():  # a folder -> take all .npz inside it
            npz_files.extend(sorted(path.glob("*.npz")))  # only direct children
        else:
            npz_files.append(path)  # a single file
    return npz_files


def pick_points(data, which):
    """Return one (N, 4) array [x, y, z, sdf] for the chosen side(s)."""
    pos = data["pos"]  # outside samples
    neg = data["neg"]  # inside samples
    if which == "pos":
        return pos  # outside only
    if which == "neg":
        return neg  # inside only
    return np.concatenate([pos, neg], axis=0)  # both sides (axis=0 = stack rows)


def color_by_sign(sdf):
    """Red for outside, blue for inside, brightness grows closer to the surface."""
    closeness = 1.0 - np.clip(np.abs(sdf) / 0.1, 0.0, 1.0)  # 1 on surface, 0 at |sdf| >= 0.1
    shade = (80 + 175 * closeness).astype(np.uint8)  # keep far points visible (min 80)
    colors = np.zeros((len(sdf), 3), dtype=np.uint8)  # one RGB triple per point
    outside = sdf > 0  # same sign rule as the preprocessing
    colors[outside, 0] = shade[outside]  # outside -> red channel
    colors[~outside, 2] = shade[~outside]  # inside -> blue channel (~ = logical not)
    return colors


def write_ply(path, points, colors):
    """Write a binary little-endian PLY with x, y, z, sdf and RGB per vertex."""
    vertex_type = np.dtype([
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("sdf", "<f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
    ])  # one record per point (<f4 = little-endian float32, u1 = uint8)
    vertices = np.empty(len(points), dtype=vertex_type)  # allocate the records
    vertices["x"] = points[:, 0]  # coordinates
    vertices["y"] = points[:, 1]
    vertices["z"] = points[:, 2]
    vertices["sdf"] = points[:, 3]  # kept so you can color by it in CloudCompare/ParaView
    vertices["red"] = colors[:, 0]  # colors
    vertices["green"] = colors[:, 1]
    vertices["blue"] = colors[:, 2]

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {len(vertices)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property float sdf\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )  # standard PLY header matching vertex_type field order
    with open(path, "wb") as f:  # binary mode for the vertex block
        f.write(header.encode("ascii"))  # header is plain text
        f.write(vertices.tobytes())  # raw records right after the header


def convert(npz_path, out_dir, which, max_points, rng):
    """Convert one .npz file and print a short summary."""
    data = np.load(npz_path)  # archive with "pos" and "neg"
    points = pick_points(data, which)  # rows [x, y, z, sdf]
    points = points[~np.isnan(points[:, 3])]  # drop NaN sdf rows
    if max_points and len(points) > max_points:  # optional thinning for big files
        keep = rng.choice(len(points), max_points, replace=False)  # random row indices
        points = points[keep]  # random subset
    colors = color_by_sign(points[:, 3])  # red outside, blue inside
    suffix = "" if which == "both" else f"_{which}"  # e.g. case_neg.ply
    out_path = out_dir / f"{npz_path.stem}{suffix}.ply"  # same name as the input
    write_ply(out_path, points, colors)  # save it
    inside = int(np.sum(points[:, 3] <= 0))  # count for the summary
    print(f"OK  {out_path}  points={len(points)}  inside={inside}")  # report


def main():
    parser = argparse.ArgumentParser(description=__doc__)  # docstring doubles as help
    parser.add_argument("inputs", type=Path, nargs="+")  # .npz files or folders (+ = one or more)
    parser.add_argument("--out_dir", type=Path, default=None)  # default: next to each input
    parser.add_argument("--which", choices=["both", "pos", "neg"], default="both")  # sides
    parser.add_argument("--max_points", type=int, default=0)  # 0 = keep every point
    parser.add_argument("--seed", type=int, default=0)  # reproducible thinning
    args = parser.parse_args()  # read the command line

    rng = np.random.default_rng(args.seed)  # random generator for --max_points
    npz_files = collect_inputs(args.inputs)  # expand folders
    if not npz_files:  # nothing matched
        raise SystemExit("no .npz files found")  # stop with a message
    for npz_path in npz_files:  # one PLY per input
        out_dir = args.out_dir or npz_path.parent  # fall back to the input's folder
        out_dir.mkdir(parents=True, exist_ok=True)  # create it if missing
        convert(npz_path, out_dir, args.which, args.max_points, rng)  # do the work


if __name__ == "__main__":
    main()  # run only when executed, not when imported