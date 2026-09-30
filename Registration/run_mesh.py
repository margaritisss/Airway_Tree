#!/usr/bin/env python
"""
Command-line runner for the mesh conversion functions in scan.py.

Place this file next to scan.py, i.e.
    /home/ids/gmargari-24/airway_project/Registration/run_mesh.py

Examples
--------
    python run_mesh.py --input-folder /path/to/AIIB23
    python run_mesh.py --input-folder /path/to/AIIB23 --output-folder /path/to/out
    python run_mesh.py --input-folder /path/to/AIIB23 --how-many 3 --workers 1
"""

import argparse
import glob
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

# Make sure `import scan` works no matter where the job was launched from.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scan import nifti2mesh  # noqa: E402


def default_output_folder(input_folder):
    """Fall back to the old naming rule when --output-folder is not given."""
    return f"{input_folder.rstrip('/')}_mesh"                          # e.g. .../AIIB23 -> .../AIIB23_mesh


def find_inputs(input_folder, recursive):
    """Return the .nii.gz files to convert, sorted for reproducibility."""
    pattern = "**/*.nii.gz" if recursive else "*.nii.gz"                # ** descends into subfolders
    hits = glob.glob(os.path.join(input_folder, pattern), recursive=recursive)
    return sorted(hits)


def convert_one(job):
    """Top-level worker function (must be importable for pickling)."""
    file_path, output_folder, level, fmt, verbose = job
    base     = os.path.basename(file_path)                             # drop the directory part
    out_path = os.path.join(output_folder, base.replace(".nii.gz", f"_mesh.{fmt}"))

    if os.path.exists(out_path):                                       # makes reruns cheap after a timeout
        return file_path, "skipped (already exists)", 0.0

    t0 = time.time()
    try:
        res = nifti2mesh(file_path, output_filename=out_path, level=level,
                         fmt=fmt, verbose=verbose)
        n_faces = len(res["mesh"].faces)                               # cheap sanity check on the result
        return file_path, f"ok ({n_faces:,} faces)", time.time() - t0
    except Exception as exc:
        return file_path, f"FAILED: {type(exc).__name__}: {exc}", time.time() - t0


def main():
    p = argparse.ArgumentParser(description="Convert NIfTI masks to meshes.")
    p.add_argument("--input-folder", required=True,
                   help="Folder containing .nii.gz files.")
    p.add_argument("--output-folder", default=None,
                   help="Where to write the meshes (default: <input-folder>_mesh).")
    p.add_argument("--how-many", default="all",
                   help="'all' or a positive integer (default: all).")
    p.add_argument("--level", type=float, default=0.5,
                   help="Marching-cubes iso-level (default: 0.5 for binary masks).")
    p.add_argument("--workers", type=int, default=1,
                   help="Parallel processes. 1 = sequential.")
    p.add_argument("--recursive", action="store_true",
                   help="Also search subfolders of --input-folder.")
    p.add_argument("--quiet", action="store_true",
                   help="Suppress the per-file inspection printout.")
    p.add_argument("--fmt", default="obj", choices=["obj", "stl", "ply"],
                   help="Output mesh format (default: obj).")
    args = p.parse_args()

    in_folder = args.input_folder
    if not os.path.isdir(in_folder):                                   # guard clause: bad path fails fast
        sys.exit(f"Input folder does not exist: {in_folder}")

    out_folder = args.output_folder or default_output_folder(in_folder)
    os.makedirs(out_folder, exist_ok=True)                             # exist_ok: harmless on rerun

    files = find_inputs(in_folder, args.recursive)
    if not files:
        sys.exit(f"No .nii.gz files found in: {in_folder}")

    if str(args.how_many).lower() != "all":
        files = files[: int(args.how_many)]                            # take the first N after sorting

    n_total = len(files)
    verbose = not args.quiet

    print(f"Input : {in_folder}")
    print(f"Output: {out_folder}")
    print(f"Level : {args.level}   Workers: {args.workers}   Files: {n_total}")
    print("-" * 70, flush=True)

    t_start = time.time()
    jobs = [(f, out_folder, args.level, args.fmt, verbose) for f in files]
    n_ok = 0

    # ---- Sequential path ---------------------------------------------------
    if args.workers <= 1:
        for i, job in enumerate(jobs, start=1):
            path, status, secs = convert_one(job)                      # same worker, just called in-process
            if not status.startswith("FAILED"):
                n_ok += 1
            print(f"[{i}/{n_total}] {os.path.basename(path)} -> {status} "
                  f"({secs:.1f}s)", flush=True)
    # ---- Parallel path: one process per file -------------------------------
    else:
        print(f"Dispatching {n_total} file(s) across {args.workers} worker(s).\n",
              flush=True)
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(convert_one, j): j[0] for j in jobs}
            for i, fut in enumerate(as_completed(futures), start=1):   # report as they land, not in order
                path, status, secs = fut.result()
                if not status.startswith("FAILED"):
                    n_ok += 1
                print(f"[{i}/{n_total}] {os.path.basename(path)} -> {status} "
                      f"({secs:.1f}s)", flush=True)

    elapsed = time.time() - t_start
    print("-" * 70)
    print(f"Converted {n_ok}/{n_total} file(s) in {elapsed/60:.1f} min.")
    print(f"Meshes are in: {out_folder}")

    # Non-zero exit code if anything failed, so Slurm marks the job FAILED.
    sys.exit(0 if n_ok == n_total else 1)


if __name__ == "__main__":
    main()
