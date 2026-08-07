#!/usr/bin/env python
"""
Command-line runner for the mesh conversion functions in scan.py.

Place this file next to scan.py, i.e.
    /home/ids/gmargari-24/airway_project/Registration/run_mesh.py

Examples
--------
    python run_mesh.py --input-folder /path/to/AIIB23
    python run_mesh.py --input-folder /path/to/AIIB23 --workers 8
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

from scan import nifti2mesh, nifti_to_mesh  # noqa: E402

def output_folder_for(input_folder):
    """Reproduce the naming rule used inside nifti_to_mesh."""
    return f"{input_folder.rstrip('/')}_mesh"

def convert_one(job):
    """Top-level worker function (must be importable for pickling)."""
    file_path, output_folder, level, fmt = job
    base     = os.path.basename(file_path)
    out_path = os.path.join(output_folder, base.replace(".nii.gz", f"_mesh.{fmt}"))

    if os.path.exists(out_path):
        return file_path, "skipped (already exists)", 0.0

    t0 = time.time()
    try:
        res = nifti2mesh(file_path, output_filename=out_path, level=level,
                         fmt=fmt, verbose=False)
        n_faces = len(res["mesh"].faces)
        return file_path, f"ok ({n_faces:,} faces)", time.time() - t0
    except Exception as exc:
        return file_path, f"FAILED: {type(exc).__name__}: {exc}", time.time() - t0

def main():
    p = argparse.ArgumentParser(description="Convert NIfTI masks to STL meshes.")
    p.add_argument("--input-folder", required=True,
                   help="Folder containing .nii.gz files.")
    p.add_argument("--how-many", default="all",
                   help="'all' or a positive integer (default: all).")
    p.add_argument("--level", type=float, default=0.5,
                   help="Marching-cubes iso-level (default: 0.5 for binary masks).")
    p.add_argument("--workers", type=int, default=1,
                   help="Parallel processes. 1 = sequential, same as the notebook.")
    p.add_argument("--quiet", action="store_true",
                   help="Suppress the per-file inspection printout.")
    p.add_argument("--fmt", default="obj", choices=["obj", "stl", "ply"],
               help="Output mesh format (default: obj).")
    args = p.parse_args()

    in_folder = args.input_folder
    if not os.path.isdir(in_folder):
        sys.exit(f"Input folder does not exist: {in_folder}")

    out_folder = output_folder_for(in_folder)
    os.makedirs(out_folder, exist_ok=True)

    print(f"Input : {in_folder}")
    print(f"Output: {out_folder}")
    print(f"Level : {args.level}   Workers: {args.workers}")
    print("-" * 70, flush=True)

    t_start = time.time()

    # ---- Sequential path: exactly what you ran in the notebook -------------
    if args.workers <= 1:
        results = nifti_to_mesh(
            input_folder=in_folder,
            how_many=args.how_many,
            level=args.level,
            fmt=args.fmt,
            verbose=not args.quiet,
        )
        candidates = sorted(glob.glob(os.path.join(in_folder, "*.nii.gz")))
        if str(args.how_many).lower() != "all":
            candidates = candidates[: int(args.how_many)]
        n_ok = len(results)
        n_total = len(candidates)
    # ---- Parallel path: one process per file -------------------------------
    else:
        files = sorted(glob.glob(os.path.join(in_folder, "*.nii.gz")))
        if not files:
            sys.exit(f"No .nii.gz files found in: {in_folder}")

        if str(args.how_many).lower() != "all":
            files = files[: int(args.how_many)]

        n_total = len(files)
        print(f"Dispatching {n_total} file(s) across {args.workers} worker(s).\n",
              flush=True)

        jobs = [(f, out_folder, args.level, args.fmt) for f in files]
        n_ok = 0
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(convert_one, j): j[0] for j in jobs}
            for i, fut in enumerate(as_completed(futures), start=1):
                path, status, secs = fut.result()
                if status.startswith("ok") or status.startswith("skipped"):
                    n_ok += 1
                print(f"[{i}/{n_total}] {os.path.basename(path)} -> {status} "
                      f"({secs:.1f}s)", flush=True)

    elapsed = time.time() - t_start
    print("-" * 70)
    print(f"Converted {n_ok}/{n_total} file(s) in {elapsed/60:.1f} min.")
    print(f"STL files are in: {out_folder}")

    # Non-zero exit code if anything failed, so Slurm marks the job FAILED.
    sys.exit(0 if n_ok == n_total else 1)


if __name__ == "__main__":
    main()
