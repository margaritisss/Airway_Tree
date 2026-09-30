"""Convert the DeepSDF C++ outputs (run_batch.sh) into the files DGCI's dataset.py reads.

Input, per case <n>, in NPY_DIR:
  <n>_surface.npy  (N, 6)  xyz + normal     from SampleVisibleMeshSurface
  <n>_sdf.npy      (M, 4)  xyz + sdf        from PreprocessMesh
  <n>_norm.npz     offset (3,), scale (1,)  x_norm = (x + offset) * scale

Output, in OUT_ROOT (set point_cloud_path to OUT_ROOT/surface_pts_n_normal):
  surface_pts_n_normal/<n>.mat  key 'p'      (N, 6)
  free_space_pts/<n>.mat        key 'p_sdf'  (M, 4)
  transinfo/<n>.json            's', 'R', 't' for implicit_skel
  case_order.txt                latent-code index -> case name

    python npy_to_dgci.py <npy_dir> <out_root>
"""
import argparse  # command-line options
import json  # transinfo files
import sys  # exit code
from pathlib import Path  # path handling

import numpy as np  # array math
from scipy.io import savemat  # dataset.py reads .mat via scipy.io.loadmat
from scipy.spatial import cKDTree  # nearest surface point for the orientation check


def clean_surface(surface):
    finite = np.isfinite(surface).all(axis=1)  # drop rows with NaN/Inf anywhere
    surface = surface[finite]  # keep good rows only
    lengths = np.linalg.norm(surface[:, 3:], axis=1)  # normal lengths
    has_normal = lengths > 1e-6  # a zero normal makes cosine similarity undefined
    surface = surface[has_normal]  # keep rows with a usable normal
    surface[:, 3:] /= lengths[has_normal, None]  # force unit length ([:, None] = column)
    return surface, int((~finite).sum() + (~has_normal).sum())  # cleaned rows, dropped count


def clean_free(free):
    finite = np.isfinite(free).all(axis=1)  # drop rows with NaN/Inf
    # dataset.py reads sdf == 0 as "surface point" and sdf == -1 as "uniform, no ground truth"
    not_sentinel = (free[:, 3] != 0.0) & (free[:, 3] != -1.0)  # values that would be misread
    keep = finite & not_sentinel  # both conditions
    return free[keep], int((~keep).sum())  # cleaned rows, dropped count


def normal_agreement(surface, free, max_dist, n_check=20000):
    near = free[np.abs(free[:, 3]) < max_dist]  # only points close enough to judge
    if len(near) == 0:  # nothing to test against
        return float("nan"), 0  # caller reports "not checked"
    pick = np.random.choice(len(near), min(n_check, len(near)), replace=False)  # subsample
    near = near[pick]  # points used for the check
    tree = cKDTree(surface[:, :3])  # index of surface positions
    _, idx = tree.query(near[:, :3])  # nearest surface point for each free point
    offset = near[:, :3] - surface[idx, :3]  # vector surface -> free point
    side = np.einsum("ij,ij->i", offset, surface[idx, 3:])  # row-wise dot with the normal (einsum)
    agree = np.sign(side) == np.sign(near[:, 3])  # outside along +n should mean sdf > 0
    return float(agree.mean()), len(near)  # share of points where normals and SDF agree


def convert_case(name, npy_dir, out_root, args):
    surface = np.load(npy_dir / f"{name}_surface.npy").astype(np.float64)  # (N, 6)
    free = np.load(npy_dir / f"{name}_sdf.npy").astype(np.float64)  # (M, 4)
    norm = np.load(npy_dir / f"{name}_norm.npz")  # normalization from SampleVisibleMeshSurface
    offset = norm["offset"].reshape(3)  # = -center
    scale = float(norm["scale"].reshape(-1)[0])  # = 1 / (max_dist * 1.03)

    surface, bad_surf = clean_surface(surface)  # unit normals, no NaN
    free, bad_free = clean_free(free)  # no NaN, no sentinel collisions

    agree, n_checked = normal_agreement(surface, free, args.check_dist)  # orientation test
    flipped = False  # whether normals were inverted below
    if agree < 0.5:  # most normals point inward relative to the SDF sign
        surface[:, 3:] *= -1.0  # flip so grad(SDF) and normals point the same way
        agree = 1.0 - agree  # agreement after the flip
        flipped = True  # remember for the report

    radius = np.linalg.norm(surface[:, :3], axis=1).max()  # should be < 1 after normalization
    inside_share = float((free[:, 3] < 0).mean())  # share of free points inside the lumen

    savemat(out_root / "surface_pts_n_normal" / f"{name}.mat", {"p": surface.astype(np.float32)})  # S_i
    savemat(out_root / "free_space_pts" / f"{name}.mat", {"p_sdf": free.astype(np.float32)})  # free
    transinfo = {"s": scale, "R": np.eye(3).tolist(), "t": (offset * scale).tolist()}  # s*x + t
    with open(out_root / "transinfo" / f"{name}.json", "w") as f:  # implicit_skel reads this
        json.dump(transinfo, f)  # one file per case

    ok = agree >= args.min_agree and radius < 1.0  # pass/fail for this case
    flag = "OK  " if ok else "WARN"  # prefix for the log line
    note = " (normals flipped)" if flipped else ""  # mention a flip explicitly
    print(f"{flag} {name}: surf {len(surface)} (-{bad_surf}), free {len(free)} (-{bad_free}), "
          f"inside {inside_share:.2f}, max |x| {radius:.3f}, "
          f"normal/SDF agreement {agree:.3f} on {n_checked} pts{note}")  # one line per case
    return ok  # used for the exit code


def main():
    parser = argparse.ArgumentParser()  # CLI definition
    parser.add_argument("npy_dir")  # output dir of run_batch.sh
    parser.add_argument("out_root")  # DGCI data root
    # free points within this normalized distance are used to check normal orientation
    parser.add_argument("--check_dist", type=float, default=0.01)
    parser.add_argument("--min_agree", type=float, default=0.9)  # below this, flag the case
    parser.add_argument("--seed", type=int, default=0)  # reproducible check subsample
    args = parser.parse_args()  # read the arguments

    np.random.seed(args.seed)  # fixed subsample for the orientation check
    npy_dir = Path(args.npy_dir)  # input directory
    out_root = Path(args.out_root)  # output root
    for sub in ["surface_pts_n_normal", "free_space_pts", "transinfo"]:
        (out_root / sub).mkdir(parents=True, exist_ok=True)  # create the layout

    names = []  # cases with all three input files
    for surf_path in sorted(npy_dir.glob("*_surface.npy")):  # sorted = dataset.py's order
        name = surf_path.name[: -len("_surface.npy")]  # strip the suffix
        have_sdf = (npy_dir / f"{name}_sdf.npy").exists()  # PreprocessMesh output
        have_norm = (npy_dir / f"{name}_norm.npz").exists()  # normalization output
        if have_sdf and have_norm:  # complete case
            names.append(name)  # include it
        else:
            print(f"SKIP {name}: missing _sdf.npy or _norm.npz")  # incomplete run

    if not names:  # wrong directory or nothing finished yet
        print(f"ERROR: no complete cases in {npy_dir}")  # say why nothing happened
        sys.exit(1)  # fail loudly instead of writing an empty dataset

    results = [convert_case(n, npy_dir, out_root, args) for n in names]  # convert every case
    (out_root / "case_order.txt").write_text("\n".join(names) + "\n")  # latent index -> case
    print(f"\n{len(names)} cases written, {results.count(False)} flagged")  # summary
    print(f"config: point_cloud_path: {out_root / 'surface_pts_n_normal'}")  # what to paste
    print(f"        num_instances: {len(names)}")  # must match the case count
    sys.exit(0 if all(results) else 2)  # non-zero if any case needs a look


if __name__ == "__main__":
    main()  # run only when executed as a script
