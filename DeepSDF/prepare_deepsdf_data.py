#!/usr/bin/env python3
"""Arrange PreprocessMesh outputs into the layout DeepSDF's training code expects.

Accepts either kind of PreprocessMesh output (pick files with --pattern):
  - <id>_sdf.npy with one (N, 4) array  (default; what run_batch.sh writes)
  - <id>.npz with "pos" / "neg" arrays  (use --pattern "*.npz" --strip_suffix "")

Produces:
  <data_dir>/SdfSamples/<dataset>/<class>/<id>.npz
  <splits_dir>/<prefix>_train.json and <prefix>_test.json
  <experiment_dir>/specs.json
"""

import argparse  # command-line options
import json  # split files and specs.json are JSON
import random  # shuffle subjects before splitting
from pathlib import Path  # path joins without os.path noise

import numpy as np  # read/write the .npy / .npz sample files


def load_samples(path):
    """Return (pos, neg) float32 arrays of rows [x, y, z, sdf]."""
    if path.suffix == ".npz":  # already in DeepSDF's format
        data = np.load(path)  # lazy archive of named arrays
        return data["pos"], data["neg"]  # outside / inside samples
    samples = np.load(path)  # single (N, 4) array from writeSDFToNPY
    positive_mask = samples[:, 3] > 0  # same sign rule as writeSDFToNPZ
    pos = samples[positive_mask]  # sdf > 0 -> outside the airway
    neg = samples[~positive_mask]  # sdf <= 0 -> inside the airway
    return pos, neg


def convert_all(sdf_dir, target_dir, pattern, strip_suffix, max_samples, rng):
    """Write one pos/neg .npz per subject; return the list of subject ids."""
    target_dir.mkdir(parents=True, exist_ok=True)  # create class folder (parents = whole chain)
    # run_batch.sh mixes <id>_sdf.npy, <id>_surface.npy and <id>_norm.npz in one folder
    sources = sorted(sdf_dir.glob(pattern))  # only the SDF files (glob, e.g. *_sdf.npy)
    subject_ids = []  # ids that produced usable data
    for source in sources:  # one file per subject
        pos, neg = load_samples(source)  # split into outside / inside
        pos = pos[~np.isnan(pos[:, 3])]  # drop NaN sdf rows (loader does too)
        neg = neg[~np.isnan(neg[:, 3])]  # same for inside samples
        total = len(pos) + len(neg)  # samples after NaN removal
        if max_samples and total > max_samples:  # smaller files -> much faster loading
            keep_share = max_samples / total  # same share from both sides keeps the ratio
            pos = pos[rng.choice(len(pos), int(len(pos) * keep_share), replace=False)]  # random subset
            neg = neg[rng.choice(len(neg), int(len(neg) * keep_share), replace=False)]  # random subset
        if len(pos) == 0 or len(neg) == 0:  # training draws half from each set
            print(f"SKIP  {source.name}: pos={len(pos)} neg={len(neg)}")  # report and move on
            continue
        subject_id = source.stem  # file name without extension
        if strip_suffix and subject_id.endswith(strip_suffix):  # "case01_sdf" -> "case01"
            subject_id = subject_id[: -len(strip_suffix)]  # cut the suffix off the end
        out_path = target_dir / f"{subject_id}.npz"  # DeepSDF looks up <id>.npz
        np.savez(out_path, pos=pos.astype(np.float32), neg=neg.astype(np.float32))  # loader keys
        inside_share = 100.0 * len(neg) / (len(pos) + len(neg))  # thin tubes -> small share
        print(f"OK    {subject_id}: pos={len(pos)} neg={len(neg)} ({inside_share:.1f}% inside)")
        subject_ids.append(subject_id)  # keep for the split files
    return subject_ids


def write_splits(subject_ids, args):
    """Shuffle subjects, split train/test, write DeepSDF split JSONs."""
    ids = sorted(subject_ids)  # fixed order so the seed is reproducible
    random.Random(args.seed).shuffle(ids)  # seeded shuffle (own RNG, no global state)
    num_test = int(round(len(ids) * args.test_fraction))  # size of the held-out set
    test_ids = sorted(ids[:num_test])  # first chunk -> test
    train_ids = sorted(ids[num_test:])  # remainder -> train
    args.splits_dir.mkdir(parents=True, exist_ok=True)  # split folder may not exist yet
    paths = {}  # "train"/"test" -> written path
    for name, subset in (("train", train_ids), ("test", test_ids)):  # same format for both
        split = {args.dataset_name: {args.class_name: subset}}  # {dataset: {class: [ids]}}
        path = args.splits_dir / f"{args.prefix}_{name}.json"  # e.g. airways_train.json
        path.write_text(json.dumps(split, indent=2))  # human-readable JSON
        paths[name] = path  # remembered for specs.json
    print(f"\nsplit: {len(train_ids)} train / {len(test_ids)} test")  # summary line
    return paths, len(train_ids)


def write_specs(args, split_paths, num_train):
    """Write specs.json based on DeepSDF's example, with batch size fitted to the data."""
    # DataLoader uses drop_last=True: a batch larger than the train set means zero iterations
    scenes_per_batch = min(args.scenes_per_batch, num_train)  # never exceed dataset size
    specs = {
        "Description": "DeepSDF on pulmonary airway trees.",  # train_deep_sdf.py expects a string
        "DataSource": str(args.data_dir),
        "TrainSplit": str(split_paths["train"]),
        "TestSplit": str(split_paths["test"]),
        "NetworkArch": "deep_sdf_decoder",
        "NetworkSpecs": {
            "dims": [512, 512, 512, 512, 512, 512, 512, 512],
            "dropout": [0, 1, 2, 3, 4, 5, 6, 7],
            "dropout_prob": 0.2,
            "norm_layers": [0, 1, 2, 3, 4, 5, 6, 7],
            "latent_in": [4],
            "xyz_in_all": False,
            "use_tanh": False,
            "latent_dropout": False,
            "weight_norm": True,
        },
        "CodeLength": 256,
        "NumEpochs": 2001,
        "SnapshotFrequency": 500,
        "AdditionalSnapshots": [100, 250],
        "LearningRateSchedule": [
            {"Type": "Step", "Initial": 0.0005, "Interval": 500, "Factor": 0.5},
            {"Type": "Step", "Initial": 0.001, "Interval": 500, "Factor": 0.5},
        ],
        "SamplesPerScene": 16384,
        "ScenesPerBatch": scenes_per_batch,
        "DataLoaderThreads": args.loader_threads,
        "ClampingDistance": 0.1,
        "LoadRam": True,  # read by the one-line patch to train_deep_sdf.py; ignored otherwise
        "CodeRegularization": True,
        "CodeRegularizationLambda": 1e-4,
        "CodeBound": 1.0,
    }  # values copied from examples/chairs/specs.json except where noted above
    args.experiment_dir.mkdir(parents=True, exist_ok=True)  # experiment folder
    specs_path = args.experiment_dir / "specs.json"  # the only file training needs
    specs_path.write_text(json.dumps(specs, indent=2))  # write it
    print(f"specs: {specs_path} (ScenesPerBatch={scenes_per_batch})")  # confirm


def main():
    parser = argparse.ArgumentParser(description=__doc__)  # docstring doubles as help text
    parser.add_argument("--sdf_dir", type=Path, required=True)  # PreprocessMesh outputs
    parser.add_argument("--pattern", default="*_sdf.npy")  # which files hold SDF samples
    parser.add_argument("--strip_suffix", default="_sdf")  # removed to get the subject id
    parser.add_argument("--max_samples", type=int, default=0)  # per-subject cap; 0 = keep all
    parser.add_argument("--data_dir", type=Path, default=Path("data"))  # DeepSDF DataSource
    parser.add_argument("--dataset_name", default="Airways")  # level 1 under SdfSamples
    parser.add_argument("--class_name", default="airway")  # level 2 under SdfSamples
    parser.add_argument("--splits_dir", type=Path, default=Path("examples/splits"))  # JSON home
    parser.add_argument("--prefix", default="airways")  # split file name prefix
    parser.add_argument("--experiment_dir", type=Path, default=Path("examples/airways"))
    parser.add_argument("--test_fraction", type=float, default=0.2)  # share held out
    parser.add_argument("--seed", type=int, default=0)  # makes the split reproducible
    parser.add_argument("--scenes_per_batch", type=int, default=64)  # upper bound
    parser.add_argument("--loader_threads", type=int, default=8)  # match --cpus-per-task
    args = parser.parse_args()  # read the command line

    target_dir = args.data_dir / "SdfSamples" / args.dataset_name / args.class_name  # loader path
    rng = np.random.default_rng(args.seed)  # reproducible subsampling
    subject_ids = convert_all(
        args.sdf_dir, target_dir, args.pattern, args.strip_suffix, args.max_samples, rng
    )  # step 1: data layout
    if not subject_ids:  # nothing usable found
        raise SystemExit(f"no usable {args.pattern} files in {args.sdf_dir}")  # stop early
    split_paths, num_train = write_splits(subject_ids, args)  # step 2: split files
    write_specs(args, split_paths, num_train)  # step 3: experiment config


if __name__ == "__main__":
    main()  # run only when executed, not when imported