"""
HDBSCAN diagnostic for VoxelVAE128 latent embeddings.

What it does
------------
Loads the .npz embedding files saved by latent_diagnostic.py and runs
HDBSCAN with a small parameter sweep on each. Reports:

  - Number of clusters found (excluding the noise class -1)
  - Noise fraction (points HDBSCAN refused to cluster)
  - Cluster size distribution
  - Density-Based Cluster Validity (DBCV) score — the HDBSCAN-native
    quality metric, analogous to silhouette but for density-based clusters

Why HDBSCAN and not just more k-means
-------------------------------------
K-means assumes clusters are roughly spherical and of similar size. For
disease-driven morphological variation that is rarely true: some
phenotypes are common, others are rare; some are tight, others are diffuse.
HDBSCAN finds clusters of varying density without assuming a shape, and
it can refuse to cluster ambiguous points (marking them as noise), which
is more honest than forcing every point into a group.

Why a parameter sweep
---------------------
min_cluster_size determines the smallest phenotype HDBSCAN will report.
For 419 patients:
  - min_cluster_size=10 → smallest cluster = 2.4% of cohort (rare phenotypes)
  - min_cluster_size=20 → 4.8% of cohort (uncommon phenotypes)
  - min_cluster_size=30 → 7.2% (common phenotypes only)
  - min_cluster_size=50 → 12% (very coarse partition)

min_samples controls how conservative the algorithm is about density.
Higher values mean more points get labeled as noise, but the clusters
that remain are tighter and more reliable.

Usage
-----
    python hdbscan_diagnostic.py \\
        --emb-a /path/to/latent_beta1.npz \\
        --emb-b /path/to/latent_beta20.npz \\
        --label-a beta1 \\
        --label-b beta20 \\
        --out-dir /path/to/hdbscan_diag

Single embedding also works — omit --emb-b.

Notes on data prep
------------------
HDBSCAN uses Euclidean distance by default, which is fine for VAE latents
in principle but sensitive to scale differences across dimensions. We
standardize (zero mean, unit variance per dim) before clustering, which
is the safe default for latents where some dims are more active than
others.
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np

try:
    import hdbscan
except ImportError:
    raise SystemExit(
        "hdbscan not installed. Install with:\n"
        "    pip install hdbscan --break-system-packages"
    )


def load_embedding(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    """Load latent embedding from .npz, tolerant of different key names.

    Tries the expected keys first ('mu', 'logsigma'). Falls back to inspecting
    the archive and picking 2D float arrays of the same shape, treating the
    first as mu and the second as logsigma. If only one array is present we
    return it as mu and None for logsigma — HDBSCAN only needs mu.
    """
    data = np.load(path, allow_pickle=False)
    keys = list(data.keys())
    print(f"Loaded {path.name}: contains keys {keys}")

    # Preferred: explicit keys
    mu = logsigma = None
    for mu_key in ("mu", "Z", "z", "mean", "latent", "embeddings"):
        if mu_key in data:
            mu = data[mu_key]
            break
    for ls_key in ("logsigma", "log_sigma", "logvar", "log_var", "sigma"):
        if ls_key in data:
            logsigma = data[ls_key]
            break

    # Fallback: pick by shape. We expect 2D float arrays with first dim = N patients.
    if mu is None:
        candidates = [(k, data[k]) for k in keys
                      if data[k].ndim == 2 and np.issubdtype(data[k].dtype, np.floating)]
        if not candidates:
            raise RuntimeError(
                f"No 2D float array in {path}. Keys+shapes: "
                f"{[(k, data[k].shape, data[k].dtype) for k in keys]}"
            )
        # Heuristic: mu first (alphabetic key), but really we just need any 2D float
        candidates.sort(key=lambda kv: kv[0])
        mu = candidates[0][1]
        print(f"  picked '{candidates[0][0]}' as mu by fallback (shape {mu.shape})")
        if len(candidates) > 1 and candidates[1][1].shape == mu.shape:
            logsigma = candidates[1][1]
            print(f"  picked '{candidates[1][0]}' as logsigma by fallback")

    print(f"  mu shape: {mu.shape}"
          + (f", logsigma shape: {logsigma.shape}" if logsigma is not None else ", logsigma: not found (OK, not needed for HDBSCAN)"))
    return mu, logsigma


def standardize(X: np.ndarray) -> np.ndarray:
    """Zero mean, unit variance per dimension. Avoid divide-by-zero on dead dims."""
    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, ddof=1, keepdims=True)
    std = np.maximum(std, 1e-9)
    return (X - mean) / std


def summarize_clustering(labels: np.ndarray) -> dict:
    """Cluster counts and size distribution, excluding the noise class -1."""
    unique, counts = np.unique(labels, return_counts=True)
    noise_count = int(counts[unique == -1].sum()) if -1 in unique else 0
    cluster_mask = unique != -1
    cluster_counts = counts[cluster_mask]
    return {
        "n_clusters": int(cluster_mask.sum()),
        "n_noise": noise_count,
        "noise_frac": float(noise_count / len(labels)),
        "cluster_sizes": sorted(cluster_counts.tolist(), reverse=True),
        "largest_cluster_frac": float(cluster_counts.max() / len(labels)) if len(cluster_counts) > 0 else 0.0,
    }


def run_hdbscan_sweep(X: np.ndarray, label: str) -> dict:
    """Run HDBSCAN at several (min_cluster_size, min_samples) settings."""
    print()
    print("=" * 80)
    print(f"HDBSCAN sweep — {label}  (N={X.shape[0]}, D={X.shape[1]})")
    print("=" * 80)

    # Grid: vary cluster-size threshold (coarseness) and min_samples (strictness)
    mcs_grid = [10, 15, 20, 30, 50]
    ms_grid  = [5, 10]

    results = []
    print(f"{'min_clust_size':>14} {'min_samples':>11} {'n_clusters':>10} "
          f"{'noise%':>8} {'biggest%':>9} {'DBCV':>8}  cluster sizes (top 6)")
    print("-" * 90)

    for mcs in mcs_grid:
        for ms in ms_grid:
            try:
                clusterer = hdbscan.HDBSCAN(
                    min_cluster_size=mcs,
                    min_samples=ms,
                    metric="euclidean",
                    cluster_selection_method="eom",  # excess of mass — the default,
                                                     # gives more stable clusters than 'leaf'
                    gen_min_span_tree=True,          # required for DBCV computation
                )
                labels = clusterer.fit_predict(X)
                summary = summarize_clustering(labels)
                # DBCV (Density-Based Cluster Validity) — relative_validity_ attribute
                # is HDBSCAN's built-in implementation. Range roughly [-1, 1], higher better.
                # Can fail / return nan if clusters are too few or too small.
                try:
                    dbcv = float(clusterer.relative_validity_)
                except Exception:
                    dbcv = float("nan")

                top_sizes = summary["cluster_sizes"][:6]
                top_str = " ".join(f"{s}" for s in top_sizes) if top_sizes else "(none)"
                print(f"{mcs:>14} {ms:>11} {summary['n_clusters']:>10} "
                      f"{summary['noise_frac']*100:>7.1f}% "
                      f"{summary['largest_cluster_frac']*100:>8.1f}% "
                      f"{dbcv:>8.4f}  {top_str}")

                results.append({
                    "min_cluster_size": mcs,
                    "min_samples": ms,
                    "labels": labels.tolist(),
                    "dbcv": dbcv,
                    **summary,
                })
            except Exception as e:
                print(f"{mcs:>14} {ms:>11}  FAILED: {e}")

    return {"label": label, "results": results}


def select_best_setting(diag: dict) -> dict | None:
    """Heuristic 'best' choice for the side-by-side: highest DBCV among settings
    that produce >=2 clusters and <70% noise. This is a defensible default
    but not gospel — the real choice depends on what cluster granularity
    you care about for the downstream clinical question."""
    valid = [
        r for r in diag["results"]
        if r["n_clusters"] >= 2 and r["noise_frac"] < 0.7 and not np.isnan(r["dbcv"])
    ]
    if not valid:
        return None
    return max(valid, key=lambda r: r["dbcv"])


def compare_runs(diag_a: dict, diag_b: dict) -> None:
    print()
    print("=" * 80)
    print(f"COMPARISON  —  {diag_a['label']}  vs  {diag_b['label']}")
    print("=" * 80)

    best_a = select_best_setting(diag_a)
    best_b = select_best_setting(diag_b)

    print(f"\nBest setting per run (criterion: highest DBCV with >=2 clusters,")
    print(f"<70% noise):\n")
    for label, best in [(diag_a["label"], best_a), (diag_b["label"], best_b)]:
        if best is None:
            print(f"  {label}: no valid clustering found across the sweep")
            continue
        print(f"  {label}: mcs={best['min_cluster_size']}, ms={best['min_samples']}")
        print(f"    -> {best['n_clusters']} clusters, "
              f"{best['noise_frac']*100:.1f}% noise, DBCV={best['dbcv']:.4f}")
        print(f"    cluster sizes: {best['cluster_sizes']}")

    if best_a and best_b:
        print()
        print("INTERPRETATION")
        print("-" * 80)
        if best_a["dbcv"] > best_b["dbcv"] + 0.05:
            print(f"{diag_a['label']} produces clearly better clusters under HDBSCAN.")
            print(f"This contradicts the k-means result — HDBSCAN found density")
            print(f"structure in {diag_a['label']} that k-means missed.")
        elif best_b["dbcv"] > best_a["dbcv"] + 0.05:
            print(f"{diag_b['label']} produces clearly better clusters under HDBSCAN.")
            print(f"This agrees with the k-means result.")
        else:
            print(f"Both runs produce similar clustering quality under HDBSCAN.")
            print(f"DBCV difference < 0.05 is within noise for a dataset of this size.")
            print(f"The qualitative difference is more in cluster *granularity*:")
            print(f"compare the cluster sizes above to see which run gives the")
            print(f"phenotype resolution you want.")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--emb-a", type=Path, required=True,
                   help="First embedding .npz (saved by latent_diagnostic.py)")
    p.add_argument("--emb-b", type=Path, default=None,
                   help="Optional second embedding .npz")
    p.add_argument("--label-a", type=str, default="A")
    p.add_argument("--label-b", type=str, default="B")
    p.add_argument("--out-dir", type=Path, default=Path("./hdbscan_diag"))
    p.add_argument("--no-standardize", action="store_true",
                   help="Skip per-dim standardization (default: standardize)")
    return p.parse_args()


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # --- Run A ---
    mu_a, _ = load_embedding(args.emb_a)
    X_a = mu_a if args.no_standardize else standardize(mu_a)
    diag_a = run_hdbscan_sweep(X_a, args.label_a)

    diag_b = None
    if args.emb_b is not None:
        mu_b, _ = load_embedding(args.emb_b)
        X_b = mu_b if args.no_standardize else standardize(mu_b)
        diag_b = run_hdbscan_sweep(X_b, args.label_b)
        compare_runs(diag_a, diag_b)

    # --- Save summary so you can inspect cluster labels later ---
    # Drop the full label arrays from JSON (they're long); save them as .npy.
    def strip_labels(d):
        return {**d, "results": [{k: v for k, v in r.items() if k != "labels"}
                                  for r in d["results"]]}

    summary = {"a": strip_labels(diag_a)}
    if diag_b is not None:
        summary["b"] = strip_labels(diag_b)

    # Save best-setting label vectors for both runs (you'll want these for plots)
    for label_diag, name in [(diag_a, args.label_a), (diag_b, args.label_b)]:
        if label_diag is None:
            continue
        best = select_best_setting(label_diag)
        if best is not None:
            np.save(args.out_dir / f"hdbscan_labels_{name}.npy",
                    np.array(best["labels"]))

    with open(args.out_dir / "hdbscan_summary.json", "w") as f:
        json.dump(summary, f, indent=2,
                  default=lambda o: float(o) if isinstance(o, np.floating) else o)

    print(f"\nSaved summary + best-setting label vectors to {args.out_dir}")


if __name__ == "__main__":
    main()
