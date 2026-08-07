from __future__ import annotations
"""
HDBSCAN clustering of the VAE latent embeddings produced by `latent_embeddings.extract_embeddings``.

The extractor saves an .npz with keys:
    mu        (N, num_latents)  float32   <- the matrix we cluster
    logsigma  (N, num_latents)  float32
    paths     (N,)              str

This module loads that file and runs HDBSCAN on ``mu`` exactly the way the
hdbscan "Basic Usage" tutorial does
(https://hdbscan.readthedocs.io/en/latest/basic_hdbscan.html):

    clusterer = hdbscan.HDBSCAN(min_cluster_size=...)
    clusterer.fit(X)
    clusterer.labels_          # cluster id per sample, -1 = noise
    clusterer.probabilities_   # soft membership score in [0, 1]

Clustering is run once, on every latent dimension.
"""

from pathlib import Path
import numpy as np
import hdbscan


def _run_hdbscan(X, *, min_cluster_size, min_samples, metric, cluster_selection_method, standardize):

    """Fit HDBSCAN on a matrix X and return a small result dict.
    This is the tutorial's core: build the clusterer, fit, read labels_ and
    probabilities_. Everything else here is bookkeeping.
    """
    if standardize:
        mean = X.mean(axis=0, keepdims=True)
        std  = X.std(axis=0, keepdims=True)
        std[std == 0] = 1.0                      # leave collapsed dims untouched
        X = (X - mean) / std

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
        cluster_selection_method=cluster_selection_method,
    )
    clusterer.fit(X)

    labels        = clusterer.labels_
    probabilities = clusterer.probabilities_
    n_clusters    = int(labels.max()) + 1 if labels.size and labels.max() >= 0 else 0
    n_noise       = int((labels == -1).sum())

    return {
        "labels": labels,
        "probabilities": probabilities,
        "clusterer": clusterer,
        "X": np.ascontiguousarray(X).astype(np.float32),
        "n_clusters": n_clusters,
        "n_noise": n_noise,
    }


def cluster_embeddings(
    embeddings_path,
    *,
    min_cluster_size: int = 7,
    min_samples: int | None = None,
    metric: str = "euclidean",
    cluster_selection_method: str = "eom",
    standardize: bool = False,
    use: str = "mu",
    verbose: bool = True,
):
    """Cluster VAE latent embeddings with HDBSCAN on all dimensions.

    Parameters
    ----------
    embeddings_path : str | Path
        Path to the ``embeddings.npz`` written by ``extract_embeddings``,
        OR the run directory that contains it (``run_.../embeddings.npz``).
    min_cluster_size : int
        Smallest grouping HDBSCAN will call a cluster (the main knob).
    min_samples : int | None
        Higher -> more points labelled noise (-1). ``None`` lets hdbscan
        default it to ``min_cluster_size``.
    metric : str
        Distance metric (default Euclidean, matching the tutorial).
    cluster_selection_method : {"eom", "leaf"}
        "eom" -> fewer, larger clusters; "leaf" -> more, finer clusters.
    standardize : bool
        Z-score each dimension before clustering. Off by default to stay
        faithful to the tutorial.
    use : {"mu", "logsigma"}
        Which stored matrix to cluster (you want "mu").
    verbose : bool
        Print a short summary of the run.

    Returns
    -------
    dict with keys:
        labels         (N,)  int      cluster id per sample, -1 = noise
        probabilities  (N,)  float    soft membership score in [0, 1]
        clusterer            HDBSCAN  the fitted clusterer object
        X              (N,D) float32  the matrix that was clustered
        n_clusters           int      number of clusters found
        n_noise              int      number of points labelled noise
        paths          (N,)  str      scan path per row (aligned to labels)
        variance       (D,)  float    per-dimension variance of `use` (diagnostic)
    """
    # ---- resolve the .npz path (accept a file or a run directory) ----
    p = Path(embeddings_path)
    if p.is_dir():
        p = p / "embeddings.npz"
    if not p.is_file():
        raise FileNotFoundError(f"No embeddings file at: {p}")

    data = np.load(p, allow_pickle=False)
    if use not in data.files:
        raise KeyError(f"'{use}' not in {p} (available: {list(data.files)})")

    X_full = np.ascontiguousarray(data[use]).astype(np.float32)
    paths = data["paths"] if "paths" in data.files else np.arange(len(X_full)).astype(str)

    if X_full.ndim != 2:
        raise ValueError(f"Expected a 2D (N, D) matrix, got shape {X_full.shape}")
    if len(paths) != len(X_full):
        raise ValueError(f"paths ({len(paths)}) and {use} rows ({len(X_full)}) disagree")

    # per-dimension variance across the dataset (kept as a diagnostic only;
    # high-beta VAEs collapse several dims to ~0 variance -> posterior collapse)
    variance = X_full.var(axis=0)

    if verbose:
        print(f"[cluster] {len(X_full)} samples from '{use}' in {p.name}")
        print(f"[cluster] {X_full.shape[1]} dims")

    res = _run_hdbscan( X_full,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
        cluster_selection_method=cluster_selection_method,
        standardize=standardize,)

    if verbose:
        frac_noise = res["n_noise"] / len(X_full) if len(X_full) else 0.0
        print(f"[ALL dims] {res['X'].shape[1]:>3d} dims -> "
              f"{res['n_clusters']} clusters, {res['n_noise']} noise "
              f"({frac_noise:.1%})")

    return {
        **res,
        "paths": paths,
        "variance": variance,
    }


if __name__ == "__main__":
    import argparse

# ===========================================================================
# Notebook helpers
# ===========================================================================
# These are convenience wrappers for interactive use. They keep heavy / optional
# imports (pandas, matplotlib, umap) INSIDE the functions, so importing this
# module in a script that only needs cluster_embeddings() stays lightweight.

def to_dataframe(result: dict):
    """Tidy the clustering result into a pandas DataFrame for inspection.

    Parameters
    ----------
    result : dict
        The object returned by cluster_embeddings().

    Returns
    -------
    pandas.DataFrame with columns: file, cluster, probability, path
    (sorted by cluster then descending membership probability).
    """
    import pandas as pd

    df = pd.DataFrame({
        "file": [Path(p).name for p in result["paths"]],
        "cluster": result["labels"],
        "probability": result["probabilities"],
        "path": [str(p) for p in result["paths"]],
    })
    return df.sort_values(["cluster", "probability"],
                          ascending=[True, False]).reset_index(drop=True)


def _reduce_2d(X, method: str, seed: int):
    """Project X to 2D for plotting. Falls back to PCA if UMAP isn't installed."""
    import numpy as _np

    if X.shape[1] == 1:                       # 1 dim: pad to 2D
        return _np.column_stack([X[:, 0], _np.zeros(len(X))])
    if X.shape[1] == 2:
        return X

    if method == "umap":
        try:
            import umap
            return umap.UMAP(n_components=2, random_state=seed).fit_transform(X)
        except Exception as e:               # not installed / failed -> PCA
            print(f"[plot] UMAP unavailable ({e}); using PCA instead.")

    from sklearn.decomposition import PCA
    return PCA(n_components=2, random_state=seed).fit_transform(X)


def plot_clusters(result: dict, method: str = "pca", figsize=(6, 5), seed: int = 0):
    """Scatter-plot the clustering in 2D, coloured by cluster (noise = grey).

    Parameters
    ----------
    result : dict
        The object returned by cluster_embeddings().
    method : {"pca", "umap"}
        2D projection for display only (does NOT affect the clustering, which
        was done in the full space). UMAP falls back to PCA if not installed.
    figsize, seed : passthrough for the figure / projection.

    Returns
    -------
    matplotlib Figure (also shown inline in a notebook).
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)

    emb2d = _reduce_2d(result["X"], method, seed)
    labels = result["labels"]
    n_clusters = result["n_clusters"]

    noise = labels == -1
    ax.scatter(emb2d[noise, 0], emb2d[noise, 1], s=12, c="lightgrey",
               label="noise", linewidths=0)
    for c in range(n_clusters):
        m = labels == c
        ax.scatter(emb2d[m, 0], emb2d[m, 1], s=16, linewidths=0,
                   label=f"cluster {c}")

    dims = result["X"].shape[1]
    ax.set_title(f"{dims}D -> {method.upper()}\n"
                 f"{n_clusters} clusters, {int(noise.sum())} noise")
    ax.set_xticks([]); ax.set_yticks([])
    if n_clusters <= 12:
        ax.legend(loc="best", fontsize=8, framealpha=0.6)

    fig.tight_layout()
    return fig