"""
visualize_embeddings.py

Drop-in helper for a Jupyter notebook. Loads the embeddings.npz produced by
extract_embeddings.py, preprocesses the latent `mu` matrix correctly for t-SNE,
and returns interactive 2D + 3D scatter plots.

Usage in a notebook cell
------------------------
    from visualize_embeddings import tsne_visualize
    result = tsne_visualize("embeddings.npz")          # shows 2D + 3D plots
    # result is a dict with the t-SNE coords + metadata if you want to reuse them

Dependencies (install once in the notebook):
    !pip install numpy scikit-learn plotly
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import plotly.express as px
import plotly.graph_objects as go


def _infer_source(paths: np.ndarray) -> np.ndarray:
    """
    Tag each row with its source cohort so we can color the plot and catch the
    AIIB23 / ATM22 domain-shift confound. Falls back to the parent folder name
    if neither token is present.
    """
    labels = []
    for p in paths:
        s = str(p)
        low = s.lower()
        if "aiib23" in low:
            labels.append("AIIB23")
        elif "atm22" in low:
            labels.append("ATM22")
        else:
            labels.append(Path(s).parent.name or "unknown")
    return np.array(labels)


def tsne_visualize(
    npz_path: str | Path,
    *, # 
    perplexity: float = 15.0,
    active_var_frac: float = 0.05,
    standardize: bool = True,
    color_by: np.ndarray | None = None,
    color_name: str = "source",
    seed: int = 0,
    show: bool = True,
):
    """
    Load embeddings and produce 2D and 3D t-SNE visualizations.

    Parameters
    ----------
    npz_path        : path to embeddings.npz (must contain `mu`; `paths` optional).
    perplexity      : t-SNE perplexity. Auto-clamped to < n_samples/3.
                      Try 5, 30, 50 and compare; structure is perplexity-sensitive.
    active_var_frac : keep only latent dims whose variance exceeds this fraction
                      of the max-variance dim. Filters out collapsed dimensions.
                      Set to 0.0 to keep all dims.
    standardize     : z-score each kept dim before t-SNE (recommended).
    color_by        : optional length-N array of labels/values to color points by
                      (e.g. disease label, lung volume). If None, colors by inferred
                      source cohort.
    color_name      : legend / colorbar title for `color_by`.
    seed            : random_state for reproducibility.
    show            : if True, call fig.show() on both plots.

    Returns
    -------
    dict with keys:
        coords_2d (N,2), coords_3d (N,3), color (N,), paths (N,),
        active_dims (list[int]), fig_2d, fig_3d
    """
    npz_path = Path(npz_path)
    data = np.load(npz_path, allow_pickle=True)

    mu = np.asarray(data["mu"], dtype=np.float32)
    if mu.ndim != 2:
        raise ValueError(f"Expected mu of shape (N, D), got {mu.shape}")
    n_samples, n_dims = mu.shape
    print(f"[info] loaded mu: {n_samples} samples x {n_dims} latent dims")

    paths = data["paths"] if "paths" in data else np.array(
        [f"sample_{i}" for i in range(n_samples)]
    )

    # ---- active-dim filtering (posterior collapse check) ----
    var = mu.var(axis=0)
    if active_var_frac > 0 and var.max() > 0:
        active_mask = var > active_var_frac * var.max()
    else:
        active_mask = np.ones(n_dims, dtype=bool)
    active_dims = np.where(active_mask)[0].tolist()
    print(f"[diag] {len(active_dims)}/{n_dims} dims active "
          f"(var > {active_var_frac:.0%} of max). "
          f"max var={var.max():.4f}, min var={var.min():.4f}")
    if len(active_dims) < 2:
        print("[warn] <2 active dims -> severe posterior collapse; "
              "t-SNE will not be meaningful. Keeping all dims as a fallback.")
        active_dims = list(range(n_dims))

    X = mu[:, active_dims]

    # ---- standardize ----
    if standardize:
        X = StandardScaler().fit_transform(X)

    # ---- color labels ----
    if color_by is not None:
        color = np.asarray(color_by)
        if color.shape[0] != n_samples:
            raise ValueError(
                f"color_by has length {color.shape[0]}, expected {n_samples}")
        legend_title = color_name
    else:
        color = _infer_source(paths)
        legend_title = "source"

    # ---- perplexity guard (t-SNE needs perplexity < n_samples/3 roughly) ----
    max_perp = max(5.0, (n_samples - 1) / 3.0)
    perp     = min(perplexity, max_perp)
    if perp != perplexity:
        print(f"[info] perplexity clamped {perplexity} -> {perp:.1f} " f"for n_samples={n_samples}")

    # ---- run t-SNE in 2D and 3D (separate fits) ----
    print("[info] running t-SNE (2D)...")
    coords_2d = TSNE(
        n_components=2, perplexity=perp, init="pca",
        random_state=seed, max_iter=1000,
    ).fit_transform(X)

    print("[info] running t-SNE (3D)...")
    coords_3d = TSNE(
        n_components=3, perplexity=perp, init="pca",
        random_state=seed, max_iter=1000,
    ).fit_transform(X)

    short_paths = [Path(str(p)).name for p in paths]

    # ---- 2D plot ----
    fig_2d = px.scatter(
        x=coords_2d[:, 0], y=coords_2d[:, 1],
        color=color, hover_name=short_paths,
        labels={"x": "t-SNE 1", "y": "t-SNE 2", "color": legend_title},
        title=f"t-SNE (2D) of latent mu  —  {len(active_dims)} active dims, "
              f"perplexity={perp:.0f}",
    )
    fig_2d.update_traces(marker=dict(size=7, opacity=0.8, line=dict(width=0.5, color="white")))
    fig_2d.update_layout(width=800, height=650, legend_title_text=legend_title)

    # ---- 3D plot ----
    fig_3d = px.scatter_3d(
        x=coords_3d[:, 0], y=coords_3d[:, 1], z=coords_3d[:, 2],
        color=color, hover_name=short_paths,
        labels={"x": "t-SNE 1", "y": "t-SNE 2", "z": "t-SNE 3",
                "color": legend_title},
        title=f"t-SNE (3D) of latent mu  —  {len(active_dims)} active dims, "
              f"perplexity={perp:.0f}",
    )
    fig_3d.update_traces(marker=dict(size=4, opacity=0.8))
    fig_3d.update_layout(width=850, height=700, legend_title_text=legend_title)

    if show:
        fig_2d.show()
        fig_3d.show()

    return {
        "coords_2d": coords_2d,
        "coords_3d": coords_3d,
        "color": color,
        "paths": paths,
        "active_dims": active_dims,
        "fig_2d": fig_2d,
        "fig_3d": fig_3d,
    }


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "embeddings.npz"
    tsne_visualize(path)
