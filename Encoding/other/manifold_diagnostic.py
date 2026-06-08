"""
Manifold diagnostic for VoxelVAE128 latent embeddings.

What it does
------------
Loads the .npz embeddings saved by latent_diagnostic.py and produces:

  1. PCA variance scree:    how concentrated is variance in the top few axes?
  2. UMAP 2D embedding:     does the latent form clusters, a continuous
                            manifold, or shapeless noise?
  3. t-SNE 2D embedding:    sanity check — same conclusion under a different
                            algorithm?
  4. Side-by-side PNG plots for both checkpoints (beta1 vs beta20).
  5. Top-component summary so you can later correlate the dominant latent
     axes with patient metadata (volume, dataset, etc.).

Why this matters
----------------
HDBSCAN said your latent doesn't have density-based clusters. Three
hypotheses remain:
  A. The data lies on a continuous manifold (no clusters to find).
  B. The latent is dominated by one global axis (size, position) and
     subgroup structure is buried below.
  C. The model has memorized individual patients without finding shared
     structure.

The PCA scree plot disambiguates A vs B: heavy concentration in PC1 -> B.
The UMAP plot disambiguates A vs C: smooth curve/cloud -> A; scattered
points with no shape -> C.

Usage
-----
    python manifold_diagnostic.py \\
        --emb-a /path/to/latent_beta1.npz \\
        --emb-b /path/to/latent_beta20.npz \\
        --label-a beta1 \\
        --label-b beta20 \\
        --out-dir /path/to/manifold_diag

Single embedding also works — omit --emb-b.
"""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np

# Lazy imports so we get clear error messages if anything is missing.
def _check_imports():
    missing = []
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        missing.append("matplotlib")
    try:
        import sklearn  # noqa: F401
    except ImportError:
        missing.append("scikit-learn")
    try:
        import umap  # noqa: F401
    except ImportError:
        missing.append("umap-learn")
    if missing:
        raise SystemExit(
            f"Missing packages: {missing}\n"
            f"Install with: pip install {' '.join(missing)} --break-system-packages"
        )


def load_embedding(path: Path) -> np.ndarray:
    """Load μ from the .npz, with the same fallback logic as the HDBSCAN script."""
    data = np.load(path, allow_pickle=False)
    keys = list(data.keys())
    print(f"Loaded {path.name}: keys = {keys}")
    for key in ("mu", "Z", "z", "mean", "latent", "embeddings"):
        if key in data:
            mu = data[key]
            print(f"  using '{key}' as mu, shape {mu.shape}")
            return mu
    # Fallback: any 2D float array
    for k in keys:
        arr = data[k]
        if arr.ndim == 2 and np.issubdtype(arr.dtype, np.floating):
            print(f"  using '{k}' as mu by fallback, shape {arr.shape}")
            return arr
    raise RuntimeError(f"No suitable mu array in {path}")


def pca_analysis(mu: np.ndarray, label: str) -> dict:
    """Run PCA and print variance concentration."""
    from sklearn.decomposition import PCA

    # Standardize so PC1 doesn't trivially capture whichever dim happens to
    # have the largest absolute scale.
    mu_std = (mu - mu.mean(axis=0)) / np.maximum(mu.std(axis=0), 1e-9)
    pca = PCA(n_components=min(mu.shape) - 1, random_state=0).fit(mu_std)
    var_ratio = pca.explained_variance_ratio_
    cumvar = np.cumsum(var_ratio)

    print()
    print(f"--- PCA on {label} (after per-dim standardization) ---")
    print(f"  Variance explained by PC1:           {var_ratio[0]*100:5.1f}%")
    print(f"  Variance explained by top 2 PCs:     {cumvar[1]*100:5.1f}%")
    print(f"  Variance explained by top 5 PCs:     {cumvar[4]*100:5.1f}%")
    print(f"  Variance explained by top 10 PCs:    {cumvar[9]*100:5.1f}%")
    print(f"  PCs needed for 50% variance:         {int(np.searchsorted(cumvar, 0.50) + 1)}")
    print(f"  PCs needed for 90% variance:         {int(np.searchsorted(cumvar, 0.90) + 1)}")
    print(f"  Effective rank (participation ratio): "
          f"{1.0 / (var_ratio ** 2).sum():.1f}")

    # Diagnostic verdict
    pc1_frac = var_ratio[0]
    if pc1_frac > 0.40:
        print(f"  -> PC1 dominates ({pc1_frac*100:.0f}% > 40%). Hypothesis B: "
              f"one global axis swamps subgroup structure.")
    elif pc1_frac < 0.10:
        print(f"  -> Variance is spread across many dims (PC1={pc1_frac*100:.0f}%). "
              f"No single global axis dominates.")
    else:
        print(f"  -> Moderate PC1 dominance ({pc1_frac*100:.0f}%). Mixed regime.")

    return {
        "var_ratio": var_ratio,
        "cumvar": cumvar,
        "pc1_frac": float(pc1_frac),
        "pcs_for_90pct": int(np.searchsorted(cumvar, 0.90) + 1),
        "mu_std": mu_std,
        "pca_coords_top5": pca.transform(mu_std)[:, :5],
    }


def umap_embed(mu: np.ndarray, label: str, seed: int = 0) -> np.ndarray:
    """Compute a 2D UMAP embedding. Uses sensible defaults for ~400-point data."""
    import umap
    # n_neighbors: balance local vs global structure. For 419 points,
    # 15-30 is the standard range; 15 reveals fine local structure,
    # 30 reveals broader manifold shape. We use 15.
    # min_dist: how tightly points are allowed to pack. 0.1 default works well.
    reducer = umap.UMAP(
        n_neighbors=15,
        min_dist=0.1,
        n_components=2,
        metric="euclidean",
        random_state=seed,
    )
    print(f"  Running UMAP on {label} ({mu.shape[0]} points, {mu.shape[1]}D)...")
    return reducer.fit_transform(mu)


def tsne_embed(mu: np.ndarray, label: str, seed: int = 0) -> np.ndarray:
    """Compute a 2D t-SNE embedding for cross-check."""
    from sklearn.manifold import TSNE
    # perplexity ~ 30 is the default and works well for ~400 points.
    tsne = TSNE(
        n_components=2,
        perplexity=30,
        learning_rate="auto",
        init="pca",
        random_state=seed,
    )
    print(f"  Running t-SNE on {label}...")
    return tsne.fit_transform(mu)


def plot_diagnostic(
    pca_info: dict,
    umap_xy: np.ndarray,
    tsne_xy: np.ndarray,
    label: str,
    out_path: Path,
) -> None:
    import matplotlib
    matplotlib.use("Agg")  # no display on cluster
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # 1. PCA scree
    ax = axes[0]
    var = pca_info["var_ratio"][:30]
    ax.bar(range(1, len(var) + 1), var * 100, color="steelblue")
    ax.axhline(5, color="gray", linestyle="--", linewidth=0.5,
               label="5% threshold")
    ax.set_xlabel("PC index")
    ax.set_ylabel("Variance explained (%)")
    ax.set_title(f"{label}: PCA scree (top 30)")
    ax.legend(fontsize=8)

    # 2. PCA cumulative
    ax = axes[1]
    ax.plot(range(1, len(pca_info["cumvar"]) + 1),
            pca_info["cumvar"] * 100,
            color="steelblue", linewidth=1.5)
    ax.axhline(90, color="red", linestyle="--", linewidth=0.5,
               label="90%")
    ax.axhline(50, color="orange", linestyle="--", linewidth=0.5,
               label="50%")
    ax.set_xlabel("Number of PCs")
    ax.set_ylabel("Cumulative variance (%)")
    ax.set_title(f"{label}: cumulative variance")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # 3. UMAP
    ax = axes[2]
    ax.scatter(umap_xy[:, 0], umap_xy[:, 1], s=12, alpha=0.6,
               c="steelblue", edgecolors="none")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.set_title(f"{label}: UMAP (n_neighbors=15)")
    ax.set_aspect("equal")

    # 4. t-SNE
    ax = axes[3]
    ax.scatter(tsne_xy[:, 0], tsne_xy[:, 1], s=12, alpha=0.6,
               c="darkorange", edgecolors="none")
    ax.set_xlabel("t-SNE-1")
    ax.set_ylabel("t-SNE-2")
    ax.set_title(f"{label}: t-SNE (perplexity=30)")
    ax.set_aspect("equal")

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved plot: {out_path}")


def interpret_umap_shape(umap_xy: np.ndarray, label: str) -> str:
    """Heuristic verdict on the UMAP layout. Replaces eyeballing for headless runs."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    # Compute a few simple shape diagnostics on the 2D UMAP coords:
    # - "elongation": ratio of std along the longer axis vs the shorter.
    #   Large -> linear/manifold-like; near 1 -> roughly isotropic blob.
    # - 2D silhouette: how clusterable does the *projection* itself look?
    #   This is generous (UMAP exaggerates clusters), so high value just
    #   means "looks visually grouped", not "real clusters in original".
    pc = umap_xy - umap_xy.mean(axis=0)
    cov = np.cov(pc.T)
    eigs = np.linalg.eigvalsh(cov)
    elongation = float(np.sqrt(eigs[1] / max(eigs[0], 1e-9)))

    sil_vals = []
    for k in [2, 3, 4, 5]:
        try:
            km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(umap_xy)
            sil_vals.append((k, float(silhouette_score(umap_xy, km.labels_))))
        except Exception:
            pass

    best_k, best_sil = max(sil_vals, key=lambda kv: kv[1]) if sil_vals else (None, 0.0)

    verdict = []
    verdict.append(f"\n--- UMAP shape verdict for {label} ---")
    verdict.append(f"  Elongation ratio (long/short axis): {elongation:.2f}")
    verdict.append(f"  Best 2D silhouette: {best_sil:.3f} at k={best_k}")

    if elongation > 3.5:
        verdict.append(f"  -> Strongly elongated. Likely a continuous "
                       f"manifold/trajectory, not clusters.")
    elif best_sil > 0.45:
        verdict.append(f"  -> Visually clustered in 2D projection. Even if "
                       f"HDBSCAN didn't find them in 100D, the *shape* is "
                       f"there — could be worth re-clustering on UMAP coords.")
    elif elongation < 1.8 and best_sil < 0.30:
        verdict.append(f"  -> Isotropic blob. Hypothesis A confirmed: data "
                       f"is one continuous cloud without subgroup structure.")
    else:
        verdict.append(f"  -> Intermediate shape. Inspect plot manually.")

    return "\n".join(verdict)


def run_one(emb_path: Path, label: str, out_dir: Path) -> dict:
    print()
    print("=" * 70)
    print(f"DIAGNOSING {label}")
    print("=" * 70)
    mu = load_embedding(emb_path)
    pca_info = pca_analysis(mu, label)
    umap_xy = umap_embed(pca_info["mu_std"], label)
    tsne_xy = tsne_embed(pca_info["mu_std"], label)

    plot_path = out_dir / f"manifold_{label}.png"
    plot_diagnostic(pca_info, umap_xy, tsne_xy, label, plot_path)

    print(interpret_umap_shape(umap_xy, label))

    # Save the 2D embeddings so you can recolor by metadata later
    np.savez(out_dir / f"embed2d_{label}.npz",
             umap=umap_xy, tsne=tsne_xy,
             pca_top5=pca_info["pca_coords_top5"])

    return {"pca_info": pca_info, "umap": umap_xy, "tsne": tsne_xy}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--emb-a", type=Path, required=True)
    p.add_argument("--emb-b", type=Path, default=None)
    p.add_argument("--label-a", type=str, default="A")
    p.add_argument("--label-b", type=str, default="B")
    p.add_argument("--out-dir", type=Path, default=Path("./manifold_diag"))
    return p.parse_args()


def main():
    _check_imports()
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    run_one(args.emb_a, args.label_a, args.out_dir)
    if args.emb_b is not None:
        run_one(args.emb_b, args.label_b, args.out_dir)

    print()
    print(f"Done. PNGs and 2D embeddings saved to {args.out_dir}")
    print(f"Inspect the PNGs visually — that's where the answer is.")


if __name__ == "__main__":
    main()
