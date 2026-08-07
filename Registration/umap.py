import numpy as np
import umap
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import hdbscan
from sklearn.metrics import silhouette_score


def apply_umap(embeddings_path: str, output_image: str = "umap_projection.png"):
    # 1. Load the data
    print(f"Loading embeddings from {embeddings_path}...")
    data = np.load(embeddings_path)
    mu = data['mu']         # The N x 100 feature matrix
    paths = data['paths']   # Array of file paths to map back to patients
    
    print(f"Loaded {mu.shape[0]} airway models of dimension {mu.shape[1]}")

    # 2. Configure and apply UMAP
    # Note: UMAP is highly sensitive to n_neighbors and min_dist. 
    print("Fitting UMAP manifold... (this might take a moment)")
    reducer = umap.UMAP(
        n_neighbors = 15,    # Balances local vs global structure, n_neig
        min_dist    = 0.0125,   # Controls cluster tightness
        n_components= 2,     # Target dimensions (2D for plotting)
        metric      = 'euclidean',  # Standard for VAE latent spaces
        random_state=42      # Fix the seed for reproducibility
    )
    
    embedding_2d = reducer.fit_transform(mu)
    print(f"Successfully projected data to shape: {embedding_2d.shape}")

    # 3. Visualize the result
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 8))
    
    # Create a scatter plot of the 2D embeddings
    plt.scatter(
        embedding_2d[:, 0], 
        embedding_2d[:, 1], 
        alpha=0.7, 
        s=15, 
        cmap='viridis')
    
    plt.title('UMAP Projection of Airway VAE Embeddings', fontsize=16)
    plt.xlabel('UMAP Dimension 1', fontsize=12)
    plt.ylabel('UMAP Dimension 2', fontsize=12)
    
    # Save and show
    plt.tight_layout()
    plt.savefig(output_image, dpi=300)
    print(f"Saved UMAP plot to {output_image}")
    
    # Optional: Return the projected array and paths for downstream HDBSCAN clustering
    return embedding_2d, paths




def apply_hdbscan(embedding_2d, output_image="hdbscan_clusters.png"):
    print("Fitting HDBSCAN clustering model...")

    # 1. Configure and apply HDBSCAN
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=7,      # Minimum points required to form a cluster
        min_samples=2,           # Controls how conservative the clustering is
        gen_min_span_tree=True,  # Required for relative_validity_ (DBCV)
    )
    cluster_labels = clusterer.fit_predict(embedding_2d)

    # Count clusters (excluding noise, label -1)
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_noise = int((cluster_labels == -1).sum())
    print(f"Found {n_clusters} clusters ({n_noise} noise points).")

    # --- Evaluation metrics ---
    # Silhouette: standard, but excludes noise and assumes globular clusters
    mask = cluster_labels != -1
    if n_clusters >= 2 and mask.sum() > n_clusters:
        sil = silhouette_score(embedding_2d[mask], cluster_labels[mask])
        print(f"Silhouette score (noise excluded): {sil:.4f}  "
              f"[{mask.sum()}/{len(cluster_labels)} points]")
    else:
        sil = None
        print("Silhouette score: not computable (need >= 2 clusters).")

    # DBCV: density-based, honest metric for HDBSCAN's cluster shapes
    dbcv = clusterer.relative_validity_
    print(f"DBCV (relative_validity_):         {dbcv:.4f}")

    # 2. Visualize the result
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 8))

    # Noise (-1) in light gray
    noise_mask = (cluster_labels == -1)
    plt.scatter(
        embedding_2d[noise_mask, 0],
        embedding_2d[noise_mask, 1],
        c='lightgray', alpha=0.5, s=10, label='Noise'
    )

    # Clusters (>= 0) in color
    cluster_mask = (cluster_labels >= 0)
    plt.scatter(
        embedding_2d[cluster_mask, 0],
        embedding_2d[cluster_mask, 1],
        c=cluster_labels[cluster_mask], cmap='Spectral', alpha=0.8, s=20
    )

    # Put the scores in the title so they're saved with the figure
    sil_str = f"{sil:.3f}" if sil is not None else "n/a"
    plt.title(
        f'HDBSCAN Clustering ({n_clusters} clusters) — '
        f'Silhouette={sil_str}, DBCV={dbcv:.3f}',
        fontsize=15
    )
    plt.xlabel('UMAP Dimension 1', fontsize=12)
    plt.ylabel('UMAP Dimension 2', fontsize=12)

    plt.tight_layout()
    plt.savefig(output_image, dpi=300)
    print(f"Saved HDBSCAN plot to {output_image}")

    return cluster_labels





import numpy as np
import umap
import hdbscan
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score
import heapq
import warnings

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(x, **kwargs):
        return x


def _plot_clustering(emb, labels, sil, dbcv, n_clusters, params, output_image):
    """Render one clustering result and save it."""
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 8))

    noise_mask = (labels == -1)
    plt.scatter(emb[noise_mask, 0], emb[noise_mask, 1],
                c='lightgray', alpha=0.5, s=10, label='Noise')

    cluster_mask = (labels >= 0)
    plt.scatter(emb[cluster_mask, 0], emb[cluster_mask, 1],
                c=labels[cluster_mask], cmap='Spectral', alpha=0.8, s=20)

    plt.title(
        f"n_neighbors={params[0]}, min_cluster_size={params[1]}, "
        f"min_samples={params[2]}\n"
        f"{n_clusters} clusters — Silhouette={sil:.3f}, DBCV={dbcv:.3f}",
        fontsize=13
    )
    plt.xlabel('UMAP Dimension 1', fontsize=12)
    plt.ylabel('UMAP Dimension 2', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_image, dpi=300)
    plt.show()


def sweep_umap_hdbscan(
    embeddings_path,
    n_neighbors_range=range(20, 21),        # UMAP needs >= 2 (1 to 419 requested)
    min_cluster_size_range=range(7, 11),    # HDBSCAN needs >= 2 (1 to 10 requested)
    min_samples_range=range(2, 5),         # 1 to 10
    top_k=10,
    output_prefix="best_clustering",
    results_csv="sweep_results.csv",
):
    # 1. Load data once
    data = np.load(embeddings_path)
    mu = data['mu']
    paths = data['paths']
    n_samples = mu.shape[0]
    print(f"Loaded {n_samples} models of dimension {mu.shape[1]}")

    results = []      # lightweight record of every config
    top = []          # min-heap of the best top_k (stores embeddings + labels)
    counter = 0       # tiebreaker so heap never compares dicts

    # 2. Outer loop: one UMAP fit per n_neighbors
    for n_nb in tqdm(list(n_neighbors_range), desc="UMAP n_neighbors"):
        if n_nb >= n_samples:
            continue  # n_neighbors must be < number of samples

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                reducer = umap.UMAP(
                    n_neighbors=n_nb,
                    min_dist=0.0125,
                    n_components=2,
                    metric='euclidean',
                    random_state=42,
                )
                emb = reducer.fit_transform(mu)
        except Exception as e:
            print(f"  UMAP failed at n_neighbors={n_nb}: {e}")
            continue

        # 3. Inner loop: cheap HDBSCAN fits on this embedding
        for mcs in min_cluster_size_range:
            for ms in min_samples_range:
                try:
                    clusterer = hdbscan.HDBSCAN(
                        min_cluster_size=mcs,
                        min_samples=ms,
                        gen_min_span_tree=True,
                    )
                    labels = clusterer.fit_predict(emb)
                except Exception:
                    continue

                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                mask = labels != -1

                if n_clusters >= 2 and mask.sum() > n_clusters:
                    sil = silhouette_score(emb[mask], labels[mask])
                else:
                    sil = np.nan  # silhouette undefined with < 2 clusters

                try:
                    dbcv = clusterer.relative_validity_
                except Exception:
                    dbcv = np.nan

                results.append({
                    'n_neighbors': n_nb,
                    'min_cluster_size': mcs,
                    'min_samples': ms,
                    'n_clusters': n_clusters,
                    'n_noise': int((labels == -1).sum()),
                    'silhouette': sil,
                    'dbcv': dbcv,
                })

                # Keep only the top_k by silhouette; copy embedding lazily
                if not np.isnan(sil):
                    if len(top) < top_k or sil > top[0][0]:
                        counter += 1
                        entry = (sil, counter, {
                            'params': (n_nb, mcs, ms),
                            'silhouette': sil, 'dbcv': dbcv,
                            'n_clusters': n_clusters,
                            'embedding': emb.copy(), 'labels': labels.copy(),
                        })
                        if len(top) < top_k:
                            heapq.heappush(top, entry)
                        else:
                            heapq.heapreplace(top, entry)

    # 4. Save the full results table
    try:
        import pandas as pd
        df = pd.DataFrame(results)
        df.sort_values('silhouette', ascending=False).to_csv(results_csv, index=False)
        print(f"\nSaved {len(df)} results to {results_csv}")
    except ImportError:
        df = None
        print("\npandas not available — skipping CSV export")

    # 5. Plot ONLY the top_k by silhouette, best first
    best = sorted(top, key=lambda x: x[0], reverse=True)
    print(f"\nTop {len(best)} configurations by silhouette score:")
    for rank, (sil, _, e) in enumerate(best, 1):
        n_nb, mcs, ms = e['params']
        print(f"  {rank}. n_neighbors={n_nb}, min_cluster_size={mcs}, "
              f"min_samples={ms} | silhouette={sil:.4f}, "
              f"dbcv={e['dbcv']:.4f}, clusters={e['n_clusters']}")
        _plot_clustering(
            e['embedding'], e['labels'], e['silhouette'], e['dbcv'],
            e['n_clusters'], e['params'],
            f"{output_prefix}_rank{rank}.png"
        )

    return df, best