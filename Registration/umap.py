import numpy as np
import umap
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

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
        n_neighbors = 5,    # Balances local vs global structure, n_neig
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


import hdbscan

def apply_hdbscan(embedding_2d, output_image="hdbscan_clusters.png"):
    print("Fitting HDBSCAN clustering model...")
    
    # 1. Configure and apply HDBSCAN
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size= 6,  # The minimum number of points required to form a cluster
        min_samples     = 4,   # Controls how conservative the clustering is
    )
    
    # Fit the model and get cluster labels
    cluster_labels = clusterer.fit_predict(embedding_2d)
    
    # Count how many clusters were found (excluding noise)
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    print(f"Found {n_clusters} clusters.")
    
    # 2. Visualize the result
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 8))
    
    # HDBSCAN assigns a label of -1 to noise (points that don't belong to any cluster).
    # We want to plot noise points in light gray, and clustered points in color.
    # Plot Noise (-1)
    noise_mask = (cluster_labels == -1)
    plt.scatter(
        embedding_2d[noise_mask, 0], 
        embedding_2d[noise_mask, 1], 
        c='lightgray', 
        alpha=0.5, 
        s=10, 
        label='Noise'
    )
    
    # Plot Clusters (everything >= 0)
    cluster_mask = (cluster_labels >= 0)
    plt.scatter(
        embedding_2d[cluster_mask, 0], 
        embedding_2d[cluster_mask, 1], 
        c=cluster_labels[cluster_mask], 
        cmap='Spectral', 
        alpha=0.8, 
        s=20
    )
    
    plt.title(f'HDBSCAN Clustering (Found {n_clusters} Clusters)', fontsize=16)
    plt.xlabel('UMAP Dimension 1', fontsize=12)
    plt.ylabel('UMAP Dimension 2', fontsize=12)
    
    # Save and show
    plt.tight_layout()
    plt.savefig(output_image, dpi=300)
    print(f"Saved HDBSCAN plot to {output_image}")
    
    return cluster_labels
