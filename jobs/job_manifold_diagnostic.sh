#!/bin/bash
#SBATCH --job-name=manifold_diag
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=manifold_diag_%j.out

# Manifold analysis on 419-point latents is tiny — runs in <1 minute on CPU.

module load python/3.11.13
source /home/ids/gmargari-24/airway_project/3env/bin/activate
cd /home/ids/gmargari-24/airway_project/Encoding

# One-time installs if missing. matplotlib + scikit-learn are likely already
# present; umap-learn often isn't.
pip install umap-learn matplotlib --break-system-packages 2>&1 | tail -3

python manifold_diagnostic.py \
    --emb-a /home/ids/gmargari-24/airway_project/Data/latent_diag_5_50/latent_beta5.npz \
    --emb-b /home/ids/gmargari-24/airway_project/Data/latent_diag_5_50/latent_beta50.npz \
    --label-a beta5 \
    --label-b beta50 \
    --out-dir /home/ids/gmargari-24/airway_project/Data/manifold_diag_5_50
