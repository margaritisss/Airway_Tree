#!/bin/bash
#SBATCH --job-name=hdbscan_diag
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=hdbscan_diag_%j.out
# HDBSCAN on the 419x100 latent embeddings is tiny — runs in seconds on CPU.
# No GPU needed.

module load python/3.11.13
source /home/ids/gmargari-24/airway_project/3env/bin/activate
cd /home/ids/gmargari-24/airway_project/Encoding

python hdbscan_diagnostic.py \
    --emb-a /home/ids/gmargari-24/airway_project/Data/latent_diag_5_50/latent_beta5.npz \
    --emb-b /home/ids/gmargari-24/airway_project/Data/latent_diag_5_50/latent_beta50.npz \
    --label-a beta5 \
    --label-b beta50 \
    --out-dir /home/ids/gmargari-24/airway_project/Data/hdbscan_diag_5_50
