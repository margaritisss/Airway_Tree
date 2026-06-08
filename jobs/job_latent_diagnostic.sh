#!/bin/bash
#SBATCH --job-name=latent_diag
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --partition=H100,A100,L40S
#SBATCH --output=latent_diag_%j.out

# Same env setup as training
module load python/3.11.13
source /home/ids/gmargari-24/airway_project/3env/bin/activate
export LD_LIBRARY_PATH=/projects/share/apps/miniconda3/25.5.1/lib:$LD_LIBRARY_PATH
cd /home/ids/gmargari-24/airway_project/Encoding
export PYTHONPATH=/home/ids/gmargari-24/airway_project:$PYTHONPATH

# Run on both checkpoints
python latent_diagnostic.py \
    --data-dirs /home/ids/gmargari-24/airway_project/Data/Registered_on_Template_22_23/Affine_registered/AIIB23_128 \
                /home/ids/gmargari-24/airway_project/Data/Registered_on_Template_22_23/Affine_registered/ATM22_128 \
    --ckpt-a /home/ids/gmargari-24/Data/vae_runs/beta_5/final.pt\
    --ckpt-b /home/ids/gmargari-24/Data/vae_runs/beta_50/final.pt \
    --label-a "beta5" \
    --label-b "beta50" \
    --out-dir /home/ids/gmargari-24/airway_project/Data/latent_diag_5_50