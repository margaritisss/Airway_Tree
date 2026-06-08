#!/bin/bash
#SBATCH --job-name=vae_hpo
#SBATCH --output=hpo_logs/%x_%A_%a.out
#SBATCH --error=hpo_logs/%x_%A_%a.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=23:00:00
#SBATCH --partition=H100,A100,L40S
#SBATCH --exclude=node54
#SBATCH --array=1-8%8
# Array of 4 workers, max 4 running at once. Adjust to your cluster's
# per-user GPU quota. SQLite on NFS handles ~10 concurrent workers; go
# higher and switch to PostgreSQL.

export PYTHONUNBUFFERED=1

echo "Worker array task $SLURM_ARRAY_TASK_ID started on $(hostname) at $(date)"
echo "GPU(s): $CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo "---"

module load python/3.11.13
source /home/ids/gmargari-24/airway_project/3env/bin/activate
export LD_LIBRARY_PATH=/projects/share/apps/miniconda3/25.5.1/lib:$LD_LIBRARY_PATH

cd /home/ids/gmargari-24/airway_project/Encoding
export PYTHONPATH=/home/ids/gmargari-24/airway_project:$PYTHONPATH

# Shared across all workers in this study
STUDY_DIR=/home/ids/gmargari-24/Data/vae_optuna
mkdir -p "$STUDY_DIR" "$STUDY_DIR/logs"
STUDY_NAME=voxelvae128_v2
STORAGE="sqlite:///${STUDY_DIR}/${STUDY_NAME}.db"

NUM_WORKERS=$(( SLURM_CPUS_PER_TASK - 1 ))

time python -u run_optuna_worker.py \
    --study-name "$STUDY_NAME" \
    --storage    "$STORAGE" \
    --data-dirs  /home/ids/gmargari-24/airway_project/Data/Registered_on_Template_22_23/Affine_registered/AIIB23_128 \
                 /home/ids/gmargari-24/airway_project/Data/Registered_on_Template_22_23/Affine_registered/ATM22_128 \
    --num-workers $NUM_WORKERS \
    --max-epochs 80 \
    --n-trials 15 \
    --global-trial-cap 10000 \
    --soft-deadline-seconds 82800 \
    --log-dir "$STUDY_DIR/logs"

echo "Worker array task $SLURM_ARRAY_TASK_ID finished at $(date)"
