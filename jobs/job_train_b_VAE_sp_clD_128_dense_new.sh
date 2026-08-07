#!/bin/bash
#SBATCH --job-name=new_128_L2048_b4
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --partition=L40S,A100
#SBATCH --exclude=node54
#Unbuffered Python stdout/stderr so the .out file updates live.
export PYTHONUNBUFFERED=1

echo "Job $SLURM_JOB_ID started on $(hostname) at $(date)"
echo "Allocated CPUs: $SLURM_CPUS_PER_TASK"
echo "GPU(s): $CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo "---"

module load python/3.11.13
module load cuda/12.1                    
source /home/ids/gmargari-24/airway_project/new_3env/bin/activate
export LD_LIBRARY_PATH=/projects/share/apps/miniconda3/25.5.1/lib:$LD_LIBRARY_PATH

cd /home/ids/gmargari-24/airway_project/Encoding
export PYTHONPATH=/home/ids/gmargari-24/airway_project:$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODE=${1:-fresh}     
OUT_DIR=/home/ids/gmargari-24/airway_project/Data/vae_runs/new_128_L2048_b4
mkdir -p "$OUT_DIR"
# ---- Weights & Biases ----
export WANDB_DIR="$OUT_DIR"     # keep run data beside the checkpoints (and for `wandb sync`)
# If the compute node has no outbound internet, uncomment to log offline and
# `wandb sync "$OUT_DIR/wandb/latest-run"` from a login node afterwards:
# export WANDB_MODE=offline

CONT_ARG=""
[ "$MODE" = "continue" ] && CONT_ARG="--continue"

NUM_WORKERS=$(( SLURM_CPUS_PER_TASK - 1 ))

time python -u train_β_VAE_clD_scale_128_new.py \
    --data-dirs /home/ids/gmargari-24/airway_project/Data/Registered_on_Template_22_23/Affine_registered/AIIB23_128 \
                /home/ids/gmargari-24/airway_project/Data/Registered_on_Template_22_23/Affine_registered/ATM22_128 \
    --out-dir "$OUT_DIR" \
    --batch-size 12 \
    --max-epochs 7000 \
    --num-workers $NUM_WORKERS \
    --num-latents 2048     \
    --seed 0 \
    --beta 4 \
    --wandb --wandb-project airway-bvae \
    $CONT_ARG

echo "Job finished at $(date)"