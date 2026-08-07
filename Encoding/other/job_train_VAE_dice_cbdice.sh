#!/bin/bash
#SBATCH --job-name=train_vae128_dice_cbdice
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --partition=H100,A100,L40S
#SBATCH --exclude=node54

# Unbuffered Python stdout/stderr so the .out file updates live.
export PYTHONUNBUFFERED=1

echo "Job $SLURM_JOB_ID started on $(hostname) at $(date)"
echo "Allocated CPUs: $SLURM_CPUS_PER_TASK"
echo "GPU(s): $CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo "---"
echo "Loss configuration: Dice + cbDice (replaces previous weighted-BCE + cbDice)"
echo "Reconstruction term: alpha_dice * SoftDice + (1 - alpha_dice) * cbDice"
echo "alpha_dice schedule: 1.0 for 15 epochs, ramp 1.0 -> 0.5 over 10 epochs, then 0.5"
echo "beta_max: 1.0 (lowered from 50 to avoid KL collapse)"
echo "---"

module load python/3.11.13
source /home/ids/gmargari-24/airway_project/3env/bin/activate
export LD_LIBRARY_PATH=/projects/share/apps/miniconda3/25.5.1/lib:$LD_LIBRARY_PATH

cd /home/ids/gmargari-24/airway_project/Encoding
export PYTHONPATH=/home/ids/gmargari-24/airway_project:$PYTHONPATH

OUT_DIR=/home/ids/gmargari-24/airway_project/Data/vae_runs/run_${SLURM_JOB_ID}
mkdir -p "$OUT_DIR"

NUM_WORKERS=$(( SLURM_CPUS_PER_TASK - 1 ))

time python -u train_VAE_cbdice_DICE.py \
    --data-dirs /home/ids/gmargari-24/airway_project/Data/Registered_on_Template_22_23/Affine_registered/AIIB23_128 \
                /home/ids/gmargari-24/airway_project/Data/Registered_on_Template_22_23/Affine_registered/ATM22_128 \
    --out-dir   "$OUT_DIR" \
    --batch-size 4 \
    --max-epochs 150 \
    --num-workers $NUM_WORKERS \
    --num-latents 100 \
    --seed 0 \
    --data-augmentation False

echo "Job finished at $(date)"
