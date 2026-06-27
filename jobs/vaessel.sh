#!/bin/bash
#SBATCH --job-name=train_vaessel
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --partition=H100,A100,L40S
#SBATCH --exclude=node54

# Unbuffered Python stdout/stderr so the .out file updates live.
export PYTHONUNBUFFERED=1

# train_2.py defaults ATTN_BACKEND to flash_attn (needs fp16/bf16; Ampere+ / Ada).
# Export it here too so it's explicit; switch to xformers if flash-attn isn't installed.
export ATTN_BACKEND=flash_attn
# export ATTN_BACKEND=xformers

echo "Job $SLURM_JOB_ID started on $(hostname) at $(date)"
echo "Allocated CPUs: $SLURM_CPUS_PER_TASK"
echo "GPU(s): $CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo "ATTN_BACKEND=$ATTN_BACKEND"
echo "---"

module load python/3.11.13
source /home/ids/gmargari-24/airway_project/3env/bin/activate
export LD_LIBRARY_PATH=/projects/share/apps/miniconda3/25.5.1/lib:$LD_LIBRARY_PATH

# Run from the directory that holds train_22.py, vaessel_22.py and data_22.py,
# so the `from vaessel_22 import ...` / `from data_22 import ...` imports resolve.
cd /home/ids/gmargari-24/airway_project/VAEsselSparse
export PYTHONPATH=/home/ids/gmargari-24/airway_project/Direct3D_S2:/home/ids/gmargari-24/airway_project:$PYTHONPATH

# Folder to collect this run's artifacts (logs + the best checkpoint).
OUT_DIR=/home/ids/gmargari-24/airway_project/Data/vae_runs/run_${SLURM_JOB_ID}
mkdir -p "$OUT_DIR"

# NOTE: train_22.py takes NO command-line arguments. All configuration lives in
# constants at the top of train_22.py (DATA_ROOTS, TARGET_SHAPE, EPOCHS, LR, BETA,
# BATCH_SIZE, NUM_WORKERS, CKPT, ...). Edit those before launching, or add argparse.
time python -u train_22.py

# train_22.py writes the best checkpoint to ./vaesselsparse_best.pt (CWD).
# Copy it into this run's folder if it was produced.
if [ -f vaesselsparse_best.pt ]; then
    cp -v vaesselsparse_best.pt "$OUT_DIR"/
fi

echo "Job finished at $(date)"