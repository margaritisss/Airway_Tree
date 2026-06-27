#!/bin/bash
#SBATCH --job-name=AirMorph
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --partition=H100,A100,L40S
#SBATCH --exclude=node54

# Stop the script if any command fails (so it won't falsely say "finished")
set -e

# --- Environment Setup ---
echo "Starting job on GPU node..."

# 1. Load the CUDA toolkit
module load cuda/12.1

# 2. Activate your Python virtual environment
source ~/airway_project/3env/bin/activate

# 3. Load GCC 14 — provides a modern libstdc++ with GLIBCXX_3.4.32 (needed by pycuda)
module load gcc/14.3.0

# 4. Force the modern libstdc++ to load first, ahead of miniconda's old one.
#    LD_PRELOAD guarantees this exact file wins regardless of path ordering.
export LD_PRELOAD=/projects/share/apps/gcc/14.3.0/lib64/libstdc++.so.6:$LD_PRELOAD

# Diagnostic: confirm the loaded libstdc++ has the required symbol
echo "=== libstdc++ check on $(hostname) ==="
strings /projects/share/apps/gcc/14.3.0/lib64/libstdc++.so.6 2>/dev/null | grep -q GLIBCXX_3.4.32 && echo "libstdc++ OK"

# --- Execution ---
echo "Environment set up. Starting AirMorph pipeline..."

cd ~/airway_project/AirMorph
python airwayatlas_pipeline.py

echo "Job finished successfully!"