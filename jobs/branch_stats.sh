#!/bin/bash
#SBATCH --job-name=Tree_Stats
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --partition=H100,A100,L40S
#SBATCH --exclude=node54

set -e

echo "Starting job on GPU node..."

# 1. Load CUDA — match the version the Python script expects (12.9)
module load cuda/12.9

# 2. Activate your Python virtual environment
source ~/airway_project/3env/bin/activate

# 3. Load GCC 14 — modern libstdc++ with GLIBCXX_3.4.32 (needed by pycuda)
module load gcc/14.3.0

# 4. Force the modern libstdc++ to load first
export LD_PRELOAD=/projects/share/apps/gcc/14.3.0/lib64/libstdc++.so.6:$LD_PRELOAD

echo "=== libstdc++ check on $(hostname) ==="
strings /projects/share/apps/gcc/14.3.0/lib64/libstdc++.so.6 2>/dev/null | grep -q GLIBCXX_3.4.32 && echo "libstdc++ OK"

# --- Execution ---
echo "Environment set up. Starting tree-stats pipeline..."

cd /home/ids/gmargari-24/airway_project/tree_skeleton
python tree_stats_pipeline.py

echo "Job finished successfully!"