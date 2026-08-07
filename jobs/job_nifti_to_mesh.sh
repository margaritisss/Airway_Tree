#!/bin/bash
#SBATCH --job-name=mesh_AIIB23
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --exclude=node54
# Add a CPU partition here if your cluster requires one, e.g.:
##SBATCH --partition=cpu

export PYTHONUNBUFFERED=1

echo "Job $SLURM_JOB_ID started on $(hostname) at $(date)"
echo "Allocated CPUs: $SLURM_CPUS_PER_TASK"
echo "---"

module load python/3.11.13
source /home/ids/gmargari-24/airway_project/new_3env/bin/activate
export LD_LIBRARY_PATH=/projects/share/apps/miniconda3/25.5.1/lib:$LD_LIBRARY_PATH

cd /home/ids/gmargari-24/airway_project/Registration
export PYTHONPATH=/home/ids/gmargari-24/airway_project:$PYTHONPATH

# Each worker is its own process, so stop the numeric libs from also spawning threads inside it (oversubscription makes this slower, not faster).
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

INPUT_FOLDER=/home/ids/gmargari-24/airway_project/Data/Registered_on_Template_22_23/Affine_registered/ATM22

# One process per file. Each holds a ~1.5 GB volume plus the mesh, so keep WORKERS well under (mem / ~5 GB). With 64G, 8 is comfortable.
WORKERS=8

time python -u run_mesh.py \
    --input-folder "$INPUT_FOLDER" \
    --how-many all \
    --level 0.5 \
    --workers $WORKERS \
    --quiet

echo "Job finished at $(date)"
