#!/bin/bash
#SBATCH --job-name=register_affine_parallel
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --cpus-per-task=48
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --partition=cpu-high

echo "Job $SLURM_JOB_ID started on $(hostname) at $(date)"
echo "Allocated CPUs: $SLURM_CPUS_PER_TASK" # Check how many CPUs were allocated to the job
echo "nproc: $(nproc)"                      # Check how many CPUs are available to the job

if [ "$SLURM_CPUS_PER_TASK" -ne 48 ]; then # Verify that the expected number of CPUs were allocated
    echo "ERROR: expected 48 CPUs, got $SLURM_CPUS_PER_TASK"
    exit 1
fi

module load python/3.11.13
source /home/ids/gmargari-24/3env/bin/activate
# Use libbz2 from the miniconda installation (shared filesystem, visible on all nodes)
export LD_LIBRARY_PATH=/projects/share/apps/miniconda3/25.5.1/lib:$LD_LIBRARY_PATH

cd /home/ids/gmargari-24/airway_project
export PYTHONPATH=/home/ids/gmargari-24/airway_project:$PYTHONPATH   # Ensure the project root is in PYTHONPATH so that imports work correctly
mkdir -p /home/ids/gmargari-24/Data/Affine_registered # Make sure the output folder exists
time python jobs/run_affine_registration_parallel.py  # Run the Python script and time its execution
echo "Job finished at $(date)"