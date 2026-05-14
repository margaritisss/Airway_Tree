#!/bin/bash
#SBATCH --job-name=register_groupwise_deformable_parallel
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --cpus-per-task=48
#SBATCH --mem=450G
#SBATCH --time=5-00:00:00
#SBATCH --partition=cpu-high

# --- Force live, unbuffered output to the .out / .err files ----------
# Without this, Python buffers stdout when it's redirected to a file,
# so progress lines may not appear until the buffer fills (or the
# process exits / crashes).
export PYTHONUNBUFFERED=1
# ---------------------------------------------------------------------

echo "Job $SLURM_JOB_ID started on $(hostname) at $(date)"
echo "Allocated CPUs: $SLURM_CPUS_PER_TASK"
echo "nproc: $(nproc)"
echo "Memory limit (cgroup): $(cat /sys/fs/cgroup/memory.max 2>/dev/null || echo 'n/a')"
echo "CPU model: $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | xargs)"
free -h
echo "---"

if [ "$SLURM_CPUS_PER_TASK" -ne 48 ]; then
    echo "ERROR: expected 48 CPUs, got $SLURM_CPUS_PER_TASK"
    exit 1
fi

module load python/3.11.13
source /home/ids/gmargari-24/3env/bin/activate
export LD_LIBRARY_PATH=/projects/share/apps/miniconda3/25.5.1/lib:$LD_LIBRARY_PATH

cd /home/ids/gmargari-24/airway_project
export PYTHONPATH=/home/ids/gmargari-24/airway_project:$PYTHONPATH
mkdir -p /home/ids/gmargari-24/Data/Groupwise_registered

# `-u` is a second belt-and-braces guarantee of unbuffered stdout.
time python -u jobs/run_groupwise_parallel_registration.py

echo "Job finished at $(date)"