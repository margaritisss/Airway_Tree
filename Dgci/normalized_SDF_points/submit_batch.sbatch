#!/bin/bash
# Slurm submission for the DeepSDF preprocessing batch.
#
#     sbatch submit_batch.sbatch <mesh_dir> <npy_out_dir>
#
# Check your partition names with `sinfo` and add -p <partition> if the default
# is not what you want. Adjust --time from the mesh count: roughly 1 minute per
# subject, plus slack.
#
#SBATCH --job-name=dgci-prep
#SBATCH --output=%x-%j.log
#SBATCH --error=%x-%j.log
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# No `set -u` here: conda's activation scripts read variables like
# CONDA_BACKUP_CXX without defaults, which is fatal under nounset in a
# non-interactive shell. run_batch.sh sets -u for itself, where it is safe.
set -o pipefail

PROJECT="$HOME/airway_project/DGCI/normalized_SDF_points"
MESHDIR="${1:?usage: sbatch submit_batch.sbatch <mesh_dir> <npy_out_dir>}"
OUTDIR="${2:?usage: sbatch submit_batch.sbatch <mesh_dir> <npy_out_dir>}"

echo "host      $(hostname)"
echo "started   $(date)"
echo "meshes    $MESHDIR"
echo "output    $OUTDIR"
echo

cd "$PROJECT" || exit 1

# A batch job starts a fresh shell, so nothing from your interactive session
# carries over - the conda env and the EGL settings must be re-established here.
source ./env.sh || { echo "env.sh failed"; exit 1; }

# Mesa's software rasterizer spawns its own threads; keep it inside the
# allocation rather than oversubscribing the node.
export LP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

# Invoked through `bash` rather than `./run_batch.sh` so a missing execute bit
# (lost whenever the file is re-downloaded or copied) cannot kill the job.
bash ./run_batch.sh "$MESHDIR" "$OUTDIR"
rc=$?

echo
echo "finished  $(date)"
echo "exit      $rc"
exit $rc
