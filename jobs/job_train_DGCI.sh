#!/bin/bash
#SBATCH --job-name=dgci_airway
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --partition=L40S,A100
#SBATCH --exclude=node54

# Unbuffered stdout/stderr so the .out file updates live.
export PYTHONUNBUFFERED=1

echo "Job $SLURM_JOB_ID started on $(hostname) at $(date)"
echo "Allocated CPUs: $SLURM_CPUS_PER_TASK"
echo "SLURM gave CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo "---"

module load python/3.11.13
module load cuda/12.1
source /home/ids/gmargari-24/airway_project/new_3env/bin/activate
export LD_LIBRARY_PATH=/projects/share/apps/miniconda3/25.5.1/lib:$LD_LIBRARY_PATH

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# train.py resolves --config and logging_root RELATIVE TO CWD, so this must be
# the DGCI repo root (the directory containing train.py and configs/).
DGCI_DIR=/home/ids/gmargari-24/airway_project/DGCI
cd "$DGCI_DIR" || { echo "FATAL: cannot cd to $DGCI_DIR"; exit 1; }
export PYTHONPATH="$DGCI_DIR:$PYTHONPATH"
export PYTHONPATH=/home/ids/gmargari-24/airway_project/shims:$PYTHONPATH

CONFIG=configs/train/airway_dci.yml

# ---------------------------------------------------------------- preflight --
# Every one of these has a failure mode that wastes a full GPU allocation.
[ -f "$CONFIG" ] || { echo "FATAL: $CONFIG not found (train.py needs it at this exact path)"; exit 1; }

echo "--- config in use ---"
cat "$CONFIG"
echo "---------------------"

# Pull the data path straight out of the YAML so this never drifts from the config.
DATA_DIR=$(python - "$CONFIG" <<'PY'
import sys, yaml
print(yaml.safe_load(open(sys.argv[1]))['point_cloud_path'])
PY
)
FREE_DIR=${DATA_DIR//surface_pts_n_normal/free_space_pts}

N_SURF=$(ls "$DATA_DIR"/*.mat 2>/dev/null | wc -l)
N_FREE=$(ls "$FREE_DIR"/*.mat 2>/dev/null | wc -l)
N_CFG=$(python - "$CONFIG" <<'PY'
import sys, yaml
print(yaml.safe_load(open(sys.argv[1]))['num_instances'])
PY
)

echo "surface .mat files : $N_SURF   ($DATA_DIR)"
echo "free .mat files    : $N_FREE   ($FREE_DIR)"
echo "num_instances      : $N_CFG"

[ "$N_SURF" -gt 0 ] || { echo "FATAL: no .mat files found -- loader would assert"; exit 1; }
if [ "$N_SURF" -lt "$N_CFG" ]; then
    echo "WARNING: num_instances ($N_CFG) exceeds available cases ($N_SURF)."
    echo "         Untrained rows will sit in nn.Embedding and mesh to garbage."
fi
if [ "$N_SURF" -gt "$N_CFG" ]; then
    echo "WARNING: $((N_SURF - N_CFG)) case(s) will be SILENTLY DISCARDED by max_num_instances."
fi

# Verify the imports the code actually performs, and how many GPUs it will see.
python - <<'PY' || { echo "FATAL: dependency/GPU preflight failed"; exit 1; }
import torch, scipy.io, yaml, configargparse, tqdm
import torchmeta.modules          # DCI_Modules
import einops.layers.torch        # still imported by dcinet.py line 6
import skimage.measure, plyfile   # pulled in via training_loop_dgci.py line 9
from torch.utils.tensorboard import SummaryWriter
print('torch', torch.__version__, '| cuda available:', torch.cuda.is_available())
print('device_count BEFORE train.py overrides CUDA_VISIBLE_DEVICES:', torch.cuda.device_count())
assert torch.cuda.is_available(), 'no GPU visible'
PY
echo "--- preflight OK ---"

# ------------------------------------------------------------------- launch --
# NOTE: train.py ignores every CLI flag except --config; the YAML wins.
time python -u train.py --config "$CONFIG"

echo "Job finished at $(date)"
