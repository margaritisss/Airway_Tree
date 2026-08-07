#!/bin/bash
#SBATCH --job-name=dgci_gen
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --partition=L40S,A100
#SBATCH --exclude=node54

export PYTHONUNBUFFERED=1

echo "Job $SLURM_JOB_ID started on $(hostname) at $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "---"

module load python/3.11.13
module load cuda/12.1
source /home/ids/gmargari-24/airway_project/new_3env/bin/activate
export LD_LIBRARY_PATH=/projects/share/apps/miniconda3/25.5.1/lib:$LD_LIBRARY_PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DGCI_DIR=/home/ids/gmargari-24/airway_project/DGCI
cd "$DGCI_DIR" || { echo "FATAL: cannot cd to $DGCI_DIR"; exit 1; }
export PYTHONPATH="$DGCI_DIR:$PYTHONPATH"
export PYTHONPATH=/home/ids/gmargari-24/airway_project/shims:$PYTHONPATH

CONFIG=configs/generate/airway_dci.yml
RESOLUTION=${RESOLUTION:-512}
SUBJECTS=${SUBJECTS:-0-4}
LEVEL=${LEVEL:-0.0}

# ---------------------------------------------------------------- preflight --
[ -f "$CONFIG" ] || { echo "FATAL: $CONFIG not found"; exit 1; }

echo "--- config in use ---"
cat "$CONFIG"
echo "---------------------"

# The grid-indexing fix in sdf_meshing.py fails SILENTLY -- it produces a
# malformed .ply rather than an error. Refuse to burn an allocation without it.
if grep -qE '^\s*samples\[:, 1\] = \(overall_index\.long\(\) / N\) % N' sdf_meshing.py; then
    echo "FATAL: sdf_meshing.py still uses float division for the voxel grid."
    echo "       Switch to torch.div(..., rounding_mode='floor') before generating."
    exit 1
fi

# Verify the checkpoint exists and that the config matches its shapes,
# otherwise load_state_dict throws only after the model is built on GPU.
python - "$CONFIG" <<'PY' || { echo "FATAL: checkpoint/config preflight failed"; exit 1; }
import sys, yaml, torch
cfg = yaml.safe_load(open(sys.argv[1]))
sd = torch.load(cfg['checkpoint_path'], map_location='cpu')
n, d = sd['latent_codes.weight'].shape
print('checkpoint latent_codes :', (n, d))
print('config num_instances    :', cfg['num_instances'])
print('config latent_dim       :', cfg['latent_dim'])
assert n == cfg['num_instances'], 'num_instances mismatch -- load_state_dict will fail'
assert d == cfg['latent_dim'],    'latent_dim mismatch -- load_state_dict will fail'
assert torch.cuda.is_available(), 'no GPU visible'
PY
echo "--- preflight OK ---"

# ------------------------------------------------------------------- launch --
time python -u generate.py \
    --config "$CONFIG" \
    --subject_idx "$SUBJECTS" \
    --resolution "$RESOLUTION" \
    --level "$LEVEL"

echo "--- output ---"
ls -lh recon/*/
echo "Job finished at $(date)"