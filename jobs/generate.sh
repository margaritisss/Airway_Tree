#!/bin/bash
#SBATCH --job-name=gen_vae128
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --partition=H100,A100,L40S
#SBATCH --exclude=node54

export PYTHONUNBUFFERED=1

echo "Job $SLURM_JOB_ID started on $(hostname) at $(date)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo "---"

module load python/3.11.13
source /home/ids/gmargari-24/airway_project/3env/bin/activate
export LD_LIBRARY_PATH=/projects/share/apps/miniconda3/25.5.1/lib:$LD_LIBRARY_PATH

cd /home/ids/gmargari-24/airway_project/Encoding
export PYTHONPATH=/home/ids/gmargari-24/airway_project:$PYTHONPATH

# Fail loudly if the script isn't actually here, instead of looping 5 silent errors.
if [ ! -f generate_vae.py ]; then
    echo "ERROR: generate_vae.py not found in $(pwd)"
    echo "Copy it here, e.g.:  scp generate_vae.py <user>@<cluster>:$(pwd)/"
    exit 1
fi

# ---------------------------------------------------------------------------
# Point at the training run whose checkpoint you want to sample from.
# ---------------------------------------------------------------------------
TRAIN_JOB_ID=843468
RUN_DIR=/home/ids/gmargari-24/airway_project/Data/vae_runs/run_${TRAIN_JOB_ID}
CKPT="$RUN_DIR/final.pt"

GEN_OUT="$RUN_DIR/generated"
mkdir -p "$GEN_OUT"

if [ ! -f "$CKPT" ]; then
    echo "ERROR: checkpoint not found: $CKPT"
    echo "Contents of $RUN_DIR:"; ls -la "$RUN_DIR"
    exit 1
fi
echo "Using checkpoint: $CKPT"
echo "Writing outputs to: $GEN_OUT"
echo "---"

# 1) Prior samples -- what the model generates from scratch (needs no inputs).
time python -u generate_vae.py sample \
    --checkpoint "$CKPT" \
    --out-dir "$GEN_OUT/samples" \
    --n 12 \
    --temperature 0.8 \
    --seed 0

# 2) Reconstructions of a couple of real masks. Input grid size must match the
#    model (128^3 for VoxelVAE128); the || keeps the job alive on a mismatch.
DATA_DIR=/home/ids/gmargari-24/airway_project/Data/Registered_on_Template_22_23/Affine_registered/ATM22_128
mapfile -t EXAMPLES < <(ls "$DATA_DIR"/*.nii.gz 2>/dev/null | head -n 2)
if [ "${#EXAMPLES[@]}" -gt 0 ]; then
    time python -u generate_vae.py reconstruct \
        --checkpoint "$CKPT" \
        --out-dir "$GEN_OUT/recon" \
        --inputs "${EXAMPLES[@]}" \
        || echo "reconstruct step failed (likely a grid-size mismatch) -- continuing."
else
    echo "No example .nii.gz found in $DATA_DIR -- skipping reconstruct."
fi

# 3) Latent traversals -- sweep a few dims (num_latents=32 -> dims 0..31).
for DIM in 0 5 10; do
    time python -u generate_vae.py traverse \
        --checkpoint "$CKPT" \
        --out-dir "$GEN_OUT/traverse" \
        --dim "$DIM" \
        --steps 9 \
        --span 3.0
done

echo "---"
echo "Job finished at $(date)"
echo "Browse results under: $GEN_OUT"