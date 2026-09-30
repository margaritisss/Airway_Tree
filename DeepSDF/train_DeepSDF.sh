#!/bin/bash
#SBATCH --job-name=deepsdf_airway
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --partition=A100,L40S
#SBATCH --exclude=node54
# NOTE: no V100/P100 partition on purpose. Your torch 2.13+cu130 only ships
# kernels for sm_75 and newer: A100 (sm_80) and L40S (sm_89) work,
# V100 (sm_70) and P100 (sm_60) crash with "no kernel image is available".

export PYTHONUNBUFFERED=1                          # stream prints so the .out file updates live

echo "Job $SLURM_JOB_ID started on $(hostname) at $(date)"      # where and when, for the log
echo "Allocated CPUs: $SLURM_CPUS_PER_TASK"                     # sanity check of the request
echo "SLURM gave CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"    # which GPU(s) we own
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader  # confirm GPU model
echo "---"

module load python/3.11.13                         # same Python as the venv was built with
module load cuda/12.1                              # same as the DGCI job (torch bundles its own runtime)
source /home/ids/gmargari-24/airway_project/new_3env/bin/activate   # the shared project venv
export LD_LIBRARY_PATH=/projects/share/apps/miniconda3/25.5.1/lib:$LD_LIBRARY_PATH  # as in DGCI job

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # less fragmentation on long runs

# DeepSDF imports "networks.<NetworkArch>" and reads specs relative to CWD,
# so we must run from the repo root (the directory holding train_deep_sdf.py).
DEEPSDF_DIR=/home/ids/gmargari-24/airway_project/DeepSDF        # repo root
cd "$DEEPSDF_DIR" || { echo "FATAL: cannot cd to $DEEPSDF_DIR"; exit 1; }  # abort if missing
export PYTHONPATH="$DEEPSDF_DIR:$PYTHONPATH"      # make deep_sdf/ and networks/ importable

EXP_DIR=examples/a_2faster                     # !!!!!!!!      # experiment folder passed with -e
SPECS="$EXP_DIR/specs.json"                        # DeepSDF's config file for this experiment

# ---------------------------------------------------------------- preflight --
[ -f "$SPECS" ] || { echo "FATAL: $SPECS not found"; exit 1; }  # train script would crash anyway

echo "--- specs in use ---"
cat "$SPECS"                                       # record the exact config in the job log
echo "--------------------"

# Check every shape listed in the train split has its SDF samples on disk.
python - "$SPECS" <<'PY' || { echo "FATAL: data preflight failed"; exit 1; }
import json, os, sys                               # stdlib only
specs = json.load(open(sys.argv[1]))               # parse specs.json
data_source = specs["DataSource"]                  # root of the processed data
split = json.load(open(specs["TrainSplit"]))       # {dataset: {class: [instances]}}
total = 0                                          # instances listed in the split
missing = []                                       # listed but no .npz on disk
for dataset, classes in split.items():             # usually a single dataset
    for class_name, instances in classes.items():  # usually a single class
        for inst in instances:                     # one entry per airway tree
            total += 1                             # count it
            npz = os.path.join(data_source, "SdfSamples", dataset, class_name, inst + ".npz")
            if not os.path.isfile(npz):            # sample file absent
                missing.append(npz)                # remember it for the report
print("instances in split :", total)               # should match "There are N scenes"
print("missing .npz files :", len(missing))        # should be 0
for path in missing[:10]:                          # show a few so you can see the pattern
    print("  missing:", path)
assert total > 0, "train split is empty"           # nothing to train on
assert not missing, "some SDF sample files are missing"  # loader would fail mid-epoch
PY

# Check imports AND that a CUDA kernel really runs on this GPU (the P100 failure).
python - <<'PY' || { echo "FATAL: dependency/GPU preflight failed"; exit 1; }
import torch, numpy, plyfile, skimage.measure      # what deep_sdf imports at load time
import deep_sdf, deep_sdf.workspace                # the repo package itself
print("torch", torch.__version__, "| cuda available:", torch.cuda.is_available())
assert torch.cuda.is_available(), "no GPU visible" # fail before wasting the allocation
print("GPU:", torch.cuda.get_device_name(0), "| capability:", torch.cuda.get_device_capability(0))
a = torch.randn(256, 256, device="cuda")           # small tensor on the GPU
b = (a @ a).sum().item()                           # forces a real kernel launch
print("CUDA kernel test OK")                       # reached only if this GPU is supported
PY
echo "--- preflight OK ---"

# ------------------------------------------------------------------- launch --
# DeepSDF saves ModelParameters/latest.pth every LogFrequency epochs.
# If it exists, resume from it so a job killed by the 24 h limit just continues.
RESUME_ARGS=""                                     # default: fresh training
if [ -f "$EXP_DIR/ModelParameters/latest.pth" ]; then
    RESUME_ARGS="--continue latest"                # pick up decoder, codes and optimizer
    echo "Found latest checkpoint -> resuming"     # make the choice visible in the log
fi

time python -u train_deep_sdf.py -e "$EXP_DIR" $RESUME_ARGS  # $RESUME_ARGS unquoted so empty = no arg

echo "Job finished at $(date)"                     # end marker for the log
