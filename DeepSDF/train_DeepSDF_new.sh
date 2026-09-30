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
#
# Usage (both variables are optional):
#   sbatch train_DeepSDF.sh
#   sbatch --export=ALL,EXP_DIR=examples/airways_dgci_dryrun,BATCH_SPLIT=4 train_DeepSDF.sh

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

EXP_DIR="${EXP_DIR:-examples/airways_new_false}"       # NEW folder: never reuse a plain-DeepSDF experiment
BATCH_SPLIT="${BATCH_SPLIT:-4}"                   # sub-batches per step; raise if CUDA runs out of memory
SPECS="$EXP_DIR/specs.json"                        # DeepSDF's config file for this experiment
echo "experiment: $EXP_DIR   batch_split: $BATCH_SPLIT"          # record the choice in the log

# ---------------------------------------------------------------- preflight --
[ -f "$SPECS" ] || { echo "FATAL: $SPECS not found"; exit 1; }  # train script would crash anyway

echo "--- specs in use ---"
cat "$SPECS"                                       # record the exact config in the job log
echo "--------------------"

# Check every shape in the train split has its files, for whichever data layout the specs use.
python - "$SPECS" <<'PY' || { echo "FATAL: data preflight failed"; exit 1; }
import json, os, sys                               # stdlib only
specs = json.load(open(sys.argv[1]))               # parse specs.json
data_source = specs["DataSource"]                  # root of the processed data
split = json.load(open(specs["TrainSplit"]))       # {dataset: {class: [instances]}}
dgci = "DGCILoss" in specs                         # which loader train_deep_sdf.py will use
total = 0                                          # instances listed in the split
missing = []                                       # listed but not on disk
size_mb = 0.0                                      # data the DGCI loader keeps in RAM
for dataset, classes in split.items():             # usually a single dataset
    for class_name, instances in classes.items():  # usually a single class
        for inst in instances:                     # one entry per airway tree
            total += 1                             # count it
            if dgci:                               # flat folder of run_batch.sh outputs
                paths = [os.path.join(data_source, inst + "_surface.npy"),
                         os.path.join(data_source, inst + "_sdf.npy")]
            else:                                  # original DeepSDF layout
                paths = [os.path.join(data_source, "SdfSamples", dataset, class_name, inst + ".npz")]
            for path in paths:
                if os.path.isfile(path):           # present: add its size to the RAM estimate
                    size_mb += os.path.getsize(path) / 2**20
                else:                              # absent: remember it for the report
                    missing.append(path)
print("loader             :", "DGCI (.npy)" if dgci else "DeepSDF (.npz)")
print("instances in split :", total)               # should match "There are N scenes"
print("missing files      :", len(missing))        # should be 0
for path in missing[:10]:                          # show a few so you can see the pattern
    print("  missing:", path)
print("training data in RAM: ~%.1f GB (compare with --mem)" % (size_mb / 1024))  # all cases are loaded
assert total > 0, "train split is empty"           # nothing to train on
assert not missing, "some sample files are missing"  # loader would fail at startup
PY

# Imports, a real CUDA kernel (the P100 failure), and one forward/backward of the
# configured decoder + DGCI loss on the GPU, so config typos fail here and not after data loading.
python - "$SPECS" <<'PY' || { echo "FATAL: dependency/GPU preflight failed"; exit 1; }
import json, sys                                   # specs
import torch, numpy, plyfile, skimage.measure      # what deep_sdf imports at load time
import deep_sdf, deep_sdf.workspace                # the repo package itself
import deep_sdf.dgci_loss, track_reconstruction    # the new modules (ImportError = wrong location)
specs = json.load(open(sys.argv[1]))               # parse specs.json
if specs.get("TrackSubjects") and specs.get("TrackOutput", "mask") in ("mask", "both"):
    import SimpleITK                               # the tracker needs it for NIfTI masks
print("torch", torch.__version__, "| cuda available:", torch.cuda.is_available())
assert torch.cuda.is_available(), "no GPU visible" # fail before wasting the allocation
print("GPU:", torch.cuda.get_device_name(0), "| capability:", torch.cuda.get_device_capability(0))
a = torch.randn(256, 256, device="cuda")           # small tensor on the GPU
b = (a @ a).sum().item()                           # forces a real kernel launch
print("CUDA kernel test OK")                       # reached only if this GPU is supported

if "DGCILoss" in specs:                            # smoke test of the new training path
    arch = __import__("networks." + specs["NetworkArch"], fromlist=["Decoder"])  # as train_deep_sdf.py
    latent = specs["CodeLength"]                   # latent size
    decoder = arch.Decoder(latent, **specs["NetworkSpecs"]).cuda()  # NetworkSpecs typos fail here
    cfg = specs["DGCILoss"]                        # loss settings
    loss_fn = deep_sdf.dgci_loss.DGCILoss(weights=cfg.get("Weights"), delta=cfg.get("PhiDelta", 100.0),
                                          clamp_dist=cfg.get("L1ClampingDistance"),
                                          phi_on=cfg.get("PhiOn", "offsurface"))
    n = 4096                                       # points: enough to exercise every term
    kind = torch.randint(0, 3, (n,))               # mix of surface / free / uniform points
    xyz = (torch.rand(n, 3, device="cuda") * 2 - 1).requires_grad_(True)  # leaf for grad F
    codes = torch.randn(n, latent, device="cuda") * 0.01  # small latent codes
    normals = torch.nn.functional.normalize(torch.randn(n, 3), dim=1)  # unit normals
    loss_fn.set_counts(kind)                       # set sizes
    loss, terms = loss_fn(decoder(torch.cat([codes, xyz], 1)), xyz, torch.zeros(n, 1), normals, kind)
    loss.backward()                                # double backprop through grad F
    print("DGCI loss smoke test OK:", {k: round(v, 2) for k, v in terms.items()})
PY
echo "--- preflight OK ---"

# ------------------------------------------------------------------- launch --
# DeepSDF saves ModelParameters/latest.pth every LogFrequency epochs.
# If it exists, resume from it so a job killed by the 24 h limit just continues.
RESUME_ARGS=""                                     # default: fresh training
MARKER="$EXP_DIR/.started_with_dgci"               # written at the first launch of a DGCI run
USES_DGCI=$(python -c "import json,sys; print(int('DGCILoss' in json.load(open(sys.argv[1]))))" "$SPECS")
if [ -f "$EXP_DIR/ModelParameters/latest.pth" ]; then
    if [ "$USES_DGCI" = "1" ] && [ ! -f "$MARKER" ]; then  # checkpoint predates the DGCI loss
        echo "FATAL: $EXP_DIR holds a checkpoint from a run without DGCILoss."
        echo "       Use a new experiment folder (EXP_DIR=...) instead of resuming it."
        exit 1                                     # resuming would mix old weights with the new loss
    fi
    RESUME_ARGS="--continue latest"                # pick up decoder, codes and optimizer
    echo "Found latest checkpoint -> resuming"     # make the choice visible in the log
elif [ "$USES_DGCI" = "1" ]; then
    touch "$MARKER"                                # mark this folder as a DGCI run from the start
fi

time python -u train_deep_sdf.py -e "$EXP_DIR" --batch_split "$BATCH_SPLIT" $RESUME_ARGS  # $RESUME_ARGS unquoted so empty = no arg

echo "Job finished at $(date)"                     # end marker for the log