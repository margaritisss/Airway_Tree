#!/usr/bin/env bash
# Run both binaries over every mesh in a directory.
#
#     source env.sh
#     ./run_batch.sh <mesh_dir> <npy_out_dir>
#
# Resumable: a subject whose outputs already exist is skipped, so if the job
# dies at #30 you can rerun and it picks up from there.
#
# Tunables (environment):
#     SURF_SAMPLES  surface points per subject      default 200000
#     SDF_SAMPLES   sdf samples per subject         default 500000
#     VAR           near-surface jitter variance    default 0.0005

# Deliberately no `set -e`: one bad mesh must not kill a 45-minute batch.
set -uo pipefail

MESHDIR="${1:?usage: run_batch.sh <mesh_dir> <npy_out_dir>}"
OUTDIR="${2:?usage: run_batch.sh <mesh_dir> <npy_out_dir>}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN="$HERE/bin"

SURF_SAMPLES="${SURF_SAMPLES:-200000}"
SDF_SAMPLES="${SDF_SAMPLES:-500000}"
VAR="${VAR:-0.0005}"

if [ ! -x "$BIN/SampleVisibleMeshSurface" ]; then
  echo "ERROR: $BIN/SampleVisibleMeshSurface not found - wrong directory?" >&2
  exit 1
fi
if [ -z "${CONDA_PREFIX:-}" ]; then
  echo "ERROR: environment not set up - run 'source env.sh' first" >&2
  exit 1
fi

mkdir -p "$OUTDIR"

shopt -s nullglob
MESHES=("$MESHDIR"/*.obj "$MESHDIR"/*.ply)
shopt -u nullglob

if [ ${#MESHES[@]} -eq 0 ]; then
  echo "ERROR: no .obj or .ply files in $MESHDIR" >&2
  echo "       (Pangolin's loader reads only those two formats)" >&2
  exit 1
fi

echo "meshes:       ${#MESHES[@]}"
echo "surface pts:  $SURF_SAMPLES"
echo "sdf samples:  $SDF_SAMPLES   (--var $VAR)"
echo "output:       $OUTDIR"
echo

total=${#MESHES[@]}
i=0
failed=0
skipped=0

for m in "${MESHES[@]}"; do
  i=$((i + 1))
  n=$(basename "${m%.*}")
  surf="$OUTDIR/${n}_surface.npy"
  sdf="$OUTDIR/${n}_sdf.npy"

  if [ -s "$surf" ] && [ -s "$sdf" ]; then
    echo "[$i/$total] $n - already done, skipping"
    skipped=$((skipped + 1))
    continue
  fi

  echo "=============================================================="
  echo "[$i/$total] $n"
  echo "=============================================================="

  if [ ! -s "$surf" ]; then
    "$BIN/SampleVisibleMeshSurface" \
        -m "$m" -o "$surf" -n "$OUTDIR/${n}_norm.npz" -s "$SURF_SAMPLES" \
      || { echo "FAILED surface $n"; failed=$((failed + 1)); rm -f "$surf"; }
  fi

  if [ ! -s "$sdf" ]; then
    "$BIN/PreprocessMesh" \
        -m "$m" -o "$sdf" --ply /dev/null -s "$SDF_SAMPLES" --var "$VAR" \
      || { echo "FAILED sdf $n"; failed=$((failed + 1)); rm -f "$sdf"; }
  fi
done

echo
echo "=============================================================="
echo "done: $total meshes, $skipped skipped, $failed failures"
echo "=============================================================="
echo "next:  python npy_to_mat.py $OUTDIR <data_root> --clip -1"
