import sys
import os
sys.path.insert(0, '/home/ids/gmargari-24/airway_project/AirMorph')
os.environ['PATH'] = '/projects/share/apps/cuda/cuda-12.9/bin:' + os.environ.get('PATH', '')
import pycuda.autoinit
import pycuda.compiler
import numpy as np

# Force pycuda to use the full absolute path to nvcc — bypasses PATH lookup entirely
_NVCC = '/projects/share/apps/cuda/cuda-12.9/bin/nvcc'
_orig_init = pycuda.compiler.SourceModule.__init__

def _patched_init(self, source, nvcc=_NVCC, options=None, *args, **kwargs):
    if options is None:
        options = []
    _orig_init(self, source, nvcc=nvcc, options=options, *args, **kwargs)

pycuda.compiler.SourceModule.__init__ = _patched_init

from util.utils import load_itk_image
from data_process.tree_parse_skelbased import tree_parse

import glob
import pandas as pd
from pathlib import Path
#/projects/AirTwin_angelini/work_dataset/dataset/AIIB23/gt
#/projects/AirTwin_angelini/work_dataset/dataset/ATM22/labelsTr
MASK_DIR = '/projects/AirTwin_angelini/work_dataset/dataset/ATM22/labelsTr'  # AIIB23 masks are in the same dir as ATM22, so we can reuse this path
SKEL_DIR = '/home/ids/gmargari-24/airway_project/Data/ATM22_skeletons'
OUT_CSV  = '/home/ids/gmargari-24/airway_project/Data/atm22_tree_statss.csv'

mask_files = sorted(glob.glob(f'{MASK_DIR}/*.nii.gz'))

# Resume: load already-done patients so we can skip them.
done = set()
if Path(OUT_CSV).exists() and Path(OUT_CSV).stat().st_size > 0:
    done = set(pd.read_csv(OUT_CSV)['patient'].astype(str))
    print(f"Resuming — {len(done)} patients already in {OUT_CSV}")

# Open CSV in append mode; write header only when creating fresh.
csv_exists = Path(OUT_CSV).exists() and len(done) > 0
csv_fh = open(OUT_CSV, 'a', buffering=1)  # line-buffered
if not csv_exists:
    csv_fh.write('patient,segments,bifurcations,max_generation\n')

n_done = 0
for i, mask_path in enumerate(mask_files):
    patient = Path(mask_path).name          # AIIB23_100.nii.gz
    patient_id = patient.replace('.nii.gz', '')
    skel_path = f'{SKEL_DIR}/{patient}'

    if patient_id in done:
        print(f"[{i+1}/{len(mask_files)}] SKIP {patient} — already done")
        n_done += 1
        continue
    if not Path(skel_path).exists():
        print(f"[{i+1}/{len(mask_files)}] SKIP {patient} — skeleton not found")
        continue
    try:
        mask, *_ = load_itk_image(mask_path)
        skel, *_ = load_itk_image(skel_path)
        mask = (mask > 0.5).astype(np.uint8)
        skel = (skel > 0).astype(np.float32)
        _, children_map, generation, _, cd, _ = tree_parse(mask, skel)
        row = {
            'patient'       : patient_id,
            'segments'      : int(cd.max()),
            'bifurcations'  : int((children_map.sum(axis=1) >= 2).sum()),
            'max_generation': int(generation.max()),
        }
        csv_fh.write(f"{row['patient']},{row['segments']},{row['bifurcations']},{row['max_generation']}\n")
        n_done += 1
        print(f"[{i+1}/{len(mask_files)}] {patient}  "
              f"seg={row['segments']}  "
              f"bifurc={row['bifurcations']}  "
              f"gen={row['max_generation']}")
    except Exception as e:
        print(f"[{i+1}/{len(mask_files)}] ERROR {patient}: {e}")

csv_fh.close()
print(f"\nSaved {n_done} rows → {OUT_CSV}")