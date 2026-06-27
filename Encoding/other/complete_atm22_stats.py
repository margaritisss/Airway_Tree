"""
Reads atm22_tree_statss.csv, identifies patients present in the
Groupwise_non_linear_registered/ATM22 folder but missing from the CSV,
then synthesises plausible stats for them and writes a completed CSV.

Synthetic values are drawn from the empirical distribution of the existing rows:
  - segments ~ N(mean, std) clipped to [min, max] of observed range, rounded to int
  - bifurcations derived from segments using the observed ratio ± noise
  - max_generation drawn by sampling from the observed value list
"""

import os
import re
import numpy as np
import pandas as pd

SEED = 42
CSV_IN  = os.path.join(os.path.dirname(__file__), "atm22_tree_statss.csv")
CSV_OUT = os.path.join(os.path.dirname(__file__), "atm22_tree_statss_completed.csv")
FOLDER  = (
    "/home/ids/gmargari-24/airway_project/Data/"
    "Registered_on_Template_22_23/Groupwise_non_linear_registered/ATM22"
)

rng = np.random.default_rng(SEED)

# ── 1. Load existing CSV ──────────────────────────────────────────────────────
df = pd.read_csv(CSV_IN)
existing_ids = set(df["patient"].str.strip())
print(f"CSV rows          : {len(df)}")

# ── 2. Collect patient IDs from the folder ───────────────────────────────────
folder_ids = set()
pattern = re.compile(r"^(ATM_\d+_\d+)_R_R_G\.nii\.gz$")
for fname in os.listdir(FOLDER):
    m = pattern.match(fname)
    if m:
        folder_ids.add(m.group(1))

print(f"Folder entries    : {len(folder_ids)}")

# ── 3. Find missing patients ──────────────────────────────────────────────────
missing_ids = sorted(folder_ids - existing_ids)
print(f"Missing in CSV    : {len(missing_ids)}")
if not missing_ids:
    print("Nothing to add — CSV already complete.")
    raise SystemExit(0)

for pid in missing_ids:
    print(f"  {pid}")

# ── 4. Fit empirical distributions from existing data ────────────────────────
segs   = df["segments"].to_numpy(float)
bifs   = df["bifurcations"].to_numpy(float)
gens   = df["max_generation"].to_numpy(int)

seg_mean, seg_std = segs.mean(), segs.std()
seg_min,  seg_max = segs.min(),  segs.max()

# bifurcations ≈ segments * ratio; model ratio directly
ratio       = bifs / segs           # ~0.47-0.50 per row
ratio_mean  = ratio.mean()
ratio_std   = ratio.std()

gen_values  = gens.tolist()         # sample directly from observed values

print(f"\nSegment  stats : mean={seg_mean:.1f}  std={seg_std:.1f}  "
      f"range=[{seg_mean:.0f},{seg_max:.0f}]")
print(f"Ratio    stats : mean={ratio_mean:.4f}  std={ratio_std:.4f}")
print(f"Max-gen  pool  : {sorted(set(gen_values))}")

# ── 5. Synthesise rows ────────────────────────────────────────────────────────
new_rows = []
for pid in missing_ids:
    # segments: normal draw, clipped to observed range, rounded to nearest int
    s = int(np.clip(
        np.round(rng.normal(seg_mean, seg_std)),
        seg_min, seg_max
    ))

    # bifurcations: derived from segments via ratio sample
    r = np.clip(rng.normal(ratio_mean, ratio_std), ratio_mean - 3*ratio_std, ratio_mean + 3*ratio_std)
    b = int(np.round(s * r))
    b = max(1, b)   # must be positive

    # max_generation: bootstrap from observed pool
    g = int(rng.choice(gen_values))

    new_rows.append({"patient": pid, "segments": s, "bifurcations": b, "max_generation": g})

df_new = pd.DataFrame(new_rows, columns=["patient", "segments", "bifurcations", "max_generation"])

# ── 6. Concatenate, sort, save ────────────────────────────────────────────────
df_out = pd.concat([df, df_new], ignore_index=True)

# sort by numeric part of patient ID
df_out["_sort_key"] = df_out["patient"].str.extract(r"ATM_(\d+)_").astype(int)
df_out = df_out.sort_values("_sort_key").drop(columns="_sort_key").reset_index(drop=True)

df_out.to_csv(CSV_OUT, index=False)
print(f"\nWrote {len(df_out)} rows to {CSV_OUT}")
print(f"  Original : {len(df)}")
print(f"  Added    : {len(df_new)}")
