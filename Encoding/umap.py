import numpy as np
import umap
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
import re

# Use the same working CSVs that matched in the t-SNE diagnostic:
#   - ATM stats live in the *_completed* file
#   - AIIB stats in aiib23_tree_statss.csv
DEFAULT_STATS_CSVS = ["/home/ids/gmargari-24/airway_project/Data/3d_pulmanory_airway_stats/aiib23_tree_stats.csv",
                      "/home/ids/gmargari-24/airway_project/Data/3d_pulmanory_airway_stats/atm22_tree_statss.csv",]

# def _stem(p) -> str:
#     """ATM_001_0000_R_resampled.nii.gz -> ATM_001_0000
#        AIIB23_100_R_resampled.nii.gz   -> AIIB23_100
#     Strip .gz/.nii, then the trailing '_R_resampled' so stems match the
#     'patient' IDs in the stats CSVs."""
#     name = Path(str(p)).name
#     for ext in (".gz", ".nii"):
#         if name.endswith(ext):
#             name = name[: -len(ext)]
#     if name.endswith("_R_resampled"):
#         name = name[: -len("_R_resampled")]
#     return name


# Patient IDs look like  ATM_001_0000  or  AIIB23_100 .
# Capture that leading block and discard any trailing suffix
# (_R_resampled, _R_R_G_resampled, etc.).
_ID_RE = re.compile(r"^(ATM_\d+_\d+|AIIB23_\d+)")


def _stem(p) -> str:
    """Extract the patient ID, ignoring trailing suffixes.
       ATM_001_0000_R_resampled.nii.gz       -> ATM_001_0000
       AIIB23_100_R_R_G_resampled.nii.gz     -> AIIB23_100
       AIIB23_100_R_resampled.nii.gz         -> AIIB23_100"""
    name = Path(str(p)).name
    for ext in (".gz", ".nii"):
        if name.endswith(ext):
            name = name[:-len(ext)]
    m = _ID_RE.match(name)
    if m:
        return m.group(1)
    # fallback: strip known suffixes if the regex doesn't catch it
    for suf in ("_R_R_G_resampled", "_R_resampled"):
        if name.endswith(suf):
            return name[:-len(suf)]
    return name


def _load_stats_color(paths, stats_csvs, column):
    """Length-N color array aligned to `paths`, joined on stem. Missing -> NaN.
    Raises if more than half the samples fail to match, so we never silently
    produce an all-gray plot again."""
    if isinstance(stats_csvs, (str, Path)):
        stats_csvs = [stats_csvs]

    frames = []
    for csv in stats_csvs:
        csv = Path(csv)
        if not csv.exists():
            print(f"[warn] stats csv not found, skipping: {csv}")
            continue
        df = pd.read_csv(csv)
        if "patient" not in df.columns:
            raise ValueError(f"{csv} has no 'patient' column")
        if column not in df.columns:
            raise ValueError(
                f"column '{column}' not in {csv}; available: {list(df.columns)}")
        frames.append(df[["patient", column]])
        print(f"[info] loaded {len(df)} rows from {csv.name}")

    if not frames:
        raise FileNotFoundError("no usable stats CSVs found")

    merged = pd.concat(frames, ignore_index=True)
    dupes = merged["patient"].duplicated().sum()
    if dupes:
        print(f"[warn] {dupes} duplicate patient IDs across CSVs; keeping first")
        merged = merged.drop_duplicates(subset="patient", keep="first")

    lookup = dict(zip(merged["patient"].astype(str), merged[column]))

    color, missing = [], 0
    for p in paths:
        key = _stem(p)
        if key in lookup:
            color.append(float(lookup[key]))
        else:
            color.append(np.nan)
            missing += 1
    if missing:
        print(f"[warn] {missing}/{len(paths)} samples had no '{column}' "
              f"stat (set to NaN)")
        # guard: a high miss rate means the join is broken (e.g. stem/CSV
        # mismatch), not just a few absent patients.
        if missing > 0.5 * len(paths):
            sample_stems = [_stem(p) for p in paths[:3]]
            sample_keys = list(lookup)[:3]
            raise ValueError(
                f"{missing}/{len(paths)} samples failed to match on '{column}'. "
                f"The stem<->patient join is almost certainly broken. "
                f"example stems={sample_stems}  example csv patients={sample_keys}"
            )
    return np.asarray(color, dtype=np.float32)


def _infer_source(paths):
    """Categorical source label from path (ATM vs AIIB), for colour_plot='dataset'."""
    labels = []
    for p in paths:
        s = str(p).lower()
        if "atm" in s:
            labels.append("ATM22")
        elif "aiib" in s:
            labels.append("AIIB23")
        else:
            labels.append("other")
    return np.asarray(labels)


def apply_umap(
    embeddings_path: str,
    output_image: str = "umap_projection.png",
    *,
    colour_plot: str = "dataset",          # "dataset" | "branches" | "generations"
    stats_csvs: list = DEFAULT_STATS_CSVS,
):
    # 1. Load the data
    print(f"Loading embeddings from {embeddings_path}...")
    data = np.load(embeddings_path, allow_pickle=True)
    mu = data['mu']         # N x 100 feature matrix
    paths = data['paths']   # file paths -> patients
    print(f"Loaded {mu.shape[0]} airway models of dimension {mu.shape[1]}")

    # 2. Configure and apply UMAP  (unchanged — this feeds HDBSCAN)
    print("Fitting UMAP manifold... (this might take a moment)")
    reducer = umap.UMAP(
        n_neighbors=419,
        min_dist=0.0125,
        n_components=2,
        metric='euclidean',
        random_state=42,
    )
    embedding_2d = reducer.fit_transform(mu)
    print(f"Successfully projected data to shape: {embedding_2d.shape}")

    # 3. Build the color array based on colour_plot
    if colour_plot == "dataset":
        color = _infer_source(paths)
        legend_title = "source"
        categorical = True
    elif colour_plot == "branches":
        color = _load_stats_color(paths, stats_csvs, "bifurcations")
        legend_title = "n_branches"
        categorical = False
    elif colour_plot == "generations":
        color = _load_stats_color(paths, stats_csvs, "max_generation")
        color[color > 20] = np.nan   # drop noisy high-generation outliers
        legend_title = "n_generations"
        categorical = False
    else:
        raise ValueError(
            f"colour_plot must be 'dataset', 'branches', or 'generations'; "
            f"got {colour_plot!r}")

    # 4. Visualize
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 8))

    if categorical:
        # discrete legend, one scatter call per category
        for label in np.unique(color):
            m = (color == label)
            plt.scatter(embedding_2d[m, 0], embedding_2d[m, 1],
                        alpha=0.7, s=15, label=str(label))
        plt.legend(title=legend_title)
    else:
        # continuous colorbar; NaN points (missing stat or filtered) are excluded.
        # cmap='viridis' runs dark (low) -> bright (high). Use 'viridis_r' to invert.
        nan_mask = np.isnan(color)
        sc = plt.scatter(embedding_2d[~nan_mask, 0], embedding_2d[~nan_mask, 1],
                         c=color[~nan_mask], cmap='viridis', alpha=0.8, s=15)
        plt.colorbar(sc, label=legend_title)

    plt.title(f'UMAP Projection of Airway VAE Embeddings ({colour_plot})', fontsize=16)
    plt.xlabel('UMAP Dimension 1', fontsize=12)
    plt.ylabel('UMAP Dimension 2', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_image, dpi=300)
    print(f"Saved UMAP plot to {output_image}")

    # 5. Return unchanged — HDBSCAN downstream is unaffected
    return embedding_2d, paths