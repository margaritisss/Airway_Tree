"""Per-case mean morphological metrics, projected onto UMAP / t-SNE latents.

Each case folder under CASES_ROOT holds one *_morphological_features.npy file of
shape (n_branches, 6). This module averages those branches per case, aligns the
result to the latent-code row order, and draws one colour-coded scatter per
metric on top of an existing 2D projection.
"""

from pathlib import Path                                        # cleaner than os.path for joins
import numpy as np                                              # arrays + nan-aware means
import pandas as pd                                             # table keyed by subject
import matplotlib.pyplot as plt                                 # the scatter plots
from matplotlib.colors import Normalize                         # shared vmin/vmax per metric

# --------------------------------------------------------------------------- #
# configuration — edit these three to match your data
# --------------------------------------------------------------------------- #

CASES_ROOT = Path("/home/ids/gmargari-24/airway_project/airmorph_cases")  # one folder per tree

METRIC_NAMES = [                                                # column order inside the .npy
    "metric_0",                                                 # <- rename: column 0
    "metric_1",                                                 # <- rename: column 1
    "metric_2",                                                 # <- rename: column 2
    "metric_3",                                                 # <- rename: column 3
    "metric_4",                                                 # <- rename: column 4
    "metric_5",                                                 # <- rename: column 5
]

INVALID_VALUE = -1.0                                            # sentinel written for unmeasurable branches


# --------------------------------------------------------------------------- #
# 1. locating a case folder from a latent-code subject id
# --------------------------------------------------------------------------- #

def subject_to_stem(subject):
    """'AIIB23_30_R_mesh.mat' -> 'AIIB23_30_R'."""
    stem = str(subject).strip()                                 # guard against stray whitespace
    for suffix in ("_mesh.mat", ".mat", "_mesh"):               # strip whichever tail is present
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]                         # chop the tail (negative slice)
            break
    return stem                                                 # what the case folder name ends with


def index_case_dirs(root=CASES_ROOT):
    """Map every case folder under root to its name, once, so lookups are cheap."""
    root = Path(root)                                           # accept str or Path
    if not root.is_dir():
        raise FileNotFoundError(f"cases root not found: {root}")
    return sorted(p for p in root.iterdir() if p.is_dir())      # list of folders, alphabetical


def find_case_dir(stem, case_dirs):
    """Folder 'AIIB23_binary_AIIB23_30_R' matches stem 'AIIB23_30_R' by suffix."""
    exact = [p for p in case_dirs if p.name == stem]             # cheapest case: names agree
    if exact:
        return exact[0]
    suffix = [p for p in case_dirs if p.name.endswith("_" + stem) or p.name.endswith(stem)]
    if len(suffix) == 1:                                         # unambiguous prefixed name
        return suffix[0]
    if len(suffix) > 1:                                          # two folders claim the same case
        raise ValueError(f"ambiguous case folders for '{stem}': {[p.name for p in suffix]}")
    return None                                                  # caller records it as missing


# --------------------------------------------------------------------------- #
# 2. loading and averaging one case
# --------------------------------------------------------------------------- #

def load_case_features(case_dir):
    """Return the (n_branches, 6) array stored in this case folder."""
    expected = case_dir / f"{case_dir.name}_morphological_features.npy"   # the documented name
    path = expected if expected.exists() else None
    if path is None:                                             # fall back to any matching file
        candidates = sorted(case_dir.glob("*_morphological_features.npy"))
        if not candidates:
            raise FileNotFoundError(f"no morphological features .npy in {case_dir}")
        path = candidates[0]
    features = np.load(path, allow_pickle=False).astype(np.float64)       # float64 keeps means stable
    if features.ndim != 2:
        raise ValueError(f"{path} has shape {features.shape}, expected 2D")
    return features


def mean_features(features, n_metrics=6, drop="row"):
    """Average branches, ignoring the -1 sentinel rows. Returns a (n_metrics,) vector."""
    if features.shape[1] != n_metrics:                           # catch a column-count mismatch early
        raise ValueError(f"expected {n_metrics} metric columns, got {features.shape[1]}")
    if drop == "row":                                            # a branch is invalid as a whole
        keep = ~np.any(features == INVALID_VALUE, axis=1)         # rows with no sentinel anywhere
        valid = features[keep]
    else:                                                        # blank out only the sentinel cells
        valid = np.where(features == INVALID_VALUE, np.nan, features)
    if valid.size == 0 or np.all(np.isnan(valid)):               # every branch was unmeasurable
        return np.full(n_metrics, np.nan)                        # NaN propagates to a grey point
    return np.nanmean(valid, axis=0)                             # per-column mean over branches


# --------------------------------------------------------------------------- #
# 3. the table: one row per subject, one column per metric
# --------------------------------------------------------------------------- #

def build_metric_table(subjects, root=CASES_ROOT, metric_names=METRIC_NAMES, drop="row"):
    """Mean metrics for every subject, in the given row order (latent-code order)."""
    case_dirs = index_case_dirs(root)                            # folder list, built once
    rows, missing = [], []                                       # means so far, and unmatched ids
    for subject in subjects:
        stem = subject_to_stem(subject)                          # strip the _mesh.mat tail
        case_dir = find_case_dir(stem, case_dirs)                # locate the folder
        if case_dir is None:
            missing.append(stem)                                 # note it, keep the row aligned
            rows.append(np.full(len(metric_names), np.nan))
            continue
        features = load_case_features(case_dir)                  # (n_branches, 6)
        rows.append(mean_features(features, len(metric_names), drop))
    table = pd.DataFrame(rows, columns=metric_names)             # row i == subjects[i] == latent row i
    table.insert(0, "subject", list(subjects))                   # keep the id next to the values
    if missing:
        print(f"[warn] no case folder for {len(missing)} subjects, e.g. {missing[:5]}")
    n_nan = int(table[metric_names].isna().all(axis=1).sum())    # cases with zero valid branches
    print(f"built metric table: {table.shape[0]} subjects, {n_nan} without usable measurements")
    return table


# --------------------------------------------------------------------------- #
# 4. one colour-coded panel per metric
# --------------------------------------------------------------------------- #

def _robust_limits(values, low=2, high=98):
    """Clip the colour range to percentiles so one outlier can't flatten the map."""
    finite = values[np.isfinite(values)]                         # percentiles ignore NaN badly, so filter
    if finite.size == 0:
        return 0.0, 1.0                                          # nothing to scale, any range will do
    vmin, vmax = np.percentile(finite, [low, high])               # robust ends of the colour bar
    if vmin == vmax:                                             # constant metric, widen slightly
        vmin, vmax = vmin - 0.5, vmax + 0.5
    return float(vmin), float(vmax)


def plot_metric_projection(coords, values, metric_name, projection_name,
                           save_dir, cmap="viridis", robust=True, dpi=300, label=None):
    """Scatter one 2D projection, coloured by a single metric. Saves and returns the path."""
    label = label if label is not None else f"mean {metric_name}"    # counts aren't means, so allow an override
    coords = np.asarray(coords, dtype=np.float64)                # (N, 2) UMAP or t-SNE coordinates
    values = np.asarray(values, dtype=np.float64)                # (N,) mean metric per subject
    if coords.shape[0] != values.shape[0]:
        raise ValueError(f"{coords.shape[0]} points but {values.shape[0]} values")

    known = np.isfinite(values)                                  # points we can colour
    vmin, vmax = _robust_limits(values) if robust else (np.nanmin(values), np.nanmax(values))
    norm = Normalize(vmin=vmin, vmax=vmax)                       # shared scale for dots and colour bar

    fig, ax = plt.subplots(figsize=(8, 6.5))                     # one metric per figure
    ax.scatter(coords[~known, 0], coords[~known, 1],             # draw missing cases underneath
               c="#d9d9d9", marker="o", s=30, linewidths=0.9,
               label="no measurement", zorder=1)
    dots = ax.scatter(coords[known, 0], coords[known, 1],        # the coloured population
                      c=values[known], cmap=cmap, norm=norm,
                      s=45, alpha=0.9, edgecolors="white", linewidths=0.5, zorder=2)

    bar = fig.colorbar(dots, ax=ax)                              # legend for the continuous colour
    bar.set_label(label, fontsize=11)
    ax.set_title(f"{projection_name}: {label}", fontsize=13)
    ax.set_xlabel(f"{projection_name} 1", fontsize=11)
    ax.set_ylabel(f"{projection_name} 2", fontsize=11)
    ax.grid(color="#e5e5e5", linewidth=0.6)
    ax.set_axisbelow(True)                                       # grid behind the points
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if (~known).any():
        ax.legend(loc="best", frameon=False, fontsize=9)         # only mention grey x's if present

    save_dir = Path(save_dir)                                    # accept str or Path
    save_dir.mkdir(parents=True, exist_ok=True)                  # create the output folder if needed
    slug = projection_name.lower().replace("-", "").replace(" ", "_")   # 't-SNE' -> 'tsne'
    out_path = save_dir / f"{slug}_metric_{metric_name}.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)                                               # close or 12 figures pile up in memory
    return out_path


def plot_metric_grid(coords, table, metric_names, projection_name, save_dir, cmap="viridis", robust=True, dpi=300):
    """All six metrics as one 2x3 overview figure — handy for a slide."""
    coords = np.asarray(coords, dtype=np.float64)
    fig, axes = plt.subplots(2, 3, figsize=(19, 11))             # 6 panels, one per metric
    for ax, metric in zip(axes.ravel(), metric_names):
        values = table[metric].to_numpy(dtype=np.float64)        # this metric's column
        known = np.isfinite(values)
        vmin, vmax = _robust_limits(values) if robust else (np.nanmin(values), np.nanmax(values))
        ax.scatter(coords[~known, 0], coords[~known, 1], c="#d9d9d9", marker="o", s=18, zorder=1)
        dots = ax.scatter(coords[known, 0], coords[known, 1], c=values[known], cmap=cmap,
                          norm=Normalize(vmin, vmax), s=26, alpha=0.9,
                          edgecolors="white", linewidths=0.35, zorder=2)
        fig.colorbar(dots, ax=ax, fraction=0.046, pad=0.03)      # per-panel bar, scales differ
        ax.set_title(f"mean {metric}", fontsize=12)
        ax.set_xticks([])                                        # coordinates are unitless, hide ticks
        ax.set_yticks([])
    fig.suptitle(f"{projection_name} of latent codes, coloured by mean morphology", fontsize=15)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    slug = projection_name.lower().replace("-", "").replace(" ", "_")
    out_path = save_dir / f"{slug}_metrics_grid.png"
    fig.tight_layout(rect=[0, 0, 1, 0.96])                       # leave room for the suptitle
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------- #
# 5. one call that does the lot
# --------------------------------------------------------------------------- #

def project_metrics(umap_2d, tsne_2d, subjects, save_dir,
                    root=CASES_ROOT, metric_names=METRIC_NAMES,
                    drop="row", cmap="viridis", make_grid=True):
    """Build the metric table, then write 6 UMAP + 6 t-SNE panels (plus overview grids)."""
    table = build_metric_table(subjects, root, metric_names, drop)        # aligned to latent rows
    written = []                                                 # paths, returned for the caller
    for coords, name in ((umap_2d, "UMAP"), (tsne_2d, "t-SNE")):
        if coords is None:                                       # allow running just one projection
            continue
        for metric in metric_names:
            values = table[metric].to_numpy(dtype=np.float64)    # column -> colour
            written.append(plot_metric_projection(coords, values, metric, name, save_dir, cmap))
        if make_grid:
            written.append(plot_metric_grid(coords, table, metric_names, name, save_dir, cmap))
    print(f"wrote {len(written)} figures to {save_dir}")
    return table, written


# --------------------------------------------------------------------------- #
# 6. whole-tree counts from tree_counts.csv (branches, generations)
# --------------------------------------------------------------------------- #

TREE_COUNT_COLUMNS = ["n_branches", "max_generation"]            # the two whole-tree quantities to plot

TREE_COUNT_LABELS = {                                            # colourbar text, these are not means
    "n_branches": "branches in the tree",
    "max_generation": "deepest generation",
    "n_generations": "number of generations",
}

TREE_COUNT_ROBUST = {                                            # whether to clip the colour range
    "n_branches": True,                                          # long right tail, clip it
    "max_generation": False,                                     # only ~11 distinct values, show them all
    "n_generations": False,
}


def build_tree_count_table(subjects, tree_counts_csv, columns=TREE_COUNT_COLUMNS):
    """Look up each subject's whole-tree counts, returned in latent-code row order."""
    counts = pd.read_csv(tree_counts_csv)                        # written by count_branches_generations.py
    missing_cols = [c for c in columns if c not in counts.columns]
    if missing_cols:
        raise KeyError(f"{tree_counts_csv} has no column(s) {missing_cols}")
    case_names = counts["case"].tolist()                         # 'AIIB23_binary_AIIB23_30_R', ...

    lookup = {}                                                  # stem -> row index in the CSV
    for row_index, case in enumerate(case_names):
        lookup[case] = row_index                                 # exact name, in case ids already match
    rows, missing = [], []                                       # one CSV row index per subject
    for subject in subjects:
        stem = subject_to_stem(subject)                          # 'AIIB23_30_R_mesh.mat' -> 'AIIB23_30_R'
        hit = lookup.get(stem)
        if hit is None:                                          # folder names carry a dataset prefix
            matches = [i for i, c in enumerate(case_names) if c.endswith("_" + stem)]
            if len(matches) > 1:
                raise ValueError(f"ambiguous case rows for '{stem}': {[case_names[i] for i in matches]}")
            hit = matches[0] if matches else None
        if hit is None:
            missing.append(stem)                                 # keep the row, fill it with NaN
            rows.append({c: np.nan for c in columns})
            continue
        rows.append({c: counts.loc[hit, c] for c in columns})    # the counts for this subject

    table = pd.DataFrame(rows, columns=columns).astype(float)    # float so missing rows can hold NaN
    table.insert(0, "subject", list(subjects))                   # row i == latent row i
    if missing:
        print(f"[warn] no tree_counts row for {len(missing)} subjects, e.g. {missing[:5]}")
    print(f"built tree-count table: {table.shape[0]} subjects, {len(missing)} without counts")
    return table


def project_tree_counts(umap_2d, tsne_2d, subjects, tree_counts_csv, save_dir,
                        columns=TREE_COUNT_COLUMNS, cmap="magma", robust=None):
    """Two extra panels per projection: branch count and deepest generation."""
    table = build_tree_count_table(subjects, tree_counts_csv, columns)    # aligned to latent rows
    written = []
    for coords, name in ((umap_2d, "UMAP"), (tsne_2d, "t-SNE")):
        if coords is None:                                       # allow running just one projection
            continue
        for column in columns:
            values = table[column].to_numpy(dtype=np.float64)    # counts -> colour
            label = TREE_COUNT_LABELS.get(column, column)        # plain wording, no 'mean' prefix
            clip = TREE_COUNT_ROBUST.get(column, True) if robust is None else robust
            written.append(plot_metric_projection(coords, values, column, name, save_dir,
                                                  cmap=cmap, robust=clip, label=label))
    print(f"wrote {len(written)} tree-count figures to {save_dir}")
    return table, written

# --------------------------------------------------------------------------- #
# 7. per-cluster summary of every metric (morphology means + tree counts)
# --------------------------------------------------------------------------- #

def merge_on_subject(*tables):
    """Join the metric and tree-count tables side by side, keeping the first one's row order."""
    if not tables:
        raise ValueError("pass at least one table")
    merged = tables[0].copy()                                    # left table sets the row order
    for other in tables[1:]:
        before = len(merged)
        merged = merged.merge(other, on="subject", how="left")   # how='left' never reorders or drops
        if len(merged) != before:                                # a duplicate subject id would fan out
            raise ValueError(f"merge changed the row count {before} -> {len(merged)}: duplicate subjects?")
    return merged


def cluster_summary(labels, *tables, save_dir=None, run_test=True, decimals=3):
    """Mean, SD and valid-n of every metric within every cluster.

    Returns (wide, detailed, merged): 'wide' is the one-row-per-cluster table of means,
    'detailed' adds SD and per-metric valid counts, 'merged' is the subject-level table.
    """
    merged = merge_on_subject(*tables)                           # subject + all value columns
    value_columns = [c for c in merged.columns if c != "subject"]
    labels = np.asarray(labels).astype(int)
    if len(labels) != len(merged):                               # the usual misalignment bug
        raise ValueError(f"{len(labels)} labels but {len(merged)} subjects")
    merged = merged.copy()
    merged["cluster"] = labels                                   # row order is still the latent order

    grouped = merged.groupby("cluster")                          # one group per phenotype
    means = grouped[value_columns].mean()                        # NaN subjects are skipped per column
    stds = grouped[value_columns].std()
    valid = grouped[value_columns].count()                       # how many subjects actually contributed
    sizes = grouped.size().rename("n")                           # cluster size, including NaN subjects

    wide = pd.concat([sizes, means.round(decimals)], axis=1)     # the compact table to read
    detailed = pd.concat(                                        # mean/sd/n side by side per metric
        {col: pd.DataFrame({"mean": means[col], "sd": stds[col], "n": valid[col]})
         for col in value_columns}, axis=1).round(decimals)

    if run_test:                                                 # does the metric differ across clusters?
        from scipy.stats import kruskal                          # non-parametric, no normality assumption
        rows = []
        for col in value_columns:
            samples = [group.dropna().to_numpy() for _, group in grouped[col]]
            samples = [s for s in samples if s.size > 0]         # kruskal rejects empty groups
            if len(samples) < 2:
                rows.append({"metric": col, "kruskal_H": np.nan, "p": np.nan})
                continue
            stat, p = kruskal(*samples)
            rows.append({"metric": col, "kruskal_H": stat, "p": p})
        test_table = pd.DataFrame(rows).set_index("metric")
        test_table["p_bonferroni"] = (test_table["p"] * len(value_columns)).clip(upper=1.0)
        print("\nmetric differs across clusters? (Kruskal-Wallis)")
        print(test_table.round(4).to_string())

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        wide.to_csv(save_dir / "cluster_metric_means.csv")       # the table for the write-up
        merged.to_csv(save_dir / "subject_metrics_with_cluster.csv", index=False)
        print(f"wrote cluster_metric_means.csv and subject_metrics_with_cluster.csv to {save_dir}")

    return wide, detailed, merged


def plot_cluster_heatmap(merged, value_columns=None, save_dir=None,
                         cmap="RdBu_r", dpi=300, filename="cluster_metric_heatmap.png"):
    """Cluster means in SD units, so metrics on wildly different scales sit on one colour bar."""
    if value_columns is None:
        value_columns = [c for c in merged.columns if c not in ("subject", "cluster")]
    values = merged[value_columns].astype(float)                 # subject-level values

    centre = values.mean(axis=0)                                 # global mean per metric
    spread = values.std(axis=0).replace(0, np.nan)               # global SD, guard against a constant column
    z_scores = (values - centre) / spread                        # each metric on a common scale
    z_scores["cluster"] = merged["cluster"].to_numpy()
    grid = z_scores.groupby("cluster")[value_columns].mean()     # cluster mean, in SD units
    raw = merged.groupby("cluster")[value_columns].mean()        # real means, for the annotations
    raw = merged.groupby("cluster")[value_columns].mean()        # real means, for the annotations
    raw_sd = merged.groupby("cluster")[value_columns].std()      # real SDs, same units as the means

    limit = float(np.nanmax(np.abs(grid.to_numpy()))) or 1.0     # symmetric colour range around 0
    fig, ax = plt.subplots(figsize=(1.5 * len(value_columns) + 3, 0.8 * len(grid) + 3))
    image = ax.imshow(grid.to_numpy(), cmap=cmap, vmin=-limit, vmax=limit, aspect="auto")

    ax.set_xticks(range(len(value_columns)))
    ax.set_xticklabels(value_columns, rotation=35, ha="right", fontsize=10)
    ax.set_yticks(range(len(grid)))
    dataset = merged["subject"].str.extract(r"(AIIB23|ATM)", expand=False)  # dataset name inside the subject id
    per_dataset = pd.crosstab(merged["cluster"], dataset).reindex(grid.index, fill_value=0)  # subjects per cluster x dataset
    row_labels = []                                              # one y-axis label per cluster
    for c in grid.index:
        n = int((merged["cluster"] == c).sum())                  # all subjects in this cluster
        parts = [f"{name} {per_dataset.loc[c, name]} ({per_dataset.loc[c, name] / n:.0%})"
                 for name in per_dataset.columns]                # e.g. 'AIIB23 120 (57%)' (:.0% = percent, no decimals)
        row_labels.append(f"cluster {c} (n={n})\n" + ", ".join(parts))  # size on line 1, dataset split on line 2
    ax.set_yticklabels(row_labels, fontsize=10)                  # replaces the plain 'cluster c' labels

    for row in range(grid.shape[0]):                             # print the real mean inside each cell
        for col in range(grid.shape[1]):
            z = grid.iat[row, col]
            text_colour = "white" if abs(z) > 0.6 * limit else "black"   # keep the label readable
            label = f"{raw.iat[row, col]:.2f}\n± {raw_sd.iat[row, col]:.2f}"  # mean on top, SD below
            ax.text(col, row, label, ha="center", va="center",
                    fontsize=9, color=text_colour)                # same styling as before

    bar = fig.colorbar(image, ax=ax)
    bar.set_label("cluster mean, in SD from the cohort mean", fontsize=10)
    ax.set_title("", fontsize=12)
    fig.tight_layout()

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        out_path = save_dir / filename
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return out_path
    return fig