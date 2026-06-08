"""
Latent diagnostic for VoxelVAE128 checkpoints.

What it does
------------
1. Loads one or two trained checkpoints.
2. Encodes all 419 patients to (mu, logsigma) vectors. Encoding uses the
   *posterior mean* mu, NOT the sampled z, because mu is what HDBSCAN/k-means
   will see at inference time.
3. Computes three diagnostics that determine whether the latent is useful
   for phenotyping:
     - Active-dimension count: how many of the 100 latents carry signal
       across patients (Var_patients(mu_d) > threshold).
     - Signal-to-noise ratio: per-dimension std(mu) divided by mean
       exp(logsigma). Values >> 1 mean inter-patient differences dominate
       within-patient encoder noise — the precondition for any clustering
       to be meaningful.
     - Cluster validity: k-means + silhouette across k in {3..8}. Uses
       only the active dimensions, not all 100.
4. If two checkpoints are given (e.g. beta=1 and beta=20), prints a
   side-by-side comparison and a recommendation.

Outputs
-------
- Per-checkpoint .npz file with mu (419, 100), logsigma (419, 100), and
  file paths, so you can reload for further analysis (UMAP, HDBSCAN
  parameter sweeps, AirMorph alignment, etc.) without re-encoding.
- Stdout report.

Usage
-----
    python latent_diagnostic.py \\
        --data-dirs /path/to/AIIB23_128 /path/to/ATM22_128 \\
        --ckpt-a /path/to/run_beta1/checkpoint_epoch149.pt \\
        --ckpt-b /path/to/run_beta20/checkpoint_epoch149.pt \\
        --label-a "beta=1" \\
        --label-b "beta=20" \\
        --out-dir /path/to/diagnostics

Single checkpoint also works — just omit --ckpt-b.
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

from VAE_CA_b import VoxelVAE128
from train_VAE_CA_b import NiftiMaskDataset


# Active-dim threshold: a latent dim counts as "active" if its variance across
# patients exceeds this. 0.01 is a common rule of thumb; the script also
# reports counts at other thresholds so you can sanity-check.
ACTIVE_VAR_THRESHOLD = 0.01


def encode_all_patients(model, loader, device):
    """Forward all patients through the encoder, collect mu and logsigma.

    Uses model.eval() so BatchNorm uses running stats; no sampling, no dropout.
    """
    model.eval()
    mus, logsigmas = [], []
    with torch.no_grad():
        for x_bin in loader:
            x_bin = x_bin.to(device, non_blocking=True)
            x_in = 3.0 * x_bin - 1.0  # same rescaling as training
            mu, logsigma = model.encode(x_in)
            mus.append(mu.cpu().numpy())
            logsigmas.append(logsigma.cpu().numpy())
    return np.concatenate(mus, axis=0), np.concatenate(logsigmas, axis=0)


def load_checkpoint_into_model(ckpt_path: Path, num_latents: int, device):
    """Robust checkpoint loader: tolerates several common save formats.

    Tries dict keys 'model_state_dict', 'state_dict', 'model', and finally
    treats the loaded object itself as a state_dict if it's a dict of tensors.
    """
    obj = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = None
    if isinstance(obj, dict):
        for k in ("model_state", "model_state_dict", "state_dict", "model"):
            if k in obj and isinstance(obj[k], dict):
                state = obj[k]
                break
        if state is None:
            # Maybe the dict IS the state dict
            if all(isinstance(v, torch.Tensor) for v in obj.values()):
                state = obj
    if state is None:
        raise RuntimeError(
            f"Could not locate a state_dict inside {ckpt_path}. "
            f"Object type: {type(obj)}, keys: {list(obj.keys()) if isinstance(obj, dict) else 'n/a'}"
        )

    model = VoxelVAE128(num_latents=num_latents).to(device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  WARNING: {len(missing)} missing keys, e.g. {missing[:3]}")
    if unexpected:
        print(f"  WARNING: {len(unexpected)} unexpected keys, e.g. {unexpected[:3]}")
    return model


def compute_diagnostics(mu: np.ndarray, logsigma: np.ndarray, label: str) -> dict:
    """Compute and print the three diagnostics for one checkpoint.

    Parameters
    ----------
    mu       : (N, D) array of posterior means per patient
    logsigma : (N, D) array of posterior log-sigmas per patient
    label    : human-readable name for printing
    """
    N, D = mu.shape

    # --- 1. Active dimensions ---
    # Variance across patients, per dimension
    per_dim_var = mu.var(axis=0, ddof=1)   # (D,)
    per_dim_std = np.sqrt(per_dim_var)     # (D,)

    n_active_001 = int((per_dim_var > 0.01).sum())
    n_active_01  = int((per_dim_var > 0.1).sum())
    n_active_1   = int((per_dim_var > 1.0).sum())

    # --- 2. Signal-to-noise ratio per dim ---
    # mean encoder noise sigma per dimension, across patients
    sigma = np.exp(logsigma)               # (N, D)
    sigma_mean_per_dim = sigma.mean(axis=0)  # (D,)
    snr_per_dim = per_dim_std / np.maximum(sigma_mean_per_dim, 1e-9)  # (D,)

    # Aggregate over active dims (the ones that actually matter)
    active_mask = per_dim_var > ACTIVE_VAR_THRESHOLD
    if active_mask.sum() > 0:
        snr_active_mean = float(snr_per_dim[active_mask].mean())
        snr_active_median = float(np.median(snr_per_dim[active_mask]))
    else:
        snr_active_mean = float("nan")
        snr_active_median = float("nan")

    # --- 3. Clustering on active dims ---
    # We try a few k values. Silhouette is between -1 and 1; higher is better.
    # Use only active dims — feeding 70 collapsed dims worth of noise hurts
    # silhouette regardless of whether real cluster structure exists.
    sil_scores = {}
    if active_mask.sum() >= 2:
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
        X = mu[:, active_mask]
        for k in [3, 4, 5, 6, 8]:
            if k >= N:
                continue
            try:
                km = KMeans(n_clusters=k, n_init=20, random_state=0).fit(X)
                # silhouette breaks if any cluster has only one sample
                _, counts = np.unique(km.labels_, return_counts=True)
                if (counts < 2).any():
                    sil_scores[k] = float("nan")
                else:
                    sil_scores[k] = float(silhouette_score(X, km.labels_))
            except Exception as e:
                sil_scores[k] = float("nan")
                print(f"    k={k} clustering failed: {e}")
    else:
        sil_scores = {k: float("nan") for k in [3, 4, 5, 6, 8]}

    # --- Print report for this checkpoint ---
    print()
    print(f"========== {label}  (N={N} patients, D={D} latent dims) ==========")
    print(f"  Active dimensions  (Var_patients(mu) > threshold)")
    print(f"    > 0.01:  {n_active_001:>3} / {D}")
    print(f"    > 0.1:   {n_active_01:>3} / {D}")
    print(f"    > 1.0:   {n_active_1:>3} / {D}")
    print(f"  Per-dim std(mu) percentiles: "
          f"p10={np.percentile(per_dim_std, 10):.3f}  "
          f"p50={np.percentile(per_dim_std, 50):.3f}  "
          f"p90={np.percentile(per_dim_std, 90):.3f}")
    print(f"  Per-dim mean sigma percentiles: "
          f"p10={np.percentile(sigma_mean_per_dim, 10):.3f}  "
          f"p50={np.percentile(sigma_mean_per_dim, 50):.3f}  "
          f"p90={np.percentile(sigma_mean_per_dim, 90):.3f}")
    if active_mask.sum() > 0:
        print(f"  SNR on active dims (std(mu) / mean sigma)")
        print(f"    mean:   {snr_active_mean:.2f}")
        print(f"    median: {snr_active_median:.2f}")
    else:
        print(f"  SNR: cannot compute (zero active dimensions — full collapse)")
    print(f"  k-means + silhouette on active dims:")
    for k, s in sil_scores.items():
        marker = ""
        if not np.isnan(s):
            if s > 0.5: marker = "   <-- strong"
            elif s > 0.25: marker = "   <-- moderate"
            elif s > 0.1: marker = "   <-- weak"
        print(f"    k={k}: silhouette = {s:.4f}{marker}")

    return {
        "label": label,
        "n_patients": int(N),
        "n_latents": int(D),
        "n_active_001": n_active_001,
        "n_active_01":  n_active_01,
        "n_active_1":   n_active_1,
        "snr_active_mean":   snr_active_mean,
        "snr_active_median": snr_active_median,
        "silhouette": sil_scores,
        "per_dim_std":  per_dim_std.tolist(),
        "per_dim_sigma": sigma_mean_per_dim.tolist(),
    }


def compare_and_recommend(d_a: dict, d_b: dict) -> None:
    """Side-by-side table and a heuristic recommendation."""
    print()
    print("=" * 70)
    print(f"SIDE BY SIDE  —  {d_a['label']}  vs  {d_b['label']}")
    print("=" * 70)
    print(f"{'metric':<35} {d_a['label']:>14} {d_b['label']:>14}")
    print("-" * 70)
    print(f"{'active dims (var > 0.01)':<35} {d_a['n_active_001']:>14} {d_b['n_active_001']:>14}")
    print(f"{'active dims (var > 0.1)':<35} {d_a['n_active_01']:>14} {d_b['n_active_01']:>14}")
    print(f"{'active dims (var > 1.0)':<35} {d_a['n_active_1']:>14} {d_b['n_active_1']:>14}")
    print(f"{'SNR mean (active dims)':<35} {d_a['snr_active_mean']:>14.2f} {d_b['snr_active_mean']:>14.2f}")
    print(f"{'SNR median (active dims)':<35} {d_a['snr_active_median']:>14.2f} {d_b['snr_active_median']:>14.2f}")
    for k in [3, 4, 5, 6, 8]:
        sa = d_a["silhouette"].get(k, float("nan"))
        sb = d_b["silhouette"].get(k, float("nan"))
        winner = ""
        if not (np.isnan(sa) or np.isnan(sb)):
            winner = f"  <-- {d_a['label'] if sa > sb else d_b['label']}"
        print(f"{'silhouette k=' + str(k):<35} {sa:>14.4f} {sb:>14.4f}{winner}")

    # --- Heuristic recommendation ---
    # Three cases worth flagging directly.
    print()
    print("INTERPRETATION")
    print("-" * 70)

    def collapsed(d):
        return d["n_active_001"] < 3 or d["snr_active_median"] < 0.5

    if collapsed(d_a) and collapsed(d_b):
        print("Both runs show signs of posterior collapse or near-collapse.")
        print("Neither latent is suitable for clustering as-is. Consider:")
        print("  - Lower beta_max (try 0.1 - 5 range)")
        print("  - Reduce num_latents (the encoder may be hiding capacity)")
        print("  - Check that the model is actually fitting the data")
    elif collapsed(d_b):
        print(f"{d_b['label']} appears collapsed (few active dims, low SNR).")
        print(f"{d_a['label']} is healthier in the geometric sense — use it as")
        print(f"the basis for clustering, or back off beta_max toward the value")
        print(f"used by {d_a['label']}.")
    elif collapsed(d_a):
        print(f"{d_a['label']} has many active dims but the latent geometry")
        print(f"may be uninformative for clustering (vanilla-autoencoder regime).")
        print(f"{d_b['label']} is a healthier VAE if its silhouette beats {d_a['label']}.")
    else:
        # Both look reasonable — judge by silhouette
        best_a = max((v for v in d_a["silhouette"].values() if not np.isnan(v)), default=float("nan"))
        best_b = max((v for v in d_b["silhouette"].values() if not np.isnan(v)), default=float("nan"))
        print(f"Both checkpoints have viable latents.")
        print(f"  Best silhouette under {d_a['label']}: {best_a:.4f}")
        print(f"  Best silhouette under {d_b['label']}: {best_b:.4f}")
        if not (np.isnan(best_a) or np.isnan(best_b)):
            if abs(best_a - best_b) < 0.02:
                print(f"  Difference is small — either is defensible.")
            else:
                better = d_a["label"] if best_a > best_b else d_b["label"]
                print(f"  {better} produces noticeably better cluster separation.")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dirs", type=Path, nargs="+", required=True,
                   help="Patient mask directories (same as training).")
    p.add_argument("--ckpt-a", type=Path, required=True,
                   help="First checkpoint (e.g. beta=1 run).")
    p.add_argument("--ckpt-b", type=Path, default=None,
                   help="Optional second checkpoint (e.g. beta=20 run). "
                        "If given, a side-by-side comparison is printed.")
    p.add_argument("--label-a", type=str, default="ckpt-A")
    p.add_argument("--label-b", type=str, default="ckpt-B")
    p.add_argument("--num-latents", type=int, default=100,
                   help="Must match the trained model's num_latents.")
    p.add_argument("--batch-size", type=int, default=16,
                   help="Encoding batch size. Larger is faster; eval-only "
                        "so memory is much lower than during training.")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--out-dir", type=Path, default=Path("./latent_diag"),
                   help="Where to save the .npz embeddings and the json summary.")
    return p.parse_args()


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # IMPORTANT: same dataset and shuffle=False so the patient order is fixed
    # across both checkpoints. Otherwise row i of mu_A and row i of mu_B are
    # different patients and any per-patient comparison is meaningless.
    ds = NiftiMaskDataset(args.data_dirs, augment=False)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=(device.type == "cuda"))
    print(f"Loaded {len(ds)} patients")

    # --- Checkpoint A ---
    print(f"\nLoading {args.label_a} from {args.ckpt_a}")
    model_a = load_checkpoint_into_model(args.ckpt_a, args.num_latents, device)
    print(f"Encoding all patients with {args.label_a}...")
    mu_a, logsigma_a = encode_all_patients(model_a, loader, device)
    print(f"  mu_a shape: {mu_a.shape}, logsigma_a shape: {logsigma_a.shape}")
    np.savez(args.out_dir / f"latent_{args.label_a.replace('=', '').replace(' ', '_')}.npz",
             mu=mu_a, logsigma=logsigma_a)
    diag_a = compute_diagnostics(mu_a, logsigma_a, args.label_a)
    del model_a
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # --- Checkpoint B, if given ---
    diag_b = None
    if args.ckpt_b is not None:
        print(f"\nLoading {args.label_b} from {args.ckpt_b}")
        model_b = load_checkpoint_into_model(args.ckpt_b, args.num_latents, device)
        print(f"Encoding all patients with {args.label_b}...")
        mu_b, logsigma_b = encode_all_patients(model_b, loader, device)
        print(f"  mu_b shape: {mu_b.shape}, logsigma_b shape: {logsigma_b.shape}")
        np.savez(args.out_dir / f"latent_{args.label_b.replace('=', '').replace(' ', '_')}.npz",
                 mu=mu_b, logsigma=logsigma_b)
        diag_b = compute_diagnostics(mu_b, logsigma_b, args.label_b)

        compare_and_recommend(diag_a, diag_b)

    # --- Save summary as json for later reference ---
    summary = {"a": diag_a, "b": diag_b}
    with open(args.out_dir / "diagnostic_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=lambda o: float(o) if isinstance(o, np.floating) else o)
    print(f"\nSaved embeddings and summary to {args.out_dir}")


if __name__ == "__main__":
    main()
