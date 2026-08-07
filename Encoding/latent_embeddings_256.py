from __future__ import annotations
from pyexpat import model
import re
from pathlib import Path
import numpy as np
import nibabel as nib
import torch

# --------------------------------------------------------------------------- #
#  latent-dim detection
# --------------------------------------------------------------------------- #
_LATENT_KEYS = ["num_latents", "n_latents", "latent_dim", "latent_size", "z_dim"]
_LATENT_RE   = re.compile(r"(?:num[_-]?latents|n[_-]?latents|latent[_-]?dim|latent[_-]?size|z[_-]?dim)" 
                        r"\s*[:=]?\s*(\d+)",  re.IGNORECASE,)

def _infer_from_ckpt(ckpt) -> int | None:
    """Look for a latent-dim value stored alongside the weights."""
    if not isinstance(ckpt, dict):
        return None
    for k in _LATENT_KEYS:
        if isinstance(ckpt.get(k), int):
            return ckpt[k]
    for container in ("config", "cfg", "args", "hparams", "hyperparameters"):
        c = ckpt.get(container)
        if c is None:
            continue
        cd = c if isinstance(c, dict) else getattr(c, "__dict__", {})
        for k in _LATENT_KEYS:
            if isinstance(cd.get(k), int):
                return cd[k]
    return None


def _infer_from_log(log_path: Path) -> int | None:
    """Grep train.log for a latent-dim declaration (uses the last match)."""
    if not log_path.is_file():
        return None
    matches = _LATENT_RE.findall(log_path.read_text(errors="ignore"))
    return int(matches[-1]) if matches else None


def _infer_from_state(state: dict) -> int | None:
    """Fall back to the output dim of the mu / logsigma linear head."""
    cands = []
    for k, v in state.items():
        if not hasattr(v, "shape") or v.ndim != 2 or not k.lower().endswith("weight"):
            continue
        kl = k.lower()
        if any(tag in kl for tag in ("mu", "logvar", "logsigma", "fc_z", "z_mean")):
            cands.append(int(v.shape[0]))
    # if the mu head is unambiguous, use it
    return cands[0] if cands and all(c == cands[0] for c in cands) else None


def _detect_num_latents(ckpt, state, log_path, verbose) -> int:
    for label, val in (
        ("checkpoint metadata", _infer_from_ckpt(ckpt)),
        ("train.log", _infer_from_log(log_path)),
        ("weight shapes", _infer_from_state(state)),
    ):
        if val:
            if verbose:
                print(f"[info] num_latents = {val}  (detected from {label})")
            return val
    raise ValueError(
        "Could not auto-detect num_latents. Pass it explicitly, e.g. "
        "extract_embeddings(run_dir, data_dirs, num_latents=64)."
    )


# --------------------------------------------------------------------------- #
#  mask loading (identical to training Dataset)
# --------------------------------------------------------------------------- #
def _load_mask(path: Path) -> np.ndarray:
    img = nib.load(str(path))
    arr = np.asarray(img.dataobj)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D array at {path}, got shape {arr.shape}")
    arr = (arr > 0.5).astype(np.float32)   # defensive binarize, matches training
    return arr[None, ...]                  # (1, D, H, W)



def _find_checkpoint(run_dir: Path, checkpoint_name: str | None = None) -> Path:
    """
    Locate the trained checkpoint inside run_dir.

    Training keeps exactly one `checkpoint_<epoch>.pt` at a time (older ones are
    unlinked on each save). `final.pt` only exists if training ran to completion.
    Priority: explicit name > final.pt > highest-epoch checkpoint_<N>.pt.
    """
    if checkpoint_name is not None:                      # explicit override
        p = run_dir / checkpoint_name
        if not p.is_file():
            raise FileNotFoundError(f"No checkpoint at {p}")
        return p

    final = run_dir / "final.pt"
    if final.is_file():
        return final

    ckpts = list(run_dir.glob("checkpoint_*.pt"))
    if not ckpts:
        raise FileNotFoundError(
            f"No checkpoint in {run_dir} (looked for final.pt and checkpoint_*.pt)."
        )

    def _epoch(p: Path) -> int:                          # sort by epoch number in the name
        m = re.search(r"checkpoint_(\d+)\.pt$", p.name)
        return int(m.group(1)) if m else -1

    return max(ckpts, key=_epoch)                        # highest epoch (usually the only one)

# --------------------------------------------------------------------------- #
#  main entry point
# --------------------------------------------------------------------------- #
def extract_embeddings(
    run_dir,
    data_dirs,
    *,
    num_latents: int | None = None,
    checkpoint_name: str | None = None,   # None -> auto-discover checkpoint_<N>.pt in run_dir
    log_name: str = "train.log",
    batch_size: int = 4,
    device=None,
    model_factory=None,
    save: bool = True,
    out_path=None,
    verbose: bool = True,
) -> dict:
    """
    Freeze a trained VAE encoder and turn each patient mask into its posterior
    mean `mu`.

    Parameters
    ----------
    run_dir : str | Path
        Folder like .../vae_runs/run_876062 holding the checkpoint (+ train.log).
    data_dirs : str | Path | list
        One directory or a list of directories of .nii.gz masks (pooled, in the
        same order used for training).
    num_latents : int, optional
        Latent dimension. Auto-detected per run if left as None.
    checkpoint_name, log_name : str
        File names inside run_dir.
    batch_size : int
        Inference batch size (eval-mode BN uses running stats regardless).
    device : torch.device | str, optional
        Defaults to cuda if available, else cpu.
    model_factory : callable, optional
        `f(num_latents) -> nn.Module`. Defaults to VoxelVAE128; override to use a
        different architecture without editing this file.
    save : bool
        Write an .npz next to the run (or to `out_path`).
    out_path : str | Path, optional
        Where to save. Defaults to run_dir / "embeddings.npz".

    Returns
    -------
    dict with keys:
        mu          (N, num_latents) float32   <- feature matrix for clustering
        logsigma    (N, num_latents) float32   <- posterior-collapse diagnostics
        paths       (N,)             str
        num_latents int
        out_path    str | None
    """
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Not a directory: {run_dir}")

    checkpoint = _find_checkpoint(run_dir, checkpoint_name)
    if verbose:
        print(f"[info] checkpoint: {checkpoint.name}")

    if isinstance(data_dirs, (str, Path)):
        data_dirs = [data_dirs]

    device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    if verbose:
        print(f"[info] run: {run_dir.name}")
        print(f"[info] device: {device}")

    # ---- gather files in a stable, reproducible order ----
    paths: list[Path] = []
    for d in data_dirs:
        d = Path(d)
        if not d.is_dir():
            raise FileNotFoundError(f"Not a directory: {d}")
        paths.extend(sorted(d.glob("*.nii.gz")))
    if not paths:
        raise ValueError(f"No .nii.gz files in {data_dirs}")
    if verbose:
        print(f"[info] found {len(paths)} masks")

    # ---- load checkpoint & resolve latent dim ----
    ckpt = torch.load(str(checkpoint), map_location=device)
    state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt

    if num_latents is None:
        num_latents = _detect_num_latents(ckpt, state, run_dir / log_name, verbose)
    elif verbose:
        print(f"[info] num_latents = {num_latents}  (supplied)")

    # ---- build model ----
    if model_factory is None:
        from Encoding.β_VAE_256 import VoxelVAE256
        model_factory = lambda n: VoxelVAE256(num_latents=n)
    model = model_factory(num_latents).to(device)
    model.load_state_dict(state)
    model.eval()  # <- mu returned directly, BN uses running stats
    if verbose:
        print(f"[info] loaded checkpoint: {checkpoint.name}")

    # ---- encode ----
    mus, logsigmas, kept_paths = [], [], []
    batch, batch_paths = [], []

    @torch.no_grad()
    def flush():
        if not batch:                       # guard: skip an empty final flush
            return
        x_in = torch.from_numpy(np.stack(batch, axis=0)).float().to(device)  # (B, 1, 256, 256, 256)
        mu, logsigma = model.encode(x_in)                                    # no 3*x-1 rescale
        mus.append(mu.cpu().numpy())
        logsigmas.append(logsigma.cpu().numpy())
        kept_paths.extend(batch_paths)
        batch.clear()
        batch_paths.clear()

    for i, p in enumerate(paths):
        try:
            batch.append(_load_mask(p))
            batch_paths.append(str(p))
        except Exception as e:
            print(f"[warn] skipping {p}: {e}")
            continue
        if len(batch) == batch_size:
            flush()
        if verbose and (i + 1) % 25 == 0:
            print(f"[info] encoded {i + 1}/{len(paths)}")
    flush()

    mu = np.concatenate(mus, axis=0).astype(np.float32)
    logsigma = np.concatenate(logsigmas, axis=0).astype(np.float32)
    paths_arr = np.array(kept_paths)

    result = {
        "mu": mu,
        "logsigma": logsigma,
        "paths": paths_arr,
        "num_latents": num_latents,
        "out_path": None,
    }

    if save:
        out_path = Path(out_path) if out_path else run_dir / "embeddings.npz"
        np.savez(out_path, mu=mu, logsigma=logsigma, paths=paths_arr)
        result["out_path"] = str(out_path)
        if verbose:
            print(f"[done] wrote {out_path}  mu shape {mu.shape}")

    # ---- posterior-collapse diagnostic ----
    if verbose:
        var = mu.var(axis=0)
        order = np.argsort(var)[::-1]
        active = int((var > 0.05 * var.max()).sum())
        print(f"[diag] per-dim variance of mu: max={var.max():.4f} min={var.min():.4f}")
        print(f"[diag] ~{active}/{num_latents} dims look active "
              f"(var > 5% of max). Top dims: {order[:10].tolist()}")

    return result