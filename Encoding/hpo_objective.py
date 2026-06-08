"""
Optuna objective for VoxelVAE128 hyperparameter search.

Design choices
--------------
- Validation set: deterministic split of the patient list using a fixed seed,
  so every trial sees the same train/val partition. Reset val DataLoader with
  `augment=False`.
- ASHA-friendly β schedule: `total_iterations` is computed from a fixed
  reference epoch count (`BETA_REFERENCE_EPOCHS`), NOT from this trial's
  `max_epochs`. A trial pruned at epoch 30 has truthfully completed
  30/REFERENCE of its β schedule, comparable across trials of any length.
- Pruning: report validation ELBO at every epoch via trial.report(...), then
  raise TrialPruned() if Optuna asks.
- Determinism within a trial: each trial gets its own seed derived from the
  trial number. The train/val split itself uses a fixed seed independent of
  the trial.
- No data augmentation by default during HPO. Augmentation interacts with
  every other hyperparameter and is best tuned after the rest. Re-enable
  for the final retrain.
"""

from __future__ import annotations
from pathlib import Path
import logging
import time
import numpy as np
import optuna
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

# These imports assume the HPO files live alongside the model + training code.
from VAE_CA_b import VoxelVAE128, vae_loss, cyclical_beta
from train_VAE_CA_b import NiftiMaskDataset, reconstruction_accuracy


# --- Fixed across all trials ---
BETA_REFERENCE_EPOCHS = 150           # the β schedule is defined over this many epochs
VAL_FRACTION          = 0.12          # 50/419 patients to validation
SPLIT_SEED            = 20260101      # fixed; never depends on trial number
TP_WEIGHT             = 0.7   # ranking weight on foreground recall; (1-TP_WEIGHT) on tn

def make_train_val_loaders(
    data_dirs: list[Path],
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
):
    """Build train/val DataLoaders with a fixed patient-level split.

    Both loaders use augment=False here: HPO compares configurations on
    clean data, augmentation is a separate question.
    """
    full_ds   = NiftiMaskDataset(data_dirs, augment=False) # full dataset, to be split into train/val subsets
    n         = len(full_ds)
    rng       = np.random.default_rng(SPLIT_SEED)
    perm      = rng.permutation(n)
    n_val     = max(1, int(round(VAL_FRACTION * n)))
    val_idx   = perm[:n_val]
    train_idx = perm[n_val:]

    train_ds = Subset(full_ds, train_idx.tolist())
    val_ds   = Subset(full_ds, val_idx.tolist())

    train_loader = DataLoader(
                            train_ds,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=num_workers,
                            pin_memory=pin_memory,
                            drop_last=True,)
    
    val_loader = DataLoader(
                            val_ds,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=max(1, num_workers // 2),
                            pin_memory=pin_memory,
                            drop_last=False,)
    
    return train_loader, val_loader, len(train_idx), len(val_idx)


@torch.no_grad()
def evaluate_elbo(model: nn.Module, loader: DataLoader, device, gamma: float) -> dict:
    model.eval()
    total_recon = 0.0
    total_kl    = 0.0
    total_acc   = 0.0
    total_tp    = 0.0
    total_tn    = 0.0
    n_samples   = 0

    for x_bin in loader:
        x_bin = x_bin.to(device, non_blocking=True)
        x_in = 3.0 * x_bin - 1.0
        logits, mu, logsigma = model(x_in)
        out          = vae_loss(logits, x_bin, mu, logsigma, beta=1.0, gamma=gamma, use_kl=True)
        bs           = x_bin.shape[0]
        total_recon  += out.recon.item() * bs
        total_kl     += out.kl.item() * bs
        acc, tp, tn  = reconstruction_accuracy(logits, x_bin)
        total_acc    += acc * bs
        total_tp     += tp  * bs
        total_tn     += tn  * bs
        n_samples    += bs

    n        = max(n_samples, 1)
    val_tp   = total_tp / n
    val_tn   = total_tn / n
    weighted = TP_WEIGHT * val_tp + (1.0 - TP_WEIGHT) * val_tn
    return { "val_recon":        total_recon / n,
             "val_kl":           total_kl / n,
             "val_neg_elbo":     total_recon / n + total_kl / n,
             "val_acc":          total_acc / n,
             "val_tp":           val_tp,
             "val_tn":           val_tn,
             "val_weighted_acc": weighted, }

def objective(trial: optuna.Trial, *, data_dirs, num_workers, max_epochs, log) -> float: # the * forces these to be passed as keyword args, not positional
   
    # --- Sample hyperparameters ---
    
    n_cycles       = trial.suggest_categorical("n_cycles", [1, 2, 4, 6, 8, 10 ,12, 15, 20, 25, 30])
    beta_ratio     = trial.suggest_float("beta_ratio", 0.3, 0.9, step=0.1)
    gamma          = trial.suggest_float("gamma", 0.95, 0.995, step=0.005)
    num_latents    = trial.suggest_categorical("num_latents", [16, 32, 40, 45, 50, 64, 70, 80, 100, 256])
    batch_size     = trial.suggest_categorical("batch_size", [2, 4, 8, 16])
    beta_max       = trial.suggest_categorical("beta_max", [1.0, 2.0, 5.0, 10.0, 25.0, 50.0, 100.0])
    optimizer_name = trial.suggest_categorical("optimizer", ["sgd", "adam"])
    warmup_start_frac = trial.suggest_float("warmup_start_frac", 0.0, 0.20, step=0.05)
    warmup_frac       = trial.suggest_float("warmup_frac", 0.0, 0.20, step=0.05)
    weight_decay = trial.suggest_categorical("weight_decay", [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 5e-3])
   
    if optimizer_name == "sgd":
        peak_lr = trial.suggest_categorical("peak_lr_sgd", [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3])
    else:
        peak_lr = trial.suggest_categorical("peak_lr_adam", [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3])

    log.info("Trial %d sampled: %s", trial.number, trial.params)

    # --- Determinism ---
    seed = 1000 + trial.number
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin    = device.type == "cuda"

    # --- Data ---
    train_loader, val_loader, n_train, n_val = make_train_val_loaders(  data_dirs  = data_dirs,
                                                                        batch_size = batch_size,
                                                                        num_workers= num_workers,
                                                                        pin_memory = pin,)
    
    log.info("Trial %d: train=%d, val=%d, train_batches=%d", trial.number, n_train, n_val, len(train_loader))

    iters_per_epoch   = len(train_loader) 
    total_train_iters = max_epochs * iters_per_epoch
    warmup_iters      = int(round(warmup_frac * total_train_iters))
    
    # --- Model ---
    model = VoxelVAE128(num_latents=num_latents).to(device)

    # --- Optimizer ---
    initial_lr = peak_lr * warmup_start_frac

    if optimizer_name == "sgd":
        optimizer = torch.optim.SGD( model.parameters(),
                                    lr=initial_lr,      
                                    momentum=0.9,
                                    nesterov=True,
                                    weight_decay=weight_decay,)
        
    else:
        optimizer = torch.optim.Adam( model.parameters(),
                                      lr=initial_lr,
                                      weight_decay=weight_decay,)

    # β schedule is over a fixed reference, NOT this trial's max_epochs
    total_iterations = BETA_REFERENCE_EPOCHS * len(train_loader)

    itr = 0
    best_val_weighted  = float("-inf")

    for epoch in range(max_epochs):
        # LR jump after one warmup epoch

        model.train()
        t0 = time.time()
        for x_bin in train_loader:
            itr += 1
        
            # Linear LR warmup: ramp from initial_lr to peak_lr over warmup_iters
            if warmup_iters > 0 and itr <= warmup_iters:
                frac = itr / warmup_iters
                lr_now = initial_lr + frac * (peak_lr - initial_lr)
            else:
                lr_now = peak_lr
            for g in optimizer.param_groups:
                g["lr"] = lr_now

            beta = cyclical_beta(
                iteration=itr,
                total_iterations=total_iterations,
                n_cycles=n_cycles,
                ratio=beta_ratio,
                beta_max=beta_max,
                shape="linear",
            )
            x_bin = x_bin.to(device, non_blocking=True)
            x_in = 3.0 * x_bin - 1.0
            logits, mu, logsigma = model(x_in)
            out = vae_loss(logits, x_bin, mu, logsigma,
                           beta=beta, gamma=gamma, use_kl=True)

            optimizer.zero_grad(set_to_none=True)
            # Guard against NaN explosions from extreme β/LR combinations
            if not torch.isfinite(out.total):
                log.warning("Trial %d epoch %d: non-finite loss, pruning", trial.number, epoch)
                raise optuna.TrialPruned()
            out.total.backward()
            optimizer.step()

        # --- Validation pass ---
        val = evaluate_elbo(model, val_loader, device, gamma=gamma)
        dt  = time.time() - t0

        log.info(
            "Trial %d epoch %d/%d β_end=%.2f val_recon=%.1f val_kl=%.2f "
            "val_acc=%.4f tp=%.4f tn=%.4f w_acc=%.4f (%.1fs)",
            trial.number, epoch, max_epochs - 1, beta,
            val["val_recon"], val["val_kl"],
            val["val_acc"], val["val_tp"], val["val_tn"],
            val["val_weighted_acc"], dt,
        )

        trial.report(val["val_weighted_acc"], step=epoch)   # report the maximized metric
        if val["val_weighted_acc"] > best_val_weighted:     # > now, not 
            best_val_weighted = val["val_weighted_acc"]

        if trial.should_prune():
            log.info("Trial %d pruned at epoch %d", trial.number, epoch)
            raise optuna.TrialPruned()
        
    del model, optimizer
    torch.cuda.empty_cache()
    return best_val_weighted
