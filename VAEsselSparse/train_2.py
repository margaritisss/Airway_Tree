import os
os.environ.setdefault("ATTN_BACKEND", "flash_attn")   # or "xformers"; set BEFORE importing the model
import random
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchsparse.utils.collate import sparse_collate_fn
from vaessel_2 import Encoder, Decoder, to_sp
from data_2 import VesselDataset, list_segmentations


# ===========================================================================
# Config (paper: 1000 epochs, AdamW, lr 1e-4, c=2, 512x512x832, 80/10/10 split)
# ===========================================================================
DATA_ROOTS  = ["/home/ids/gmargari-24/airway_project/Data/Registered_on_Template_22_23/Affine_registered/ATM22_512",
               "/home/ids/gmargari-24/airway_project/Data/Registered_on_Template_22_23/Affine_registered/AIIB23_512",]          # <-- point these at your segmentations
TARGET_SHAPE = (512, 512, 832)
RESAMPLE     = False        # set True (+ scipy) if your masks are not already 0.5 mm isotropic
LABELS       = None         # e.g. {1} to keep a specific label; None -> mask > threshold

EPOCHS       = 1000
LR           = 1e-4
BETA         = 10         # KL weight; the paper does not specify a value -> tune this
BATCH_SIZE   = 1            # organ-level 512^3 volumes are big; start at 1, raise if memory allows
NUM_WORKERS  = 4
GRAD_CLIP    = 1.0
CKPT         = "vaesselsparse_best.pt"
SEED         = 0
device       = "cuda"
amp          = torch.autocast("cuda", dtype=torch.bfloat16)   # FlashAttention needs fp16/bf16


# ===========================================================================
# Loss: Eq. 6 reconstruction on Omega = Cx ∪ Cx_hat, + Eq. 5 KL.
# ===========================================================================
def _vox_key(coords, grid):
    """Unique int64 key per voxel. coords (N,4)=[b,x,y,z]; grid=(H,W,D)."""
    H, W, D    = grid
    b, x, y, z = (coords[:, 0].long(), coords[:, 1].long(), coords[:, 2].long(), coords[:, 3].long()) # .long() is important for large volumes (512^3) what it does
    return ((b * H + x) * W + y) * D + z # we 


def reconstruction_loss(logits, gt_coords, grid, absent_logit=-9.0):
    """BCE over Omega = Cx ∪ Cx_hat (Eq. 6). With the subdivide decoder, every GT
    voxel's coarse parent is in Cz and subdivide regenerates that parent's full
    8x8x8 block, so Cx ⊆ Cx_hat and `absent_logit` is essentially never used --
    it stays only as a safety term for any GT voxel the decoder failed to cover."""
    pred_keys = _vox_key(logits.coords, grid) # 
    pred_log  = logits.feats[:, 0].float()    #
    gt_keys   = _vox_key(gt_coords, grid)

    omega       = torch.unique(torch.cat([pred_keys, gt_keys]))   # sorted
    target      = torch.isin(omega, gt_keys).float()              # 1 on Cx, else 0
    logit_omega = torch.full_like(target, float(absent_logit))
    pos         = torch.searchsorted(omega, pred_keys)
    logit_omega = logit_omega.index_copy(0, pos, pred_log)        # diff. wrt pred_log
    return F.binary_cross_entropy_with_logits(logit_omega, target)


def kl_loss(mu, logvar):
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())


@torch.no_grad()
def dice(logits, gt_coords, grid, thr=0.5):
    """Hard Dice between thresholded reconstruction and GT (over their key sets)."""
    
    keep = torch.sigmoid(logits.feats[:, 0].float()) > thr
    pred = _vox_key(logits.coords, grid)[keep]
    gt   = _vox_key(gt_coords, grid)
    if pred.numel() == 0 and gt.numel() == 0:
        return 1.0
    inter = torch.isin(pred, gt).sum().item()
    return 2.0 * inter / (pred.numel() + gt.numel() + 1e-8)


# ===========================================================================
@torch.no_grad()
def evaluate(enc, dec, loader, grid):
    enc.eval(); dec.eval()
    tot_loss = tot_dice = n = 0
    for batch in loader:
        ts = batch["input"].to(device)
        x  = to_sp(ts)                          # subdivide decoder needs no spatial_range
        gt = x.coords
        with amp:
            z, mu, logvar = enc(x)              # eval mode -> z = mu (no sampling)
            logits = dec(z)
        tot_loss += (reconstruction_loss(logits, gt, grid) + BETA * kl_loss(mu, logvar)).item()
        tot_dice += dice(logits, gt, grid)
        n += 1
    enc.train(); dec.train()
    return tot_loss / max(n, 1), tot_dice / max(n, 1)


def main():
    random.seed(SEED); torch.manual_seed(SEED)

    paths = list_segmentations(DATA_ROOTS)
    assert len(paths) > 0, f"No segmentations found under {DATA_ROOTS}"
    random.shuffle(paths)
    n_val       = max(1, int(0.10 * len(paths)))                # 80/10/10 (test held out separately)
    n_test      = max(1, int(0.10 * len(paths)))
    val_paths   = paths[:n_val]
    test_paths  = paths[n_val:n_val + n_test]               # reserved; not used in training
    train_paths = paths[n_val + n_test:]
    print(f"train/val/test = {len(train_paths)}/{len(val_paths)}/{len(test_paths)}")

    common   = dict(target_shape=TARGET_SHAPE, resample=RESAMPLE, labels=LABELS)
    train_ds = VesselDataset(train_paths, **common) 
    val_ds   = VesselDataset(val_paths,   **common)
    GRID     = train_ds.grid

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  collate_fn=sparse_collate_fn, num_workers=NUM_WORKERS, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=1,          shuffle=False, collate_fn=sparse_collate_fn, num_workers=NUM_WORKERS)

    enc = Encoder(in_channels=1).to(device)
    dec = Decoder(out_channels=1).to(device)
    opt = torch.optim.AdamW(list(enc.parameters()) + list(dec.parameters()), lr=LR)

    best_dice = -1.0
    for epoch in range(EPOCHS):
        enc.train(); dec.train()
        running = 0.0
        for it, batch in enumerate(train_loader):
            ts = batch["input"].to(device)               # raw torchsparse tensor (full res)
            x  = to_sp(ts)
            gt = x.coords                                # Cx (full res) for the loss

            with amp:                                    # FlashAttention -> bf16
                z, mu, logvar = enc(x)                   # Eq. 1-2
                logits        = dec(z)                   # Eq. 3-4 (logits at full res)
            loss = reconstruction_loss(logits, gt, GRID) + BETA * kl_loss(mu, logvar)

            opt.zero_grad(set_to_none=True)  # what is does is 
            loss.backward() 
            torch.nn.utils.clip_grad_norm_(list(enc.parameters()) + list(dec.parameters()), GRAD_CLIP)
            opt.step()
            running += loss.item()

        val_loss, val_dice = evaluate(enc, dec, val_loader, GRID)
        print(f"epoch {epoch:04d} | train_loss {running/max(len(train_loader),1):.4f} " f"| val_loss {val_loss:.4f} | val_dice {val_dice:.4f}", flush=True)


if __name__ == "__main__":
    main()