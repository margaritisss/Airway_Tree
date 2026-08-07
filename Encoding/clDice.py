"""
clDice for the Voxel VAE — a soft (differentiable) loss term and a hard
(exact) evaluation metric.

clDice = "centerline Dice" (Shit et al., CVPR 2021). It measures whether the
skeleton (centerline) of the prediction lies inside the ground-truth volume
and vice-versa, so it rewards correct connectivity/topology of tubular trees
(airways, vessels) rather than raw voxel overlap.

Two entry points:

  * SoftClDiceLoss  -> differentiable, for the training objective. Uses a soft
                      skeletonization built from min/max pooling. Takes raw
                      decoder LOGITS (it applies sigmoid internally).

  * hard_cldice_batch -> exact, non-differentiable, for evaluation/logging.
                      Uses real 3D skeletonization from scikit-image on the
                      binarized volumes.

Both expect shape (B, 1, D, H, W); the target is {0, 1}.
"""
from __future__ import annotations   # lets type hints like "-> torch.Tensor" be lazy strings, so annotations never fail at import time on older Python
import numpy as np                   # NumPy: used on the eval path, where scikit-image works on CPU ndarrays, not torch tensors
import torch                         # core PyTorch: tensors, sigmoid, min, no_grad, etc.
import torch.nn as nn                # nn.Module base class, so SoftClDiceLoss integrates like any other layer/loss
import torch.nn.functional as F      # functional ops (max_pool3d, relu) — the pooling primitives that build the soft skeleton


# ---------------------------------------------------------------------------
#  Soft skeletonization (differentiable) — morphological open via min/max pool
# ---------------------------------------------------------------------------

# Erosion of a [0,1] map = min-filter = -maxpool(-x).
# We do it separably along each axis (cheaper and matches the clDice reference implementation).

def _soft_erode(img: torch.Tensor) -> torch.Tensor:
    # Erosion shrinks bright regions by replacing each voxel with the MIN of its neighbourhood.
    # PyTorch has no min-pool, so we use the identity min(x) = -max(-x): negate, max-pool, negate back.
    p1 = -F.max_pool3d(-img, (3, 1, 1), stride=1, padding=(1, 0, 0)) # (3,1,1) kernel pools only along depth (axis D); padding (1,0,0) keeps size.
    p2 = -F.max_pool3d(-img, (1, 3, 1), stride=1, padding=(0, 1, 0)) # (1,3,1) pools only along height (axis H); padding (0,1,0) keeps size.
    p3 = -F.max_pool3d(-img, (1, 1, 3), stride=1, padding=(0, 0, 1)) # (1,1,3) pools only along width (axis W); padding (0,0,1) keeps size.
    # Combining the three separable 1-D erosions by taking their elementwise min  reproduces a full 3x3x3 erosion,
    # but far cheaper than a dense 3D min-filter.
    return torch.min(torch.min(p1, p2), p3)


def _soft_dilate(img: torch.Tensor) -> torch.Tensor:
    # Dilation grows bright regions = MAX over the neighbourhood, which is just a
    # plain 3x3x3 max-pool with stride 1 and padding 1 to preserve spatial size.
    return F.max_pool3d(img, (3, 3, 3), stride=1, padding=1)


def _soft_open(img: torch.Tensor) -> torch.Tensor:
    # Morphological "opening" = erode then dilate. It removes thin protrusions and
    # smooths the shape; (img - opening(img)) exposes the thin parts that opening
    # deleted — i.e. skeleton-like structure. That difference is the key trick below.
    return _soft_dilate(_soft_erode(img))


def soft_skel(img: torch.Tensor, iters: int) -> torch.Tensor:
    """Iterative soft skeleton. `iters` must be >= the thickest radius (in
    voxels) you need to thin down to a centerline; too few leaves tubes only
    partially skeletonized, too many just wastes compute/memory."""
    img1 = _soft_open(img)                       # opening of the original volume
    skel = F.relu(img - img1)                    # first skeleton estimate = the thin bits opening removed (relu clamps negatives to 0)
    for _ in range(iters):                       # each iteration peels one voxel-thick layer off, exposing deeper centerline
        img = _soft_erode(img)                   # erode: shrink the shape by one layer
        img1 = _soft_open(img)                   # re-open the now-thinner shape
        delta = F.relu(img - img1)               # newly exposed skeleton pixels at this thinning level
        skel = skel + F.relu(delta - skel * delta)  # union-like accumulation: add new skeleton, but don't double-count voxels already in `skel` (skel*delta approximates the overlap)
    return skel                                  # accumulated soft centerline map, still differentiable end-to-end


# ---------------------------------------------------------------------------
#  Soft clDice loss (differentiable)
# ---------------------------------------------------------------------------

class SoftClDiceLoss(nn.Module):                 # subclass nn.Module so it plugs into the training loop like any loss
    def __init__(self, iters: int = 8, smooth: float = 1.0,
                 apply_sigmoid: bool = True):
        super().__init__()                       # required: initialise the nn.Module machinery
        self.iters = iters                       # how many thinning iterations soft_skel runs (cover your thickest tube radius)
        self.smooth = smooth                     # Laplace smoothing constant, avoids 0/0 when a skeleton is empty
        self.apply_sigmoid = apply_sigmoid       # if True, treat inputs as logits and squash to [0,1] internally

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # logits: (B,1,D,H,W) raw decoder output; target: (B,1,D,H,W) in {0,1}
        pred = torch.sigmoid(logits) if self.apply_sigmoid else logits  # map logits to probabilities in [0,1]; skeletonization needs a soft mask, not raw logits

        skel_pred = soft_skel(pred, self.iters)  # centerline of the PREDICTION — this carries gradients back to the decoder
        # The target skeleton is constant w.r.t. the weights -> no grad needed.
        # Wrapping it saves a large chunk of autograd memory at 128^3.
        with torch.no_grad():                    # disable autograd tracking for the block below
            skel_true = soft_skel(target, self.iters)  # centerline of the GROUND TRUTH; fixed, so no graph is stored for it

        def _sum(x):                             # helper: collapse each sample's volume to a single scalar
            return x.flatten(1).sum(dim=1)       # flatten(1) merges C,D,H,W into one axis; sum(dim=1) sums per sample -> shape (B,)

        # Topology PRECISION: fraction of the predicted centerline that falls inside the true volume (are the predicted branches real?).
        tprec = (_sum(skel_pred * target) + self.smooth) / (_sum(skel_pred) + self.smooth)
        # Topology SENSITIVITY/recall: fraction of the true centerline covered by the prediction (did we miss branches?).
        tsens = (_sum(skel_true * pred) + self.smooth) / (_sum(skel_true) + self.smooth)
        cldice = 2.0 * tprec * tsens / (tprec + tsens)  # harmonic mean of the two (same F1/Dice-style combination), per sample
        return (1.0 - cldice).mean()             # convert similarity (1=perfect) to a loss (0=perfect), then average over the batch


# ---------------------------------------------------------------------------
#  Hard clDice (exact, non-differentiable) — for evaluation only
# ---------------------------------------------------------------------------

def _skeletonize_3d(vol_bool: np.ndarray) -> np.ndarray:
    """Real 3D skeleton. `skeletonize` handles 3D in skimage >= 0.19; older
    versions expose `skeletonize_3d`."""
    from skimage.morphology import skeletonize   # imported lazily so training never requires scikit-image unless you actually evaluate
    return skeletonize(vol_bool)                 # exact topological thinning to a 1-voxel-wide centerline (not differentiable)


@torch.no_grad()                                 # decorator: no gradients anywhere in this function (it's pure measurement)
def hard_cldice_one(pred_bin: np.ndarray, true_bin: np.ndarray,
                    eps: float = 1e-7) -> float:
    """clDice for a single (D,H,W) pair of binary volumes."""
    pred_bin = pred_bin.astype(bool)             # ensure boolean dtype so & below is a logical AND, not arithmetic
    true_bin = true_bin.astype(bool)             # same for the ground truth
    sp = _skeletonize_3d(pred_bin)               # centerline of the predicted mask
    sl = _skeletonize_3d(true_bin)               # centerline of the true mask
    tprec = (sp & true_bin).sum() / (sp.sum() + eps)   # pred centerline inside GT: (# skeleton voxels landing in GT) / (# skeleton voxels); eps avoids /0
    tsens = (sl & pred_bin).sum() / (sl.sum() + eps)   # GT centerline inside pred: recall of the true tree
    if tprec + tsens < eps:                      # guard: both terms ~0 (e.g. empty prediction) would make the ratio 0/0
        return 0.0                               # define score as 0 in that degenerate case
    return float(2.0 * tprec * tsens / (tprec + tsens))  # harmonic mean -> the clDice score in [0,1]; float() unwraps the numpy scalar


@torch.no_grad()                                 # evaluation only — never builds a graph
def hard_cldice_batch(logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    """Mean hard clDice over a batch of (B,1,D,H,W) logits/targets.
    `threshold` is on the probability (sigmoid(logits)); 0.5 == logits >= 0,matching your existing tp/tn metrics. Lower it if distal branches drop out."""

    probs = torch.sigmoid(logits)                # logits -> probabilities in [0,1]
    pred  = (probs >= threshold).cpu().numpy()   # binarise at the threshold, move to CPU, convert to ndarray for scikit-image
    tgt   = (target >= 0.5).cpu().numpy()        # binarise the target the same way and convert to ndarray
    # loop over the batch (index 0) and channel (index 0, since C=1), scoring each volume independently
    scores = [hard_cldice_one(pred[b, 0], tgt[b, 0]) for b in range(pred.shape[0])]
    return float(np.mean(scores)) if scores else 0.0  # average the per-sample scores; guard against an empty batch