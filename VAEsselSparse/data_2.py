import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from torchsparse import SparseTensor

try:
    import nibabel as nib
except ImportError:
    nib = None


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
def list_segmentations(roots, exts=(".nii.gz", ".nii", ".nrrd", ".mha")):
    """Recursively collect segmentation file paths under one or more directories."""
    if isinstance(roots, str):
        roots = [roots]
    paths = []
    for r in roots:
        for e in exts:
            paths += glob.glob(os.path.join(r, "**", f"*{e}"), recursive=True)
    return sorted(set(paths))


# ---------------------------------------------------------------------------
# One segmentation -> active voxel coordinates in the padded target frame.
# Never materializes the full target_shape dense grid (we keep only coords).
# ---------------------------------------------------------------------------
def load_mask_coords(path, target_shape=(640, 640, 832), threshold=0.5, resample=False, target_spacing=0.5, labels=None):
    """
    Load a binary/segmentation mask, optionally resample to `target_spacing` mm
    isotropic (paper preprocessing), then CENTER-PAD/CROP into `target_shape`,
    returning active-voxel coords (N,3) int32 in the target frame.

    labels: if given (e.g. {1,2}), keep only those label values; else mask > threshold.
    """
    assert nib is not None, "pip install nibabel  (needed to read .nii.gz)"
    img = nib.load(path)
    vol = np.asarray(img.dataobj)
    if vol.ndim == 4:                      # drop a singleton channel/time axis if present
        vol = vol[..., 0]
    mask = np.isin(vol, list(labels)) if labels is not None else (vol > threshold)

    if resample:
        from scipy.ndimage import zoom
        zooms = img.header.get_zooms()[:3]
        factors = tuple(float(z) / target_spacing for z in zooms)
        mask = zoom(mask.astype(np.uint8), factors, order=0) > 0   # order=0: nearest (labels)

    nshape = np.array(mask.shape[:3], dtype=np.int64)
    tshape = np.array(target_shape,   dtype=np.int64)

    coords = np.argwhere(mask).astype(np.int64)          # (N,3) native indices [x,y,z]
    del mask, vol
    if coords.shape[0] == 0:
        return torch.zeros((0, 3), dtype=torch.int32)

    offset = (tshape - nshape) // 2                       # center; negative => crop
    coords = coords + offset
    inb    = np.all((coords >= 0) & (coords < tshape), axis=1)
    coords = coords[inb]
    return torch.from_numpy(coords).contiguous().int()    # (N,3) [x,y,z], NO batch col


# ---------------------------------------------------------------------------
# Dataset: yields the paper's sparse tensor X = (Cx, Fx=1) per volume.
# sparse_collate_fn prepends the batch index -> (N,4)=[b,x,y,z].
# ---------------------------------------------------------------------------
class VesselDataset(Dataset):
    def __init__(self, paths, target_shape=(512, 512, 832), threshold=0.5, resample=False, target_spacing=0.5, labels=None):
        self.paths          = list(paths)
        self.grid           = tuple(int(s) for s in target_shape)   # used to set GRID in train
        self.threshold      = threshold
        self.resample       = resample
        self.target_spacing = target_spacing
        self.labels         = labels

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        coords = load_mask_coords(self.paths[i], self.grid, self.threshold, self.resample, self.target_spacing, self.labels)
        feats  = torch.ones((coords.shape[0], 1), dtype=torch.float32)   # Fx = 1 (occupancy)
        return {"input": SparseTensor(coords=coords, feats=feats), "path": self.paths[i]}


# ---------------------------------------------------------------------------
# Kept for the sanity check / synthetic use: ONE dense volume -> SparseTensor.
# ---------------------------------------------------------------------------
def dense_to_sparse(x: torch.Tensor, feats: torch.Tensor = None, threshold: float = 0.0) -> SparseTensor:
    assert x.dim() == 3, f"expected (H, W, D), got {tuple(x.shape)}"
    coords = torch.nonzero(x > threshold, as_tuple=False).int()        # (N,3)=[x,y,z]
    if feats is None:
        feat = torch.ones((coords.shape[0], 1), dtype=torch.float32, device=x.device)
    else:
        i, j, k = coords[:, 0].long(), coords[:, 1].long(), coords[:, 2].long()
        feat    = feats[:, i, j, k].t().contiguous().float()
    return SparseTensor(coords=coords, feats=feat)