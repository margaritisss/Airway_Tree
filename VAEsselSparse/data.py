import torch
from torch.utils.data import Dataset
from torchsparse import SparseTensor


def dense_to_sparse(
    x: torch.Tensor,
    feats: torch.Tensor = None,
    threshold: float = 0.0,
) -> SparseTensor:
    """
    Convert ONE dense volume into the paper's sparse tensor X = (Cx, Fx),
    as a torchsparse SparseTensor.

    x:     (H, W, D) occupancy/segmentation volume. Active set is
           Cx = {p | x(p) > threshold}.  For a {0,1} mask this is x(p) == 1;
           threshold=0.0 is robust to float volumes loaded from .nii.gz.
    feats: optional (C, H, W, D) feature volume. If given, Fx = feats sampled
           at the active voxels. If None, Fx = 1 (one-channel occupancy),
           matching in_channels=1 in your Encoder.

    Returns a SparseTensor with
        coords (N, 3) int32  -> [x, y, z]   (NO batch column)
        feats  (N, C) float32
    The batch index is prepended later by sparse_collate_fn -> (N, 4) = [b,x,y,z],
    which is why the Encoder reads spatial coords as x.C[:, 1:4].
    """
    assert x.dim() == 3, f"expected (H, W, D), got {tuple(x.shape)}"

    coords = torch.nonzero(x > threshold, as_tuple=False).int()   # (N, 3) = [x, y, z]

    if feats is None:
        feat = torch.ones((coords.shape[0], 1), dtype=torch.float32, device=x.device)
    else:
        assert feats.dim() == 4, f"expected feats (C, H, W, D), got {tuple(feats.shape)}"
        i, j, k = coords[:, 0].long(), coords[:, 1].long(), coords[:, 2].long()
        feat = feats[:, i, j, k].t().contiguous().float()          # (N, C)

    return SparseTensor(coords=coords, feats=feat)


class VesselDataset(Dataset):
    def __init__(self, volumes):          # volumes: list/array of (H, W, D) grids
        self.volumes = volumes

    def __len__(self):
        return len(self.volumes)

    def __getitem__(self, i):
        vol = torch.as_tensor(self.volumes[i])   # (H, W, D)
        x   = dense_to_sparse(vol)                 # SparseTensor, coords (N, 3)
        return {"input": x}                      # sparse_collate_fn adds the batch dim