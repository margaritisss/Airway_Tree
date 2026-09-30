#!/usr/bin/env python3
"""DGCI SDF loss (Zhang et al., MICCAI 2024) for the DeepSDF auto-decoder, without beta.

    L_sdf = L_sdf-val + L_sdf-geo

    L_sdf-val = w_s   * sum_{p in Omega}      |F(p; a_i) - s'|
              + w_phi * sum_{p in Omega \\ S}  exp(-delta * |F(p; a_i)|)
    L_sdf-geo = w_n   * sum_{p in S}          (1 - cos(grad F(p; a_i), n'))
              + w_eik * sum_{p in Omega}      | ||grad F(p; a_i)||_2 - 1 |

Each sum is normalised by the size of its point set over the whole batch, so the
weights keep the scale they have in DGCI / SIREN (w_s=3e3, w_phi=5e2, w_n=1e2, w_eik=5e1)
and the result does not depend on --batch_split.

Point kinds (column 7 of a packed sample row, see deep_sdf.data.SDFSurfaceSamples):
    KIND_FREE    = 0   off-surface point with ground-truth SDF   (Omega \\ S, has s')
    KIND_SURFACE = 1   surface point with ground-truth normal    (S, s' = 0, has n')
    KIND_UNIFORM = 2   uniform point without ground truth        (Omega \\ S, no s')
"""
import torch  # autograd
import torch.nn.functional as F  # cosine similarity

KIND_FREE = 0  # off-surface, SDF known
KIND_SURFACE = 1  # on-surface, normal known
KIND_UNIFORM = 2  # off-surface, nothing known (only phi + Eikonal)

DEFAULT_WEIGHTS = {"sdf": 3e3, "phi": 5e2, "normal": 1e2, "eikonal": 5e1}  # paper, Sec. 3


def sdf_gradient(pred_sdf, xyz):
    ones = torch.ones_like(pred_sdf)  # sum of outputs -> per-point input gradients
    # create_graph=True keeps the gradient differentiable, so losses on it train the net
    grads = torch.autograd.grad(pred_sdf, xyz, grad_outputs=ones, create_graph=True)
    return grads[0]  # (N, 3) = grad_p F(p; a_i)


class DGCILoss:
    def __init__(self, weights=None, delta=100.0, clamp_dist=None, phi_on="offsurface"):
        self.w = dict(DEFAULT_WEIGHTS)  # start from the paper's weights
        self.w.update(weights or {})  # override any given in specs.json
        self.delta = delta  # sharpness of phi (paper: delta >> 1)
        self.clamp_dist = clamp_dist  # None = plain L1 as in Eq. 3
        if phi_on not in ("offsurface", "uniform"):  # guard against typos in specs.json
            raise ValueError("PhiOn must be 'offsurface' or 'uniform'")
        self.phi_on = phi_on  # which off-surface points phi penalises

    def set_counts(self, kind):
        """Call once per batch, before chunking: sizes of each point set in the whole batch."""
        has_sdf = (kind == KIND_FREE) | (kind == KIND_SURFACE)  # points with a known s'
        if self.phi_on == "offsurface":  # paper: every point in Omega \ S
            phi_set = (kind == KIND_FREE) | (kind == KIND_UNIFORM)
        else:  # DGCI's released code: only the uniform points
            phi_set = kind == KIND_UNIFORM
        self.n_sdf = max(int(has_sdf.sum()), 1)  # max(.., 1) avoids dividing by zero
        self.n_phi = max(int(phi_set.sum()), 1)  # denominator for the phi term
        self.n_surface = max(int((kind == KIND_SURFACE).sum()), 1)  # denominator, normal term
        self.n_all = max(int(kind.numel()), 1)  # denominator for the Eikonal term

    def __call__(self, pred_sdf, xyz, sdf_gt, normals_gt, kind):
        """pred_sdf (N,1) raw decoder output; xyz (N,3) the leaf that fed the decoder;
        sdf_gt (N,1); normals_gt (N,3); kind (N,) int. Returns (loss, dict of weighted terms)."""
        device = pred_sdf.device  # compute everything where the prediction is
        sdf_gt = sdf_gt.to(device)  # s'
        normals_gt = normals_gt.to(device)  # n'
        kind = kind.to(device)  # point kinds
        pred = pred_sdf.squeeze(-1)  # (N,) for easier masking
        target = sdf_gt.squeeze(-1)  # (N,)

        grad = sdf_gradient(pred_sdf, xyz).to(device)  # gradient of the UNclamped output

        # ---- Eq. 3, first term: |F - s'| over points with known SDF ----
        has_sdf = (kind == KIND_FREE) | (kind == KIND_SURFACE)  # uniform points have no s'
        pred_l1 = pred  # default: unclamped, as written in the paper
        target_l1 = target  # default: unclamped
        if self.clamp_dist is not None:  # optional DeepSDF-style truncation
            pred_l1 = pred.clamp(-self.clamp_dist, self.clamp_dist)  # truncate prediction
            target_l1 = target.clamp(-self.clamp_dist, self.clamp_dist)  # truncate target
        l1 = (pred_l1 - target_l1).abs()  # per-point absolute error
        loss_sdf = (l1 * has_sdf).sum() / self.n_sdf  # mean over points with s'

        # ---- Eq. 3, second term: phi(F) over off-surface points ----
        if self.phi_on == "offsurface":  # paper's Omega \ S
            phi_mask = (kind == KIND_FREE) | (kind == KIND_UNIFORM)
        else:  # released DGCI code
            phi_mask = kind == KIND_UNIFORM
        phi = torch.exp(-self.delta * pred.abs())  # ~1 when F ~ 0 away from the surface
        loss_phi = (phi * phi_mask).sum() / self.n_phi  # mean over the phi set

        # ---- Eq. 4, first term: normal alignment on the surface ----
        is_surface = kind == KIND_SURFACE  # the set S
        cos = F.cosine_similarity(grad, normals_gt, dim=-1, eps=1e-8)  # S_cos(grad F, n')
        loss_normal = ((1.0 - cos) * is_surface).sum() / self.n_surface  # mean over S

        # ---- Eq. 4, second term: Eikonal over every point ----
        eik = (grad.norm(dim=-1) - 1.0).abs()  # | ||grad F|| - 1 |
        loss_eik = eik.sum() / self.n_all  # mean over Omega

        terms = {  # weighted terms, as they enter the total
            "sdf": self.w["sdf"] * loss_sdf,
            "phi": self.w["phi"] * loss_phi,
            "normal": self.w["normal"] * loss_normal,
            "eikonal": self.w["eikonal"] * loss_eik,
        }
        total = terms["sdf"] + terms["phi"] + terms["normal"] + terms["eikonal"]  # L_sdf
        return total, {k: v.item() for k, v in terms.items()}  # tensor for backward, floats for logs
