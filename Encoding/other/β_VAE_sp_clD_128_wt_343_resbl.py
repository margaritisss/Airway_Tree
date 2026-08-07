from __future__ import annotations
import torch
import torch.nn as nn
from torchsparse import nn as spnn
from torchsparse.nn.utils import fapply
from dataclasses import dataclass

# ---------------------------------------------------------------------------
#                               Building blocks
# ---------------------------------------------------------------------------

class ResConvBlock(nn.Module):
    """Stride-1 conv block with a projection shortcut.

    body: Conv3d(k=3,s=1,p=1) -> BN -> ELU  (spatial size preserved)
    skip: 1x1 conv to fix channels (identity if in==out)
    out = body(x) + proj(x)
    """
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1):
        super().__init__()
        self.body = _conv_block(in_ch, out_ch, stride=1, padding=padding,
                                kernel_size=kernel_size)
        self.proj = (nn.Conv3d(in_ch, out_ch, kernel_size=1, bias=False)
                     if in_ch != out_ch else nn.Identity())

    def forward(self, x):
        return self.body(x) + self.proj(x)

class SparseELU(nn.ELU):
    def forward(self, x):
        return fapply(x, super().forward)

def _sparse_conv_block(in_ch, out_ch, stride, kernel_size=3):
    """ENCODER block: spnn.Conv3d -> spnn.BatchNorm -> ELU. (no padding arg)"""
    return nn.Sequential( spnn.Conv3d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, bias=False),
                          spnn.BatchNorm(out_ch),
                          SparseELU(inplace=True),)

def _conv_block(in_ch: int, out_ch: int, stride: int, padding: int, kernel_size: int = 3) -> nn.Sequential:
    """Conv3d -> BatchNorm3d -> ELU."""
    return nn.Sequential(nn.Conv3d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
                         nn.BatchNorm3d(out_ch),  
                         nn.ELU(inplace=True),) 

def _deconv_block(in_ch: int, out_ch: int, kernel_size: int = 3, stride: int = 1, padding: int = 1, output_padding: int = 0, activation: bool = True) -> nn.Sequential:
    """ConvTranspose3d -> BatchNorm3d -> (optional ELU)."""
    layers = [ nn.ConvTranspose3d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding, bias=False), nn.BatchNorm3d(out_ch),]
    
    if activation:
        layers.append(nn.ELU(inplace=True))
    return nn.Sequential(*layers) 

# ---------------------------------------------------------------------------
#                                     Model
# ---------------------------------------------------------------------------

class VoxelVAE128(nn.Module):
    """
    Voxel VAE on 128x128x128 occupancy grids. Encoder: 128 -> 64 -> 32 -> 16 -> 8 -> 4 with channels 16/32/64/128/128.
    """

    def __init__(self, num_latents: int = 100, n_channels: int = 1, base_ch: int = 4, fc_dim: int = 343):
        super().__init__()
        self.num_latents = num_latents # number of dimensions in the latent space
        
        # channel sizes -- Doubling at EVERY layer
        c1 = base_ch          # 4
        c2 = base_ch * 2      # 8
        c3 = base_ch * 4      # 16
        c4 = base_ch * 8      # 32
        c5 = base_ch * 16     # 64
        c6 = base_ch * 32     # 128
        c7 = base_ch * 64     # 256
        c8 = base_ch * 128    # 512

        
        # ---- Encoder: SPARSE. Only stride-2 layers downsample (net /16). ----
        self.enc1 = _sparse_conv_block(n_channels, c1, stride=1)
        self.enc2 = _sparse_conv_block(c1, c2, stride=2)
        self.enc3 = _sparse_conv_block(c2, c3, stride=1)
        self.enc4 = _sparse_conv_block(c3, c4, stride=2)
        self.enc5 = _sparse_conv_block(c4, c5, stride=1)
        self.enc6 = _sparse_conv_block(c5, c6, stride=2)
        self.enc7 = _sparse_conv_block(c6, c7, stride=1)
        self.enc8 = _sparse_conv_block(c7, c8, stride=2)

        # ---- Spatially-resolved bottleneck (replaces GlobalAvgPool) ----------
        # GlobalAvgPool collapses the whole /16 feature map to a single [B, c8]
        # vector, throwing away *where* features are — for sparse branching
        # airways that makes distinct trees look alike to the latent, so the
        # decoder falls back on an "average" tree. Instead we scatter the sparse
        # bottleneck onto a fixed dense (B, c8, G, G, G) grid and flatten it, so
        # the FC sees spatial layout — the same thing the dense model does when
        # it flattens its 7^3 bottleneck. Net encoder stride = 2**(#stride-2 layers) = 2**4 = 16, so a 128^3
        # input lands on an 8^3 grid at the bottleneck.

        self._enc_net_stride = 16 
        self.bottleneck_grid = 128 // self._enc_net_stride     # = 8
        flat_dim             = c8 * self.bottleneck_grid ** 3  # 512 * 8^3 = 262,144

        # self.enc_fc       = nn.Sequential(nn.Linear(flat_dim, fc_dim, bias=False), nn.BatchNorm1d(fc_dim), nn.ELU(inplace=True),)  # flattened dense bottleneck -> fc_dim                     
        self.enc_mu       = nn.Linear(flat_dim, num_latents, bias=True)
        self.enc_logsigma = nn.Linear(flat_dim, num_latents, bias=True)

        # ---- Decoder: 9 Layers (Mirror image) ----
        self.dec_fc               = nn.Sequential(nn.Linear(num_latents, fc_dim, bias=False), nn.BatchNorm1d(fc_dim),nn.ELU(inplace=True),) # 100 -> 343
        self._dec_unflatten_shape = (1, 7, 7, 7)

        # Note: We halve the channels at every step now as we work our way back up
        self.dec1 = ResConvBlock( 1, c8, kernel_size=3, padding=1)              # 7 -> 7     | Ch: 1 -> c8
        self.dec2 = _deconv_block(c8, c7, kernel_size=3, stride=2, padding=0)   # 7 -> 15    | Ch: c8 -> c7
        self.dec3 = ResConvBlock(c7, c6, kernel_size=3, padding=1)              # 15 -> 15   | Ch: c7 -> c6
        self.dec4 = _deconv_block(c6, c5, kernel_size=3, stride=2, padding=0)   # 15 -> 31   | Ch: c6 -> c5
        self.dec5 = ResConvBlock(c5, c4, kernel_size=3, padding=1)              # 31 -> 31   | Ch: c5 -> c4
        self.dec6 = _deconv_block(c4, c3, kernel_size=3, stride=2, padding=0)   # 31 -> 63   | Ch: c4 -> c3
        self.dec7 = ResConvBlock(c3, c2, kernel_size=3, padding=1)              # 63 -> 63   | Ch: c3 -> c2
        self.dec8 = _deconv_block(c2, c1, kernel_size=4, stride=2, padding=0)   # 63 -> 128  | Ch: c2 -> c1 (KERNEL=4!)
        
        # Final layer: standard convolution to map down to output channel (1)
        self.dec9 = nn.Sequential( nn.Conv3d(c1, n_channels, kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm3d(n_channels))

        self._init_weights()

    def _to_dense_bottleneck(self, h):
        """Scatter the sparse bottleneck onto a fixed dense (B, C, G, G, G) grid
        and flatten, so the latent sees spatial layout instead of a single
        global average.

        Coordinate convention (torchsparse >= 2.1): ``h.C`` is (N, 4) ordered
        ``[batch, x, y, z]``. On torchsparse < 2.1 the order is ``[x, y, z, batch]``
        AND the bottleneck coords are spaced by the net stride rather than
        contracted to 0..G-1; both are handled below. If you are on < 2.1, swap
        the two marked lines.
        """
        grid       = self.bottleneck_grid     # 8
        net_stride = self._enc_net_stride      # 16

        coords = h.C.long()                    # (N, 4)
        feats  = h.F                           # (N, C)
        C = feats.shape[1]

        batch = coords[:, 0]                   # torchsparse >= 2.1: batch first
        xyz   = coords[:, 1:4]
        # torchsparse < 2.1 instead:
        #   batch = coords[:, 3]; xyz = coords[:, 0:3]

        # Map raw coords onto a 0..grid-1 integer grid. The spacing is a known
        # property of the net (the net stride), NOT something to infer from the
        # occupied cells — inferring from gaps silently squashes the grid when a
        # sample's active sites aren't adjacent.
        xyz = xyz - xyz.amin(dim=0, keepdim=True)        # 0-based, shared frame
        if int(xyz.max()) >= grid:                       # stride-spaced (< 2.1)
            xyz = xyz // net_stride
        xyz = xyz.clamp(0, grid - 1)

        B = int(batch.max().item()) + 1
        dense = feats.new_zeros((B, grid, grid, grid, C))
        # accumulate=True: if two sites ever collapse to one cell, sum them
        dense.index_put_((batch, xyz[:, 0], xyz[:, 1], xyz[:, 2]), feats,
                         accumulate=True)
        dense = dense.permute(0, 4, 1, 2, 3).contiguous()  # (B, C, X, Y, Z)
        return dense.flatten(1)                            # (B, C * grid^3)

    def encode(self, x):
        h = self.enc1(x)
        h = self.enc2(h)
        h = self.enc3(h)
        h = self.enc4(h)
        h = self.enc5(h)
        h = self.enc6(h)
        h = self.enc7(h)
        h = self.enc8(h)
        h = self._to_dense_bottleneck(h)   # sparse /16 feature map -> flat dense (B, c8*G^3)
        # h = self.enc_fc(h)
        mu       = self.enc_mu(h)
        logsigma = self.enc_logsigma(h).clamp(-10.0, 10.0)
        return mu, logsigma

    def reparameterize(self, mu, logsigma):
        if self.training:
            return mu + torch.exp(logsigma) * torch.randn_like(mu)
        return mu

    def decode(self, z):
        h = self.dec_fc(z)
        h = h.view(-1, *self._dec_unflatten_shape)
        h = self.dec1(h)
        h = self.dec2(h)
        h = self.dec3(h)
        h = self.dec4(h)
        h = self.dec5(h)
        h = self.dec6(h)
        h = self.dec7(h)
        h = self.dec8(h)
        logits = self.dec9(h)
        return logits

    def forward(self, x):
        mu, logsigma = self.encode(x)
        z = self.reparameterize(mu, logsigma)
        return self.decode(z), mu, logsigma

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d, nn.Linear)):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, spnn.Conv3d):
                nn.init.xavier_normal_(m.kernel)   # optional; else keep its default

# ---------------------------------------------------------------------------
#                                   Loss
# ---------------------------------------------------------------------------

@dataclass # dataclass is a convenient way to bundle multiple outputs together 
class VAELossOutput:
                    total: torch.Tensor
                    recon: torch.Tensor
                    kl:    torch.Tensor

def weighted_bce_with_logits(logits, target_binary, gamma=0.99, eps=1e-7):

    o   = torch.clamp(torch.sigmoid(logits), eps, 1.0 - eps)
    t   = target_binary
    pos = gamma * t * torch.log(o)
    neg = (1.0 - gamma) * (1.0 - t) * torch.log(1.0 - o)
    return -(pos + neg).flatten(1).sum(dim=1).mean()     # Sum over voxels, average over batch -> per-sample reconstruction NLL

def kl_divergence(mu, logsigma):
    # Sum over latent dims, average over batch -> per-sample KL
    return (-0.5 * (1.0 + 2.0 * logsigma - mu.pow(2) - torch.exp(2.0 * logsigma))).sum(dim=1).mean()

def vae_loss(logits: torch.Tensor, 
             target_binary: torch.Tensor, 
             mu: torch.Tensor,
             logsigma: torch.Tensor,
             beta: float = 1.0,
             gamma: float = 0.99,
             use_kl: bool = True) -> VAELossOutput:
    """
    Full VAE objective: weighted BCE reconstruction + (optional) KL.

    L2 weight decay is NOT included here — add it via the optimizer's
    `weight_decay` argument (the original used `cfg['reg'] = 0.001`).

    Parameters
    ----------
    logits        : raw decoder output, (B, 1, 32, 32, 32)
    target_binary : binary {0,1} target voxels, same shape as logits
    mu, logsigma  : latent means and log-sigmas, (B, num_latents)
    beta          : weight for the KL divergence term
    gamma         : positive-class weight in the BCE; 0.98 matches the released
                    code, 0.97 matches the paper text.
    use_kl        : whether to add the KL term. The released code makes this
                    optional via `cfg['kl_div']` (default False), but the paper
                    describes it as part of the loss, so default True here.
    """
    recon = weighted_bce_with_logits(logits, target_binary, gamma=gamma)
    kl    = kl_divergence(mu, logsigma)
    total = recon + beta *kl if use_kl else recon
    return VAELossOutput(total=total, recon=recon, kl=kl)