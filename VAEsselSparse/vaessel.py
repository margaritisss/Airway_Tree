from torch import nn
import torch
import torch.nn.functional as F
from torchsparse import SparseTensor
from torchsparse import nn as spnn
import numpy as np
from Direct3D_S2.direct3d_s2.modules import sparse as sp
from Direct3D_S2.direct3d_s2.modules.sparse.transformer import SparseTransformerBlock

class AbsolutePositionEmbedder(nn.Module):
    """Parameter-free 3D sinusoidal positional encoding (Direct3D-S2 style)."""
    def __init__(self, channels: int, in_channels: int = 3):
        super().__init__()
        self.channels    = channels
        self.in_channels = in_channels
        self.freq_dim    = channels // in_channels // 2 
        freqs            = torch.arange(self.freq_dim, dtype=torch.float32) / self.freq_dim
        self.register_buffer("freqs", 1.0 / (10000 ** freqs), persistent=False)

    def _sin_cos(self, x):
        out = torch.outer(x, self.freqs.to(x.device))
        return torch.cat([torch.sin(out), torch.cos(out)], dim=-1)

    def forward(self, coords, factor: float = None):   # coords: (N, 3) float/int xyz
        x = coords.float()
        if factor is not None:
            x = x * factor
        N, D = x.shape
        assert D == self.in_channels
        embed = self._sin_cos(x.reshape(-1)).reshape(N, -1)
        if embed.shape[1] < self.channels:
            embed = torch.cat(
                [embed, torch.zeros(N, self.channels - embed.shape[1], device=embed.device)], dim=-1)
        return embed


class ResSparseBlock(nn.Module):
    """Residual sparse 3D CNN block: sparsity-preserving Conv3d + GroupNorm + ReLU."""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, num_groups=8, dilation=1):
        super().__init__()
        
        self.main = nn.Sequential( spnn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, dilation=dilation),
                                   spnn.GroupNorm(num_groups, out_channels),
                                   spnn.ReLU(True),
                                   spnn.Conv3d(out_channels, out_channels, kernel_size, dilation=dilation),
                                   spnn.GroupNorm(num_groups, out_channels),)
        # projection shortcut when shape/stride changes
        if in_channels != out_channels or np.prod(stride) != 1:  
            self.shortcut = nn.Sequential( spnn.Conv3d(in_channels, out_channels, 1, stride=stride), spnn.GroupNorm(num_groups, out_channels),)
        else:
            self.shortcut = nn.Identity() 

        self.relu = spnn.ReLU(True)

    def forward(self, x: SparseTensor) -> SparseTensor:
        return self.relu(self.main(x) + self.shortcut(x))
    

class EncoderStage(nn.Module):
    
    def __init__(self, in_channels, out_channels, num_groups=8):
        super().__init__()
        
        self.res  = ResSparseBlock(in_channels, in_channels, num_groups=num_groups)                  # ResSparseBlock^(l): residual, stride 1, stays at current resolution
        self.down = nn.Sequential( spnn.Conv3d(in_channels, out_channels, kernel_size=2, stride=2),  # Down^(l): strided sparse conv that halves resolution
                                   spnn.GroupNorm(num_groups, out_channels),
                                   spnn.ReLU(True),)
        
    def forward(self, x):
        x = self.res(x)    # X̃^(l+1)
        x = self.down(x)   # X^(l+1)
        return x


class Encoder(nn.Module):   

    """Sparse encoder: three residual downsampling blocks (rs = 8 = 2x2x2)."""

    def __init__(self, in_channels, channels=(64, 128, 256, 384),latent_channels = 2, num_groups=8): #is the numebr of channels corect?
        super().__init__()

        # stem: 1 -> 64, submanifold (stride 1 preserves sparsity), k=3 aggregates local geometry
        self.stem = nn.Sequential(spnn.Conv3d(in_channels, channels[0], 3, stride=1),
                                  spnn.GroupNorm(num_groups, channels[0]),
                                  spnn.ReLU(True),)
        
         # three residual downsampling blocks -> overall spatial factor 2*2*2 = 8
        self.stages = nn.ModuleList([EncoderStage(channels[0], channels[1], num_groups=num_groups),   # res@64,  down 64->128
                                     EncoderStage(channels[1], channels[2], num_groups=num_groups),   # res@128, down 128->256
                                     EncoderStage(channels[2], channels[3], num_groups=num_groups),]) # res@256, down 256->384 # Stage 4: 256 channels -> 384 channels. # Reaches the target bottleneck dimension for the transformers.

        self.pos_embedder = AbsolutePositionEmbedder(channels[-1])  # 384
        self.window_size  = 8
        num_blocks        = 3
        num_heads         = 8

        self.attn_blocks = nn.ModuleList([SparseTransformerBlock(channels    = channels[-1],          # 384
                                                                num_heads    = num_heads,              # 6
                                                                mlp_ratio    = 4.0,
                                                                attn_mode    = "windowed",             # <-- windowed sparse attn (NOT ssa/full)
                                                                window_size  = self.window_size,       # 8 -> 8x8x8 windows
                                                                shift_window = (self.window_size // 2) * (i % 2),  # 0,4,0,4,... Swin shifted windows
                                                                use_rope     = False,                  # we add APE explicitly (paper uses PE)
                                                                qk_rms_norm  = False,)
        for i in range(num_blocks)])

        # VAE head: "two linear layers on {t_n}". Paper uses c = 2.
        self.to_mu     = sp.SparseLinear(channels[-1], latent_channels)   # 384 -> 2
        self.to_logvar = sp.SparseLinear(channels[-1], latent_channels)   # 384 -> 2
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps                       # z = µ + σ ⊙ ε

    def forward(self, x):
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        # x now is the bottleneck sparse tensor X^(L) = (C_z, F_z)
        coords  = x.C[:, 1:4].float()                      # (N, 3) xyz coords --- p_n = (x, y, z); check your column order!
        pe      = self.pos_embedder(coords)
        h       = sp.SparseTensor(x.F + pe, x.C.int())  # (N, 4) with batch index preserved
        # h       = sp.SparseTensor(x.F + pe, coords.int())  # repo wrapper, scale (1,1,1)
        # x.F     = x.F + pe                               # propavly you need only this line, test it, you might not need the rest of them 
        # x       = SparseTensor(x.F + pe, stride = x.s)   # add position embedding to features -- t̃_n = F_z(p_n) + PE(p_n)
        # x.cmaps = x.cmaps                                # (preserve coord maps if your version needs it)
        for block in self.attn_blocks:
            h = block(h)
        
        h       = h.replace(F.layer_norm(h.feats, h.feats.shape[-1:]))   # optional, matches reference -- I am not sure if I need this
        mu      = self.to_mu(h).feats                        # (N, 2)
        logvar  = self.to_logvar(h).feats                    # (N, 2)
        z_feats = self.reparameterize(mu, logvar) if self.training else mu
        z       = h.replace(z_feats)                         # keeps Cz, stride/scale, layout, caches

        return z, mu, logvar
    

class DecoderStage(nn.Module):
    """Mirror of EncoderStage, run backwards: Up^(l) then ResSparseBlock^(l-1). EncoderStage did:  res (stride 1)  ->  down (strided conv, /2).
    DecoderStage does: up (transposed conv, x2)  ->  res (stride 1). This matches Eq. 4: Y~^(l-1) = Up(Y^(l)),  Y^(l-1) = ResSparseBlock(Y~^(l-1))."""
 
    def __init__(self, in_channels, out_channels, num_groups=8):
        super().__init__()
        # Up^(l): transposed strided sparse conv that DOUBLES resolution. transposed=True makes this a generative deconvolution that EXPANDS the
        # active coordinate set (the decoder predicts which fine voxels exist). If instead you want to reuse the encoder's coordinates (U-Net style, only
        # valid when an encoder pass is available), pair this with the cached kmap from the matching Down conv instead of generating coords.
        self.up = nn.Sequential(spnn.Conv3d(in_channels, out_channels, kernel_size=2, stride=2, transposed=True),
                                spnn.GroupNorm(num_groups, out_channels),
                                spnn.ReLU(True), )
        
        # ResSparseBlock^(l-1): residual refinement at the new (finer) resolution.
        self.res = ResSparseBlock(out_channels, out_channels, num_groups=num_groups)
 
    def forward(self, x: SparseTensor) -> SparseTensor:
        x = self.up(x)     # Y~^(l-1)
        x = self.res(x)    # Y^(l-1)
        return x
 
 
class Decoder(nn.Module):
    """Sparse decoder: 
    attention refinement at the bottleneck (Eq. 3),then three transposed-conv upsampling stages (Eq. 4), then a 1x1x1 head.
    Channels mirror the encoder reversed: 384 -> 256 -> 128 -> 64.
    """
 
    def __init__(self, out_channels=1, channels=(384, 256, 128, 64), latent_channels=2, num_groups=8):
        super().__init__()
 
        # ---- input projection: lift z (latent_channels) back to working width ----
        # Inverse of the encoder's to_mu (384 -> 2). Needed because attention runs at channels[0] (384) but z only has `latent_channels` (2) features, so we
        # cannot add a 384-dim PE to a 2-dim feature without this first.
        self.from_latent = sp.SparseLinear(latent_channels, channels[0]) # 2 -> 384
 
        # ---- bottleneck attention (Eq. 3): refine latent tokens BEFORE upsampling ----
        self.pos_embedder = AbsolutePositionEmbedder(channels[0])  # 384
        self.window_size  = 8
        num_blocks        = 3      
        num_heads         = 8
 
        self.attn_blocks = nn.ModuleList([ SparseTransformerBlock(channels     = channels[0],                         # 384
                                                                  num_heads    = num_heads,                           # 8
                                                                  mlp_ratio    = 4.0,
                                                                  attn_mode    = "windowed",                          # windowed sparse attn
                                                                  window_size  = self.window_size,                    # 8x8x8 windows
                                                                  shift_window = (self.window_size // 2) * (i % 2),   # 0,4,0,... Swin shift
                                                                  use_rope     = False,                               # APE added explicitly
                                                                  qk_rms_norm  = False,)
                                                                
        for i in range(num_blocks)])
 
        # ---- upsampling stages (Eq. 4): mirror encoder reversed ----
        self.stages = nn.ModuleList([DecoderStage(channels[0], channels[1], num_groups=num_groups),   # up 384 -> 256
                                     DecoderStage(channels[1], channels[2], num_groups=num_groups),   # up 256 -> 128
                                     DecoderStage(channels[2], channels[3], num_groups=num_groups),]) # up 128 ->  64
 
        # ---- final 1x1x1 sparse conv -> occupancy LOGITS (apply sigmoid outside) ----
        # Return logits, not probabilities, so you can use BCEWithLogitsLoss on the union support Omega = Cx U Cx_hat. Apply sigmoid only at inference.
        self.head = spnn.Conv3d(channels[3], out_channels, kernel_size=1, stride=1)
 
    def forward(self, z: SparseTensor) -> SparseTensor:
        # z: bottleneck sparse tensor on coords Cz, with `latent_channels` features.
        h = self.from_latent(z)                          # (N, 384), still on Cz
 
        coords = h.C[:, 1:4].float()                     # (N, 3) xyz; col 0 is batch idx
        pe     = self.pos_embedder(coords)
        h      = h.replace(h.feats + pe)                 # u~_n = z(p_n) + PE(p_n)
 
        for block in self.attn_blocks:                   # Eq. 3
            h = block(h)
 
        for stage in self.stages:                        # Eq. 4
            h = stage(h)                                 # each doubles resolution
 
        logits = self.head(h)                            # occupancy logits at full res
        return logits                                    # -> sigmoid / BCEWithLogits outside
 
