from torch import nn
import torch
import torch.nn.functional as F
import torchsparse  # kept only for the to_sp type hint
from direct3d_s2.modules import sparse as sp
from direct3d_s2.modules.sparse.transformer import SparseTransformerBlock


def _expand_spatial_range(x: 'sp.SparseTensor', factor: int = 2) -> 'sp.SparseTensor':
    """Multiply the carried torchsparse spatial_range by `factor` on the spatial dims,
    so the generative transposed conv doesn't clip the upsampled coords to the
    downsampled range (the cause of the 79->39->19->9 collapse)."""
    sr = getattr(x.data, "spatial_range", None)
    if sr is None:
        return x
    sr = list(sr)
    start = 1 if len(sr) == 4 else 0          # skip batch dim if present: (B,X,Y,Z) vs (X,Y,Z)
    for i in range(start, len(sr)):
        sr[i] = int(sr[i]) * factor
    x.data.spatial_range = tuple(sr)
    return x

def to_sp(x: torchsparse.SparseTensor, spatial_range=None) -> sp.SparseTensor:
    """
    Wraps a raw torchsparse.SparseTensor into the direct3d_s2 sp.SparseTensor,
    and optionally sets the spatial_range to prevent coordinate collapse during upsampling.
    """
    if spatial_range is not None:
        x.spatial_range = spatial_range
    return sp.SparseTensor(x)

class AbsolutePositionEmbedder(nn.Module):
    """Parameter-free 3D sinusoidal positional encoding."""
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

    def forward(self, coords, factor: float = None):   # coords: (N, 3) xyz
        x = coords.float()
        if factor is not None:
            x = x * factor
        N, D = x.shape
        assert D == self.in_channels
        embed = self._sin_cos(x.reshape(-1)).reshape(N, -1)
        if embed.shape[1] < self.channels:                      # safety pad
            embed = torch.cat(
                [embed, torch.zeros(N, self.channels - embed.shape[1], device=embed.device)], dim=-1)
        return embed


class ResSparseBlock(nn.Module):
    """Residual sparse 3D CNN block: (Conv3d k3 -> GN -> ReLU -> Conv3d k3 -> GN) + shortcut, then ReLU.

    Stays at the current resolution (stride 1). padding = k//2 keeps the active
    coordinate set unchanged (submanifold-like), so `main(x)` and `shortcut(x)`
    share coords and the wrapper's `+` (element-wise on feats) is well-defined.
    Downsampling is handled OUTSIDE this block (in EncoderStage), matching the
    paper's separation of ResSparseBlock (Eq.1 left) and Down (Eq.1 right).
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, num_groups=8):
        super().__init__()
        pad = kernel_size // 2 
        self.main = nn.Sequential( sp.SparseConv3d(in_channels,  out_channels, kernel_size, padding=pad),
                                   sp.SparseGroupNorm(num_groups, out_channels),
                                   sp.SparseReLU(),
                                   sp.SparseConv3d(out_channels, out_channels, kernel_size, padding=pad),
                                   sp.SparseGroupNorm(num_groups, out_channels),)
        
        # projection shortcut only when channels change (stride is always 1 here)
        if in_channels != out_channels:
            self.shortcut = sp.SparseConv3d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()
        self.relu = sp.SparseReLU()

    def forward(self, x: 'sp.SparseTensor') -> 'sp.SparseTensor':
        return self.relu(self.main(x) + self.shortcut(x))


class EncoderStage(nn.Module):
    """Eq. 1:  X~ = ResSparseBlock(X);  X' = Down(X~).  Halves resolution once."""
    def __init__(self, in_channels, out_channels, num_groups=8):
        super().__init__()
        self.res  = ResSparseBlock(in_channels, in_channels, num_groups=num_groups)   # refine @ current res
        # Down: strided sparse conv (k=2, s=2) -> halves resolution (paper Eq. 1).
        # Works standalone (no cache needed). To mirror Direct3D-S2 exactly and
        # keep _scale symmetric with SparseSubdivide, replace the line below with:
        #     self.down = nn.Sequential(sp.SparseDownsample(2),
        #                               sp.SparseConv3d(in_channels, out_channels, 3, padding=1),
        #                               sp.SparseGroupNorm(num_groups, out_channels),
        #                               sp.SparseReLU())
        self.down = nn.Sequential( sp.SparseConv3d(in_channels, out_channels, 2, stride=2),
            sp.SparseGroupNorm(num_groups, out_channels),
            sp.SparseReLU(),
        )

    def forward(self, x: 'sp.SparseTensor') -> 'sp.SparseTensor':
        x = self.res(x)    # X~^(l+1)
        x = self.down(x)   # X^(l+1)
        return x


class Encoder(nn.Module):
    """Sparse encoder: stem + 3 residual downsampling stages (rs = 2*2*2 = 8) + windowed attention + VAE head."""
    def __init__(self, in_channels=1, channels=(64, 128, 256, 384), latent_channels=2, num_groups=8):
        super().__init__()

        # stem: in -> 64, k=3 stride 1 (padding=1 keeps coords), aggregates local geometry
        self.stem = nn.Sequential(
            sp.SparseConv3d(in_channels, channels[0], 3, padding=1),
            sp.SparseGroupNorm(num_groups, channels[0]),
            sp.SparseReLU(),
        )

        # three residual downsampling stages -> overall spatial factor 8
        self.stages = nn.ModuleList([ EncoderStage(channels[0], channels[1], num_groups=num_groups),   # res@64,  down 64->128
                                      EncoderStage(channels[1], channels[2], num_groups=num_groups),   # res@128, down 128->256
                                      EncoderStage(channels[2], channels[3], num_groups=num_groups),])   # res@256, down 256->384

        self.pos_embedder = AbsolutePositionEmbedder(channels[-1])  # 384
        self.window_size  = 8
        num_blocks        = 3   # 3 here + 3 in decoder = 6 total (paper)
        num_heads         = 8

        # Eq. 2: windowed sparse self-attention at the bottleneck. attn_mode="windowed"
        # is valid (SparseMultiHeadAttention asserts {"full","serialized","windowed"});
        # int shift_window is broadcast to (s,s,s) internally -> Swin-style shift.
        self.attn_blocks = nn.ModuleList([
            SparseTransformerBlock(
                channels     = channels[-1],                          # 384
                num_heads    = num_heads,                             # 8
                mlp_ratio    = 4.0,
                attn_mode    = "windowed",
                window_size  = self.window_size,                      # 8x8x8 windows
                shift_window = (self.window_size // 2) * (i % 2),     # 0,4,0,... shifted windows
                use_rope     = False,                                 # APE added explicitly
                qk_rms_norm  = False,
            )
            for i in range(num_blocks)])

        # VAE head: "two linear layers on {t_n}" (paper). c = latent_channels (= 2).
        self.to_mu     = sp.SparseLinear(channels[-1], latent_channels)   # 384 -> 2
        self.to_logvar = sp.SparseLinear(channels[-1], latent_channels)   # 384 -> 2
        # Optional but recommended (reference zero-inits its output layer so the
        # initial posterior ~ prior and KL is well-behaved at the start of training):
        # nn.init.zeros_(self.to_mu.weight);     nn.init.zeros_(self.to_mu.bias)
        # nn.init.zeros_(self.to_logvar.weight); nn.init.zeros_(self.to_logvar.bias)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps                       # z = µ + σ ⊙ ε

    def forward(self, x: 'sp.SparseTensor'):
        # x must be an sp.SparseTensor: feats (N,in_channels), coords (N,4)=[batch,x,y,z]
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)                                   # X^(L) = (Cz, Fz)

        coords = x.coords[:, 1:4].float()                  # (N,3) xyz; col 0 = batch
        pe     = self.pos_embedder(coords)
        h      = x.replace(x.feats + pe)                   # t~_n = Fz(p_n) + PE(p_n); keeps scale/layout/caches

        for block in self.attn_blocks:                     # Eq. 2
            h = block(h)

        h       = h.replace(F.layer_norm(h.feats, h.feats.shape[-1:]))   # matches reference pre-head LN
        mu      = self.to_mu(h).feats                      # (N, c)
        logvar  = self.to_logvar(h).feats                  # (N, c)
        z_feats = self.reparameterize(mu, logvar) if self.training else mu
        z       = h.replace(z_feats)                       # latent on Cz, keeps scale/layout/caches
        return z, mu, logvar


class DecoderStage(nn.Module):
 
    def __init__(self, in_channels, out_channels, num_groups=8):
        super().__init__()
      
        self.up   = sp.SparseInverseConv3d(in_channels, out_channels, kernel_size=2, stride=2) # the upsampling layer. Because both the kernel_size and stride are set to 2, this layer effectively doubles the spatial resolution of the 3D voxel grid while mapping the feature dimension from in_channels to out_channels
        self.norm = sp.SparseGroupNorm(num_groups, out_channels) 
        self.act  = sp.SparseReLU()              
        self.res  = ResSparseBlock(out_channels, out_channels, num_groups=num_groups)

    def forward(self, x):
            x = _expand_spatial_range(x, factor=2)               
            y = self.up(x)                                        
            if getattr(x.data, "spatial_range", None) is not None:
                y.data.spatial_range = x.data.spatial_range       
            y = self.act(self.norm(y))
            return self.res(y)                                    


class Decoder(nn.Module):

    def __init__(self, out_channels=1, channels=(384, 256, 128, 64), latent_channels=2, num_groups=8):
        super().__init__()

        # lift z (latent_channels) back to working width so we can add the 384-d PE
        # and run 384-channel attention (inverse of the encoder's to_mu 384->2).
        self.from_latent = sp.SparseLinear(latent_channels, channels[0])  # 2 -> 384

        self.pos_embedder = AbsolutePositionEmbedder(channels[0])  # 384
        self.window_size  = 8
        num_blocks        = 3   
        num_heads         = 8

        # Eq. 3: refine latent tokens with windowed sparse attention BEFORE upsampling.
        self.attn_blocks = nn.ModuleList([SparseTransformerBlock( channels     = channels[0],                           # 384
                                                                  num_heads    = num_heads,                             # 8
                                                                  mlp_ratio    = 4.0,
                                                                  attn_mode    = "windowed",
                                                                  window_size  = self.window_size,                      # 8x8x8 windows
                                                                  shift_window = (self.window_size // 2) * (i % 2),     # 0,4,0,... shifted windows
                                                                  use_rope     = False,
                                                                  qk_rms_norm  = False,)
            for i in range(num_blocks)
        ])

        # Eq. 4: upsampling stages (mirror encoder reversed)
        self.stages = nn.ModuleList([ DecoderStage(channels[0], channels[1], num_groups=num_groups),     # up 384 -> 256
                                      DecoderStage(channels[1], channels[2], num_groups=num_groups),     # up 256 -> 128
                                      DecoderStage(channels[2], channels[3], num_groups=num_groups),])   # up 128 ->  64
    
        self.head = sp.SparseConv3d(channels[3], out_channels, 1) # 

    def forward(self, z: 'sp.SparseTensor') -> 'sp.SparseTensor':
        h = self.from_latent(z)                          # (N, 384) on Cz

        coords = h.coords[:, 1:4].float()                # (N,3) xyz; col 0 = batch
        pe     = self.pos_embedder(coords)
        h      = h.replace(h.feats + pe)                 # u~_n = z(p_n) + PE(p_n)

        for block in self.attn_blocks:                   # Eq. 3
            h = block(h)

        for stage in self.stages:                        # Eq. 4 (each doubles resolution)
            h = stage(h)

        logits = self.head(h)                            # occupancy logits at full res
        return logits                                    # logits.feats = per-voxel logits, logits.coords = Cx_hat