from __future__ import annotations 
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

def _conv_block(in_ch: int, out_ch: int, stride: int, padding: int, kernel_size: int = 3) -> nn.Sequential:
    """Conv3d -> BatchNorm3d -> ELU."""
    return nn.Sequential(nn.Conv3d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
                         nn.InstanceNorm3d(out_ch, affine=True),  # Using your updated norm
                         nn.ELU(inplace=True),) 


def _upsample_conv_block(in_ch: int, out_ch: int, target_size: int, kernel_size: int = 3, padding: int = 1, activation: bool = True) -> nn.Sequential:
    """Nearest-neighbor Upsample -> Conv3d -> BatchNorm3d -> (optional ELU)."""
    layers = [
        # 1. Spatially scale the volume to the exact target dimension
        nn.Upsample(size=(target_size, target_size, target_size), mode='nearest'),
        
        # 2. Process features without changing the spatial size (stride=1, padding=1)
        nn.Conv3d(in_ch, out_ch, kernel_size=kernel_size, stride=1, padding=padding, bias=False),
        nn.InstanceNorm3d(out_ch, affine=True)  # Using your updated norm,
    ]
    
    if activation:
        layers.append(nn.ELU(inplace=True))
    return nn.Sequential(*layers)

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class AE256(nn.Module):
    """
    AE on 128x128x128 occupancy grids. Encoder: 128 -> 64 -> 32 -> 16 -> 8 -> 4 with channels 16/32/64/128/128.
    """

    def __init__(self, num_latents: int = 100, n_channels: int = 1, base_ch: int = 1    , fc_dim: int = 343):
        super().__init__()
        self.num_latents = num_latents # number of dimensions in the latent space
        
        # channel sizes -- Doubling at EVERY layer
        c1  = base_ch          # 4
        c2  = base_ch * 2      # 8
        c3  = base_ch * 4      # 16
        c4  = base_ch * 8      # 32
        c5  = base_ch * 16     # 64
        c6  = base_ch * 32     # 256
        c7  = base_ch * 64     # 512
        c8  = base_ch * 128    # 1024
        c9  = base_ch * 256    # 1024
        c10 = base_ch * 512    # 2048

         # ---- Encoder: 10 Layers (Alternating strides, strictly doubling channels) ----
        self.enc1  = _conv_block(n_channels, c1, kernel_size=3, stride=1, padding=0)  # 256 -> 254 | Ch: 1  -> c1
        self.enc2  = _conv_block(c1, c2,  kernel_size=3, stride=2, padding=1)         # 254 -> 127 | Ch: c1 -> c2
        self.enc3  = _conv_block(c2, c3,  kernel_size=3, stride=1, padding=0)         # 127 -> 125 | Ch: c2 -> c3
        self.enc4  = _conv_block(c3, c4,  kernel_size=3, stride=2, padding=1)         # 125 -> 63  | Ch: c3 -> c4
        self.enc5  = _conv_block(c4, c5,  kernel_size=3, stride=1, padding=0)         # 63 -> 61   | Ch: c4 -> c5
        self.enc6  = _conv_block(c5, c6,  kernel_size=3, stride=2, padding=1)         # 61 -> 31   | Ch: c5 -> c6
        self.enc7  = _conv_block(c6, c7,  kernel_size=3, stride=1, padding=0)         # 31 -> 29   | Ch: c6 -> c7
        self.enc8  = _conv_block(c7, c8,  kernel_size=3, stride=2, padding=1)         # 29 -> 15   | Ch: c7 -> c8
        self.enc9  = _conv_block(c8, c9,  kernel_size=3, stride=1, padding=0)         # 15 -> 13
        self.enc10 = _conv_block(c9, c10, kernel_size=3, stride=2, padding=1)         # 13 -> 7
        # 1x1x1 Conv to reduce channels before flattening
        self.enc_reduce = nn.Sequential(nn.Conv3d(c10, 32, kernel_size=1, stride=1, padding=0, bias=False),
            nn.InstanceNorm3d(32, affine=True), # Using your updated norm
            nn.ELU(inplace=True))

        # New flattened dimension: 32 channels * 7 * 7 * 7 spatial grid = 10,976
        flat_dim = 32 * 7 * 7 * 7   

        self.enc_z = nn.Linear(flat_dim, num_latents, bias=True)
        # ---- Decoder: 8 Layers (Mirror image) ----
        # Map latents back to the correct flattened dimension (10,976)
        self.dec_fc = nn.Sequential( 
            nn.Linear(num_latents, flat_dim, bias=False), 
            nn.LayerNorm(flat_dim),
            nn.ELU(inplace=True))
               
        # Unflatten to match the encoder's 32-channel reduction
        self._dec_unflatten_shape = (32, 7, 7, 7)

        # Shift the 32-channel expansion to start at dec3, mapping up to c8
        self.dec1  = _conv_block(32, c10, kernel_size=3, stride=1, padding=1)   # 7 -> 7
        self.dec2  = _upsample_conv_block(c10, c9, target_size=15)              # 7 -> 15
        self.dec3  = _conv_block(  c9, c8, kernel_size=3, stride=1, padding=1)  # 15 -> 15
        self.dec4  = _upsample_conv_block(c8, c7, target_size=31)               # 15 -> 31
        self.dec5  = _conv_block(  c7, c6, kernel_size=3, stride=1, padding=1)  # 31 -> 31
        self.dec6  = _upsample_conv_block(c6, c5, target_size=63)               # 31 -> 63
        self.dec7  = _conv_block(  c5, c4, kernel_size=3, stride=1, padding=1)  # 63 -> 63
        self.dec8  = _upsample_conv_block(c4, c3, target_size=127)              # 63 -> 127
        self.dec9  = _conv_block(  c3, c2, kernel_size=3, stride=1, padding=1)  # 127 -> 127
        self.dec10 = _upsample_conv_block(c2, c1, target_size=256)             # 127 -> 256
        self.dec11 = nn.Conv3d(c1, n_channels, kernel_size=3, stride=1, padding=1, bias=True)
        self._init_weights()

    def encode(self, x):
        h = self.enc1(x)
        h = self.enc2(h) 
        h = self.enc3(h)
        h = self.enc4(h)
        h = self.enc5(h)
        h = self.enc6(h)
        h = self.enc7(h)
        h = self.enc8(h)
        h = self.enc9(h)
        h = self.enc10(h)
        h = self.enc_reduce(h)    # squashing the 2048 channels down to 32 while keeping the 7x7x7 grid
        h = h.flatten(1)
        z = self.enc_z(h)
        return z

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
        h = self.dec9(h)
        h = self.dec10(h)
        logits = self.dec11(h)
        return logits

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)
    
    def _init_weights(self):
        """Glorot/Xavier normal init."""
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d, nn.Linear)):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

@dataclass
class AELossOutput:
    total: torch.Tensor
    recon: torch.Tensor

def weighted_bce_with_logits(logits, target_binary, gamma=0.97, eps=1e-7):

    o   = torch.clamp(torch.sigmoid(logits), eps, 1.0 - eps)
    t   = target_binary
    pos = gamma * t * torch.log(o)
    neg = (1.0 - gamma) * (1.0 - t) * torch.log(1.0 - o)
    return -(pos + neg).flatten(1).sum(dim=1).mean()     # Sum over voxels, average over batch -> per-sample reconstruction NLL

# REPLACE the whole vae_loss(...) with:
def ae_loss(logits: torch.Tensor,
            target_binary: torch.Tensor,
            gamma: float = 0.99) -> AELossOutput:
    recon = weighted_bce_with_logits(logits, target_binary, gamma=gamma)
    return AELossOutput(total=recon, recon=recon)




