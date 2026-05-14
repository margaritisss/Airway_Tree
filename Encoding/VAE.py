from __future__ import annotations 
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

def _conv_block(in_ch: int, out_ch: int, stride: int, padding: int, kernel_size: int = 3) -> nn.Sequential:
    """Conv3d -> BatchNorm3d -> ELU."""
    return nn.Sequential(nn.Conv3d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
        nn.BatchNorm3d(out_ch),  
        nn.ELU(inplace=True),
    )

def _deconv_block(in_ch: int, out_ch: int, kernel_size: int = 3, stride: int = 1, padding: int = 1, output_padding: int = 0,
                  activation: bool = True) -> nn.Sequential:
    
    """ConvTranspose3d -> BatchNorm3d -> (optional ELU)."""
    layers = [
        nn.ConvTranspose3d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding, bias=False),
        nn.BatchNorm3d(out_ch),
    ]
    if activation:
        layers.append(nn.ELU(inplace=True))
    return nn.Sequential(*layers) 

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class VoxelVAE128(nn.Module):
    """
    Voxel VAE on 128x128x128 occupancy grids.
    Encoder: 128 -> 64 -> 32 -> 16 -> 8 -> 4 with channels 16/32/64/128/128.
    """

    def __init__(self, num_latents: int = 256, n_channels: int = 1, base_ch: int = 16, fc_dim: int = 1024):
        super().__init__()
        self.num_latents = num_latents # number of dimensions in the latent space
        
        # channel sizes -- Doubling at EVERY layer
        c1 = base_ch          # 8
        c2 = base_ch * 2      # 16
        c3 = base_ch * 4      # 32
        c4 = base_ch * 8      # 64
        c5 = base_ch * 16     # 128
        c6 = base_ch * 32     # 256
        c7 = base_ch * 64     # 512
        c8 = base_ch * 128    # 1024

        # ---- Encoder: 8 Layers (Alternating strides, strictly doubling channels) ----

        self.enc1 = _conv_block(n_channels, c1, kernel_size=3, stride=1, padding=0)  # 128 -> 126 | Ch: 1 -> c1
        self.enc2 = _conv_block(c1, c2, kernel_size=3, stride=2, padding=1)          # 126 -> 63  | Ch: c1 -> c2
        self.enc3 = _conv_block(c2, c3, kernel_size=3, stride=1, padding=0)          # 63 -> 61   | Ch: c2 -> c3
        self.enc4 = _conv_block(c3, c4, kernel_size=3, stride=2, padding=1)          # 61 -> 31   | Ch: c3 -> c4
        self.enc5 = _conv_block(c4, c5, kernel_size=3, stride=1, padding=0)          # 31 -> 29   | Ch: c4 -> c5
        self.enc6 = _conv_block(c5, c6, kernel_size=3, stride=2, padding=1)          # 29 -> 15   | Ch: c5 -> c6
        self.enc7 = _conv_block(c6, c7, kernel_size=3, stride=1, padding=0)          # 15 -> 13   | Ch: c6 -> c7
        self.enc8 = _conv_block(c7, c8, kernel_size=3, stride=2, padding=1)          # 13 -> 7    | Ch: c7 -> c8

        flat_dim = c8 * 7 * 7 * 7   # Bottleneck shape is now 1024 * 7 * 7 * 7 = 351,232

        self.enc_fc = nn.Sequential(
                      nn.Linear(flat_dim, fc_dim, bias=False),
                      nn.BatchNorm1d(fc_dim),
                      nn.ELU(inplace=True),
        )
        self.enc_mu = nn.Sequential(
                      nn.Linear(fc_dim, num_latents, bias=False),
                      nn.BatchNorm1d(num_latents),
        )
        self.enc_logsigma = nn.Sequential(
                            nn.Linear(fc_dim, num_latents, bias=False),
                            nn.BatchNorm1d(num_latents),
        )

        # ---- Decoder: 9 Layers (Mirror image) ----
        self.dec_fc = nn.Sequential(
            nn.Linear(num_latents, fc_dim, bias=False),
            nn.BatchNorm1d(fc_dim),
            nn.ELU(inplace=True),
            nn.Linear(fc_dim, flat_dim, bias=False),
            nn.BatchNorm1d(flat_dim),
            nn.ELU(inplace=True),
        )
        self._dec_unflatten_shape = (c8, 7, 7, 7)

        # Note: We halve the channels at every step now as we work our way back up
        self.dec1 = _conv_block(c8, c8, kernel_size=3, stride=1, padding=1)         # 7 -> 7     | Ch: c8 -> c8 (Maintain start size)
        self.dec2 = _deconv_block(c8, c7, kernel_size=3, stride=2, padding=0)       # 7 -> 15    | Ch: c8 -> c7
        self.dec3 = _conv_block(c7, c6, kernel_size=3, stride=1, padding=1)         # 15 -> 15   | Ch: c7 -> c6
        self.dec4 = _deconv_block(c6, c5, kernel_size=3, stride=2, padding=0)       # 15 -> 31   | Ch: c6 -> c5
        self.dec5 = _conv_block(c5, c4, kernel_size=3, stride=1, padding=1)         # 31 -> 31   | Ch: c5 -> c4
        self.dec6 = _deconv_block(c4, c3, kernel_size=3, stride=2, padding=0)       # 31 -> 63   | Ch: c4 -> c3
        self.dec7 = _conv_block(c3, c2, kernel_size=3, stride=1, padding=1)         # 63 -> 63   | Ch: c3 -> c2
        self.dec8 = _deconv_block(c2, c1, kernel_size=4, stride=2, padding=0)       # 63 -> 128  | Ch: c2 -> c1 (KERNEL=4!)
        
        # Final layer: standard convolution to map down to output channel (1)
        self.dec9 = nn.Sequential(
            nn.Conv3d(c1, n_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(n_channels)
        )

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
        h = h.flatten(1)
        h = self.enc_fc(h)
        return self.enc_mu(h), self.enc_logsigma(h)

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
        """Glorot/Xavier normal init."""
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d, nn.Linear)):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

@dataclass # dataclass is a convenient way to bundle multiple outputs together 
class VAELossOutput:
    total: torch.Tensor
    recon: torch.Tensor
    kl: torch.Tensor

def weighted_bce_with_logits(logits: torch.Tensor,
                             target_binary: torch.Tensor,
                             gamma: float = 0.98,
                             eps: float = 1e-7) -> torch.Tensor:
    """
    Modified weighted BCE from the paper.

    The original code:
        L = -(98 * t * log(o) + 2 * (1-t) * log(1-o)) / 100
    where o = clip(sigmoid(logits), 1e-7, 1-1e-7), t in {0,1}.

    `gamma` is the weight on the false-negative (positive) term. The paper text
    says γ=0.97; the released code uses γ=0.98 (= 98/100). Default is 0.98 to
    match the code; pass gamma=0.97 to match the paper text.
    """
    o = torch.clamp(torch.sigmoid(logits), eps, 1.0 - eps)
    t = target_binary
    pos = gamma * t * torch.log(o)
    neg = (1.0 - gamma) * (1.0 - t) * torch.log(1.0 - o)
    return -(pos + neg).mean()

def kl_divergence(mu: torch.Tensor, logsigma: torch.Tensor) -> torch.Tensor:
    """
    KL( N(mu, sigma^2) || N(0, I) ), with logsigma = log σ.

    Original Theano:
        -0.5 * mean(1 + 2*logsigma - mu^2 - exp(2*logsigma))
    """
    return -0.5 * torch.mean(1.0 + 2.0 * logsigma - mu.pow(2) - torch.exp(2.0 * logsigma))\

def vae_loss(logits: torch.Tensor,
             target_binary: torch.Tensor,
             mu: torch.Tensor,
             logsigma: torch.Tensor,
             gamma: float = 0.98,
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
    gamma         : positive-class weight in the BCE; 0.98 matches the released
                    code, 0.97 matches the paper text.
    use_kl        : whether to add the KL term. The released code makes this
                    optional via `cfg['kl_div']` (default False), but the paper
                    describes it as part of the loss, so default True here.
    """
    recon = weighted_bce_with_logits(logits, target_binary, gamma=gamma)
    kl = kl_divergence(mu, logsigma)
    total = recon + kl if use_kl else recon
    return VAELossOutput(total=total, recon=recon, kl=kl)

# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(0) 
    model = VoxelVAE128()
    model.train() 

    # Fake binary occupancy grid
    B = 4 # B = batch size which is 4 here for testing; in practice, you might use a larger batch size like 16 or 32 depending on your GPU memory.
    x_bin = (torch.rand(B, 1, 128, 128, 128) > 0.97).float()
    x_in  = 3.0 * x_bin - 1.0                           # rescale to {-1, 2}

    logits, mu, logsigma = model(x_in)
    print(f"logits   : {tuple(logits.shape)}")
    print(f"mu       : {tuple(mu.shape)}")
    print(f"logsigma : {tuple(logsigma.shape)}")

    out = vae_loss(logits, x_bin, mu, logsigma)
    print(f"recon : {out.recon.item():.4f}")
    print(f"kl    : {out.kl.item():.4f}")
    print(f"total : {out.total.item():.4f}")

    # Backward pass to confirm everything is connected.
    out.total.backward()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params: {n_params:,}")