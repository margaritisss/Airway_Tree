import torch
from torch import nn
from torch.utils.data import DataLoader
from torchsparse.utils.collate import sparse_collate_fn
from vaessel import Encoder
from data import VesselDataset


def main():
    device = "cuda"
    dataset = VesselDataset(volumes=...)        # your data
    loader = DataLoader(dataset,
                        batch_size=4,
                        shuffle=True,
                        collate_fn=sparse_collate_fn,   # <-- THE bridge: batches SparseTensors + adds batch idx
                        num_workers=4,)                 # dense_to_sparse runs here, off the GPU

    model     = Encoder(in_channels=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for epoch in range(num_epochs):
        for batch in loader:
            x = batch["input"].to(device)       # SparseTensor.to() moves coords+feats
            B = int(x.coords[:, 0].max()) + 1 # number of samples in this batch
            x.spatial_range = (B, *GRID)      # REQUIRED before the generative decoder runs
            z = model(x)                         # bottleneck SparseTensor X^(L)

            loss = compute_loss(z, ...)          # your VAE/decoder loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


if __name__ == "__main__":
    main()