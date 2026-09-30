#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import torch.nn as nn
import torch
import torch.nn.functional as F
import math  # sqrt / pi for the geometric init


class Decoder(nn.Module):
    def __init__(
        self,
        latent_size,
        dims,
        dropout=None,
        dropout_prob=0.0,
        norm_layers=(),
        latent_in=(),
        weight_norm=False,
        xyz_in_all=None,
        use_tanh=False,
        latent_dropout=False,
        final_tanh=True,  # DGCI: set false, tanh caps |F| < 1 and fights the Eikonal term
        activation="relu",  # DGCI: "softplus" gives smooth input gradients (IGR)
        softplus_beta=100.0,  # sharpness of softplus; 100 is IGR's value
        add_xyz_norm=False,  # DGCI: output += ||xyz||, so F starts as a distance field
        geometric_init=False,  # IGR: start as the SDF of a sphere (unit gradient everywhere)
        geometric_init_radius=0.5,  # radius of that starting sphere, in normalised units
    ):
        super(Decoder, self).__init__()

        def make_sequence():
            return []

        dims = [latent_size + 3] + dims + [1]

        self.num_layers = len(dims)
        self.norm_layers = norm_layers
        self.latent_in = latent_in
        self.latent_dropout = latent_dropout
        if self.latent_dropout:
            self.lat_dp = nn.Dropout(0.2)

        self.xyz_in_all = xyz_in_all
        self.weight_norm = weight_norm

        for layer in range(0, self.num_layers - 1):
            if layer + 1 in latent_in:
                out_dim = dims[layer + 1] - dims[0]
            else:
                out_dim = dims[layer + 1]
                if self.xyz_in_all and layer != self.num_layers - 2:
                    out_dim -= 3

            if weight_norm and layer in self.norm_layers:
                setattr(
                    self,
                    "lin" + str(layer),
                    nn.utils.weight_norm(nn.Linear(dims[layer], out_dim)),
                )
            else:
                setattr(self, "lin" + str(layer), nn.Linear(dims[layer], out_dim))

            if (
                (not weight_norm)
                and self.norm_layers is not None
                and layer in self.norm_layers
            ):
                setattr(self, "bn" + str(layer), nn.LayerNorm(out_dim))

        self.use_tanh = use_tanh
        if use_tanh:
            self.tanh = nn.Tanh()
        if activation == "relu":  # original DeepSDF behaviour
            self.relu = nn.ReLU()  # name kept so old checkpoints still load
        elif activation == "softplus":  # smooth ReLU, second derivative exists
            self.relu = nn.Softplus(beta=softplus_beta)  # parameter-free, checkpoint-safe
        else:
            raise ValueError("activation must be 'relu' or 'softplus'")  # catch typos

        self.dropout_prob = dropout_prob
        self.dropout = dropout
        if final_tanh:  # forward() applies it only if the attribute exists
            self.th = nn.Tanh()  # original DeepSDF output squashing
        self.add_xyz_norm = add_xyz_norm  # used at the end of forward()
        if geometric_init:  # overwrite PyTorch's default init, which starts almost flat
            self._geometric_init(dims, geometric_init_radius)

    def _geometric_init(self, dims, radius):
        """IGR geometric initialisation (Gropp et al. 2020), adapted to the latent input.

        Hidden layers ~ N(0, sqrt(2 / fan_out)), zero bias; last layer ~ N(sqrt(pi / fan_in), 1e-5),
        bias -radius. Weights reading the latent code (first layer) and the skip-connection input
        start at zero, so every shape starts as the same sphere and the per-layer scaling that makes
        the output ~ ||x|| - radius is not disturbed. They become non-zero after the first step.
        """
        latent_size = dims[0] - 3  # input is [latent, xyz]
        last = self.num_layers - 2  # index of the output layer
        for layer in range(self.num_layers - 1):
            lin = getattr(self, "lin" + str(layer))  # the (possibly weight-normed) layer
            out_dim, in_dim = lin.weight.shape  # current weight shape
            weight = torch.empty(out_dim, in_dim)  # fresh weights
            if layer == last:  # output layer: mean chosen so the output grows like ||x||
                nn.init.normal_(weight, mean=math.sqrt(math.pi) / math.sqrt(in_dim), std=1e-5)
                bias_value = -radius  # shifts the zero set to the sphere
            else:  # hidden layer: keeps the activation scale constant through depth
                nn.init.normal_(weight, mean=0.0, std=math.sqrt(2.0) / math.sqrt(out_dim))
                bias_value = 0.0  # no offset
            if layer == 0:  # first layer sees [latent, xyz]
                weight[:, :latent_size] = 0.0  # latent columns start at zero
            if layer in self.latent_in:  # skip layer sees [hidden, latent, xyz]
                weight[:, in_dim - dims[0]:] = 0.0  # skip-input columns start at zero
            with torch.no_grad():
                if hasattr(lin, "weight_g"):  # weight norm: weight = g * v / ||v||
                    lin.weight_v.copy_(weight)  # direction
                    lin.weight_g.copy_(weight.norm(dim=1, keepdim=True))  # per-row length
                else:
                    lin.weight.copy_(weight)  # plain linear layer
                lin.bias.fill_(bias_value)  # set the bias

    # input: N x (L+3)
    def forward(self, input):
        xyz = input[:, -3:]

        if input.shape[1] > 3 and self.latent_dropout:
            latent_vecs = input[:, :-3]
            latent_vecs = F.dropout(latent_vecs, p=0.2, training=self.training)
            x = torch.cat([latent_vecs, xyz], 1)
        else:
            x = input

        for layer in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(layer))
            if layer in self.latent_in:
                x = torch.cat([x, input], 1)
            elif layer != 0 and self.xyz_in_all:
                x = torch.cat([x, xyz], 1)
            x = lin(x)
            # last layer Tanh
            if layer == self.num_layers - 2 and self.use_tanh:
                x = self.tanh(x)
            if layer < self.num_layers - 2:
                if (
                    self.norm_layers is not None
                    and layer in self.norm_layers
                    and not self.weight_norm
                ):
                    bn = getattr(self, "bn" + str(layer))
                    x = bn(x)
                x = self.relu(x)
                if self.dropout is not None and layer in self.dropout:
                    x = F.dropout(x, p=self.dropout_prob, training=self.training)

        if hasattr(self, "th"):
            x = self.th(x)

        if self.add_xyz_norm:  # same trick as DGCI's SDFBVPNet
            # ||xyz|| has unit gradient everywhere, so the Eikonal term is met at step 0
            # and the MLP only has to learn a residual; without it the net starts flat
            x = x + torch.norm(xyz, dim=1, keepdim=True)

        return x
