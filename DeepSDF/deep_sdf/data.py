#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import glob
import logging
import numpy as np
import os
import random
import torch
import torch.utils.data

import deep_sdf.workspace as ws


def get_instance_filenames(data_source, split):
    npzfiles = []
    for dataset in split:
        for class_name in split[dataset]:
            for instance_name in split[dataset][class_name]:
                instance_filename = os.path.join(
                    dataset, class_name, instance_name + ".npz"
                )
                if not os.path.isfile(
                    os.path.join(data_source, ws.sdf_samples_subdir, instance_filename)
                ):
                    # raise RuntimeError(
                    #     'Requested non-existent file "' + instance_filename + "'"
                    # )
                    logging.warning(
                        "Requested non-existent file '{}'".format(instance_filename)
                    )
                npzfiles += [instance_filename]
    return npzfiles


class NoMeshFileError(RuntimeError):
    """Raised when a mesh file is not found in a shape directory"""

    pass


class MultipleMeshFileError(RuntimeError):
    """"Raised when a there a multiple mesh files in a shape directory"""

    pass


def find_mesh_in_directory(shape_dir):
    mesh_filenames = list(glob.iglob(shape_dir + "/**/*.obj")) + list(
        glob.iglob(shape_dir + "/*.obj")
    )
    if len(mesh_filenames) == 0:
        raise NoMeshFileError()
    elif len(mesh_filenames) > 1:
        raise MultipleMeshFileError()
    return mesh_filenames[0]


def remove_nans(tensor):
    tensor_nan = torch.isnan(tensor[:, 3])
    return tensor[~tensor_nan, :]


def read_sdf_samples_into_ram(filename):
    npz = np.load(filename)
    pos_tensor = torch.from_numpy(npz["pos"])
    neg_tensor = torch.from_numpy(npz["neg"])

    return [pos_tensor, neg_tensor]


def unpack_sdf_samples(filename, subsample=None):
    npz = np.load(filename)
    if subsample is None:
        return npz
    pos_tensor = remove_nans(torch.from_numpy(npz["pos"]))
    neg_tensor = remove_nans(torch.from_numpy(npz["neg"]))

    # split the sample into half
    half = int(subsample / 2)

    random_pos = (torch.rand(half) * pos_tensor.shape[0]).long()
    random_neg = (torch.rand(half) * neg_tensor.shape[0]).long()

    sample_pos = torch.index_select(pos_tensor, 0, random_pos)
    sample_neg = torch.index_select(neg_tensor, 0, random_neg)

    samples = torch.cat([sample_pos, sample_neg], 0)

    return samples


def unpack_sdf_samples_from_ram(data, subsample=None):
    if subsample is None:
        return data
    pos_tensor = data[0]
    neg_tensor = data[1]

    # split the sample into half
    half = int(subsample / 2)

    pos_size = pos_tensor.shape[0]
    neg_size = neg_tensor.shape[0]

    pos_start_ind = random.randint(0, pos_size - half)
    sample_pos = pos_tensor[pos_start_ind : (pos_start_ind + half)]

    if neg_size <= half:
        random_neg = (torch.rand(half) * neg_tensor.shape[0]).long()
        sample_neg = torch.index_select(neg_tensor, 0, random_neg)
    else:
        neg_start_ind = random.randint(0, neg_size - half)
        sample_neg = neg_tensor[neg_start_ind : (neg_start_ind + half)]

    samples = torch.cat([sample_pos, sample_neg], 0)

    return samples


class SDFSamples(torch.utils.data.Dataset):
    def __init__(
        self,
        data_source,
        split,
        subsample,
        load_ram=False,
        print_filename=False,
        num_files=1000000,
    ):
        self.subsample = subsample

        self.data_source = data_source
        self.npyfiles = get_instance_filenames(data_source, split)

        logging.debug(
            "using "
            + str(len(self.npyfiles))
            + " shapes from data source "
            + data_source
        )

        self.load_ram = load_ram

        if load_ram:
            self.loaded_data = []
            for f in self.npyfiles:
                filename = os.path.join(self.data_source, ws.sdf_samples_subdir, f)
                npz = np.load(filename)
                pos_tensor = remove_nans(torch.from_numpy(npz["pos"]))
                neg_tensor = remove_nans(torch.from_numpy(npz["neg"]))
                self.loaded_data.append(
                    [
                        pos_tensor[torch.randperm(pos_tensor.shape[0])],
                        neg_tensor[torch.randperm(neg_tensor.shape[0])],
                    ]
                )

    def __len__(self):
        return len(self.npyfiles)

    def __getitem__(self, idx):
        filename = os.path.join(
            self.data_source, ws.sdf_samples_subdir, self.npyfiles[idx]
        )
        if self.load_ram:
            return (
                unpack_sdf_samples_from_ram(self.loaded_data[idx], self.subsample),
                idx,
            )
        else:
            return unpack_sdf_samples(filename, self.subsample), idx


# ---------------------------------------------------------------------------
# DGCI-style samples: surface points with normals + off-surface SDF samples.
# Reads the outputs of the modified C++ tools, from a flat directory:
#   <DataSource>/<name>_surface.npy  (N, 6) xyz + outward normal  (SampleVisibleMeshSurface)
#   <DataSource>/<name>_sdf.npy      (M, 4) xyz + sdf             (PreprocessMesh)
# Each item is (samples, index); samples is (n_surface + n_free + n_uniform, 8):
#   cols 0:3 xyz | col 3 sdf | cols 4:7 normal | col 7 kind (see deep_sdf.dgci_loss)
# ---------------------------------------------------------------------------
from deep_sdf.dgci_loss import KIND_FREE, KIND_SURFACE, KIND_UNIFORM  # point-kind codes


def get_instance_names(split):
    names = []  # flat list of case names, in split order
    for dataset in split:  # same nesting as DeepSDF's split files
        for class_name in split[dataset]:
            names += list(split[dataset][class_name])  # only the names are used
    return names  # index i here = latent code i


def load_dgci_case(data_source, name):
    surface = np.load(os.path.join(data_source, name + "_surface.npy"))  # (N, 6)
    free = np.load(os.path.join(data_source, name + "_sdf.npy"))  # (M, 4)

    surface = surface[np.isfinite(surface).all(axis=1)]  # drop NaN/Inf rows
    lengths = np.linalg.norm(surface[:, 3:6], axis=1)  # normal lengths
    surface = surface[lengths > 1e-6]  # a zero normal has no direction
    surface[:, 3:6] /= lengths[lengths > 1e-6, None]  # unit normals ([:, None] = column)

    free = free[np.isfinite(free).all(axis=1)]  # drop NaN/Inf rows
    pos = free[free[:, 3] > 0]  # outside the airway
    neg = free[free[:, 3] <= 0]  # inside the lumen (the rarer kind for thin tubes)
    if len(surface) == 0 or len(pos) == 0 or len(neg) == 0:  # unusable case
        raise RuntimeError("case '{}' has no surface, outside or inside samples".format(name))
    return surface.astype(np.float32), pos.astype(np.float32), neg.astype(np.float32)


def sample_dgci_case(case, n_surface, n_free, n_uniform):
    """Draw one packed (n_surface + n_free + n_uniform, 8) tensor from a loaded case."""
    # torch RNG, not numpy: DataLoader reseeds torch per worker but not numpy,
    # so numpy draws would repeat the same "random" points in every worker
    surface, pos, neg = case  # the case's pools
    half = n_free // 2  # balance inside/outside, as DeepSDF does
    s_rows = surface[torch.randint(len(surface), (n_surface,)).numpy()]  # random subset
    p_rows = pos[torch.randint(len(pos), (n_free - half,)).numpy()]  # outside subset
    n_rows = neg[torch.randint(len(neg), (half,)).numpy()]  # inside subset
    f_rows = np.concatenate([p_rows, n_rows], axis=0)  # all SDF samples

    out = np.zeros((n_surface + n_free + n_uniform, 8), dtype=np.float32)  # packed rows
    a = n_surface  # end of the surface block
    b = a + n_free  # end of the free block

    out[:a, 0:3] = s_rows[:, 0:3]  # surface xyz (sdf column stays 0)
    out[:a, 4:7] = s_rows[:, 3:6]  # surface normals
    out[:a, 7] = KIND_SURFACE  # set S

    out[a:b, 0:4] = f_rows  # xyz + ground-truth sdf
    out[a:b, 7] = KIND_FREE  # Omega minus S, with s'

    uniform = torch.rand(n_uniform, 3).numpy() * 2.0 - 1.0  # same [-1, 1] cube as meshing
    out[b:, 0:3] = uniform  # uniform xyz
    out[b:, 7] = KIND_UNIFORM  # Omega minus S, without s'
    return torch.from_numpy(out)  # (N, 8) float tensor


class SDFSurfaceSamples(torch.utils.data.Dataset):
    def __init__(self, data_source, split, n_surface, n_free, n_uniform):
        self.names = get_instance_names(split)  # case names, index = latent code
        self.n_surface = n_surface  # surface points per case per iteration
        self.n_free = n_free  # off-surface SDF points per case per iteration
        self.n_uniform = n_uniform  # uniform points without ground truth
        self.cases = []  # (surface, pos, neg) arrays per case, kept in RAM
        for name in self.names:
            self.cases.append(load_dgci_case(data_source, name))  # fail early on a bad case
            logging.debug("loaded {}".format(name))  # progress in --debug mode

    def __len__(self):
        return len(self.names)  # one item per case, as in DeepSDF

    def __getitem__(self, idx):
        samples = sample_dgci_case(
            self.cases[idx], self.n_surface, self.n_free, self.n_uniform
        )  # fresh random subset every call
        return samples, idx  # same (data, index) contract as SDFSamples
