#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import argparse
import json
import logging
import os
import random
import time
import torch

import deep_sdf
import deep_sdf.workspace as ws
import deep_sdf.dgci_loss  # DGCI SDF loss, used when specs.json has DGCILoss


def reconstruct(
    decoder,
    num_iterations,
    latent_size,
    test_sdf,
    stat,
    clamp_dist,
    num_samples=30000,
    lr=5e-4,
    l2reg=False,
):
    def adjust_learning_rate(
        initial_lr, optimizer, num_iterations, decreased_by, adjust_lr_every
    ):
        lr = initial_lr * ((1 / decreased_by) ** (num_iterations // adjust_lr_every))
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

    decreased_by = 10
    adjust_lr_every = int(num_iterations / 2)

    if type(stat) == type(0.1):
        latent = torch.ones(1, latent_size).normal_(mean=0, std=stat).cuda()
    else:
        latent = torch.normal(stat[0].detach(), stat[1].detach()).cuda()

    latent.requires_grad = True

    optimizer = torch.optim.Adam([latent], lr=lr)

    loss_num = 0
    loss_l1 = torch.nn.L1Loss()

    for e in range(num_iterations):

        decoder.eval()
        sdf_data = deep_sdf.data.unpack_sdf_samples_from_ram(
            test_sdf, num_samples
        ).cuda()
        xyz = sdf_data[:, 0:3]
        sdf_gt = sdf_data[:, 3].unsqueeze(1)

        sdf_gt = torch.clamp(sdf_gt, -clamp_dist, clamp_dist)

        adjust_learning_rate(lr, optimizer, e, decreased_by, adjust_lr_every)

        optimizer.zero_grad()

        latent_inputs = latent.expand(num_samples, -1)

        inputs = torch.cat([latent_inputs, xyz], 1).cuda()

        pred_sdf = decoder(inputs)

        # TODO: why is this needed?
        if e == 0:
            pred_sdf = decoder(inputs)

        pred_sdf = torch.clamp(pred_sdf, -clamp_dist, clamp_dist)

        loss = loss_l1(pred_sdf, sdf_gt)
        if l2reg:
            loss += 1e-4 * torch.mean(latent.pow(2))
        loss.backward()
        optimizer.step()

        if e % 50 == 0:
            logging.debug(loss.cpu().data.numpy())
            logging.debug(e)
            logging.debug(latent.norm())
        loss_num = loss.cpu().data.numpy()

    return loss_num, latent


def reconstruct_dgci(
    decoder, num_iterations, latent_size, case, loss_fn,
    n_surface, n_free, n_uniform, lr=5e-3, l2reg=True, init_std=0.01,
):
    """Fit one latent code to a test case with the DGCI loss; decoder weights stay fixed."""
    decreased_by = 10  # same step schedule as reconstruct()
    adjust_lr_every = int(num_iterations / 2)  # drop the lr once, half way

    latent = torch.ones(1, latent_size).normal_(mean=0, std=init_std).cuda()  # small random a_i
    latent.requires_grad = True  # the only thing being optimised
    optimizer = torch.optim.Adam([latent], lr=lr)  # optimiser over the code only

    decoder.eval()  # no dropout at test time
    for param in decoder.parameters():
        param.requires_grad_(False)  # skip decoder weight gradients (they would be discarded)

    loss_value = 0.0  # last loss, returned for logging
    for e in range(num_iterations):
        step_lr = lr * ((1 / decreased_by) ** (e // adjust_lr_every))  # stepped lr
        for group in optimizer.param_groups:
            group["lr"] = step_lr  # apply it

        samples = deep_sdf.data.sample_dgci_case(case, n_surface, n_free, n_uniform)  # fresh draw
        xyz = samples[:, 0:3].cuda().requires_grad_(True)  # still needed for grad F
        kind = samples[:, 7].long()  # point kinds
        loss_fn.set_counts(kind)  # set sizes for this draw

        optimizer.zero_grad()  # clear the previous step
        latent_inputs = latent.expand(xyz.shape[0], -1)  # same code for every point
        pred_sdf = decoder(torch.cat([latent_inputs, xyz], 1))  # F(p; a)
        loss, _ = loss_fn(pred_sdf, xyz, samples[:, 3:4], samples[:, 4:7], kind)  # Eq. 3 + 4
        if l2reg:  # same code prior as the original reconstruct()
            loss = loss + 1e-4 * loss_fn.w["sdf"] * torch.mean(latent.pow(2))  # scaled like the L1 term
        loss.backward()  # gradient reaches the latent code only
        optimizer.step()  # update the code

        if e % 50 == 0:
            logging.debug("iter {} loss {:.5f}".format(e, loss.item()))  # progress in --debug
        loss_value = loss.item()  # keep the last value

    return loss_value, latent  # same return shape as reconstruct()


def run_dgci_reconstruction(args, specs, decoder, latent_size, split, meshes_dir, codes_dir):
    """Reconstruct every case in the split from <name>_surface.npy and <name>_sdf.npy."""
    dgci_specs = specs["DGCILoss"]  # same settings as training
    loss_fn = deep_sdf.dgci_loss.DGCILoss(
        weights=dgci_specs.get("Weights"),  # None = paper weights
        delta=dgci_specs.get("PhiDelta", 100.0),  # phi sharpness
        clamp_dist=dgci_specs.get("L1ClampingDistance"),  # None = unclamped
        phi_on=dgci_specs.get("PhiOn", "offsurface"),  # which points phi uses
    )
    n_surface = dgci_specs.get("SurfaceSamplesPerScene", 4096)  # points per iteration
    n_free = dgci_specs.get("FreeSamplesPerScene", 4096)  # points per iteration
    n_uniform = dgci_specs.get("UniformSamplesPerScene", 0)  # points per iteration

    for name in deep_sdf.data.get_instance_names(split):  # every test case
        mesh_filename = os.path.join(meshes_dir, name)  # create_mesh appends .ply
        latent_filename = os.path.join(codes_dir, name + ".pth")  # fitted code
        if args.skip and os.path.isfile(mesh_filename + ".ply") and os.path.isfile(latent_filename):
            continue  # already done

        logging.info("reconstructing {}".format(name))  # progress
        case = deep_sdf.data.load_dgci_case(args.data_source, name)  # surface / pos / neg pools
        start = time.time()  # timing
        err, latent = reconstruct_dgci(
            decoder, int(args.iterations), latent_size, case, loss_fn,
            n_surface, n_free, n_uniform,
        )  # fit the code
        logging.debug("reconstruct time: {:.1f}s, loss {:.5f}".format(time.time() - start, err))

        torch.save(latent.unsqueeze(0), latent_filename)  # save first: meshing can fail
        try:
            with torch.no_grad():  # meshing needs no gradients
                deep_sdf.mesh.create_mesh(
                    decoder, latent, mesh_filename, N=256, max_batch=int(2 ** 18)
                )  # normalised-space mesh
        except ValueError as err:  # marching cubes finds no zero crossing (undertrained model)
            logging.warning("no mesh for {}: {}".format(name, err))  # keep going with the next case


if __name__ == "__main__":

    arg_parser = argparse.ArgumentParser(
        description="Use a trained DeepSDF decoder to reconstruct a shape given SDF "
        + "samples."
    )
    arg_parser.add_argument(
        "--experiment",
        "-e",
        dest="experiment_directory",
        required=True,
        help="The experiment directory which includes specifications and saved model "
        + "files to use for reconstruction",
    )
    arg_parser.add_argument(
        "--checkpoint",
        "-c",
        dest="checkpoint",
        default="latest",
        help="The checkpoint weights to use. This can be a number indicated an epoch "
        + "or 'latest' for the latest weights (this is the default)",
    )
    arg_parser.add_argument(
        "--data",
        "-d",
        dest="data_source",
        required=True,
        help="The data source directory.",
    )
    arg_parser.add_argument(
        "--split",
        "-s",
        dest="split_filename",
        required=True,
        help="The split to reconstruct.",
    )
    arg_parser.add_argument(
        "--iters",
        dest="iterations",
        default=800,
        help="The number of iterations of latent code optimization to perform.",
    )
    arg_parser.add_argument(
        "--skip",
        dest="skip",
        action="store_true",
        help="Skip meshes which have already been reconstructed.",
    )
    deep_sdf.add_common_args(arg_parser)

    args = arg_parser.parse_args()

    deep_sdf.configure_logging(args)

    def empirical_stat(latent_vecs, indices):
        lat_mat = torch.zeros(0).cuda()
        for ind in indices:
            lat_mat = torch.cat([lat_mat, latent_vecs[ind]], 0)
        mean = torch.mean(lat_mat, 0)
        var = torch.var(lat_mat, 0)
        return mean, var

    specs_filename = os.path.join(args.experiment_directory, "specs.json")

    if not os.path.isfile(specs_filename):
        raise Exception(
            'The experiment directory does not include specifications file "specs.json"'
        )

    specs = json.load(open(specs_filename))

    arch = __import__("networks." + specs["NetworkArch"], fromlist=["Decoder"])

    latent_size = specs["CodeLength"]

    decoder = arch.Decoder(latent_size, **specs["NetworkSpecs"])

    decoder = torch.nn.DataParallel(decoder)

    saved_model_state = torch.load(
        os.path.join(
            args.experiment_directory, ws.model_params_subdir, args.checkpoint + ".pth"
        )
    )
    saved_model_epoch = saved_model_state["epoch"]

    decoder.load_state_dict(saved_model_state["model_state_dict"])

    decoder = decoder.module.cuda()

    with open(args.split_filename, "r") as f:
        split = json.load(f)

    if "DGCILoss" in specs:  # .npy data: the .npz lookup would only print warnings
        npz_filenames = []  # handled by run_dgci_reconstruction below
    else:
        npz_filenames = deep_sdf.data.get_instance_filenames(args.data_source, split)

    random.shuffle(npz_filenames)

    logging.debug(decoder)

    err_sum = 0.0
    repeat = 1
    save_latvec_only = False
    rerun = 0

    reconstruction_dir = os.path.join(
        args.experiment_directory, ws.reconstructions_subdir, str(saved_model_epoch)
    )

    if not os.path.isdir(reconstruction_dir):
        os.makedirs(reconstruction_dir)

    reconstruction_meshes_dir = os.path.join(
        reconstruction_dir, ws.reconstruction_meshes_subdir
    )
    if not os.path.isdir(reconstruction_meshes_dir):
        os.makedirs(reconstruction_meshes_dir)

    reconstruction_codes_dir = os.path.join(
        reconstruction_dir, ws.reconstruction_codes_subdir
    )
    if not os.path.isdir(reconstruction_codes_dir):
        os.makedirs(reconstruction_codes_dir)

    if "DGCILoss" in specs:  # model trained with the DGCI loss on .npy data
        run_dgci_reconstruction(
            args, specs, decoder, latent_size, split,
            reconstruction_meshes_dir, reconstruction_codes_dir,
        )  # replaces the .npz loop below
        npz_filenames = []  # nothing left for the original loop

    for ii, npz in enumerate(npz_filenames):

        if "npz" not in npz:
            continue

        full_filename = os.path.join(args.data_source, ws.sdf_samples_subdir, npz)

        logging.debug("loading {}".format(npz))

        data_sdf = deep_sdf.data.read_sdf_samples_into_ram(full_filename)

        for k in range(repeat):

            if rerun > 1:
                mesh_filename = os.path.join(
                    reconstruction_meshes_dir, npz[:-4] + "-" + str(k + rerun)
                )
                latent_filename = os.path.join(
                    reconstruction_codes_dir, npz[:-4] + "-" + str(k + rerun) + ".pth"
                )
            else:
                mesh_filename = os.path.join(reconstruction_meshes_dir, npz[:-4])
                latent_filename = os.path.join(
                    reconstruction_codes_dir, npz[:-4] + ".pth"
                )

            if (
                args.skip
                and os.path.isfile(mesh_filename + ".ply")
                and os.path.isfile(latent_filename)
            ):
                continue

            logging.info("reconstructing {}".format(npz))

            data_sdf[0] = data_sdf[0][torch.randperm(data_sdf[0].shape[0])]
            data_sdf[1] = data_sdf[1][torch.randperm(data_sdf[1].shape[0])]

            start = time.time()
            err, latent = reconstruct(
                decoder,
                int(args.iterations),
                latent_size,
                data_sdf,
                0.01,  # [emp_mean,emp_var],
                0.1,
                num_samples=8000,
                lr=5e-3,
                l2reg=True,
            )
            logging.debug("reconstruct time: {}".format(time.time() - start))
            err_sum += err
            logging.debug("current_error avg: {}".format((err_sum / (ii + 1))))
            logging.debug(ii)

            logging.debug("latent: {}".format(latent.detach().cpu().numpy()))

            decoder.eval()

            if not os.path.exists(os.path.dirname(mesh_filename)):
                os.makedirs(os.path.dirname(mesh_filename))

            if not save_latvec_only:
                start = time.time()
                with torch.no_grad():
                    deep_sdf.mesh.create_mesh(
                        decoder, latent, mesh_filename, N=256, max_batch=int(2 ** 18)
                    )
                logging.debug("total time: {}".format(time.time() - start))

            if not os.path.exists(os.path.dirname(latent_filename)):
                os.makedirs(os.path.dirname(latent_filename))

            torch.save(latent.unsqueeze(0), latent_filename)
