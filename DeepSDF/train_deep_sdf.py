#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import torch
import torch.utils.data as data_utils
import signal
import sys
import os
import logging
import math
import json
import time

import deep_sdf
import deep_sdf.workspace as ws
import deep_sdf.dgci_loss  # DGCI SDF loss (optional, enabled by specs.json)
import track_reconstruction  # airway: periodic reconstruction of chosen training subjects


class LearningRateSchedule:
    def get_learning_rate(self, epoch):
        pass


class ConstantLearningRateSchedule(LearningRateSchedule):
    def __init__(self, value):
        self.value = value

    def get_learning_rate(self, epoch):
        return self.value


class StepLearningRateSchedule(LearningRateSchedule):
    def __init__(self, initial, interval, factor):
        self.initial = initial
        self.interval = interval
        self.factor = factor

    def get_learning_rate(self, epoch):

        return self.initial * (self.factor ** (epoch // self.interval))


class WarmupLearningRateSchedule(LearningRateSchedule):
    def __init__(self, initial, warmed_up, length):
        self.initial = initial
        self.warmed_up = warmed_up
        self.length = length

    def get_learning_rate(self, epoch):
        if epoch > self.length:
            return self.warmed_up
        return self.initial + (self.warmed_up - self.initial) * epoch / self.length


def get_learning_rate_schedules(specs):

    schedule_specs = specs["LearningRateSchedule"]

    schedules = []

    for schedule_specs in schedule_specs:

        if schedule_specs["Type"] == "Step":
            schedules.append(
                StepLearningRateSchedule(
                    schedule_specs["Initial"],
                    schedule_specs["Interval"],
                    schedule_specs["Factor"],
                )
            )
        elif schedule_specs["Type"] == "Warmup":
            schedules.append(
                WarmupLearningRateSchedule(
                    schedule_specs["Initial"],
                    schedule_specs["Final"],
                    schedule_specs["Length"],
                )
            )
        elif schedule_specs["Type"] == "Constant":
            schedules.append(ConstantLearningRateSchedule(schedule_specs["Value"]))

        else:
            raise Exception(
                'no known learning rate schedule of type "{}"'.format(
                    schedule_specs["Type"]
                )
            )

    return schedules


def save_model(experiment_directory, filename, decoder, epoch):

    model_params_dir = ws.get_model_params_dir(experiment_directory, True)

    torch.save(
        {"epoch": epoch, "model_state_dict": decoder.state_dict()},
        os.path.join(model_params_dir, filename),
    )


def save_optimizer(experiment_directory, filename, optimizer, epoch):

    optimizer_params_dir = ws.get_optimizer_params_dir(experiment_directory, True)

    torch.save(
        {"epoch": epoch, "optimizer_state_dict": optimizer.state_dict()},
        os.path.join(optimizer_params_dir, filename),
    )


def load_optimizer(experiment_directory, filename, optimizer):

    full_filename = os.path.join(
        ws.get_optimizer_params_dir(experiment_directory), filename
    )

    if not os.path.isfile(full_filename):
        raise Exception(
            'optimizer state dict "{}" does not exist'.format(full_filename)
        )

    data = torch.load(full_filename)

    optimizer.load_state_dict(data["optimizer_state_dict"])

    return data["epoch"]


def save_latent_vectors(experiment_directory, filename, latent_vec, epoch):

    latent_codes_dir = ws.get_latent_codes_dir(experiment_directory, True)

    all_latents = latent_vec.state_dict()

    torch.save(
        {"epoch": epoch, "latent_codes": all_latents},
        os.path.join(latent_codes_dir, filename),
    )


# TODO: duplicated in workspace
def load_latent_vectors(experiment_directory, filename, lat_vecs):

    full_filename = os.path.join(
        ws.get_latent_codes_dir(experiment_directory), filename
    )

    if not os.path.isfile(full_filename):
        raise Exception('latent state file "{}" does not exist'.format(full_filename))

    data = torch.load(full_filename)

    if isinstance(data["latent_codes"], torch.Tensor):

        # for backwards compatibility
        if not lat_vecs.num_embeddings == data["latent_codes"].size()[0]:
            raise Exception(
                "num latent codes mismatched: {} vs {}".format(
                    lat_vecs.num_embeddings, data["latent_codes"].size()[0]
                )
            )

        if not lat_vecs.embedding_dim == data["latent_codes"].size()[2]:
            raise Exception("latent code dimensionality mismatch")

        for i, lat_vec in enumerate(data["latent_codes"]):
            lat_vecs.weight.data[i, :] = lat_vec

    else:
        lat_vecs.load_state_dict(data["latent_codes"])

    return data["epoch"]


def save_logs(
    experiment_directory,
    loss_log,
    lr_log,
    timing_log,
    lat_mag_log,
    param_mag_log,
    epoch,
):

    torch.save(
        {
            "epoch": epoch,
            "loss": loss_log,
            "learning_rate": lr_log,
            "timing": timing_log,
            "latent_magnitude": lat_mag_log,
            "param_magnitude": param_mag_log,
        },
        os.path.join(experiment_directory, ws.logs_filename),
    )


def load_logs(experiment_directory):

    full_filename = os.path.join(experiment_directory, ws.logs_filename)

    if not os.path.isfile(full_filename):
        raise Exception('log file "{}" does not exist'.format(full_filename))

    data = torch.load(full_filename)

    return (
        data["loss"],
        data["learning_rate"],
        data["timing"],
        data["latent_magnitude"],
        data["param_magnitude"],
        data["epoch"],
    )


def clip_logs(loss_log, lr_log, timing_log, lat_mag_log, param_mag_log, epoch):

    iters_per_epoch = len(loss_log) // len(lr_log)

    loss_log = loss_log[: (iters_per_epoch * epoch)]
    lr_log = lr_log[:epoch]
    timing_log = timing_log[:epoch]
    lat_mag_log = lat_mag_log[:epoch]
    for n in param_mag_log:
        param_mag_log[n] = param_mag_log[n][:epoch]

    return (loss_log, lr_log, timing_log, lat_mag_log, param_mag_log)


def get_spec_with_default(specs, key, default):
    try:
        return specs[key]
    except KeyError:
        return default


def get_mean_latent_vector_magnitude(latent_vectors):
    return torch.mean(torch.norm(latent_vectors.weight.data.detach(), dim=1))


def append_parameter_magnitudes(param_mag_log, model):
    for name, param in model.named_parameters():
        if len(name) > 7 and name[:7] == "module.":
            name = name[7:]
        if name not in param_mag_log.keys():
            param_mag_log[name] = []
        param_mag_log[name].append(param.data.norm().item())


def dgci_train_step(
    decoder, lat_vecs, sdf_data, indices, loss_fn, batch_split,
    do_code_regularization, code_reg_lambda, epoch, optimizer, grad_clip,
):
    """One optimizer step with L_sdf = L_sdf-val + L_sdf-geo (DGCI, no beta)."""
    num_per_scene = sdf_data.shape[1]  # points per case in this batch
    sdf_data = sdf_data.reshape(-1, 8)  # flatten cases: (B * N, 8)
    num_sdf_samples = sdf_data.shape[0]  # used by DeepSDF's code regulariser

    xyz = sdf_data[:, 0:3]  # coordinates p
    sdf_gt = sdf_data[:, 3:4]  # s' (0 on the surface, unused for uniform points)
    normals_gt = sdf_data[:, 4:7]  # n' (only meaningful on the surface)
    kind = sdf_data[:, 7].long()  # surface / free / uniform
    scene_ids = indices.unsqueeze(-1).repeat(1, num_per_scene).view(-1)  # latent index per point

    loss_fn.set_counts(kind)  # set sizes over the WHOLE batch, before chunking

    xyz_c = torch.chunk(xyz, batch_split)  # sub-batches that fit in GPU memory
    sdf_c = torch.chunk(sdf_gt, batch_split)  # matching splits
    nrm_c = torch.chunk(normals_gt, batch_split)  # matching splits
    kind_c = torch.chunk(kind, batch_split)  # matching splits
    ids_c = torch.chunk(scene_ids, batch_split)  # matching splits

    optimizer.zero_grad()  # gradients accumulate over the chunks below
    batch_loss = 0.0  # for Logs.pth
    term_sums = {}  # per-term values for the log line

    for i in range(len(xyz_c)):  # torch.chunk may return fewer than batch_split pieces
        xyz_i = xyz_c[i].cuda().requires_grad_(True)  # leaf we differentiate F with respect to
        batch_vecs = lat_vecs(ids_c[i]).cuda()  # a_i for every point
        decoder_input = torch.cat([batch_vecs, xyz_i], dim=1)  # DeepSDF input: [latent, xyz]
        pred_sdf = decoder(decoder_input)  # raw F(p; a_i), never clamped before the gradient

        chunk_loss, terms = loss_fn(pred_sdf, xyz_i, sdf_c[i], nrm_c[i], kind_c[i])  # Eq. 3 + 4

        if do_code_regularization:  # DeepSDF's latent prior (DGCI's L_intra-reg on a_i)
            l2_size_loss = torch.sum(torch.norm(batch_vecs, dim=1))  # summed code norms
            reg_loss = code_reg_lambda * min(1, epoch / 100) * l2_size_loss / num_sdf_samples  # warm-up
            chunk_loss = chunk_loss + reg_loss  # add to the objective
            terms["code_reg"] = reg_loss.item()  # log it next to the SDF terms

        chunk_loss.backward()  # double backprop through the gradient terms happens here
        batch_loss += chunk_loss.item()  # running total for the log
        for name, value in terms.items():
            term_sums[name] = term_sums.get(name, 0.0) + value  # sum over chunks

    if grad_clip is not None:  # same optional clipping as the original loop
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), grad_clip)  # stabilise large weights

    optimizer.step()  # update decoder + latent codes
    return batch_loss, term_sums  # total and per-term values


def main_function(experiment_directory, continue_from, batch_split):

    logging.debug("running " + experiment_directory)

    specs = ws.load_experiment_specifications(experiment_directory)

    description = specs["Description"]  # airway: example specs store a list of strings
    if isinstance(description, list):  # join so the concatenation below works
        description = " ".join(description)
    logging.info("Experiment description: \n" + description)

    data_source = specs["DataSource"]
    train_split_file = specs["TrainSplit"]

    arch = __import__("networks." + specs["NetworkArch"], fromlist=["Decoder"])

    logging.debug(specs["NetworkSpecs"])

    latent_size = specs["CodeLength"]

    checkpoints = list(
        range(
            specs["SnapshotFrequency"],
            specs["NumEpochs"] + 1,
            specs["SnapshotFrequency"],
        )
    )

    for checkpoint in specs["AdditionalSnapshots"]:
        checkpoints.append(checkpoint)
    checkpoints.sort()

    lr_schedules = get_learning_rate_schedules(specs)

    grad_clip = get_spec_with_default(specs, "GradientClipNorm", None)
    if grad_clip is not None:
        logging.debug("clipping gradients to max norm {}".format(grad_clip))

    def save_latest(epoch):

        save_model(experiment_directory, "latest.pth", decoder, epoch)
        save_optimizer(experiment_directory, "latest.pth", optimizer_all, epoch)
        save_latent_vectors(experiment_directory, "latest.pth", lat_vecs, epoch)

    def save_checkpoints(epoch):

        save_model(experiment_directory, str(epoch) + ".pth", decoder, epoch)
        save_optimizer(experiment_directory, str(epoch) + ".pth", optimizer_all, epoch)
        save_latent_vectors(experiment_directory, str(epoch) + ".pth", lat_vecs, epoch)

    def signal_handler(sig, frame):
        logging.info("Stopping early...")
        sys.exit(0)

    def adjust_learning_rate(lr_schedules, optimizer, epoch):

        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedules[i].get_learning_rate(epoch)

    def empirical_stat(latent_vecs, indices):
        lat_mat = torch.zeros(0).cuda()
        for ind in indices:
            lat_mat = torch.cat([lat_mat, latent_vecs[ind]], 0)
        mean = torch.mean(lat_mat, 0)
        var = torch.var(lat_mat, 0)
        return mean, var

    signal.signal(signal.SIGINT, signal_handler)

    num_samp_per_scene = specs["SamplesPerScene"]
    scene_per_batch = specs["ScenesPerBatch"]
    clamp_dist = specs["ClampingDistance"]
    minT = -clamp_dist
    maxT = clamp_dist
    enforce_minmax = True

    do_code_regularization = get_spec_with_default(specs, "CodeRegularization", True)
    code_reg_lambda = get_spec_with_default(specs, "CodeRegularizationLambda", 1e-4)

    code_bound = get_spec_with_default(specs, "CodeBound", None)

    dgci_specs = get_spec_with_default(specs, "DGCILoss", None)  # absent = original DeepSDF
    use_dgci = dgci_specs is not None  # switch between the two training paths

    decoder = arch.Decoder(latent_size, **specs["NetworkSpecs"]).cuda()

    logging.info("training with {} GPU(s)".format(torch.cuda.device_count()))

    # if torch.cuda.device_count() > 1:
    decoder = torch.nn.DataParallel(decoder)

    num_epochs = specs["NumEpochs"]
    log_frequency = get_spec_with_default(specs, "LogFrequency", 10)

    with open(train_split_file, "r") as f:
        train_split = json.load(f)

    if use_dgci:  # surface points + normals + SDF samples, from the .npy files
        n_surface = dgci_specs.get("SurfaceSamplesPerScene", 4096)  # paper: 4K surface points
        n_free = dgci_specs.get("FreeSamplesPerScene", 4096)  # paper: 4K free points
        n_uniform = dgci_specs.get("UniformSamplesPerScene", 0)  # extra points without s'
        sdf_dataset = deep_sdf.data.SDFSurfaceSamples(
            data_source, train_split, n_surface, n_free, n_uniform
        )
        num_samp_per_scene = n_surface + n_free + n_uniform  # replaces SamplesPerScene
        dgci_loss = deep_sdf.dgci_loss.DGCILoss(
            weights=dgci_specs.get("Weights"),  # None = paper weights
            delta=dgci_specs.get("PhiDelta", 100.0),  # phi(s) = exp(-delta |s|)
            clamp_dist=dgci_specs.get("L1ClampingDistance"),  # None = unclamped, as in Eq. 3
            phi_on=dgci_specs.get("PhiOn", "offsurface"),  # "offsurface" (paper) or "uniform"
        )
        logging.info("DGCI loss: {}".format(dgci_specs))  # record the settings in the log
    else:
        sdf_dataset = deep_sdf.data.SDFSamples(
            data_source, train_split, num_samp_per_scene,
            load_ram=get_spec_with_default(specs, "LoadRam", False),  # airway: RAM caching
        )

    num_data_loader_threads = get_spec_with_default(specs, "DataLoaderThreads", 1)
    logging.debug("loading data with {} threads".format(num_data_loader_threads))

    sdf_loader = data_utils.DataLoader(
        sdf_dataset,
        batch_size=scene_per_batch,
        shuffle=True,
        num_workers=num_data_loader_threads,
        drop_last=True,
    )

    logging.debug("torch num_threads: {}".format(torch.get_num_threads()))

    num_scenes = len(sdf_dataset)

    logging.info("There are {} scenes".format(num_scenes))

    logging.debug(decoder)

    lat_vecs = torch.nn.Embedding(num_scenes, latent_size, max_norm=code_bound)
    torch.nn.init.normal_(
        lat_vecs.weight.data,
        0.0,
        get_spec_with_default(specs, "CodeInitStdDev", 1.0) / math.sqrt(latent_size),
    )

    # DGCI dataset lists case names, the original one .npz paths; row i = latent code i in both
    instance_files = sdf_dataset.names if use_dgci else sdf_dataset.npyfiles
    tracker = track_reconstruction.Tracker(  # airway: reads Track* keys from specs.json
        specs, experiment_directory, instance_files, data_source, ws.sdf_samples_subdir
    )  # disabled when specs has no "TrackSubjects"

    logging.debug(
        "initialized with mean magnitude {}".format(
            get_mean_latent_vector_magnitude(lat_vecs)
        )
    )

    loss_l1 = torch.nn.L1Loss(reduction="sum")

    optimizer_all = torch.optim.Adam(
        [
            {
                "params": decoder.parameters(),
                "lr": lr_schedules[0].get_learning_rate(0),
            },
            {
                "params": lat_vecs.parameters(),
                "lr": lr_schedules[1].get_learning_rate(0),
            },
        ]
    )

    loss_log = []
    lr_log = []
    lat_mag_log = []
    timing_log = []
    param_mag_log = {}

    start_epoch = 1

    if continue_from is not None:

        logging.info('continuing from "{}"'.format(continue_from))

        lat_epoch = load_latent_vectors(
            experiment_directory, continue_from + ".pth", lat_vecs
        )

        model_epoch = ws.load_model_parameters(
            experiment_directory, continue_from, decoder
        )

        optimizer_epoch = load_optimizer(
            experiment_directory, continue_from + ".pth", optimizer_all
        )

        loss_log, lr_log, timing_log, lat_mag_log, param_mag_log, log_epoch = load_logs(
            experiment_directory
        )

        if not log_epoch == model_epoch:
            loss_log, lr_log, timing_log, lat_mag_log, param_mag_log = clip_logs(
                loss_log, lr_log, timing_log, lat_mag_log, param_mag_log, model_epoch
            )

        if not (model_epoch == optimizer_epoch and model_epoch == lat_epoch):
            raise RuntimeError(
                "epoch mismatch: {} vs {} vs {} vs {}".format(
                    model_epoch, optimizer_epoch, lat_epoch, log_epoch
                )
            )

        start_epoch = model_epoch + 1

        logging.debug("loaded")

    logging.info("starting from epoch {}".format(start_epoch))

    logging.info(
        "Number of decoder parameters: {}".format(
            sum(p.data.nelement() for p in decoder.parameters())
        )
    )
    logging.info(
        "Number of shape code parameters: {} (# codes {}, code dim {})".format(
            lat_vecs.num_embeddings * lat_vecs.embedding_dim,
            lat_vecs.num_embeddings,
            lat_vecs.embedding_dim,
        )
    )

    for epoch in range(start_epoch, num_epochs + 1):

        start = time.time()

        logging.info("epoch {}...".format(epoch))

        decoder.train()

        adjust_learning_rate(lr_schedules, optimizer_all, epoch)

        epoch_terms = {}  # per-term sums over this epoch (DGCI path)
        num_batches = 0  # batches seen this epoch

        for sdf_data, indices in sdf_loader:

            if use_dgci:  # DGCI path; the original L1 path below is left untouched
                batch_loss, terms = dgci_train_step(
                    decoder, lat_vecs, sdf_data, indices, dgci_loss, batch_split,
                    do_code_regularization, code_reg_lambda, epoch, optimizer_all, grad_clip,
                )
                loss_log.append(batch_loss)  # keeps Logs.pth / plot_log.py working
                for name, value in terms.items():
                    epoch_terms[name] = epoch_terms.get(name, 0.0) + value  # epoch totals
                num_batches += 1  # for the epoch average
                continue  # skip the original L1 code for this batch

            # Process the input data
            sdf_data = sdf_data.reshape(-1, 4)

            num_sdf_samples = sdf_data.shape[0]

            sdf_data.requires_grad = False

            xyz = sdf_data[:, 0:3]
            sdf_gt = sdf_data[:, 3].unsqueeze(1)

            if enforce_minmax:
                sdf_gt = torch.clamp(sdf_gt, minT, maxT)

            xyz = torch.chunk(xyz, batch_split)
            indices = torch.chunk(
                indices.unsqueeze(-1).repeat(1, num_samp_per_scene).view(-1),
                batch_split,
            )

            sdf_gt = torch.chunk(sdf_gt, batch_split)

            batch_loss = 0.0

            optimizer_all.zero_grad()

            for i in range(batch_split):

                batch_vecs = lat_vecs(indices[i])

                input = torch.cat([batch_vecs, xyz[i]], dim=1)

                # NN optimization
                pred_sdf = decoder(input)

                if enforce_minmax:
                    pred_sdf = torch.clamp(pred_sdf, minT, maxT)

                chunk_loss = loss_l1(pred_sdf, sdf_gt[i].cuda()) / num_sdf_samples

                if do_code_regularization:
                    l2_size_loss = torch.sum(torch.norm(batch_vecs, dim=1))
                    reg_loss = (
                        code_reg_lambda * min(1, epoch / 100) * l2_size_loss
                    ) / num_sdf_samples

                    chunk_loss = chunk_loss + reg_loss.cuda()

                chunk_loss.backward()

                batch_loss += chunk_loss.item()

            logging.debug("loss = {}".format(batch_loss))

            loss_log.append(batch_loss)

            if grad_clip is not None:

                torch.nn.utils.clip_grad_norm_(decoder.parameters(), grad_clip)

            optimizer_all.step()

        if use_dgci and num_batches > 0:  # average weighted terms over the epoch
            summary = ", ".join(
                "{} {:.4f}".format(k, v / num_batches) for k, v in epoch_terms.items()
            )  # e.g. "sdf 12.3, phi 0.4, ..."
            logging.info("epoch {} loss terms: {}".format(epoch, summary))  # track each term

        end = time.time()

        seconds_elapsed = end - start
        timing_log.append(seconds_elapsed)

        lr_log.append([schedule.get_learning_rate(epoch) for schedule in lr_schedules])

        lat_mag_log.append(get_mean_latent_vector_magnitude(lat_vecs))

        append_parameter_magnitudes(param_mag_log, decoder)

        if epoch in checkpoints:
            save_checkpoints(epoch)

        tracker.run(epoch, decoder, lat_vecs)  # airway: meshes + metrics every TrackFrequency epochs

        if epoch % log_frequency == 0:

            save_latest(epoch)
            save_logs(
                experiment_directory,
                loss_log,
                lr_log,
                timing_log,
                lat_mag_log,
                param_mag_log,
                epoch,
            )


if __name__ == "__main__":

    import argparse

    arg_parser = argparse.ArgumentParser(description="Train a DeepSDF autodecoder")
    arg_parser.add_argument(
        "--experiment",
        "-e",
        dest="experiment_directory",
        required=True,
        help="The experiment directory. This directory should include "
        + "experiment specifications in 'specs.json', and logging will be "
        + "done in this directory as well.",
    )
    arg_parser.add_argument(
        "--continue",
        "-c",
        dest="continue_from",
        help="A snapshot to continue from. This can be 'latest' to continue"
        + "from the latest running snapshot, or an integer corresponding to "
        + "an epochal snapshot.",
    )
    arg_parser.add_argument(
        "--batch_split",
        dest="batch_split",
        default=1,
        help="This splits the batch into separate subbatches which are "
        + "processed separately, with gradients accumulated across all "
        + "subbatches. This allows for training with large effective batch "
        + "sizes in memory constrained environments.",
    )

    deep_sdf.add_common_args(arg_parser)

    args = arg_parser.parse_args()

    deep_sdf.configure_logging(args)

    main_function(args.experiment_directory, args.continue_from, int(args.batch_split))