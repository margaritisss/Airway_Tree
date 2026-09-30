#!/usr/bin/env python3
"""Reconstruct a few training subjects every N epochs while DeepSDF trains.

The default output is a binary NIfTI mask on each subject's own CT grid, built the
same way as create_mask in the DGCI sdf_meshing.py: one centre probe per voxel,
supersampling only in the thin shell around the surface, same LPS/RAS handling.

Settings are read from specs.json (no "TrackSubjects" = tracker off):
  "TrackSubjects":     ["AIIB23_100_mesh", "ATM22_005_mesh"]  or  "auto"
                       "auto" = first training subject of each dataset prefix (text before "_")
  "TrackFrequency":    10          every this many epochs
  "TrackOutput":       "mask"      "mask", "mesh" or "both"
  "TrackNormDir":      "/path"     folder(s) with <subject><TrackNormSuffix>, comma separated
  "TrackNormSuffix":   "_norm.npz"
  "TrackReferenceDir": "/a,/b"     folder(s) with the GT binary volumes (defines the output grid)
  "TrackReference":    {"AIIB23_100_mesh": "/path/AIIB23_100.nii.gz"}   optional explicit map
  "TrackWorld":        "ras"       world convention the meshes were built in ("ras" or "lps")
  "TrackSupersample":  2           probes per voxel edge near the surface
  "TrackOccupancy":    0.5         fraction of probes inside to fill a voxel
  "TrackLevel":        0.0         iso-level
  "TrackResolution":   256         marching-cubes grid, mesh output only

Outputs under <experiment>/TrainingProgress/<subject>/:
  epoch_0010.nii.gz   uint8 mask with the reference's origin, spacing and direction
  epoch_0010.ply      only with "mesh"/"both", in CT coordinates
and one line per subject and check in TrainingProgress/metrics.csv:
  l1_clamped, sign_accuracy (on the subject's own SDF samples) and dice against the reference,
  plus, when <name>_surface.npy exists (DGCI loss runs), normal_cos and grad_norm on the
  subject's surface points: the quantities the normal and Eikonal terms push towards 1.

Works with both training paths of train_deep_sdf.py:
  original DeepSDF  instance files are .npz with pos/neg under <DataSource>/SdfSamples/
  DGCILoss          instance names; samples read from <DataSource>/<name>_sdf.npy (+ _surface.npy)
"""

import csv  # metrics log
import glob  # reference file lookup
import logging  # same logger the training script uses
import os  # paths
import time  # timing

import numpy as np  # geometry on the CPU side
import torch  # decoder evaluation

REFERENCE_EXTENSIONS = (".nii.gz", ".nii", ".mha", ".nrrd")  # formats SimpleITK reads
METRIC_COLUMNS = ["epoch", "subject", "l1_clamped", "sign_accuracy",
                  "normal_cos", "grad_norm", "dice"]  # metrics.csv layout


# ------------------------------------------------------------------ subject bookkeeping

def _pick_subjects(requested, instance_filenames):
    """Map requested names (or "auto") to their row in the latent-code table."""
    names = [os.path.splitext(os.path.basename(f))[0] for f in instance_filenames]  # "AIIB23_100_mesh"
    if requested == "auto":  # one subject per dataset prefix
        chosen = {}  # prefix -> first name seen
        for name in names:  # names are in latent-code order
            chosen.setdefault(name.split("_")[0], name)  # keep only the first per prefix
        requested = list(chosen.values())  # e.g. ["AIIB23_100_mesh", "ATM22_001_mesh"]
    picked = []  # (name, latent index)
    for name in requested:  # validate every requested subject
        if name not in names:  # latent codes exist only for training subjects
            raise ValueError(f"TrackSubjects: '{name}' is not in the training split")
        picked.append((name, names.index(name)))  # row index == dataset index == latent index
    return picked


def _split_dirs(value):
    """Accept a list or a comma separated string of folders."""
    if isinstance(value, str):  # "a,b" as in generate.py
        return [d for d in value.split(",") if d]
    return list(value or [])  # list or None


def _find_norm(name, norm_dirs, suffix):
    """Return (offset, scale) from <dir>/<name><suffix>, or None if not found."""
    for folder in norm_dirs:  # search every given folder
        path = os.path.join(folder, name + suffix)  # e.g. .../AIIB23_100_mesh_norm.npz
        if os.path.isfile(path):  # found it
            data = np.load(path)  # written by SampleVisibleMeshSurface
            offset = data["offset"].astype(np.float64).reshape(3)  # = -bounding-box centre
            scale = float(data["scale"].reshape(-1)[0])  # = 1 / (max radius * 1.03)
            return offset, scale
    return None


def _find_reference(name, reference_dirs, explicit):
    """Find the GT volume for a subject: explicit map first, then name-based search."""
    if name in explicit:  # user said exactly which file
        return explicit[name]
    stems = [name]  # "AIIB23_100_mesh"
    if name.endswith("_mesh"):  # volumes are usually named without the mesh suffix
        stems.append(name[: -len("_mesh")])  # "AIIB23_100"
    for folder in reference_dirs:  # exact names first, in every folder
        for stem in stems:
            for extension in REFERENCE_EXTENSIONS:
                path = os.path.join(folder, stem + extension)  # e.g. .../AIIB23_100.nii.gz
                if os.path.isfile(path):
                    return path
    for folder in reference_dirs:  # then "<stem>_anything.ext", e.g. AIIB23_100_binary.nii.gz
        for extension in REFERENCE_EXTENSIONS:
            hits = sorted(glob.glob(os.path.join(folder, stems[-1] + "_*" + extension)))  # prefix match
            if len(hits) == 1:  # only accept an unambiguous match
                return hits[0]
    return None


def _load_eval_samples(data_source, sdf_subdir, instance_file, name, rng, n_each=50000):
    """Fixed evaluation points: (100k, 4) SDF samples and, if available, (50k, 6) surface points."""
    if instance_file.endswith(".npz"):  # original DeepSDF layout
        samples = np.load(os.path.join(data_source, sdf_subdir, instance_file))  # pos/neg arrays
        pos, neg = samples["pos"], samples["neg"]  # outside / inside
    else:  # DGCI layout: flat folder of .npy files from run_batch.sh
        free = np.load(os.path.join(data_source, name + "_sdf.npy"))  # (M, 4)
        free = free[np.isfinite(free).all(axis=1)]  # drop NaN rows, as the dataset does
        pos, neg = free[free[:, 3] > 0], free[free[:, 3] <= 0]  # same split as SDFSurfaceSamples
    sdf_eval = np.concatenate([pos[rng.choice(len(pos), n_each)],  # outside points
                               neg[rng.choice(len(neg), n_each)]])  # inside points

    surface_eval = None  # only for DGCI data
    surface_path = os.path.join(data_source, name + "_surface.npy")  # SampleVisibleMeshSurface
    if not instance_file.endswith(".npz") and os.path.isfile(surface_path):
        surface = np.load(surface_path)  # (N, 6) xyz + normal
        surface = surface[np.isfinite(surface).all(axis=1)]  # drop NaN rows
        surface_eval = surface[rng.choice(len(surface), n_each)]  # fixed subset
    return sdf_eval, surface_eval


# ------------------------------------------------------------------ geometry (pure numpy)

def _subvoxel_offsets(factor):
    """Probe positions inside one voxel, in voxel units, centred on the voxel."""
    steps = (np.arange(factor) + 0.5) / factor - 0.5  # factor=1 collapses to the centre
    grid = np.meshgrid(steps, steps, steps, indexing="ij")  # all combinations
    return np.stack([axis.ravel() for axis in grid], axis=1)  # (factor**3, 3)


def voxelize(image, offset, scale, signs, sdf_fn, level=0.0, supersample=2, occupancy=0.5):
    """Binary mask on the grid of `image` (SimpleITK), same algorithm as DGCI create_mask.

    sdf_fn maps (P, 3) normalized points to (P,) SDF values (numpy in, numpy out).
    """
    spacing = np.array(image.GetSpacing())  # sitk order is (x, y, z)
    origin = np.array(image.GetOrigin())  # LPS millimetres
    direction = np.array(image.GetDirection()).reshape(3, 3)  # index axes -> world axes
    n_x, n_y, n_z = image.GetSize()  # sitk size is (x, y, z)

    def probe(voxel_positions):
        """SDF at fractional voxel positions; +inf outside the trained cube."""
        world = origin + np.dot(voxel_positions * spacing, direction.T)  # SimpleITK gives LPS
        world = world * signs  # match the convention the mesh was built in
        normalized = (world + offset) * scale  # the C++ BoundingCubeNormalization, forward
        sdf = np.full(len(normalized), np.inf, dtype=np.float32)  # outside cube = outside shape
        in_cube = np.all(np.abs(normalized) <= 1.0, axis=1)  # network only trained in [-1, 1]^3
        if in_cube.any():
            sdf[in_cube] = sdf_fn(normalized[in_cube])  # decoder call
        return sdf

    mask = np.zeros((n_z, n_y, n_x), dtype=np.uint8)  # numpy order (z, y, x) like GetArrayFromImage
    offsets = _subvoxel_offsets(supersample)  # probe pattern for the surface shell
    margin = 0.5 * np.linalg.norm(spacing) * scale  # half a voxel diagonal, in cube units
    rows, cols = np.meshgrid(np.arange(n_y), np.arange(n_x), indexing="ij")  # one slice
    base = np.stack([cols.ravel(), rows.ravel(), np.zeros(cols.size)], axis=1).astype(np.float64)
    for z in range(n_z):  # one slice at a time keeps RAM flat
        base[:, 2] = z  # only the z column changes
        centre = probe(base)  # one probe per voxel
        solid = centre <= level - margin  # safely inside, no refinement needed
        edge = np.flatnonzero(np.abs(centre - level) <= margin)  # straddles the surface
        if len(edge) > 0:
            hits = np.zeros(len(edge), dtype=np.int16)  # inside-probe counter
            for probe_offset in offsets:  # supersample the shell
                hits += probe(base[edge] + probe_offset) <= level
            solid[edge] = hits >= occupancy * len(offsets)  # majority vote
        mask[z] = solid.reshape(n_y, n_x)  # store the slice
    return mask


def dice(prediction, reference):
    """Dice overlap of two boolean arrays; 1.0 when both are empty."""
    total = prediction.sum() + reference.sum()  # voxel counts
    if total == 0:
        return 1.0
    return 2.0 * np.logical_and(prediction, reference).sum() / total


def _write_ply(vertices, faces, path):
    """Vectorized PLY writer (DeepSDF's version loops over every vertex in Python)."""
    import plyfile  # only needed for mesh output

    vertex_array = np.empty(len(vertices), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])  # PLY layout
    vertex_array["x"], vertex_array["y"], vertex_array["z"] = vertices.T  # fill all at once
    face_array = np.empty(len(faces), dtype=[("vertex_indices", "i4", (3,))])  # triangle list
    face_array["vertex_indices"] = faces  # (F, 3) indices
    elements = [plyfile.PlyElement.describe(vertex_array, "vertex"),  # vertex block
                plyfile.PlyElement.describe(face_array, "face")]  # face block
    plyfile.PlyData(elements).write(path)  # binary PLY


# ------------------------------------------------------------------ the tracker

class Tracker:
    """Created once before the epoch loop; call run(epoch, decoder, lat_vecs) every epoch."""

    def __init__(self, specs, experiment_directory, instance_filenames, data_source, sdf_subdir):
        requested = specs.get("TrackSubjects")  # None or [] -> disabled
        self.enabled = bool(requested)  # everything below is skipped when off
        if not self.enabled:
            return
        self.frequency = int(specs.get("TrackFrequency", 10))  # epochs between checks
        self.output = specs.get("TrackOutput", "mask")  # "mask", "mesh" or "both"
        assert self.output in ("mask", "mesh", "both"), "TrackOutput must be mask, mesh or both"
        self.level = float(specs.get("TrackLevel", 0.0))  # iso-level
        self.supersample = int(specs.get("TrackSupersample", 2))  # shell refinement
        self.occupancy = float(specs.get("TrackOccupancy", 0.5))  # vote threshold
        self.resolution = int(specs.get("TrackResolution", 256))  # mesh grid
        self.clamp = float(specs.get("ClampingDistance", 0.1))  # same clamp as the loss
        world = specs.get("TrackWorld", "ras")  # your DGCI runs use WORLD=ras
        assert world in ("ras", "lps"), "TrackWorld must be ras or lps"
        self.signs = np.array([-1.0, -1.0, 1.0] if world == "ras" else [1.0, 1.0, 1.0])  # as generate.py
        norm_dirs = _split_dirs(specs.get("TrackNormDir", ""))  # norm npz folders
        suffix = specs.get("TrackNormSuffix", "_norm.npz")  # run_batch.sh naming
        reference_dirs = _split_dirs(specs.get("TrackReferenceDir", ""))  # GT volume folders
        explicit = specs.get("TrackReference", {}) or {}  # optional name -> file map
        self.out_dir = os.path.join(experiment_directory, "TrainingProgress")  # all outputs here
        os.makedirs(self.out_dir, exist_ok=True)
        self.metrics_path = os.path.join(self.out_dir, "metrics.csv")  # appended every check
        if os.path.exists(self.metrics_path):  # file from a run before the DGCI columns existed
            with open(self.metrics_path) as handle:
                header = handle.readline().strip().split(",")  # its column names
            if header != METRIC_COLUMNS:  # appending would misalign the columns
                self.metrics_path = os.path.join(self.out_dir, "metrics_v2.csv")  # start a new file

        self.subjects = []  # everything needed per subject
        for name, index in _pick_subjects(requested, instance_filenames):  # validated list
            norm = _find_norm(name, norm_dirs, suffix)  # (offset, scale) or None
            if norm is None:  # both outputs need it to reach CT coordinates
                raise FileNotFoundError(f"TrackNormDir: no {name}{suffix} in {norm_dirs}")
            reference = None  # sitk image and GT array, mask output only
            if self.output in ("mask", "both"):
                path = _find_reference(name, reference_dirs, explicit)  # GT volume path
                if path is None:  # fail at startup, not after hours of training
                    raise FileNotFoundError(
                        f"no reference volume for {name} in {reference_dirs}; "
                        f'add it to "TrackReference" in specs.json')
                import SimpleITK as sitk  # only needed for mask output

                image = sitk.ReadImage(path)  # defines the output grid
                reference = (image, sitk.GetArrayFromImage(image) > 0)  # GT as boolean (z, y, x)
                logging.info(f"track {name}: reference {path} size {image.GetSize()}")
            rng = np.random.default_rng(0)  # same evaluation points every check
            sdf_eval, surface_eval = _load_eval_samples(
                data_source, sdf_subdir, instance_filenames[index], name, rng)  # either data layout
            os.makedirs(os.path.join(self.out_dir, name), exist_ok=True)  # one folder per subject
            self.subjects.append({
                "name": name, "index": index, "norm": norm, "reference": reference,
                "eval": torch.from_numpy(sdf_eval).float(),  # (100k, 4)
                "surface": None if surface_eval is None else torch.from_numpy(surface_eval).float(),
            })
            logging.info(f"tracking {name} (latent {index}) every {self.frequency} epochs -> {self.output}")

    def run(self, epoch, decoder, lat_vecs):
        """Reconstruct all tracked subjects if this epoch is due; restores train mode after."""
        if not self.enabled or epoch % self.frequency != 0:  # nothing to do this epoch
            return
        device = next(decoder.parameters()).device  # wherever training put the decoder
        decoder.eval()  # dropout off, as at test time
        try:
            with torch.no_grad():  # no gradients, no effect on the optimizer
                for subject in self.subjects:  # usually two
                    latent = lat_vecs.weight[subject["index"]].detach().to(device)  # current code
                    row = {"epoch": epoch, "subject": subject["name"]}  # metrics for this check
                    row.update(self._sample_metrics(subject, decoder, latent, device))  # cheap
                    if subject["surface"] is not None:  # DGCI data: check the geometric terms
                        row.update(self._surface_metrics(subject, decoder, latent, device))
                    if self.output in ("mask", "both"):
                        row["dice"] = self._mask(epoch, subject, decoder, latent, device)  # NIfTI
                    if self.output in ("mesh", "both"):
                        self._mesh(epoch, subject, decoder, latent, device)  # PLY
                    self._log(row)  # CSV + training log
        finally:
            decoder.train()  # always back to training mode, even after an error

    # -------------------------------------------------------------- pieces

    def _decode(self, decoder, latent, points, device):
        """SDF at (P, 3) torch points on `device`, in chunks of 262k."""
        values = []  # chunk results
        for chunk in torch.split(points, 2 ** 18):  # same batch size as DGCI max_batch
            inputs = torch.cat([latent.expand(len(chunk), -1), chunk], dim=1)  # [code, xyz]
            values.append(decoder(inputs).squeeze(1))  # (chunk,)
        return torch.cat(values)  # (P,)

    def _sample_metrics(self, subject, decoder, latent, device):
        """L1 (clamped, like training) and inside/outside accuracy on 100k own samples."""
        samples = subject["eval"].to(device)  # (100k, 4) x, y, z, sdf
        predicted = self._decode(decoder, latent, samples[:, :3], device)  # network output
        target = samples[:, 3]  # stored sdf
        l1 = torch.mean(torch.abs(predicted.clamp(-self.clamp, self.clamp)
                                  - target.clamp(-self.clamp, self.clamp))).item()  # training-style
        accuracy = torch.mean(((predicted <= 0) == (target <= 0)).float()).item()  # sign agreement
        return {"l1_clamped": round(l1, 6), "sign_accuracy": round(accuracy, 4)}

    def _surface_metrics(self, subject, decoder, latent, device):
        """Mean cos(grad F, n') and mean ||grad F|| on 50k own surface points (targets: 1 and 1)."""
        surface = subject["surface"].to(device)  # (50k, 6) xyz + normal
        cos_sum, norm_sum = 0.0, 0.0  # running sums over chunks
        with torch.enable_grad():  # run() is under no_grad; the gradient needs autograd back on
            for chunk in torch.split(surface, 2 ** 15):  # small chunks: backward keeps activations
                xyz = chunk[:, :3].clone().requires_grad_(True)  # leaf to differentiate against
                inputs = torch.cat([latent.expand(len(xyz), -1), xyz], dim=1)  # [code, xyz]
                sdf = decoder(inputs)  # F(p; a_i)
                grad = torch.autograd.grad(sdf.sum(), xyz)[0]  # (chunk, 3); no graph kept
                cos_sum += torch.nn.functional.cosine_similarity(grad, chunk[:, 3:], dim=1).sum().item()
                norm_sum += grad.norm(dim=1).sum().item()  # Eikonal target is 1
        n = len(surface)  # number of points
        return {"normal_cos": round(cos_sum / n, 4), "grad_norm": round(norm_sum / n, 4)}

    def _mask(self, epoch, subject, decoder, latent, device):
        """Voxel mask on the reference CT grid; returns Dice against the reference."""
        import SimpleITK as sitk  # imported at init too, cheap here

        start = time.time()  # timing for the log
        image, truth = subject["reference"]  # grid + GT boolean array
        offset, scale = subject["norm"]  # CT -> cube transform

        def sdf_fn(points):  # numpy (P, 3) -> numpy (P,), what voxelize expects
            tensor = torch.from_numpy(points).float().to(device)  # to the GPU
            return self._decode(decoder, latent, tensor, device).cpu().numpy()  # back to numpy

        mask = voxelize(image, offset, scale, self.signs, sdf_fn,
                        self.level, self.supersample, self.occupancy)  # (z, y, x) uint8
        out = sitk.GetImageFromArray(mask)  # numpy -> image
        out.CopyInformation(image)  # origin, spacing and direction of the reference
        path = os.path.join(self.out_dir, subject["name"], f"epoch_{epoch:04d}.nii.gz")  # sortable
        sitk.WriteImage(out, path, True)  # True = compress
        score = dice(mask > 0, truth)  # overlap with the GT segmentation
        logging.info(f"track {subject['name']} epoch {epoch}: mask in {time.time() - start:.1f}s -> {path}")
        return round(float(score), 4)

    def _mesh(self, epoch, subject, decoder, latent, device):
        """Dense cube grid -> marching cubes -> PLY in CT coordinates."""
        import skimage.measure  # only needed for mesh output

        n = self.resolution  # points per axis
        axis = torch.linspace(-1.0, 1.0, n, device=device)  # grid spans the training cube
        grid_y, grid_z = torch.meshgrid(axis, axis, indexing="ij")  # one x-slice
        yz = torch.stack([grid_y.reshape(-1), grid_z.reshape(-1)], dim=1)  # (n*n, 2)
        volume = torch.empty(n, n * n, device=device)  # filled slice by slice
        for i in range(n):  # x index
            points = torch.cat([axis[i].expand(n * n, 1), yz], dim=1)  # (n*n, 3) x, y, z
            volume[i] = self._decode(decoder, latent, points, device)  # one slice
        volume = volume.reshape(n, n, n).cpu().numpy()  # axes (x, y, z)
        if not (volume.min() < self.level < volume.max()):  # no crossing -> nothing to mesh
            logging.info(f"track {subject['name']} epoch {epoch}: no surface yet")
            return
        spacing = 2.0 / (n - 1)  # grid step in cube units
        vertices, faces, _, _ = skimage.measure.marching_cubes(volume, level=self.level, spacing=(spacing,) * 3)
        offset, scale = subject["norm"]  # back to CT coordinates
        vertices = (vertices - 1.0) / scale - offset  # grid starts at -1; inverse normalization
        path = os.path.join(self.out_dir, subject["name"], f"epoch_{epoch:04d}.ply")  # sortable
        _write_ply(vertices.astype(np.float32), faces.astype(np.int32), path)  # save

    def _log(self, row):
        """Append one row to metrics.csv and echo it to the training log."""
        columns = METRIC_COLUMNS  # fixed layout; missing values stay empty
        new_file = not os.path.exists(self.metrics_path)  # header only once
        with open(self.metrics_path, "a", newline="") as handle:  # survives restarts
            writer = csv.DictWriter(handle, fieldnames=columns)  # missing keys -> empty cells
            if new_file:
                writer.writeheader()
            writer.writerow(row)
        logging.info("track " + ", ".join(f"{k}={v}" for k, v in row.items()))  # one-line summary
