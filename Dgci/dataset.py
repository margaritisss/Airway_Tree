# -*- coding: utf-8 -*-                                                                          # Declares UTF-8 source encoding (Python 2 leftover, harmless here).
import os                                                                                        # Filesystem helpers: listdir, path.join, path.isfile.
import numpy as np                                                                               # All sampling and concatenation happens in NumPy, not torch.
import torch                                                                                     # Used only at the end, to convert NumPy arrays into tensors.
from torch.utils.data import Dataset                                                             # Base class for PyTorch datasets; requires __len__ and __getitem__.
from scipy.io import loadmat                                                                     # Reads MATLAB .mat files (v5/v7 only, NOT v7.3/HDF5).


class PointCloud_with_FreePoints(Dataset):                                                       # Holds ONE case: its surface points+normals and its free-space points+SDF.
    def __init__(self, pointcloud_path, on_surface_points, instance_idx=None, expand=-1,
             max_points=-1, split='train', val_ratio=0.1, split_seed=0, val_items=4):            # Constructor for a single case.
        super().__init__()                                                                       # Calls Dataset's (essentially empty) init; convention.

        self.instance_idx = instance_idx                                                         # Case ID; indexes nn.Embedding to pick this subject's latent code. Model input, not a label.
        self.expand       = expand                                                               # Surface-dilation amount along normals; -1 disables it.

        print("Loading point cloud of subject%04d" % self.instance_idx)                          # Startup progress print; %04d zero-pads to 4 digits.
        point_cloud = loadmat(pointcloud_path)                                                   # Loads the .mat into a dict (also holds MATLAB header keys).
        point_cloud = point_cloud['p']                                                           # Extracts the array under key 'p'; expected shape (N, 6) = x,y,z,nx,ny,nz.

        free_points = loadmat(pointcloud_path.replace('surface_pts_n_normal', 'free_space_pts')) # Derives the free-space path by STRING SUBSTITUTION; folder names and filenames must match exactly.
        free_points = free_points['p_sdf']                                                       # Extracts the array under key 'p_sdf'; expected shape (M, 4) = x,y,z,sdf.
        print("Finished loading point cloud")                                                    # Paired progress print.
        free_points_coords = free_points[:, :3]                                                  # Columns 0-2 → free-space XYZ, shape (M, 3).
        free_points_sdf    = free_points[:, 3:]                                                  # Column 3 onward → SDF; slice (not int) keeps it 2-D as (M, 1).

        self.coords             = point_cloud[:, :3]                                             # Columns 0-2 → on-surface XYZ, (N, 3). No normalization applied; must already be in [-1,1].
        self.normals            = point_cloud[:, 3:]                                             # Columns 3 onward → normals, (N, 3). GOTCHA: '3:' not '3:6', so a 7th column gets swallowed here.
        # self.free_points_coords = free_points_coords                                             # Stores free-space XYZ on the instance.
        # self.free_points_sdf    = free_points_sdf                                                # Stores free-space SDF on the instance.
        self.on_surface_points  = on_surface_points                                              # Surface points to draw per item (4000 in the shipped config).
        self.max_points         = max_points                                                     # If set, overrides the epoch length; see __len__.

        rng    = np.random.RandomState(split_seed + instance_idx)
        s_perm = rng.permutation(self.coords.shape[0])
        f_perm = rng.permutation(free_points_coords.shape[0])
        n_s    = max(1, int(round(val_ratio * len(s_perm))))
        n_f    = max(1, int(round(val_ratio * len(f_perm))))
        s_idx, f_idx = (s_perm[n_s:], f_perm[n_f:]) if split == 'train' else (s_perm[:n_s], f_perm[:n_f])

        self.coords             = self.coords[s_idx]
        self.normals            = self.normals[s_idx]
        self.free_points_coords = free_points_coords[f_idx]
        self.free_points_sdf    = free_points_sdf[f_idx]
        self.split, self.val_items = split, val_items


    # def __len__(self):                                                                           # Tells the DataLoader how many items this case contributes per epoch.
    #     if self.max_points != -1:                                                                # Branch taken with the shipped config (max_points = 300000).
    #         return self.max_points // self.on_surface_points                                     # 300000 // 4000 = 75 items. IGNORES the real file size, so a tiny file silently resamples itself.
    #     return self.coords.shape[0] // self.on_surface_points                                    # Fallback: scales with actual point count; under 4000 points returns 0 and the case contributes nothing.

    def __len__(self):
        if self.split == 'val':
            return self.val_items
        if self.max_points != -1:
            return self.max_points // self.on_surface_points
        return self.coords.shape[0] // self.on_surface_points

    def __getitem__(self, idx):                                                                  # Builds ONE training sample. 'idx' is accepted but never used — every call resamples randomly.
        rng = np.random.RandomState(1000 * self.instance_idx + idx) if self.split == 'val' else np.random
        point_cloud_size    = self.coords.shape[0]                                               # N: how many surface points are available.
        free_point_size     = self.free_points_coords.shape[0]                                   # M: how many free-space points are available.
        off_surface_samples = self.on_surface_points                                             # Off-surface budget is set EQUAL to the on-surface budget → 4000.

        total_samples       = self.on_surface_points + off_surface_samples                       # 4000 + 4000 = 8000 points in the final sample.
        rand_idcs           = rng.choice(point_cloud_size, size=self.on_surface_points)     # line 67   # Draws 4000 random surface row indices, WITH replacement (numpy default) → some duplicates.
        on_surface_coords   = self.coords[rand_idcs, :]                                          # Fancy-indexes those rows → (4000, 3). Fancy indexing returns a COPY, which makes the += below safe.
        on_surface_normals  = self.normals[rand_idcs, :]                                         # The matching 4000 normals → (4000, 3), same row order as the coords.
        

        if self.expand != -1:                                                                    # Only if dilation is enabled.
            on_surface_coords += on_surface_normals * self.expand                                # Pushes each surface point outward along its own normal; only correct if normals are unit length.

        off_surface_coords  = rng.uniform(-1, 1, size=(off_surface_samples // 2, 3))      # SYNTHESIZES 2000 uniform points in [-1,1]^3; not from your files. This hardcodes the [-1,1] domain. 
        free_rand_idcs      = rng.choice(free_point_size, size=off_surface_samples // 2)  # Draws 2000 random free-space indices, also with replacement. Only 2000 of your free points per item, not 4000.
        free_points_coords  = self.free_points_coords[free_rand_idcs, :]                         # XYZ of those 2000 free-space points → (2000, 3).
        off_surface_normals = np.ones((off_surface_samples, 3)) * -1                             # Dummy filler (4000, 3) of -1 covering BOTH off-surface blocks; never used (normal loss is gated on sdf == 0).

        sdf = np.zeros((total_samples, 1))                                                       # Ground-truth SDF buffer (8000, 1); zeros are already correct for the first 4000 (on the zero level set).
        sdf[self.on_surface_points:, :] = -1                                                     # Rows 4000-7999 set to -1: a SENTINEL meaning "no SDF supervision", not a distance. Half is overwritten next.

        if self.expand != -1:                                                                    # Dilation branch.
            sdf[self.on_surface_points + off_surface_samples // 2:, :] = (                       # Targets rows 6000-7999...
                    self.free_points_sdf[free_rand_idcs] - self.expand)                          # ...with the real SDF shifted by 'expand', keeping it consistent with the dilated surface.
        else:                                                                                    # Default branch (expand = -1).
            sdf[self.on_surface_points + off_surface_samples // 2:, :] = self.free_points_sdf[free_rand_idcs]  # Rows 6000-7999 get your real SDF verbatim; rows 4000-5999 keep the -1 sentinel. A true SDF of exactly -1.0 would be misread here.

        coords  = np.concatenate((on_surface_coords, off_surface_coords, free_points_coords), axis=0)  # Stacks to (8000, 3). ORDER IS THE CONTRACT: [0:4000] surface, [4000:6000] uniform, [6000:8000] free.
        normals = np.concatenate((on_surface_normals, off_surface_normals), axis=0)                    # Stacks 4000 real + 4000 dummy → (8000, 3), row-aligned with coords.

        return {'coords':       torch.from_numpy(coords).float(),                                # NumPy → tensor, cast to float32 (loadmat gives float64).
                'sdf':          torch.from_numpy(sdf).float(),                                   # (8000, 1) float32 distances and sentinels.
                'normals':      torch.from_numpy(normals).float(),                               # (8000, 3) float32 normals plus filler.
                'instance_idx': torch.Tensor([self.instance_idx]).squeeze().long()}              # ID as a scalar int64 tensor; squeeze drops the length-1 dim, .long() is required by nn.Embedding.


class PointCloudMultitrain(Dataset):                                                             # Wraps ALL cases and maps a global index → (which case, which sample).
    def __init__(self, root_dir, on_surface_points, max_num_instances=-1, expand=-1, max_points=-1,
                 split='train', val_ratio=0.1, split_seed=0, val_items=4, **kwargs):  # Constructor for the full training set.
        super().__init__()                                                                       # Dataset base-class init.

        self.root_dir = root_dir                                                                 # Either the surface_pts_n_normal folder path, or an explicit list of .mat paths.
        print(root_dir)                                                                          # Debug print of that path/list.
        if isinstance(root_dir, list):                                                           # If a list was passed...
            self.instance_dirs = root_dir                                                        # ...use it verbatim: no sorting, no pairing check. This is the hook for an explicit train/test split.
        else:                                                                                    # Otherwise scan the directory.
            self.instance_dirs = []                                                              # Accumulator for the discovered surface files.
            for file in sorted(os.listdir(root_dir)):                                            # ALPHABETICAL order — this defines instance_idx, so renaming files reshuffles latent codes. Note 'case_10' sorts before 'case_2'.
                if file.endswith('mat'):                                                         # Skips non-.mat entries (README, .DS_Store, ...).
                    if os.path.isfile(os.path.join(root_dir, file).replace('surface_pts_n_normal', 'free_space_pts')):  # Pairing check; a missing free-space twin is skipped SILENTLY, with no warning.
                        self.instance_dirs.append(os.path.join(root_dir, file))                  # Records the full path to the surface file.

        assert (len(self.instance_dirs) != 0), "No objects in the data directory"                # Hard stop if nothing matched — usually a wrong path or wrong folder names.

        if max_num_instances != -1:                                                              # If a cap was given (config: num_instances = 50)...
            self.instance_dirs = self.instance_dirs[:max_num_instances]                          # ...keep only the first K. Must be <= the model's nn.Embedding size.

        self.all_instances = [PointCloud_with_FreePoints(instance_idx     =idx,
                                                         pointcloud_path  =dir,
                                                         on_surface_points=on_surface_points, expand=expand,
                                                         max_points       =max_points,
                                                         split=split, val_ratio=val_ratio,
                                                         split_seed=split_seed, val_items=val_items)
                              for idx, dir in enumerate(self.instance_dirs)]                                    # MEMORY: every .mat (both halves, all cases) is loaded into RAM right here, before training starts.

        self.num_instances = len(self.all_instances)                                             # How many cases were actually loaded.
        self.num_per_instance_observations = [len(obj) for obj in self.all_instances]            # Calls __len__ on each case → e.g. [75, 75, 75, ...]; used for the index arithmetic below.

    def __len__(self):                                                                           # Total items per epoch across all cases.
        return np.sum(self.num_per_instance_observations)                                        # e.g. 50 cases x 75 = 3750.

    def get_instance_idx(self, idx):                                                             # Translates a global index into (case index, offset within that case).
        """Maps an index into all tuples of all objects to the idx of the tuple relative to the other tuples of that
        object
        """                                                                                      # Original docstring.
        obj_idx = 0                                                                              # Start at the first case.
        while idx >= 0:                                                                          # Walk forward until idx goes negative.
            idx -= self.num_per_instance_observations[obj_idx]                                   # Subtract this case's item count.
            obj_idx += 1                                                                         # Advance to the next case; the loop always overshoots by one.
        return obj_idx - 1, int(idx + self.num_per_instance_observations[obj_idx - 1])           # Undo the overshoot and rebuild the offset. The offset is decorative: __getitem__ ignores it.

    def collate_fn(self, batch_list):                                                            # Custom batching, needed because __getitem__ returns a tuple of lists of dicts.
        batch_list = zip(*batch_list)                                                            # Transposes [(obs_0, gt_0), (obs_1, gt_1), ...] into two groups: all observations, then all ground truths.

        all_parsed = []                                                                          # Will hold one merged dict per group.
        for entry in batch_list:                                                                 # 'entry' is a tuple of length batch_size; each element is a list of dicts (length 1 here).
            ret = {}                                                                             # Merged dict for this group.
            for k in entry[0][0].keys():                                                         # Reads key names off the first dict of the first sample.
                ret[k] = []                                                                      # Seeds an empty list per key ('coords', 'sdf', 'normals', 'instance_idx').
            for b in entry:                                                                      # Iterates over the samples in the batch.
                for k in entry[0][0].keys():                                                     # And over each key.
                    ret[k].extend([bi[k] for bi in b])                                           # Flattens the inner lists, collecting each sample's tensor under its key.
            for k in ret.keys():                                                                 # Now turn the lists into tensors.
                if type(ret[k][0]) == torch.Tensor:                                              # Only for tensor-valued entries.
                    ret[k] = torch.stack(ret[k])                                                 # Stacks along a NEW batch dim → coords (16,8000,3), sdf (16,8000,1), normals (16,8000,3), instance_idx (16,).
            all_parsed.append(ret)                                                               # Appends this group's merged dict.
        return tuple(all_parsed)                                                                 # Returns (model_input, ground_truth), the two dicts the training loop unpacks.

    def __getitem__(self, idx):                                                                  # Fetches one item by global index.
        """Each __getitem__ call yields a list of self.samples_per_instance observations of a single scene (each a dict),
        as well as a list of ground-truths for each observation (also a dict)."""                # Original docstring.
        obj_idx, rel_idx = self.get_instance_idx(idx)                                            # Global index → (case, offset).

        observations = []                                                                        # Container for this scene's observations.
        observations.append(self.all_instances[obj_idx][rel_idx])                                # Calls PointCloud_with_FreePoints.__getitem__ on the chosen case; listed because the codebase allows several observations per scene, though it's always one here.
        ground_truth = [{'sdf': obj['sdf'],                                                      # Duplicates the SDF into a separate GT dict...
                         'normals': obj['normals']} for obj in observations]                     # ...along with the normals. These are the SAME tensor objects, not copies.

        return observations, ground_truth                                                        # The tuple collate_fn expects.