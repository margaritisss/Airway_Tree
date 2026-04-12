import os
import ants

def register_airway_trees(input_folder, which='all', ref_tree=None, output_folder=None):
    """
    Linearly (affine) register .nii.gz files from a folder.

    Parameters
    ----------
    input_folder : str
        Path to folder containing .nii.gz files.
    which : int or 'all'
        Number of files to register. Use 'all' to register every file.
    fixed_image : str, optional
        Path to the fixed reference image. If None, the first file in the
        folder is used as the fixed image.
    output_folder : str, optional
        Where to save results. Defaults to <input_folder>/registered.

    Returns
    -------
    list of str
        Paths to the saved registered images.
    """
    # Collect .nii.gz files (sorted for reproducibility)
    files = sorted(f for f in os.listdir(input_folder) if f.endswith('.nii.gz'))
    if not files:
        raise ValueError(f"No .nii.gz files found in {input_folder}")

    # Validate `which`
    if which == 'all':
        n = len(files)
    elif isinstance(which, int) and 1 <= which <= len(files):
        n = which
    else:
        raise ValueError(
            f"`which` must be 'all' or an int in 1..{len(files)}, got {which!r}"
        )

    # Output folder
    if output_folder is None:
        output_folder = os.path.join(input_folder, 'registered')
    os.makedirs(output_folder, exist_ok=True)

    # Fixed image: provided path, or first file in the folder
    if ref_tree is None:
        fixed_path = os.path.join(input_folder, files[0])
    else:
        fixed_path = ref_tree
    fixed = ants.image_read(fixed_path)

    saved = []
    for fname in files[:n]:
        moving_path = os.path.join(input_folder, fname)
        moving = ants.image_read(moving_path)

        reg = ants.registration(
            fixed=fixed,
            moving=moving,
            type_of_transform='Affine',
        )

        base = fname[:-len('.nii.gz')]
        out_path = os.path.join(output_folder, f"{base}_R.nii.gz")
        ants.image_write(reg['warpedmovout'], out_path)
        saved.append(out_path)
        print(f"Registered: {fname} -> {os.path.basename(out_path)}")

    return saved

# Example usage:
# register_airway_tree('/path/to/data', which='all')
# register_airway_tree('/path/to/data', which=5)
# register_airway_tree('/path/to/data', which=10, fixed_image='/path/to/template.nii.gz')


def register_groupwise_deformable(
    input_folder,
    which='all',
    output_folder=None,
    groupwise_iters=3,
    gradient_step=0.2,
    blending_weight=0.75,
    verbose=False,
):
    """
    Groupwise + deformable (SyN) registration of .nii.gz files.

    Builds an unbiased group template via iterative SyN registration
    (ants.build_template), then deformably registers every selected image
    to that template.  Results are saved as <stem>_R_G.nii.gz.

    Parameters
    ----------
    input_folder : str
        Path to a folder that contains .nii.gz source files.
    which : int or 'all'
        How many files to register.  Pass an integer 1 … N to take the
        first N files (sorted alphabetically), or 'all' to include every
        .nii.gz file found in the folder.
    output_folder : str, optional
        Destination folder for the registered volumes.
        Defaults to <input_folder>/registered.
    groupwise_iters : int
        Template-building iterations passed to ants.build_template (default 3).
    gradient_step : float
        Shape-update gradient step used during template building (default 0.2).
    blending_weight : float
        Template blending weight in [0, 1].  Higher values produce a
        smoother template (default 0.75).
    verbose : bool
        Forward ANTs verbosity to the internal registration calls (default False).

    Returns
    -------
    list of str
        Absolute paths to every saved _R_G.nii.gz file, in the same order
        as the selected source files.

    Examples
    --------
    # Register all files found in the data folder
    from register_groupwise import register_groupwise_deformable
    paths = register_groupwise_deformable('/Data/MEDICAL/data', which='all')

    # Register only the first 3 files
    paths = register_groupwise_deformable('/Data/MEDICAL/data', which=3)
    """
    # ------------------------------------------------------------------
    # 1. Discover .nii.gz files
    # ------------------------------------------------------------------
    all_files = sorted(
        f for f in os.listdir(input_folder) if f.endswith('.nii.gz')
    )
    if not all_files:
        raise ValueError(f"No .nii.gz files found in '{input_folder}'")

    if which == 'all':
        n = len(all_files)
    elif isinstance(which, int) and 1 <= which <= len(all_files):
        n = which
    else:
        raise ValueError(
            f"`which` must be 'all' or an integer in 1..{len(all_files)}, "
            f"got {which!r}"
        )

    selected = all_files[:n]

    # ------------------------------------------------------------------
    # 2. Prepare output folder
    # ------------------------------------------------------------------
    if output_folder is None:
        output_folder = os.path.join(input_folder, 'registered')
    os.makedirs(output_folder, exist_ok=True)

    # ------------------------------------------------------------------
    # 3. Load images with ANTs
    # ------------------------------------------------------------------
    print(f"Loading {n} image(s) from '{input_folder}' ...")
    images = []
    for fname in selected:
        img = ants.image_read(os.path.join(input_folder, fname))
        images.append(img)
        print(f"  {fname}  shape={img.shape}")

    # ------------------------------------------------------------------
    # 4. Build an unbiased groupwise template
    #
    #    ants.build_template iteratively registers every image to the
    #    current template (using SyN by default) and updates the template
    #    as the weighted average of the warped images.  The result is an
    #    unbiased common reference space — the core of groupwise
    #    registration as described in the VoxelMorph template-learning
    #    literature (Dalca et al., NeurIPS 2019).
    # ------------------------------------------------------------------
    if n == 1:
        # Degenerate case: one image is its own template.
        print("\nOnly one image selected — using it as its own template.")
        template = images[0]
    else:
        print(
            f"\nBuilding groupwise template "
            f"({groupwise_iters} iteration(s), SyN) ..."
        )
        template = ants.build_template( # 
            image_list=images,
            iterations=groupwise_iters,
            gradient_step=gradient_step,
            blending_weight=blending_weight,
        )
        print("Groupwise template built successfully.")

    # ------------------------------------------------------------------
    # 5. Deformably register every selected image to the template (SyN)
    #
    #    SyN (Symmetric Normalization) is ANTs' diffeomorphic deformable
    #    registration algorithm — the same family of deformable transforms
    #    that VoxelMorph's VxmPairwise network is trained to approximate.
    # ------------------------------------------------------------------
    print(f"\nDeformably registering {n} image(s) to the template (SyN) ...")
    saved = []
    for fname, img in zip(selected, images):
        print(f"  {fname} ...", end=' ', flush=True)
        reg = ants.registration(
            fixed=template,
            moving=img,
            type_of_transform='SyN',
            verbose=verbose,
        )
        stem = fname[:-len('.nii.gz')]          # strip .nii.gz
        out_name = f"{stem}_R_G.nii.gz"
        out_path = os.path.join(output_folder, out_name)
        ants.image_write(reg['warpedmovout'], out_path)
        saved.append(os.path.abspath(out_path))
        print(f"saved -> {out_name}")

    print(f"\nDone!  {n} volume(s) saved to '{output_folder}'")
    return saved


# ---------------------------------------------------------------------------
# Quick self-test when run directly
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import sys
    folder = sys.argv[1] if len(sys.argv) > 1 else '/Data/MEDICAL/data'
    which  = sys.argv[2] if len(sys.argv) > 2 else 'all'
    if which != 'all':
        which = int(which)
    register_groupwise_deformable(folder, which=which)
