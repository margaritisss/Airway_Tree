import glob
import os
import ants
from joblib import Parallel, delayed
import nibabel as nib
import numpy as np
from skimage import measure
import trimesh

def register_pairwise_affine(input_folder, which='all', ref_tree=None, output_folder=None, sdt_folder=None):
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
    
    # Helper: SDT file path
    def sdt_path(fname):
        base = fname[:-len('.nii.gz')] # strip .nii.gz to get base name which is used for naming the SDT file
        return os.path.join(sdt_folder, f"{base}_sdt.nii.gz")

    # Fixed image: provided path, or first file in the folder
    if ref_tree is None:
        fixed_path = os.path.join(input_folder, files[0])
        fixed_fname = files[0] # for logging purposes, we do that before stripping .nii.gz to avoid confusion with the SDT file names
    else:
        fixed_path  = ref_tree
        fixed_fname = os.path.basename(ref_tree)
    fixed = ants.image_read(fixed_path)
    
    # If using SDTs, also load the fixed SDT (used to drive the registration)
    if sdt_folder is not None:
        fixed_sdt_path = sdt_path(fixed_fname)
        if not os.path.isfile(fixed_sdt_path):
            raise FileNotFoundError(
                f"Expected fixed SDT at {fixed_sdt_path} but it does not exist."
            )
        fixed_sdt = ants.image_read(fixed_sdt_path)

    saved = []
    for fname in files[:n]:
        moving_path = os.path.join(input_folder, fname)
        moving = ants.image_read(moving_path)

        if sdt_folder is None:
            # Original behavior: register the binary masks directly
            reg = ants.registration(
                fixed=fixed,
                moving=moving,
                type_of_transform='Affine',
            )
            warped = reg['warpedmovout']
        else:
            # SDT-driven: estimate transform on distance maps, apply to mask
            moving_sdt_path = sdt_path(fname)
            if not os.path.isfile(moving_sdt_path):
                raise FileNotFoundError(f"Expected moving SDT at {moving_sdt_path} but it does not exist.")
            moving_sdt = ants.image_read(moving_sdt_path)

            reg = ants.registration(
                fixed=fixed_sdt,
                moving=moving_sdt,
                type_of_transform='Affine',
                aff_metric='meansquares',
            )
            # Apply the estimated transform to the original binary mask
            warped = ants.apply_transforms(
                fixed=fixed,
                moving=moving,
                transformlist=reg['fwdtransforms'],
                interpolator='nearestNeighbor',
            )

        base = fname[:-len('.nii.gz')]
        out_path = os.path.join(output_folder, f"{base}_R.nii.gz")
        ants.image_write(warped, out_path)
        saved.append(out_path)
        print(f"Registered: {fname} -> {os.path.basename(out_path)}")

    return saved

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
            image_list      = images,
            iterations      = groupwise_iters,
            gradient_step   = gradient_step,
            blending_weight = blending_weight,
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
        stem     = fname[:-len('.nii.gz')]    # strip .nii.gz
        out_name = f"{stem}_R_G.nii.gz"
        out_path = os.path.join(output_folder, out_name)
        ants.image_write(reg['warpedmovout'], out_path)
        saved.append(os.path.abspath(out_path))
        print(f"saved -> {out_name}")

    print(f"\nDone!  {n} volume(s) saved to '{output_folder}'")
    return saved

def register_groupwise_affine(
    input_folder,
    which='all',
    output_folder=None,
    groupwise_iters=3,
    gradient_step=0.2,
    blending_weight=0.75,
    verbose=False,
):
    """
    Groupwise + affine (linear) registration of .nii.gz files.

    Builds an unbiased group template via iterative affine registration
    (ants.build_template), then affinely registers every selected image
    to that template.  Anatomy is preserved (no local bending) — only
    rotation, translation, scaling, and shear are applied.

    Results are saved as <stem>_R_G.nii.gz.
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
    # 4. Build an unbiased groupwise template (AFFINE only)
    #
    #    By forcing type_of_transform='Affine', the template-building
    #    step uses only linear transforms (rotation, translation,
    #    scaling, shear) — no diffeomorphic warping.  Anatomy is
    #    preserved as a rigid-plus-scale body.
    # ------------------------------------------------------------------
    if n == 1:
        print("\nOnly one image selected — using it as its own template.")
        template = images[0]
    else:
        print(
            f"\nBuilding groupwise template "
            f"({groupwise_iters} iteration(s), Affine) ..."
        )
        template = ants.build_template(
            image_list=images,
            iterations=groupwise_iters,
            gradient_step=gradient_step,
            blending_weight=blending_weight,
            type_of_transform='Affine',   # <-- linear only
        )
        print("Groupwise template built successfully.")

    # ------------------------------------------------------------------
    # 5. Affinely register every selected image to the template
    #
    #    'Affine' = 12 degrees of freedom (rotation + translation +
    #    scaling + shear).  No local deformation; airway branching
    #    geometry, lumen diameters, and bifurcation angles are preserved.
    # ------------------------------------------------------------------
    print(f"\nAffinely registering {n} image(s) to the template ...")
    saved = []
    for fname, img in zip(selected, images):
        print(f"  {fname} ...", end=' ', flush=True)
        reg = ants.registration(
            fixed=template,
            moving=img,
            type_of_transform='Affine',   # <-- linear only
            verbose=verbose,
        )
        stem = fname[:-len('.nii.gz')]
        out_name = f"{stem}_R_G.nii.gz"
        out_path = os.path.join(output_folder, out_name)
        ants.image_write(reg['warpedmovout'], out_path)
        saved.append(os.path.abspath(out_path))
        print(f"saved -> {out_name}")

    print(f"\nDone!  {n} volume(s) saved to '{output_folder}'")
    return saved

def nifti_to_aligned_stl(file_path, output_filename=None, level=0.5, verbose=True):
    """
    Convert a NIfTI file (.nii.gz) to a spatially aligned STL mesh.
    
    Parameters
    ----------
    file_path : str
        Path to the input NIfTI file (.nii.gz).
    output_filename : str, optional
        Path for the output STL file. If None, it is auto-generated by
        replacing '.nii.gz' with '_aligned.stl' in the input path.
    level : float, optional
        Iso-surface level used by the Marching Cubes algorithm (default: 0.5,
        suitable for binary segmentation masks).
    verbose : bool, optional
        If True, print inspection and progress information (default: True).
    
    Returns
    -------
    dict
        Dictionary containing:
            - 'mesh'            : the trimesh.Trimesh object
            - 'output_path'     : path to the saved STL file
            - 'affine'          : the 4x4 affine matrix
            - 'voxel_size'      : voxel dimensions in mm
            - 'data_shape'      : shape of the NIfTI data grid
            - 'is_segmentation' : True if the volume appears to be a segmentation mask
    """
    # Auto-generate output path if not provided
    if output_filename is None:
        output_filename = file_path.replace('.nii.gz', '_aligned.stl')
    
    # ==========================================
    # PART 1: LOAD & INSPECT
    # ==========================================
    if verbose:
        print(f"Loading {file_path}...\n")
    
    img           = nib.load(file_path)
    data          = img.get_fdata()
    voxel_size    = img.header.get_zooms()
    affine_matrix = img.affine
    
    unique_values   = np.unique(data)
    is_segmentation = len(unique_values) <= 10
    
    if verbose:
        print("--- NIfTI File Inspection ---")
        print(f"Data Shape (Grid Dimensions): {data.shape}")
        print(f"Voxel Size (mm): {voxel_size}")
        print("\nAffine Matrix (Position in space):")
        print(affine_matrix)
        print("\n--- Data Value Analysis ---")
        print(f"Minimum Value: {np.min(data)}")
        print(f"Maximum Value: {np.max(data)}")
        if is_segmentation:
            print("\nCONCLUSION: This is likely a **Segmentation Mask**.")
            print(f"It only contains these specific labels: {unique_values}")
        else:
            print("\nCONCLUSION: This is likely a **Raw CT/MRI Scan**.")
    
    # ==========================================
    # PART 2: PROCESS & ALIGN MESH
    # ==========================================
    if verbose:
        print("\n--- Mesh Generation & Alignment ---")
        print("Applying Marching Cubes algorithm...")
    
    # Extract raw voxel coordinates; the affine matrix handles all spatial transformation
    verts, faces, normals, _ = measure.marching_cubes(data, level=level)
    
    if verbose:
        print("Constructing 3D Mesh Object...")
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
    
    if verbose:
        print("Applying the full Affine Matrix spatial transformation...")
    mesh.apply_transform(affine_matrix)
    
    if verbose:
        print("Exporting Mesh...")
    mesh.export(output_filename)
    
    if verbose:
        print(f"\nSuccess! Spatially aligned mesh saved to:\n{output_filename}")
    
    return {
        'mesh': mesh,
        'output_path': output_filename,
        'affine': affine_matrix,
        'voxel_size': voxel_size,
        'data_shape': data.shape,
        'is_segmentation': is_segmentation,
    }

def batch_nifti_to_stl(input_folder, how_many='all', level=0.5, verbose=True):
    """
    Convert NIfTI files (.nii.gz) in a folder to spatially aligned STL meshes.
    Each STL is saved in the same input folder, with the suffix '_aligned.stl'.
    
    Parameters
    ----------
    input_folder : str
        Path to the folder containing .nii.gz files. STL outputs are saved here.
    how_many : int or str, optional
        How many files to process. Either a positive integer, or the string
        'all' to process every .nii.gz file in the folder (default: 'all').
    level : float, optional
        Iso-surface level passed to Marching Cubes (default: 0.5).
    verbose : bool, optional
        If True, print progress information (default: True).
    
    Returns
    -------
    list of dict
        A list of result dictionaries, one per successfully converted file.
    """
    # Validate input folder
    if not os.path.isdir(input_folder):
        raise NotADirectoryError(f"Input folder does not exist: {input_folder}")
    
    # Collect all .nii.gz files (sorted for reproducibility)
    all_files = sorted(glob.glob(os.path.join(input_folder, '*.nii.gz')))
    
    if not all_files:
        print(f"No .nii.gz files found in: {input_folder}")
        return []
    
    # Resolve how_many
    if isinstance(how_many, str):
        if how_many.lower() == 'all':
            files_to_process = all_files
        else:
            raise ValueError(
                f"Invalid value for how_many: {how_many!r}. "
                "Use a positive integer or the string 'all'."
            )
    elif isinstance(how_many, int):
        if how_many <= 0:
            raise ValueError("how_many must be a positive integer.")
        files_to_process = all_files[:how_many]
    else:
        raise TypeError("how_many must be an int or the string 'all'.")
    
    if verbose:
        print(f"Found {len(all_files)} file(s) in '{input_folder}'.")
        print(f"Processing {len(files_to_process)} file(s). Outputs saved in the same folder.\n")
    
    # Process each selected file
    results = []
    for i, file_path in enumerate(files_to_process, start=1):
        if verbose:
            print(f"[{i}/{len(files_to_process)}]")
        
        try:
            # output_filename=None -> auto-generated next to input file
            result = nifti_to_aligned_stl(
                file_path,
                output_filename=None,
                level  =level,
                verbose=verbose,
            )
            results.append(result)
        except Exception as e:
            print(f"  ERROR processing {file_path}: {e}\n")
    
    if verbose:
        print(f"Done. {len(results)}/{len(files_to_process)} file(s) converted successfully.")
    
    return results

def pairwise_affine_parallel(input_folder, which='all', ref_tree=None, output_folder=None, sdt_folder=None):
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
    
    # Helper: SDT file path
    def sdt_path(fname):
        base = fname[:-len('.nii.gz')] # strip .nii.gz to get base name which is used for naming the SDT file
        return os.path.join(sdt_folder, f"{base}_sdt.nii.gz")

    # Fixed image: provided path, or first file in the folder
    if ref_tree is None:
        fixed_path = os.path.join(input_folder, files[0])
        fixed_fname = files[0] # for logging purposes, we do that before stripping .nii.gz to avoid confusion with the SDT file names
    else:
        fixed_path  = ref_tree
        fixed_fname = os.path.basename(ref_tree)
    fixed = ants.image_read(fixed_path)
    
    # If using SDTs, also load the fixed SDT (used to drive the registration)
    if sdt_folder is not None:
        fixed_sdt_path = sdt_path(fixed_fname)
        if not os.path.isfile(fixed_sdt_path):
            raise FileNotFoundError(
                f"Expected fixed SDT at {fixed_sdt_path} but it does not exist."
            )
        fixed_sdt = ants.image_read(fixed_sdt_path)

    saved = []
    n_workers = int(os.environ.get('SLURM_CPUS_PER_TASK', 1))
    saved = Parallel(n_jobs=n_workers)(delayed(_register_one)(fname, input_folder, output_folder, fixed, fixed_sdt, sdt_folder, sdt_path)
       for fname in files[:n])
    
    return saved

def _register_one(fname, input_folder, output_folder, fixed, fixed_sdt, sdt_folder, sdt_path):
    
    moving_path = os.path.join(input_folder, fname) # we do that inside the function to avoid issues with parallelization and shared state
    moving      = ants.image_read(moving_path)      # we do that inside the function to avoid issues with parallelization and shared state

    if sdt_folder is None:
        reg    = ants.registration(fixed=fixed, moving=moving, type_of_transform='Affine')
        warped = reg['warpedmovout']
    else:
        moving_sdt = ants.image_read(sdt_path(fname))
        reg        = ants.registration(fixed=fixed_sdt, moving=moving_sdt,type_of_transform='Affine', aff_metric='meansquares',)
        warped = ants.apply_transforms(fixed=fixed, moving=moving, transformlist=reg['fwdtransforms'], interpolator ='nearestNeighbor',)

    base     = fname[:-len('.nii.gz')] # 
    out_path = os.path.join(output_folder, f"{base}_R.nii.gz")
    ants.image_write(warped, out_path)
    print(f"Registered: {fname} -> {os.path.basename(out_path)}")
    return out_path