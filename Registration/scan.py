import glob
import os
import ants
from scipy.ndimage import zoom
from joblib import Parallel, delayed
import nibabel as nib
import numpy as np
from skimage import measure
import trimesh
from nibabel.filebasedimages import ImageFileError
from nibabel.filebasedimages import ImageFileError
from scipy.ndimage import zoom


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
    

    # Fixed image: provided path, or first file in the folder
    if ref_tree is None:
        fixed_path = os.path.join(input_folder, files[0])
        fixed_fname = files[0] # for logging purposes, we do that before stripping .nii.gz to avoid confusion with the SDT file names
    else:
        fixed_path  = ref_tree
        fixed_fname = os.path.basename(ref_tree)
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
        warped = reg['warpedmovout']

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

def nifti_to_mesh(input_folder, how_many='all', level=0.5, verbose=True):
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

def export_nifti_resolutions(input_folder, output_filename="resolutions.txt"):
    """
    Safely reads .nii.gz files, ignoring directories, non-NIfTI files, 
    and handling corrupted data gracefully.
    """
    output_dir = "/home/ids/gmargari-24"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_filename)
    
    # 1. Find files matching the pattern
    search_pattern = os.path.join(input_folder, '*.nii.gz')
    
    # 2. Robustness check: Ensure it is actually a file, not a directory 
    # that happens to be named '.nii.gz'
    nii_files = [f for f in sorted(glob.glob(search_pattern)) if os.path.isfile(f)]
    
    if not nii_files:
        print(f"No valid '.nii.gz' files were found in {input_folder}")
        return

    with open(output_path, 'w') as f:
        f.write("Index | Filename | Dimensions (Voxels) | Voxel Spacing (mm)\n")
        f.write("-" * 80 + "\n")
        
        successful_reads = 0
        
        for index, file_path in enumerate(nii_files, start=1):
            filename = os.path.basename(file_path)
            try:
                # Attempt to load the file
                img = nib.load(file_path)
                
                # Extract data
                shape_str = "x".join(map(str, img.shape))
                zooms_str = "x".join([f"{z:.3f}" for z in img.header.get_zooms()])
                
                f.write(f"{index}. {filename} | Dims: {shape_str} | Spacing: {zooms_str}\n")
                successful_reads += 1
                
            # 3. Robustness check: Catch files that say they are NIfTI but aren't (e.g., renamed meshes)
            except ImageFileError:
                print(f"Warning: '{filename}' is corrupted or not a true NIfTI file.")
                f.write(f"{index}. {filename} | ERROR: Invalid/Corrupted NIfTI Format\n")
                
            # 4. Robustness check: Catch any other unexpected errors (e.g., read permissions)
            except Exception as e:
                print(f"Error processing '{filename}': {e}")
                f.write(f"{index}. {filename} | ERROR: {str(e)}\n")
                
    print(f"\nProcessing complete! Successfully read {successful_reads} out of {len(nii_files)} files.")
    print(f"Results saved to: {output_path}")

def crop_airway_masks(input_folder, which='all'):
    """
    Reads a specified number of .nii.gz masks, calculates the bounding box 
    of the non-zero elements, crops the image, and saves it.
    
    Args:
        input_folder (str): Directory containing the original .nii.gz files.
        which (str or int): 'all' to process every file, or an integer (e.g., 5) 
                            to process the first N files.
    """
    # Set up output directory
    base_output_dir = "/home/ids/gmargari-24"
    cropped_dir = os.path.join(base_output_dir, "cropped_masks")
    os.makedirs(cropped_dir, exist_ok=True)
    
    # Find valid NIfTI files
    search_pattern = os.path.join(input_folder, '*.nii.gz')
    nii_files = [f for f in sorted(glob.glob(search_pattern)) if os.path.isfile(f)]
    
    if not nii_files:
        print(f"No '.nii.gz' files found in {input_folder}")
        return

    # Handle the 'which' argument
    if str(which).lower() != 'all':
        try:
            limit = int(which)
            nii_files = nii_files[:limit]
        except ValueError:
            print("Error: 'which' must be 'all' or an integer (e.g., 10).")
            return

    print(f"Starting to process {len(nii_files)} files...")
    
    successful_crops = 0

    for index, file_path in enumerate(nii_files, start=1):
        filename = os.path.basename(file_path)
        
        try:
            # 1. Load the NIfTI file and extract the data array
            img = nib.load(file_path)
            data = img.get_fdata()
            
            # 2. Check if the mask is empty (all zeros)
            if not np.any(data):
                print(f"[{index}] Skipping '{filename}': Mask is completely empty (no airways found).")
                continue
                
            # 3. Compute the Bounding Box
            # np.nonzero returns the indices of elements that are not zero
            non_zero_indices = np.nonzero(data)
            
            # Get min and max coordinates for X, Y, and Z dimensions
            x_min, x_max = np.min(non_zero_indices[0]), np.max(non_zero_indices[0])
            y_min, y_max = np.min(non_zero_indices[1]), np.max(non_zero_indices[1])
            z_min, z_max = np.min(non_zero_indices[2]), np.max(non_zero_indices[2])
            
            # 4. Crop the data array (+1 because array slicing is exclusive at the end)
            cropped_data = data[x_min:x_max+1, y_min:y_max+1, z_min:z_max+1]
            
            # 5. Adjust the Affine Matrix (Crucial for medical imaging!)
            # We must shift the origin by multiplying the original affine by the shift vector
            original_affine = img.affine
            shift_vector = [x_min, y_min, z_min, 1]
            new_origin = original_affine.dot(shift_vector)
            
            # Create a copy of the affine and update its translation column
            new_affine = original_affine.copy()
            new_affine[:3, 3] = new_origin[:3]
            
            # 6. Save the new cropped NIfTI file
            # Create the new NIfTI object. Nibabel automatically updates the header shape.
            cropped_img = nib.Nifti1Image(cropped_data, new_affine, img.header)
            
            # Construct output path (e.g., scan_001_cropped.nii.gz)
            name, ext = filename.split('.nii.gz')
            output_filename = f"{name}_cropped.nii.gz"
            output_path = os.path.join(cropped_dir, output_filename)
            
            nib.save(cropped_img, output_path)
            print(f"[{index}] Cropped '{filename}' -> Original: {data.shape}, Cropped: {cropped_data.shape}")
            successful_crops += 1

        except ImageFileError:
            print(f"[{index}] Error: '{filename}' is corrupted or not a true NIfTI file.")
        except Exception as e:
            print(f"[{index}] Error processing '{filename}': {e}")

    print(f"\nFinished! Successfully cropped {successful_crops} out of {len(nii_files)} files.")
    print(f"Cropped files are saved in: {cropped_dir}")

def get_dimension_extremes(input_folder):


    """
    Reads all .nii.gz files and finds the absolute lowest and highest values 
    for X, Y, and Z, while retaining the original full dimension of that file.
    """
    search_pattern = os.path.join(input_folder, '*.nii.gz')
    nii_files = [f for f in sorted(glob.glob(search_pattern)) if os.path.isfile(f)]
    
    if not nii_files:
        print(f"No valid '.nii.gz' files found in {input_folder}")
        return None

    # Dictionaries to keep track of the extreme value AND the full shape/file
    # Mins start at infinity, Maxes start at 0
    extremes = {
        'min_x': {'val': float('inf'), 'shape': None, 'file': None},
        'max_x': {'val': 0,             'shape': None, 'file': None},
        'min_y': {'val': float('inf'), 'shape': None, 'file': None},
        'max_y': {'val': 0,             'shape': None, 'file': None},
        'min_z': {'val': float('inf'), 'shape': None, 'file': None},
        'max_z': {'val': 0,             'shape': None, 'file': None},
    }
    
    successful_reads = 0

    print(f"Scanning {len(nii_files)} files for detailed dimension extremes...")

    for file_path in nii_files:
        filename = os.path.basename(file_path)
        try:
            img = nib.load(file_path)
            shape = img.shape
            
            if len(shape) < 3:
                continue
                
            x, y, z = shape[0], shape[1], shape[2]
            
            # --- Check X Axis ---
            if x < extremes['min_x']['val']:
                extremes['min_x'] = {'val': x, 'shape': shape, 'file': filename}
            if x > extremes['max_x']['val']:
                extremes['max_x'] = {'val': x, 'shape': shape, 'file': filename}
                
            # --- Check Y Axis ---
            if y < extremes['min_y']['val']:
                extremes['min_y'] = {'val': y, 'shape': shape, 'file': filename}
            if y > extremes['max_y']['val']:
                extremes['max_y'] = {'val': y, 'shape': shape, 'file': filename}
                
            # --- Check Z Axis ---
            if z < extremes['min_z']['val']:
                extremes['min_z'] = {'val': z, 'shape': shape, 'file': filename}
            if z > extremes['max_z']['val']:
                extremes['max_z'] = {'val': z, 'shape': shape, 'file': filename}
            
            successful_reads += 1
            
        except ImageFileError:
            pass  # Silently skip non-NIfTI files
        except Exception as e:
            print(f"Error reading {filename}: {e}")

    if successful_reads == 0:
        print("Could not read any valid 3D dimensions from the files.")
        return None

    # Print the requested 6 values
    print("\n" + "=" * 80)
    print(f"Successfully analyzed {successful_reads} files.")
    print("=" * 80)
    
    print(f"1. Lowest X:  {extremes['min_x']['val']:<5} | Original Dimension: {str(extremes['min_x']['shape']):<18} | File: {extremes['min_x']['file']}")
    print(f"2. Highest X: {extremes['max_x']['val']:<5} | Original Dimension: {str(extremes['max_x']['shape']):<18} | File: {extremes['max_x']['file']}")
    
    print("-" * 80)
    print(f"3. Lowest Y:  {extremes['min_y']['val']:<5} | Original Dimension: {str(extremes['min_y']['shape']):<18} | File: {extremes['min_y']['file']}")
    print(f"4. Highest Y: {extremes['max_y']['val']:<5} | Original Dimension: {str(extremes['max_y']['shape']):<18} | File: {extremes['max_y']['file']}")
    
    print("-" * 80)
    print(f"5. Lowest Z:  {extremes['min_z']['val']:<5} | Original Dimension: {str(extremes['min_z']['shape']):<18} | File: {extremes['min_z']['file']}")
    print(f"6. Highest Z: {extremes['max_z']['val']:<5} | Original Dimension: {str(extremes['max_z']['shape']):<18} | File: {extremes['max_z']['file']}")
    print("=" * 80 + "\n")
    
    return extremes

def pad_and_downsample(input_folder, target_pad=672, target_downsample=256):
    """
    Pads NIfTI masks symmetrically to a target size (default 672^3) 
    and then downsamples them to a new target size (default 256^3).
    """
    # Set up output directories exactly where you want them
    base_dir = "/home/ids/gmargari-24"
    pad_dir = os.path.join(base_dir, f"padded_{target_pad}")
    down_dir = os.path.join(base_dir, f"downsampled_{target_downsample}")
    
    os.makedirs(pad_dir, exist_ok=True)
    os.makedirs(down_dir, exist_ok=True)
    
    # Find valid files
    search_pattern = os.path.join(input_folder, '*.nii.gz')
    nii_files = [f for f in sorted(glob.glob(search_pattern)) if os.path.isfile(f)]
    
    if not nii_files:
        print(f"No '.nii.gz' files found in {input_folder}")
        return

    print(f"Processing {len(nii_files)} files...")
    print(f"1. Padding to {target_pad}x{target_pad}x{target_pad}")
    print(f"2. Downsampling to {target_downsample}x{target_downsample}x{target_downsample}\n")

    successful_files = 0

    for index, file_path in enumerate(nii_files, start=1):
        filename = os.path.basename(file_path)
        
        try:
            # --- 1. LOAD DATA ---
            img = nib.load(file_path)
            data = img.get_fdata()
            original_affine = img.affine
            x, y, z = data.shape
            
            # --- 2. PADDING STEP ---
            # Calculate how much to pad on each axis. max(0, ...) prevents negative padding
            pad_x = max(0, target_pad - x)
            pad_y = max(0, target_pad - y)
            pad_z = max(0, target_pad - z)
            
            # Symmetrical padding (split the padding evenly before and after)
            px_before, px_after = pad_x // 2, pad_x - (pad_x // 2)
            py_before, py_after = pad_y // 2, pad_y - (pad_y // 2)
            pz_before, pz_after = pad_z // 2, pad_z - (pad_z // 2)
            
            # Apply padding (using 0 for the background of masks)
            padded_data = np.pad(
                data, 
                pad_width=((px_before, px_after), (py_before, py_after), (pz_before, pz_after)), 
                mode='constant', 
                constant_values=0
            )
            
            # Update Affine for Padding
            # Padding shifts the array index relative to the physical origin, 
            # so we must subtract the padding applied at the 'before' edges.
            pad_shift_matrix = np.eye(4)
            pad_shift_matrix[0, 3] = -px_before
            pad_shift_matrix[1, 3] = -py_before
            pad_shift_matrix[2, 3] = -pz_before
            padded_affine = original_affine.dot(pad_shift_matrix)
            
            # Save Padded Image
            padded_img = nib.Nifti1Image(padded_data, padded_affine, img.header)
            nib.save(padded_img, os.path.join(pad_dir, filename))

            # --- 3. DOWNSAMPLING STEP ---
            # Calculate the scaling factor
            scale_factor = target_downsample / target_pad
            
            # Apply downsampling using scipy.ndimage.zoom
            # CRITICAL: order=0 is Nearest Neighbor. Use this for MASKS so you don't create decimals!
            # If you run this on original CT scans, change order to 1 (trilinear) or 3 (cubic).
            downsampled_data = zoom(padded_data, zoom=scale_factor, order=0)
            
            # Update Affine for Downsampling
            # Voxels are now larger, so the physical spacing multiplier goes up
            downsample_scale_matrix = np.diag([1/scale_factor, 1/scale_factor, 1/scale_factor, 1])
            downsampled_affine = padded_affine.dot(downsample_scale_matrix)
            
            # Save Downsampled Image
            downsampled_img = nib.Nifti1Image(downsampled_data, downsampled_affine, padded_img.header)
            nib.save(downsampled_img, os.path.join(down_dir, filename))
            
            print(f"[{index}] Success: '{filename}' | Original: {data.shape} -> Padded: {padded_data.shape} -> Downsampled: {downsampled_data.shape}")
            successful_files += 1

        except ImageFileError:
            print(f"[{index}] Error: '{filename}' is corrupted or not a true NIfTI file.")
        except Exception as e:
            print(f"[{index}] Error processing '{filename}': {e}")

    print(f"\nPipeline Complete! Processed {successful_files} out of {len(nii_files)} files.")
    print(f"Padded masks saved in:      {pad_dir}")
    print(f"Downsampled masks saved in: {down_dir}")

def crop_and_resample_airway_masks(input_folders, which='all',
                                   target_shape=(128, 128, 128),
                                   common_bbox_padding=8):
    """
    Two-pass pipeline applied jointly across one or more input folders:
      Pass 1: scan all masks in every input folder, find each one's airway
              bounding box, determine a SINGLE common bbox size that fits every
              tree across all folders (so all output volumes share the same
              physical extent).
      Pass 2: for each mask, center the common bbox on the airway centroid,
              extract (padding with zeros if needed), and resample to target_shape.
 
    Output:
      For each input folder, a sibling folder is created in the same parent
      directory with the suffix "_<target_shape[0]>" (e.g. "masks" ->
      "masks_128"). Resampled masks from that input folder are written there.
 
    Args:
        input_folders (str or list[str]): A single folder path or a list of
            folder paths containing the registered .nii.gz masks.
        which (str or int): 'all' or an integer N to process only the first N
            files from EACH input folder.
        target_shape (tuple): Final grid size after resampling, e.g.
            (128, 128, 128).
        common_bbox_padding (int): Extra voxels added to each side of the common
            bbox size (small safety margin).
    """
    # Normalize input_folders to a list
    if isinstance(input_folders, (str, os.PathLike)):
        input_folders = [input_folders]
    input_folders = [os.path.abspath(p) for p in input_folders]
 
    # Build the file list per folder and the matching output directories
    files_per_folder = {}   # input_folder -> list of file paths
    output_dirs = {}        # input_folder -> output directory path
 
    for in_folder in input_folders:
        if not os.path.isdir(in_folder):
            print(f"Warning: input folder does not exist, skipping: {in_folder}")
            continue
 
        search_pattern = os.path.join(in_folder, '*.nii.gz')
        nii_files = [f for f in sorted(glob.glob(search_pattern))
                     if os.path.isfile(f)]
 
        if not nii_files:
            print(f"No '.nii.gz' files found in {in_folder}")
            continue
 
        # Apply 'which' filter per folder
        if str(which).lower() != 'all':
            try:
                limit = int(which)
                nii_files = nii_files[:limit]
            except ValueError:
                print("Error: 'which' must be 'all' or an integer (e.g., 10).")
                return
 
        files_per_folder[in_folder] = nii_files
 
        # Output directory: sibling of the input folder, with "_<size>" suffix
        parent_dir = os.path.dirname(in_folder.rstrip(os.sep))
        folder_name = os.path.basename(in_folder.rstrip(os.sep))
        out_folder = os.path.join(parent_dir,
                                  f"{folder_name}_{target_shape[0]}")
        os.makedirs(out_folder, exist_ok=True)
        output_dirs[in_folder] = out_folder
 
    if not files_per_folder:
        print("No valid input folders / files. Aborting.")
        return
 
    total_files = sum(len(v) for v in files_per_folder.values())
 
    # =========================================================================
    # PASS 1 — gather per-scan bboxes and centroids across ALL folders
    # =========================================================================
    print(f"\n=== Pass 1: scanning {total_files} files across "
          f"{len(files_per_folder)} folders for bbox statistics ===")
 
    bbox_info = {}  # file_path -> dict with shape + bbox_size + centroid + source folder
    max_dx = max_dy = max_dz = 0
 
    global_index = 0
    for in_folder, nii_files in files_per_folder.items():
        print(f"\n-- Folder: {in_folder} ({len(nii_files)} files)")
        for file_path in nii_files:
            global_index += 1
            filename = os.path.basename(file_path)
            try:
                img = nib.load(file_path)
                data = img.get_fdata()
 
                if not np.any(data):
                    print(f"[{global_index}] Skipping '{filename}': empty mask.")
                    continue
 
                nz = np.nonzero(data)
                x_min, x_max = nz[0].min(), nz[0].max()
                y_min, y_max = nz[1].min(), nz[1].max()
                z_min, z_max = nz[2].min(), nz[2].max()
 
                dx = x_max - x_min + 1
                dy = y_max - y_min + 1
                dz = z_max - z_min + 1
 
                # Centroid (center of mass of non-zero voxels), as integers
                cx = int(round(nz[0].mean()))
                cy = int(round(nz[1].mean()))
                cz = int(round(nz[2].mean()))
 
                bbox_info[file_path] = {
                    'shape': data.shape,
                    'bbox_size': (dx, dy, dz),
                    'centroid': (cx, cy, cz),
                    'source_folder': in_folder,
                }
 
                max_dx = max(max_dx, dx)
                max_dy = max(max_dy, dy)
                max_dz = max(max_dz, dz)
 
            except ImageFileError:
                print(f"[{global_index}] Error: '{filename}' is corrupted.")
            except Exception as e:
                print(f"[{global_index}] Error in pass 1 for '{filename}': {e}")
 
    if not bbox_info:
        print("No valid masks found. Aborting.")
        return
 
    # Common bbox size: max dimension across all scans + small padding, rounded
    # to even numbers so centering is symmetric.
    common_dx = int(np.ceil((max_dx + 2 * common_bbox_padding) / 2.0)) * 2
    common_dy = int(np.ceil((max_dy + 2 * common_bbox_padding) / 2.0)) * 2
    common_dz = int(np.ceil((max_dz + 2 * common_bbox_padding) / 2.0)) * 2
    common_bbox = (common_dx, common_dy, common_dz)
 
    print(f"\nMax per-axis bbox size across dataset: ({max_dx}, {max_dy}, {max_dz})")
    print(f"Chosen common bbox size:                {common_bbox}")
    print(f"Resampling target shape:                {target_shape}")
 
    # =========================================================================
    # PASS 2 — crop to common bbox (centered on centroid), then resample
    # =========================================================================
    print(f"\n=== Pass 2: cropping + resampling to {target_shape} ===")
 
    successful = 0
    half = (common_dx // 2, common_dy // 2, common_dz // 2)
 
    for index, (file_path, info) in enumerate(bbox_info.items(), start=1):
        filename = os.path.basename(file_path)
        src_folder = info['source_folder']
        out_dir = output_dirs[src_folder]
        try:
            img = nib.load(file_path)
            data = img.get_fdata()
            orig_shape = data.shape
            cx, cy, cz = info['centroid']
 
            # Desired crop window in original coordinates (may extend outside volume)
            x_start = cx - half[0]
            y_start = cy - half[1]
            z_start = cz - half[2]
            x_end = x_start + common_dx
            y_end = y_start + common_dy
            z_end = z_start + common_dz
 
            # Allocate zero-padded output of common bbox size
            cropped = np.zeros(common_bbox, dtype=data.dtype)
 
            # Source slice (clipped to image bounds)
            sx0 = max(x_start, 0)
            sy0 = max(y_start, 0)
            sz0 = max(z_start, 0)
            sx1 = min(x_end, orig_shape[0])
            sy1 = min(y_end, orig_shape[1])
            sz1 = min(z_end, orig_shape[2])
 
            # Destination slice (offset in case the window started before 0)
            dx0 = sx0 - x_start
            dy0 = sy0 - y_start
            dz0 = sz0 - z_start
            dx1 = dx0 + (sx1 - sx0)
            dy1 = dy0 + (sy1 - sy0)
            dz1 = dz0 + (sz1 - sz0)
 
            cropped[dx0:dx1, dy0:dy1, dz0:dz1] = data[sx0:sx1, sy0:sy1, sz0:sz1]
 
            # Resample to target_shape using nearest-neighbor (order=0) for binary masks
            zoom_factors = (
                target_shape[0] / common_bbox[0],
                target_shape[1] / common_bbox[1],
                target_shape[2] / common_bbox[2],
            )
            resampled = zoom(cropped, zoom_factors, order=0, prefilter=False)
 
            # Ensure shape is exactly target_shape (zoom can be off-by-one occasionally)
            if resampled.shape != target_shape:
                fixed = np.zeros(target_shape, dtype=resampled.dtype)
                s = tuple(slice(0, min(resampled.shape[i], target_shape[i]))
                          for i in range(3))
                fixed[s] = resampled[s]
                resampled = fixed
 
            # Re-binarize defensively (nearest-neighbor should already be binary)
            resampled = (resampled > 0).astype(np.uint8)
 
            # ---- Adjust the affine ----
            # Original affine maps voxel -> physical. After cropping, the new
            # origin corresponds to the cropped volume's (0,0,0), which is at
            # (x_start, y_start, z_start) in original voxel coordinates (note:
            # this may be negative, which is mathematically fine).
            original_affine = img.affine
            shift_vector = np.array([x_start, y_start, z_start, 1.0])
            new_origin = original_affine.dot(shift_vector)
 
            # Resampling scales the spacing: new_spacing = old_spacing / zoom
            new_affine = original_affine.copy()
            for i in range(3):
                new_affine[:3, i] = original_affine[:3, i] / zoom_factors[i]
            new_affine[:3, 3] = new_origin[:3]
 
            # ---- Save ----
            out_img = nib.Nifti1Image(resampled, new_affine)
            out_img.set_qform(new_affine, code=1)
            out_img.set_sform(new_affine, code=1)
 
            name = filename.split('.nii.gz')[0]
            output_path = os.path.join(out_dir, f"{name}_resampled.nii.gz")
            nib.save(out_img, output_path)
 
            print(f"[{index}] {filename}: {orig_shape} "
                  f"-> crop {common_bbox} -> resample {target_shape}  "
                  f"(voxels: {int(resampled.sum())})  "
                  f"-> {os.path.basename(out_dir)}")
            successful += 1
 
        except Exception as e:
            print(f"[{index}] Error in pass 2 for '{filename}': {e}")
 
    print(f"\nFinished! Successfully processed {successful} of "
          f"{len(bbox_info)} masks.")
    print("Output directories:")
    for in_folder, out_dir in output_dirs.items():
        print(f"  {in_folder}  ->  {out_dir}")


def isotropic_center_pad(input_folders, 
                                        target_spacing=(0.5, 0.5, 0.5), 
                                        target_shape=(640, 640, 832)):
    """
    Scans input folder(s) for .nii.gz masks, resamples each to true physical 
    isotropic resolution, center-pads (or crops) to a fixed grid size, and 
    saves them in a sibling output folder ending in _<target_shape>.
    """
    # Normalize input_folders to a list
    if isinstance(input_folders, (str, os.PathLike)):
        input_folders = [input_folders]
    input_folders = [os.path.abspath(p) for p in input_folders]

    for in_folder in input_folders:
        if not os.path.isdir(in_folder):
            print(f"Warning: input folder does not exist, skipping: {in_folder}")
            continue
        
        # Build the file list
        search_pattern = os.path.join(in_folder, '*.nii.gz')
        nii_files = sorted(glob.glob(search_pattern))
        
        if not nii_files:
            print(f"No '.nii.gz' files found in {in_folder}")
            continue
            
        # Create output directory: sibling of the input folder with _640 suffix
        parent_dir = os.path.dirname(in_folder.rstrip(os.sep))
        folder_name = os.path.basename(in_folder.rstrip(os.sep))
        
        # Automatically uses the first dimension of your target_shape (e.g., 640)
        out_folder = os.path.join(parent_dir, f"{folder_name}_{target_shape[0]}")
        os.makedirs(out_folder, exist_ok=True)
        
        print(f"\n=== Processing {len(nii_files)} files in: {in_folder} ===")
        print(f"Outputting to: {out_folder}")
        
        successful = 0
        
        for index, file_path in enumerate(nii_files, start=1):
            filename = os.path.basename(file_path)
            out_path = os.path.join(out_folder, filename.replace('.nii.gz', '_resampled.nii.gz'))
            
            try:
                # ---------------------------------------------------------
                # 1. Load Data and Metadata
                # ---------------------------------------------------------
                img = nib.load(file_path)
                data = img.get_fdata()
                original_affine = img.affine
                orig_spacing = img.header.get_zooms()[:3]
                
                # ---------------------------------------------------------
                # 2. Resample to Isotropic Resolution
                # ---------------------------------------------------------
                zoom_factors = [orig_spacing[i] / target_spacing[i] for i in range(3)]
                
                resampled_data = zoom(data, zoom_factors, order=0, prefilter=False)
                current_shape = resampled_data.shape
                
                # ---------------------------------------------------------
                # 3. Center Pad (or Crop) to Target Shape
                # ---------------------------------------------------------
                final_data = np.zeros(target_shape, dtype=resampled_data.dtype)
                shift_in_voxels = [0, 0, 0]
                
                start_idx_current = [0, 0, 0]
                end_idx_current = list(current_shape)
                start_idx_final = [0, 0, 0]
                end_idx_final = list(target_shape)

                for i in range(3):
                    diff = target_shape[i] - current_shape[i]
                    if diff > 0:
                        # Pad
                        pad_start = diff // 2
                        start_idx_final[i] = pad_start
                        end_idx_final[i] = pad_start + current_shape[i]
                        shift_in_voxels[i] = -pad_start 
                    elif diff < 0:
                        # Crop
                        crop_start = abs(diff) // 2
                        start_idx_current[i] = crop_start
                        end_idx_current[i] = crop_start + target_shape[i]
                        shift_in_voxels[i] = crop_start 

                s_final = tuple(slice(start_idx_final[i], end_idx_final[i]) for i in range(3))
                s_current = tuple(slice(start_idx_current[i], end_idx_current[i]) for i in range(3))
                
                final_data[s_final] = resampled_data[s_current]
                
                # Re-binarize defensively
                final_data = (final_data > 0).astype(np.uint8)

                # ---------------------------------------------------------
                # 4. Compute New Affine
                # ---------------------------------------------------------
                new_affine = np.copy(original_affine)
                
                for i in range(3):
                    new_affine[:3, i] = new_affine[:3, i] / zoom_factors[i]
                
                shift_vector = np.array([shift_in_voxels[0], shift_in_voxels[1], shift_in_voxels[2], 1.0])
                new_origin = new_affine.dot(shift_vector)
                new_affine[:3, 3] = new_origin[:3]

                # ---------------------------------------------------------
                # 5. Save Output
                # ---------------------------------------------------------
                out_img = nib.Nifti1Image(final_data, new_affine)
                out_img.set_qform(new_affine, code=1)
                out_img.set_sform(new_affine, code=1)
                
                out_img.header.set_zooms(target_spacing)
                
                nib.save(out_img, out_path)
                print(f"[{index}/{len(nii_files)}] {filename} resampled to {current_shape} -> padded to {target_shape}")
                successful += 1
                
            except Exception as e:
                print(f"[{index}/{len(nii_files)}] Error processing '{filename}': {e}")
                
        print(f"\nFinished! Successfully processed {successful} of {len(nii_files)} files in {folder_name}.")