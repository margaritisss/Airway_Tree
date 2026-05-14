import ants
from joblib import Parallel, delayed
import os
import time
import resource
import paramiko
from pathlib import Path

def register_affine_pairwise(input_folder, which='all', ref_tree=None, output_folder=None, sdt_folder=None):
    
    files = sorted(f for f in os.listdir(input_folder) if f.endswith('.nii.gz'))
    if not files:
        raise ValueError(f"No .nii.gz files found in {input_folder}")

    if which == 'all':
        n = len(files)
    elif isinstance(which, int) and 1 <= which <= len(files):
        n = which
    else:
        raise ValueError(
            f"`which` must be 'all' or an int in 1..{len(files)}, got {which!r}"
        )
        
    if output_folder is None:
        output_folder = os.path.join(input_folder, 'registered')
    os.makedirs(output_folder, exist_ok=True)
    
    if ref_tree is None:
        fixed_path = os.path.join(input_folder, files[0])
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

def _get_cpu_affinity():
    """Return the list of CPU IDs this process is allowed to run on."""
    try:
        return sorted(os.sched_getaffinity(0))
    except AttributeError:
        return None  

def _format_cpus(cpus):
    """Compact representation: [0,1,2,3] -> '0-3', [0,2,4] -> '0,2,4'."""
    if not cpus:
        return "?"
    cpus = sorted(cpus)
    ranges = []
    start = prev = cpus[0]
    for c in cpus[1:]:
        if c == prev + 1:
            prev = c
        else:
            ranges.append(f"{start}-{prev}" if start != prev else f"{start}")
            start = prev = c
    ranges.append(f"{start}-{prev}" if start != prev else f"{start}")
    return ",".join(ranges)

def _register_one(fname, input_folder, fixed_path, output_folder, itk_threads):
  
    os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(itk_threads)
    pid         = os.getpid()                       # get current process ID for logging
    cpus        = _format_cpus(_get_cpu_affinity()) # get CPU affinity for logging. Affinity =  which CPUs this process is allowed to run on. This can help understand how the OS is scheduling the processes and if there are any bottlenecks due to CPU contention.
    t_start     = time.time()
    start_stamp = time.strftime("%H:%M:")

    print(f"[{start_stamp}] START  pid={pid} cpus=[{cpus}] threads={itk_threads} file={fname}",flush=True)

    moving_path = os.path.join(input_folder, fname)
    fixed       = ants.image_read(fixed_path)
    moving      = ants.image_read(moving_path)

    reg = ants.registration(fixed =fixed, moving=moving, type_of_transform='Affine',)

    warped   = reg['warpedmovout'] # the registered image (warped moving image) is stored in the 'warpedmovout' key of the registration result dictionary returned by ants.registration().
    base     = fname[:-len('.nii.gz')]
    out_path = os.path.join(output_folder, f"{base}_R.nii.gz")
    ants.image_write(warped, out_path)

    elapsed     = time.time() - t_start # total time taken for this registration task
    peak_mem_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss /(1024 * 1024) # peak memory usage in GB. ru_maxrss is in bytes, so we convert to GB.
    end_stamp   = time.strftime("%H:%M:")

    print(f"[{end_stamp}] DONE   pid = {pid} file ={ fname} "
          f"elapsed={elapsed:6.1f}s ({elapsed/60:.1f}min) peak_mem={peak_mem_mb:6.0f}GB",
          flush=True)

    return out_path

def register_affine_pairwise_parallel(input_folder, which='all', ref_tree=None,
                                      output_folder=None, n_jobs=12, itk_threads=4):
    files = sorted(f for f in os.listdir(input_folder) if f.endswith('.nii.gz'))
    if not files:
        raise ValueError(f"No .nii.gz files found in {input_folder}")

    if which == 'all':
        n = len(files)
    elif isinstance(which, int) and 1 <= which <= len(files):
        n = which
    else:
        raise ValueError(f"`which` must be 'all' or int in 1..{len(files)}")

    if output_folder is None:
        output_folder = os.path.join(input_folder, 'registered')
    os.makedirs(output_folder, exist_ok=True)

    fixed_path = ref_tree if ref_tree else os.path.join(input_folder, files[0])

    todo = [] # list of files to process (skip already done)
    for fname in files[:n]: # [:n] 
        base = fname[:-len('.nii.gz')] # remove .nii.gz
        out_path = os.path.join(output_folder, f"{base}_R.nii.gz") # expected output path
        if not os.path.exists(out_path): # if output file doesn't exist, we need to process it
            todo.append(fname)  
              

    print(f"=== BATCH START ===", flush=True)
    print(f"Total files: {n}, already done: {n - len(todo)}, to process: {len(todo)}", flush=True)
    print(f"Parallelism: n_jobs={n_jobs}, itk_threads={itk_threads} "
          f"(total threads = {n_jobs * itk_threads})", flush=True)
    print(f"Fixed (reference): {os.path.basename(fixed_path)}", flush=True)
    print(f"Output: {output_folder}", flush=True)
    print(f"===================", flush=True)

    t0 = time.time()
    saved = Parallel(n_jobs=n_jobs, backend='loky', verbose=10)(
        delayed(_register_one)(fname, input_folder, fixed_path, output_folder, itk_threads)
        for fname in todo
    )
    total = time.time() - t0
    print(f"=== BATCH DONE ===", flush=True)
    print(f"Processed {len(saved)} files in {total/60:.1f} min "
          f"({total/max(len(saved),1):.1f}s per file on average)", flush=True)
    return saved

def download_files(
    remote_input_folder: str,
    local_output_folder: str,
    file_range: tuple,
    hostname: str,
    username: str,
    password: str = None,
    key_filename: str = None,
    port: int = 22,
):
    """
    Download a range of STL files from a remote cluster to the local machine.

    Parameters
    ----------
    remote_input_folder : str
        Path to the folder on the cluster containing the STL files.
    local_output_folder : str
        Path on the local machine where files will be saved (created if missing).
    file_range : tuple(int, int)
        (start, end) inclusive 1-based indices, e.g. (1, 10) for the first 10 files.
    hostname : str
        Cluster hostname or IP (e.g. "cluster.university.edu").
    username : str
        Your SSH username.
    password : str, optional
        SSH password (use either this or key_filename).
    key_filename : str, optional
        Path to your SSH private key (recommended over password).
    port : int
        SSH port, default 22.
    """
    # Create local folder if it doesn't exist
    local_path = Path(local_output_folder)
    local_path.mkdir(parents=True, exist_ok=True)

    # Connect via SSH
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(
        hostname=hostname,
        port=port,
        username=username,
        password=password,
        key_filename=key_filename,
    )

    try:
        sftp = ssh.open_sftp()

        # List & filter STL files (sorted for deterministic ordering)
        all_files = sorted(sftp.listdir(remote_input_folder))
        stl_files = [f for f in all_files if f.lower().endswith(".stl")]

        if not stl_files:
            print(f"No STL files found in {remote_input_folder}")
            return

        # Slice the requested range (1-based, inclusive)
        start, end = file_range
        selected = stl_files[start - 1 : end]

        print(f"Found {len(stl_files)} STL files. Downloading {len(selected)} "
              f"(indices {start}-{min(end, len(stl_files))}).")

        for i, filename in enumerate(selected, start=start):
            remote_file = f"{remote_input_folder.rstrip('/')}/{filename}"
            local_file = local_path / filename
            print(f"  [{i}] {filename} ...", end=" ")
            sftp.get(remote_file, str(local_file))
            print("done")

        sftp.close()
        print(f"\n✅ All files saved to: {local_path.resolve()}")

    finally:
        ssh.close()