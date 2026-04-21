# Standard library module for filesystem path operations (join, listdir, etc.)
import os
# PyTorch: main deep learning library, used here for tensors and GPU acceleration
import torch
# PyTorch's functional API: provides affine_grid and grid_sample for image warping
import torch.nn.functional as F
# NumPy: numerical arrays, used as intermediate format between NIfTI and torch
import numpy as np
# NiBabel: library for reading/writing medical imaging files in NIfTI (.nii.gz) format
import nibabel as nib
# torchreg's affine registration class: performs gradient-based affine alignment on the GPU
from torchreg import AffineRegistration

# Standard library thread primitives — Thread runs the sampler in the background,
# Event is a thread-safe flag we use to tell the sampler when to stop.
import threading
# time.monotonic() gives a steady elapsed-time clock that never goes backwards,
# even if the system clock is adjusted mid-run.
import time
# sys provides stdout and a way to detect whether stdout is a real terminal
# (isatty) — escape codes only work on real terminals, not in log files.
import sys

# psutil: cross-platform access to CPU and RAM usage (system-wide and per-process).
import psutil

# pynvml: NVIDIA's official Python bindings for querying GPU state. Import is
# wrapped in try/except so the module still works on machines without an
# NVIDIA GPU (or without the NVML library installed).
try:
    import pynvml
    _HAS_NVML = True
except ImportError:
    _HAS_NVML = False


# ---------------------------------------------------------------------------
# ANSI escape codes for controlling the terminal cursor and scrolling region.
# These are interpreted by the terminal emulator, not by Python. They let us
# reserve the top line for the monitor while letting all other output scroll
# normally in the remaining rows.
# ---------------------------------------------------------------------------
# CSI = "Control Sequence Introducer", the prefix for every ANSI sequence.
CSI = "\033["
# Save / restore the current cursor position. We use this every time the
# monitor updates: save where the scrolling cursor was, jump to the top,
# rewrite the monitor line, then jump back so scrolling output continues
# exactly where it left off.
SAVE_CURSOR    = CSI + "s"
RESTORE_CURSOR = CSI + "u"
# Move the cursor to row 1, column 1 (the "home" position).
CURSOR_HOME    = CSI + "H"
# Clear the entire current line without moving the cursor. Used before
# rewriting the monitor line so old content doesn't bleed through when the
# new line is shorter than the previous one.
CLEAR_LINE     = CSI + "2K"
# Reset the scrolling region to the full terminal (used during cleanup).
RESET_SCROLL_REGION = CSI + "r"


def _set_scroll_region(top, bottom):
    """Build an ANSI sequence that restricts scrolling to rows top..bottom."""
    return f"{CSI}{top};{bottom}r"


class ResourceMonitor:
    """
    Background resource monitor implemented as a context manager.

    Reserves the TOP LINE of the terminal for a live CPU / RAM / GPU readout
    and lets everything else (registration prints, torchreg progress bars,
    tracebacks, etc.) scroll normally in the rows below.

    Usage
    -----
        with ResourceMonitor(interval=1.0):
            register_pairwise_affine_gpu(...)

    On entry: sets up the reserved region. On exit (normally OR via an
    exception): tears down the region and restores normal terminal behaviour.
    """

    def __init__(self, interval=1.0, gpu_index=0, stream=sys.stdout):
        # Sampling period in seconds.
        self.interval = interval
        # Which GPU to monitor when multiple are present. 0 is the first one.
        self.gpu_index = gpu_index
        # Where to write the live line and control codes. Defaults to stdout.
        self.stream = stream

        # Decide whether fancy terminal control is even possible. If stdout
        # is redirected to a file or a pipe (no TTY), ANSI codes would appear
        # as literal garbage, so we fall back to a plain scrolling mode that
        # just prints one line per sample.
        self._is_tty = hasattr(stream, "isatty") and stream.isatty()

        # Thread control primitives.
        self._stop_event = threading.Event()
        self._thread = None
        # NVML handle for the chosen GPU; None means GPU monitoring disabled.
        self._gpu_handle = None

    # ------------------------------------------------------------------
    # Context manager protocol
    # ------------------------------------------------------------------
    def __enter__(self):
        # Initialize NVML; tolerate machines without NVIDIA GPUs.
        if _HAS_NVML:
            try:
                pynvml.nvmlInit()
                self._gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_index)
            except pynvml.NVMLError:
                self._gpu_handle = None

        # Prime psutil's delta-based CPU sampler (first call always returns 0).
        psutil.cpu_percent(interval=None)

        # Set up the reserved top line, but only if we're on a real terminal.
        if self._is_tty:
            # Step 1: emit a newline so the shell prompt / previous output
            # doesn't get consumed by the scrolling region. The newline pushes
            # the cursor down by one row, and that row is where scrolling
            # output will start from.
            self.stream.write("\n")
            # Step 2: restrict scrolling to rows 2..end. Row 1 is now
            # "outside" the scrolling region — writes can target it, but
            # normal output will never scroll over it.
            #
            # Note: 9999 is a cheap trick for "bottom of screen". Most
            # terminals silently clamp to the actual last row. Using the
            # real height would require querying it (shutil.get_terminal_size),
            # which is fine but adds a dependency for marginal benefit.
            self.stream.write(_set_scroll_region(2, 9999))
            # Step 3: move cursor into the scrolling region (row 2) so the
            # first print from the main thread doesn't overwrite the reserved
            # row. The "2;1H" sequence means "row 2, column 1".
            self.stream.write(f"{CSI}2;1H")
            self.stream.flush()

        # Start the sampling thread.
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Stop the sampler and wait for it to exit.
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval + 1.0)

        # Release NVML.
        if self._gpu_handle is not None:
            try:
                pynvml.nvmlShutdown()
            except pynvml.NVMLError:
                pass

        # Tear down terminal state. This is critical — if we skip it, the
        # user's shell prompt would keep the scrolling-region restriction
        # for the rest of the terminal session.
        if self._is_tty:
            # Clear the reserved top line so the final monitor reading
            # doesn't linger as a random-looking leftover at the top.
            self.stream.write(SAVE_CURSOR)
            self.stream.write(CURSOR_HOME)
            self.stream.write(CLEAR_LINE)
            self.stream.write(RESTORE_CURSOR)
            # Restore full-screen scrolling, so the shell behaves normally.
            self.stream.write(RESET_SCROLL_REGION)
            # Ensure subsequent prints start on a fresh row.
            self.stream.write("\n")
            self.stream.flush()

        # Don't suppress exceptions from the wrapped block.
        return False

    # ------------------------------------------------------------------
    # Background sampling loop
    # ------------------------------------------------------------------
    def _run(self):
        start = time.monotonic()

        # Event.wait(timeout) returns True if the stop event fires before
        # timeout, False if the timeout elapses. We loop while it keeps
        # returning False — i.e. while no stop has been requested.
        while not self._stop_event.wait(self.interval):
            # --- Sample CPU (system-wide %) over the last interval ---
            cpu_pct = psutil.cpu_percent(interval=None)

            # --- Sample RAM ---
            vm = psutil.virtual_memory()
            ram_pct = vm.percent
            ram_used_gb = vm.used / (1024 ** 3)
            ram_total_gb = vm.total / (1024 ** 3)

            # --- Sample GPU (if available) ---
            if self._gpu_handle is not None:
                try:
                    util = pynvml.nvmlDeviceGetUtilizationRates(self._gpu_handle)
                    gpu_pct = util.gpu
                    mem = pynvml.nvmlDeviceGetMemoryInfo(self._gpu_handle)
                    gpu_used_gb = mem.used / (1024 ** 3)
                    gpu_total_gb = mem.total / (1024 ** 3)
                    gpu_str = (f"GPU {gpu_pct:3d}% | "
                               f"VRAM {gpu_used_gb:5.2f}/{gpu_total_gb:5.2f} GiB")
                except pynvml.NVMLError:
                    gpu_str = "GPU   --% | VRAM   --/--  GiB"
            else:
                gpu_str = "GPU  n/a"

            elapsed = time.monotonic() - start

            # Build the status line (no CR/LF — positioning is handled below).
            line = (f"[{elapsed:6.1f}s] "
                    f"CPU {cpu_pct:5.1f}% | "
                    f"RAM {ram_pct:5.1f}% "
                    f"({ram_used_gb:5.2f}/{ram_total_gb:5.2f} GiB) | "
                    f"{gpu_str}")

            if self._is_tty:
                # The real trick: SAVE current cursor, HOME to top-left,
                # CLEAR the reserved line, write fresh content, then RESTORE
                # the cursor back to wherever scrolling output had left it.
                # The whole sequence is written as one string so the terminal
                # processes it atomically — no flicker, no race with the
                # main thread's own writes.
                self.stream.write(
                    SAVE_CURSOR + CURSOR_HOME + CLEAR_LINE
                    + line
                    + RESTORE_CURSOR
                )
                self.stream.flush()
            else:
                # Non-TTY fallback: just scroll one line per sample. No cursor
                # tricks because the output is probably going to a log file.
                self.stream.write(line + "\n")
                self.stream.flush()



def _load_as_tensor(path, device):
    """Load a NIfTI into a [1, 1, D, H, W] float tensor on the chosen device."""
    # Open the NIfTI file (lazy load, doesn't read voxel data yet)
    img    = nib.load(path)
    # Read the actual voxel values into a NumPy array as float32 (saves memory vs float64)
    data   = img.get_fdata(dtype=np.float32)
    # Convert the NumPy array to a torch tensor, add two leading dimensions
    # to get shape [batch=1, channel=1, D, H, W] (required by PyTorch conv/grid ops),
    # and move it to the chosen device (GPU or CPU)
    tensor = torch.from_numpy(data)[None, None].to(device)
    # Return the tensor plus the affine matrix (voxel->world transform) and header metadata,
    # which we'll need to save results back as a valid NIfTI
    return tensor, img.affine, img.header


def _save_tensor_as_nifti(tensor, affine, header, path):
    # Detach from the autograd graph, move back to CPU, convert to NumPy,
    # and drop the batch and channel dimensions to get a 3D volume
    arr = tensor.detach().cpu().numpy()[0, 0]
    # Wrap the NumPy array as a NIfTI image (reusing original affine + header)
    # and write it to disk at the given path
    nib.save(nib.Nifti1Image(arr, affine=affine, header=header), path)


def _apply_affine_nearest(moving, affine_matrix, out_shape):
    """
    Apply a torchreg affine matrix to a moving volume using nearest-neighbor
    interpolation. This mirrors what reg.transform() does internally, but lets
    us pick the interpolation mode (which reg.transform() does not expose).

    Parameters
    ----------
    moving : Tensor [1, 1, D, H, W]
    affine_matrix : Tensor [1, 3, 4]  (from reg.get_affine())
    out_shape : tuple (D, H, W)
    """
    # Build the target grid shape: [batch=1, channel=1, D, H, W]
    # This tells affine_grid how big the output volume should be
    grid_shape = (1, 1) + tuple(out_shape)
    # Generate a sampling grid of the given shape from the 3x4 affine matrix.
    # Each voxel in the output grid is mapped to a coordinate in the input volume.
    grid       = F.affine_grid(affine_matrix, size=grid_shape, align_corners=False)
    # Sample the moving volume at those coordinates using nearest-neighbor interpolation.
    # Nearest-neighbor preserves binary/integer labels (no interpolated fractional values).
    # 'zeros' padding means any sample falling outside the volume returns 0.
    warped     = F.grid_sample(
        moving, grid,
        mode='nearest',
        padding_mode='zeros',
        align_corners=False,
    )
    # Return the warped volume, same shape as out_shape
    return warped


def register_pairwise_affine_gpu(
    input_folder,
    which='all',
    ref_tree=None,
    output_folder=None,
    device=None,
):
    """
    GPU-accelerated affine registration, drop-in replacement for the ANTs version.
    Requires: torch (with CUDA), torchreg, nibabel.
    """
    # If the caller didn't specify a device, pick CUDA when available, else fall back to CPU
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # List all .nii.gz files in the input folder and sort them so the order is deterministic
    files = sorted(f for f in os.listdir(input_folder) if f.endswith('.nii.gz'))
    # Fail early if the folder is empty — otherwise we'd silently do nothing
    if not files:
        raise ValueError(f"No .nii.gz files found in {input_folder}")

    # Decide how many files to process:
    # 'all' means every file; an int means only the first N files
    if which == 'all':
        n = len(files)
    elif isinstance(which, int) and 1 <= which <= len(files):
        n = which
    else:
        # Anything else is a user error — reject it explicitly
        raise ValueError(f"which must be 'all' or int in 1..{len(files)}, got {which!r}")

    # Default output folder is a 'registered' subdirectory inside the input folder
    if output_folder is None:
        output_folder = os.path.join(input_folder, 'registered')
    # Create the output folder if it doesn't already exist (no error if it does)
    os.makedirs(output_folder, exist_ok=True)

    # --- Fixed image (the reference all others get aligned to) ---
    if ref_tree is None:
        # No explicit reference given → use the first file in the folder as the fixed image
        fixed_path  = os.path.join(input_folder, files[0])
        fixed_fname = files[0]
    else:
        # Use the reference path supplied by the caller
        fixed_path = ref_tree
        fixed_fname = os.path.basename(ref_tree)

    # Load the fixed image onto the device; keep its affine and header for saving later
    fixed_tensor, fixed_affine, fixed_header = _load_as_tensor(fixed_path, device)
    # Extract just the spatial dimensions (D, H, W) — used as the output shape for warping
    fixed_shape = fixed_tensor.shape[2:]

    # Announce the fixed reference once, up front. flush=True forces the line
    # out immediately so it appears before any monitoring output overwrites it.
    print(f"Fixed reference: {fixed_fname}  |  device: {device}  |  {n} file(s) to register",
          flush=True)

    # Keep track of where each registered file ends up, so we can return the list
    saved = []
    # Iterate over the first n files; enumerate with start=1 so i is a human-friendly counter
    for i, fname in enumerate(files[:n], start=1):
        # Build the full path to the current moving image and load it onto the device
        moving_path = os.path.join(input_folder, fname)
        # Announce the current file BEFORE work begins, so the user sees
        # progress during the slow optimization step (not only after it ends).
        # The leading \n keeps this line from being overwritten by a live
        # monitor line that may be running on the previous row.
        print(f"\n[{i}/{n}] Registering: {fname}  →  {fixed_fname}", flush=True)
        moving_tensor, _, _ = _load_as_tensor(moving_path, device)

        # Register the moving mask directly onto the fixed mask.
        # torchreg's AffineRegistration uses MSE loss by default in 3D mode.
        reg = AffineRegistration(is_3d=True)
        # Run the optimization: torchreg iteratively updates the affine matrix
        # to minimize the loss between the warped moving and the static fixed image
        _ = reg(moving=moving_tensor, static=fixed_tensor)
        # Pull out the learned 3x4 affine transformation matrix
        affine_mat = reg.get_affine()
        # Re-apply that affine to the original mask using nearest-neighbor
        # so the saved output stays strictly binary (no interpolated gray values)
        warped = _apply_affine_nearest(moving_tensor, affine_mat, fixed_shape)

        # Strip '.nii.gz' from the filename to build a cleaner output name
        base = fname[:-len('.nii.gz')]
        # Output filename: original name + '_R' suffix (for 'Registered')
        out_path = os.path.join(output_folder, f"{base}_R.nii.gz")
        # Save the warped volume, reusing the fixed image's affine/header so it sits
        # in the same world-space as the reference
        _save_tensor_as_nifti(warped, fixed_affine, fixed_header, out_path)
        # Remember this output path for the final return value
        saved.append(out_path)
        # Print a completion line. The leading \n is for the same reason as
        # the pre-registration print: it protects this message from being
        # overwritten by an active live monitor line.
        print(f"\nDone:       {fname} -> {os.path.basename(out_path)}", flush=True)

    # Return the list of all registered output paths
    return saved


if __name__ == "__main__":
    # Folder containing the .nii.gz binary masks to register.
    input_folder = "/Data/MEDICAL/data/unregistered"

    # Optional: pick a specific reference. If left as None, the first file
    # (alphabetically) in input_folder will be used as the fixed reference.
    ref_tree = None

    # Wrap the registration call in ResourceMonitor. The background thread
    # starts on `with` entry and stops on block exit — even if registration
    # raises an exception. interval=1.0 samples once per second.
    with ResourceMonitor(interval=1.0):
        saved_paths = register_pairwise_affine_gpu(
            input_folder=input_folder,
            which='all',
            ref_tree=ref_tree,
            output_folder='/Data/MEDICAL/data/registered_gpu',
        )

    # Normal program output resumes here, after monitoring has cleanly stopped.
    print(f"Finished. {len(saved_paths)} file(s) written.")