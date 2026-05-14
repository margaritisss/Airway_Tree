import os
import sys
import time
import threading
import socket
from contextlib import contextmanager
from datetime import datetime

# Set BEFORE importing ants/ITK so the thread count takes effect.
os.environ.setdefault(
    'ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS',
    str(os.cpu_count() or 1),
)
os.environ.setdefault('OMP_NUM_THREADS', str(os.cpu_count() or 1))

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import ants

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False

# ----------------------------------------------------------------------
# Logging helpers
# ----------------------------------------------------------------------
def log(msg):
    """Timestamped, immediately-flushed print. Use this everywhere."""
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')[:-3]
    print(f"[{ts}] {msg}", flush=True)

def log_hardware():
    """Print one-time hardware / environment summary at job start."""
    log(f"Host: {socket.gethostname()}")
    log(f"PID:  {os.getpid()}")
    log(f"Python: {sys.version.split()[0]}")
    log(f"ANTsPy: {getattr(ants, '__version__', 'unknown')}")

    # SLURM-visible CPU count (what we're actually allowed to use)
    slurm_cpus = os.environ.get('SLURM_CPUS_PER_TASK')
    log(f"os.cpu_count(): {os.cpu_count()}")
    log(f"SLURM_CPUS_PER_TASK: {slurm_cpus}")
    log(f"len(os.sched_getaffinity(0)): {len(os.sched_getaffinity(0))}")

    if _HAS_PSUTIL:
        vm = psutil.virtual_memory()
        log(f"Total RAM: {vm.total / 1e9:.1f} GB, "
            f"available: {vm.available / 1e9:.1f} GB")
    else:
        log("psutil not installed -- resource monitor will be disabled. "
            "Install with: pip install psutil")
# ----------------------------------------------------------------------
# Per-stage timer context manager
# ----------------------------------------------------------------------
@contextmanager
def stage(name):
    """Context manager that logs entry, exit and elapsed wall time."""
    log(f"=== STAGE START: {name} ===")
    t0 = time.monotonic()
    try:
        yield
    finally:
        dt = time.monotonic() - t0
        h, rem = divmod(dt, 3600)
        m, s = divmod(rem, 60)
        log(f"=== STAGE END:   {name}  (elapsed {int(h)}h{int(m):02d}m{s:05.2f}s) ===")

# ----------------------------------------------------------------------
# Background resource monitor
#
# Runs in a daemon thread and periodically logs:
#   - elapsed wall time
#   - aggregate RSS of the parent + all descendant processes (this is
#     what SLURM's cgroup OOM killer cares about)
#   - node-wide CPU utilisation
#   - number of live child processes
# ----------------------------------------------------------------------
class ResourceMonitor:
    def __init__(self, interval_sec=30):
        self.interval = interval_sec
        self._stop = threading.Event()
        self._thread = None
        self._t0 = None
        self._peak_rss_gb = 0.0

    def start(self):
        if not _HAS_PSUTIL:
            log("ResourceMonitor disabled (psutil missing).")
            return
        self._t0 = time.monotonic()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        log(f"ResourceMonitor started (every {self.interval}s).")

    def stop(self):
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=self.interval + 5)
        log(f"ResourceMonitor stopped. Peak tree RSS: {self._peak_rss_gb:.2f} GB")

    def _run(self):
        parent = psutil.Process(os.getpid())
        # Prime cpu_percent so the next call returns a real value.
        psutil.cpu_percent(interval=None)
        while not self._stop.wait(self.interval):
            try:
                kids = parent.children(recursive=True)
                rss = parent.memory_info().rss
                for c in kids:
                    try:
                        rss += c.memory_info().rss
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                rss_gb = rss / 1e9
                self._peak_rss_gb = max(self._peak_rss_gb, rss_gb)

                vm = psutil.virtual_memory()
                cpu = psutil.cpu_percent(interval=None)
                elapsed = time.monotonic() - self._t0
                log(
                    f"[monitor] elapsed={elapsed/60:6.1f}min  "
                    f"workers={len(kids):2d}  "
                    f"tree_rss={rss_gb:6.2f}GB  "
                    f"node_avail={vm.available/1e9:6.2f}GB  "
                    f"cpu={cpu:5.1f}%"
                )
            except Exception as e:
                log(f"[monitor] error: {e}")


# ----------------------------------------------------------------------
# Worker-side helpers (must be module-level so they pickle cleanly)
# ----------------------------------------------------------------------
def _init_worker(threads):
    """
    Initializer run once per worker process.
    """
    os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = str(threads)
    os.environ['OMP_NUM_THREADS'] = str(threads)
    import ants  # noqa: F401

def _register_one(args):
    """
    Worker function: deformably register a single image to the template.
    Returns a small dict of timing + path info so the parent can log it.
    """
    in_path, template_path, out_path, verbose = args
    pid = os.getpid()
    t_start = time.monotonic()

    # We can't use the parent's `log()` here cleanly because each worker
    # has its own stdout buffer; printing with flush=True is enough.
    print(f"[worker {pid}] START  {os.path.basename(in_path)}", flush=True)

    img = ants.image_read(in_path)
    template = ants.image_read(template_path)
    t_loaded = time.monotonic()

    reg = ants.registration(
        fixed=template,
        moving=img,
        type_of_transform='SyNOnly',
        verbose=verbose,
    )
    t_regdone = time.monotonic()

    ants.image_write(reg['warpedmovout'], out_path)
    t_end = time.monotonic()

    print(
        f"[worker {pid}] DONE   {os.path.basename(in_path)}  "
        f"load={t_loaded-t_start:.1f}s  "
        f"reg={t_regdone-t_loaded:.1f}s  "
        f"write={t_end-t_regdone:.1f}s  "
        f"total={t_end-t_start:.1f}s",
        flush=True,
    )

    return {
        'out_path': os.path.abspath(out_path),
        'in_name': os.path.basename(in_path),
        'load_s': t_loaded - t_start,
        'reg_s': t_regdone - t_loaded,
        'write_s': t_end - t_regdone,
        'total_s': t_end - t_start,
    }

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def register_groupwise_deformable(
    input_folder,
    which='all',
    output_folder=None,
    groupwise_iters=3,
    gradient_step=0.2,
    blending_weight=0.75,
    verbose=False,
    n_workers=8,
    threads_per_worker=6,
    template_threads=None,
    monitor_interval_sec=30,
    existing_template_path=None,
):
    """
    Groupwise + deformable (SyN) registration of .nii.gz files. 
    """
    log_hardware()
    monitor = ResourceMonitor(interval_sec=monitor_interval_sec)
    monitor.start()

    try:
        # --------------------------------------------------------------
        # 1. Discover .nii.gz files
        # --------------------------------------------------------------
        with stage("1: discover input files"):
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
            log(f"Selected {n} of {len(all_files)} files.")

        # --------------------------------------------------------------
        # 2. Prepare output folder
        # --------------------------------------------------------------
        with stage("2: prepare output folder"):
            if output_folder is None:
                output_folder = os.path.join(input_folder, 'registered')
            os.makedirs(output_folder, exist_ok=True)
            log(f"Output folder: {output_folder}")

        # --------------------------------------------------------------
        # 3+4. Either reuse an existing template, or build one groupwise
        # --------------------------------------------------------------
        if existing_template_path is not None:
            with stage("3+4: reuse existing template (skipping groupwise build)"):
                if not os.path.isfile(existing_template_path):
                    raise FileNotFoundError(
                        f"existing_template_path not found: {existing_template_path}"
                    )
                template = ants.image_read(existing_template_path)
                log(f"Loaded template from {existing_template_path}  "
                    f"shape={template.shape}")
            # Make sure children inherit the per-worker thread count
            os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = str(threads_per_worker)
            os.environ['OMP_NUM_THREADS'] = str(threads_per_worker)
        elif n == 1:
            with stage("3: load single image (degenerate template)"):
                only_path = os.path.join(input_folder, selected[0])
                template = ants.image_read(only_path)
                log(f"  {selected[0]} shape={template.shape}")
        else:
            with stage(f"3: load {n} images"):
                images = []
                for fname in selected:
                    t_load = time.monotonic()
                    img = ants.image_read(os.path.join(input_folder, fname))
                    images.append(img)
                    log(f"  loaded {fname}  shape={img.shape}  "
                        f"({time.monotonic()-t_load:.1f}s)")

            if template_threads is None:
                template_threads = os.cpu_count() or 1
            os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = str(template_threads)
            os.environ['OMP_NUM_THREADS'] = str(template_threads)

            with stage(f"4: build template ({groupwise_iters} iters, "
                       f"{template_threads} threads)"):
                template = ants.build_template(
                    image_list=images,
                    iterations=groupwise_iters,
                    gradient_step=gradient_step,
                    blending_weight=blending_weight,
                    type_of_transform='SyNOnly',
                )
            del images

            os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = str(threads_per_worker)
            os.environ['OMP_NUM_THREADS'] = str(threads_per_worker)

        # --------------------------------------------------------------
        # 5. Per-image SyN registration in parallel
        # --------------------------------------------------------------
        
        if existing_template_path is not None:
            template_path = existing_template_path   # reuse on disk, no rewrite
        else:
            template_path = os.path.join(output_folder, '_template.nii.gz')
            ants.image_write(template, template_path)
        del template

        n_workers = min(n_workers, n)
        tasks = []
        for fname in selected:
            stem = fname[:-len('.nii.gz')]
            tasks.append((
                os.path.join(input_folder, fname),
                template_path,
                os.path.join(output_folder, f"{stem}_R_G.nii.gz"),
                verbose,
            ))

        results = [None] * len(tasks)
        with stage(f"5: SyN registration ({n} images, {n_workers} workers x "
                   f"{threads_per_worker} threads)"):
            ctx = mp.get_context('spawn')
            with ProcessPoolExecutor(
                max_workers=n_workers,
                mp_context=ctx,
                initializer=_init_worker,
                initargs=(threads_per_worker,),
            ) as ex:
                future_to_idx = {
                    ex.submit(_register_one, t): i for i, t in enumerate(tasks)
                }
                completed = 0
                for fut in as_completed(future_to_idx):
                    i = future_to_idx[fut]
                    info = fut.result()
                    results[i] = info['out_path']
                    completed += 1
                    log(
                        f"  [{completed}/{n}] finished: {info['in_name']}  "
                        f"(reg={info['reg_s']:.1f}s, total={info['total_s']:.1f}s)"
                    )

        log(f"All {n} volumes saved to '{output_folder}'")
        return results

    finally:
        monitor.stop()