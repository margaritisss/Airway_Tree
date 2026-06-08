"""
Optuna worker script. One SLURM array task runs this script.

The worker connects to a shared SQLite study DB, asks for a trial, runs it,
reports the result, then asks for the next trial. It exits when either
the global trial budget is reached or it gets close to the SLURM wallclock.

Resuming: re-running the same script with the same study name + storage
URL just picks up where things left off. Workers coordinate via the DB.
"""

from __future__ import annotations
import argparse
import logging
import os
import signal
import sys
import time
from functools import partial
from pathlib import Path

import optuna
from optuna.pruners import HyperbandPruner
from optuna.samplers import TPESampler
from optuna.storages import RDBStorage

from hpo_objective import objective


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--study-name", required=True)
    p.add_argument("--storage", required=True,
                   help="e.g. sqlite:////abs/path/to/optuna_study.db")
    p.add_argument("--data-dirs", type=Path, nargs="+", required=True)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--max-epochs", type=int, default=80,
                   help="Per-trial max epochs. With Hyperband this is the "
                        "max budget for the longest rung. Pruned trials "
                        "stop earlier.")
    p.add_argument("--n-trials", type=int, default=20,
                   help="Per-worker trial budget. The shared study DB caps "
                        "the global trial count separately if you set it.")
    p.add_argument("--global-trial-cap", type=int, default=60,
                   help="Worker exits if the study already has this many "
                        "completed+running trials.")
    p.add_argument("--soft-deadline-seconds", type=int, default=82_800,
                   help="Worker stops launching new trials this long after "
                        "startup. Default ~16.6h leaves margin under an 18h "
                        "SLURM wallclock.")
    p.add_argument("--log-dir", type=Path, default=Path("optuna_logs"))
    return p.parse_args()


def make_logger(log_dir: Path, worker_tag: str) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger(f"optuna_worker.{worker_tag}")
    log.setLevel(logging.INFO)
    fh = logging.FileHandler(log_dir / f"worker_{worker_tag}.log")
    sh = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
    for h in (fh, sh):
        h.setFormatter(fmt)
        log.addHandler(h)
    return log


def main():
    args = parse_args()
    worker_tag = (
        f"slurm{os.environ.get('SLURM_JOB_ID', 'local')}"
        f"_arr{os.environ.get('SLURM_ARRAY_TASK_ID', '0')}"
    )
    log = make_logger(args.log_dir, worker_tag)
    log.info("Worker %s starting. PID=%d", worker_tag, os.getpid())
    log.info("Args: %s", vars(args))

    # On the first worker, create_study creates the DB. Subsequent workers
    # join the existing study. load_if_exists=True handles both cases.
    sampler = TPESampler(
        n_startup_trials=10,       # random search for first 10 trials
        multivariate=True,         # model correlations between hyperparams
        group=True,                # honor conditional spaces
        seed=42,
    )
    pruner = HyperbandPruner(
        min_resource=10,           # don't prune before epoch 10
        max_resource=args.max_epochs,
        reduction_factor=3,        # standard SHA / Hyperband
    )

    storage = RDBStorage(
    url=args.storage,
    engine_kwargs={"connect_args": {"timeout": 100}},  # wait up to 100s for a lock
)
    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        sampler=sampler,
        pruner=pruner,
        direction="maximize",      # maximize validation weighted accuracy
        load_if_exists=True,
    )

    log.info(
        "Joined study '%s'. Trials so far: %d completed, %d pruned, %d running.",
        args.study_name,
        sum(1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE),
        sum(1 for t in study.trials if t.state == optuna.trial.TrialState.PRUNED),
        sum(1 for t in study.trials if t.state == optuna.trial.TrialState.RUNNING),
    )

    deadline = time.time() + args.soft_deadline_seconds

    def near_deadline_or_full() -> bool:
        if time.time() > deadline:
            log.info("Soft deadline reached; not starting another trial.")
            return True
        n_done = sum(
            1 for t in study.trials
            if t.state in (
                optuna.trial.TrialState.COMPLETE,
                optuna.trial.TrialState.PRUNED,
                optuna.trial.TrialState.RUNNING,
            )
        )
        if n_done >= args.global_trial_cap:
            log.info("Global trial cap (%d) reached.", args.global_trial_cap)
            return True
        return False

    # Graceful shutdown on SIGTERM (SLURM sends this before SIGKILL on timeout)
    stopper = {"flag": False}
    def handle_term(signum, frame):
        log.warning("Received signal %d, will stop after current trial.", signum)
        stopper["flag"] = True
    signal.signal(signal.SIGTERM, handle_term)

    obj = partial(
        objective,
        data_dirs=args.data_dirs,
        num_workers=args.num_workers,
        max_epochs=args.max_epochs,
        log=log,
    )

    trials_run = 0
    while trials_run < args.n_trials and not stopper["flag"]:
        if near_deadline_or_full():
            break
        try:
            study.optimize(obj, n_trials=1, gc_after_trial=True,  catch=(RuntimeError,))
            trials_run += 1
        except KeyboardInterrupt:
            log.info("KeyboardInterrupt; exiting.")
            break

    log.info("Worker done. Ran %d trials.", trials_run)
    log.info("Best so far: %s", study.best_trial.params if study.best_trial else "none")


if __name__ == "__main__":
    main()
