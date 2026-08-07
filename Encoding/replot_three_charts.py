#!/usr/bin/env python
"""
Rebuild EXACTLY three W&B charts for the 256 run, covering BOTH training
segments (epochs 0..219) as one continuous history:

    1. Loss (train vs val)
    2. Dice (train vs val)
    3. clDice hard (train vs val)

Style matches your existing `_line_series` charts (solid training line, dashed
validation line, "training"/"validation" legend).

Why online: `wandb.plot.line_series` panels do NOT render after an offline
`wandb sync`. Logging them to a fresh ONLINE run (this script) makes them render.

Source of truth is train.log (it holds the full history for both runs);
metrics.jsonl only holds the resumed segment, so it's used just to refine
precision on epochs 100..219.

Run from a node WITH internet (your login node):

    python replot_three_charts.py \
        --log     /home/ids/gmargari-24/airway_project/Data/vae_runs/new_256_L32/train.log \
        --metrics /home/ids/gmargari-24/airway_project/Data/vae_runs/new_256_L32/metrics.jsonl \
        --project airway-bvae \
        --name    new_256_L32_full

Add --dry-run to check the parsed point counts without uploading.
"""
from __future__ import annotations
import argparse
import json
import math
import re
from pathlib import Path

TRAIN_RE = re.compile(
    r"Epoch (\d+)/\d+\s+lr=[\d.eE+-]+\s+β=[\d.eE+-]+\s+v_loss=([\d.eE+-]+)\s+"
    r"D_kl=[\d.eE+-]+\s+acc=[\d.eE+-]+\s+tp=[\d.eE+-]+\s+tn=[\d.eE+-]+\s+"
    r"clD_soft\(tr\)=[\w.eE+-]+\s+clD_hard\(tr\)=([\w.eE+-]+)\s+dice=([\d.eE+-]+)"
)
VAL_RE = re.compile(
    r"Val eval @ epoch (\d+) over \d+ samples: clD_hard=([\w.eE+-]+)\s+"
    r"clD_soft=[\w.eE+-]+\s+val_loss=([\d.eE+-]+)\s+val_kl=[\d.eE+-]+\s+val_dice=([\d.eE+-]+)"
)


def _num(s: str) -> float:
    try:
        return float(s)
    except ValueError:
        return float("nan")


def parse(log_path: Path, metrics_path: Path | None) -> dict[int, dict]:
    """{epoch: {vloss, dice, cldice_hard_train, val_vloss, val_dice, cldice_hard}}.
    Later train.log lines overwrite earlier ones (resume wins on overlap)."""
    rec: dict[int, dict] = {}
    with open(log_path) as f:
        for line in f:
            m = TRAIN_RE.search(line)
            if m:
                e = int(m.group(1))
                rec.setdefault(e, {}).update(
                    vloss=_num(m.group(2)),
                    cldice_hard_train=_num(m.group(3)),
                    dice=_num(m.group(4)),
                )
                continue
            v = VAL_RE.search(line)
            if v:
                e = int(v.group(1))
                rec.setdefault(e, {}).update(
                    cldice_hard=_num(v.group(2)),
                    val_vloss=_num(v.group(3)),
                    val_dice=_num(v.group(4)),
                )
    # Refine precision of the TRAIN loss/dice on the resumed segment from
    # metrics.jsonl. Do NOT pull val_* or cldice_hard from here: metrics.jsonl
    # carries the last eval values forward on every epoch, which would add stale
    # duplicate validation points and turn the val line into flat stair-steps.
    # Validation stays sourced only from genuine "Val eval @ epoch" lines.
    if metrics_path and metrics_path.is_file():
        with open(metrics_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    j = json.loads(line)
                except json.JSONDecodeError:
                    continue
                e = j.get("epoch")
                if e is None:
                    continue
                e = int(e)
                rec.setdefault(e, {})
                for key in ("vloss", "dice"):
                    if key in j and isinstance(j[key], (int, float)):
                        rec[e][key] = j[key]
    return rec


def series(rec: dict[int, dict], key: str) -> tuple[list[int], list[float]]:
    """Only real (finite) points for this key, in epoch order -> connected line."""
    xs, ys = [], []
    for e in sorted(rec):
        v = rec[e].get(key)
        if v is not None and math.isfinite(v):
            xs.append(e)
            ys.append(float(v))
    return xs, ys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", type=Path, required=True)
    ap.add_argument("--metrics", type=Path, default=None)
    ap.add_argument("--project", type=str, default="airway-bvae")
    ap.add_argument("--name", type=str, default=None)
    ap.add_argument("--entity", type=str, default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not args.log.is_file():
        raise SystemExit(f"log file not found: {args.log}")

    rec = parse(args.log, args.metrics)

    # Build the three (train, val) pairs.
    tr_loss_x, tr_loss_y = series(rec, "vloss")
    va_loss_x, va_loss_y = series(rec, "val_vloss")
    tr_dice_x, tr_dice_y = series(rec, "dice")
    va_dice_x, va_dice_y = series(rec, "val_dice")
    tr_cld_x,  tr_cld_y  = series(rec, "cldice_hard_train")
    va_cld_x,  va_cld_y  = series(rec, "cldice_hard")

    print("Point counts (train / val):")
    print(f"  Loss        : {len(tr_loss_x):3d} / {len(va_loss_x):3d}")
    print(f"  Dice        : {len(tr_dice_x):3d} / {len(va_dice_x):3d}")
    print(f"  clDice hard : {len(tr_cld_x):3d} / {len(va_cld_x):3d}")

    if args.dry_run:
        print("Dry run: not uploading.")
        return

    import wandb

    # Try to keep the workspace to ONLY the three charts (no System panels).
    try:
        settings = wandb.Settings(x_disable_stats=True)
    except TypeError:
        try:
            settings = wandb.Settings(_disable_stats=True)  # older wandb
        except TypeError:
            settings = None

    run_name = args.name or (args.log.parent.name + "_full")
    run = wandb.init(project=args.project, entity=args.entity, name=run_name,
                     settings=settings,
                     config={"note": "three continuous charts, both runs, from train.log"})

    def line_series(title, trx, try_, vax, vay):
        return wandb.plot.line_series(
            xs=[trx, vax], ys=[try_, vay],
            keys=["training", "validation"], title=title, xname="epoch")

    run.log({
        "loss_combined":   line_series("Loss (train vs val)",       tr_loss_x, tr_loss_y, va_loss_x, va_loss_y),
        "dice_combined":   line_series("Dice (train vs val)",       tr_dice_x, tr_dice_y, va_dice_x, va_dice_y),
        "cldice_combined": line_series("clDice hard (train vs val)", tr_cld_x,  tr_cld_y,  va_cld_x,  va_cld_y),
    })
    run.finish()
    print(f"Done: 3 charts uploaded to run '{run_name}'.")


if __name__ == "__main__":
    main()