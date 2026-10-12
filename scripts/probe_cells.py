"""Offline prompt-sensitivity check for a hosted checkpoint (no robot): does the planned aim change with the cell?

  .venv/bin/python scripts/probe_cells.py full-v1 --checkpoint-step 2999
  .venv/bin/python scripts/probe_cells.py --rank        # rank every probed checkpoint (runs/probe/*.json)

Sends the same real rig frames (rear: data/camera_check/cam0.jpg, gun: crop of docs/img/record-web-done.jpg with the
red mark drawn) and the same joint state with every "shoot <cell>" prompt, and reads where each plan ends (step 49).
A model that listens to the prompt spreads its final pan across columns A→E and its final wrist pitch across rows 1→5.
The frames are from the recording night, so absolute aims are only indicative; the spread and ordering are the signal.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from autobot import schema as s
from autobot.hardware import draw_gun_crosshair, load_gun_mark
from autobot.policy import PiPolicy, gateway_url_for

ROOT = Path(__file__).resolve().parents[1]
STATE = np.array([5.6, -110.6, 97.7, -0.4, -91.8], dtype=np.float32)  # a typical pose the models return to
CELLS = [c for c in s.CELLS if c not in ("A5", "E5")]


def frames() -> dict[str, np.ndarray]:
    rear = cv2.cvtColor(cv2.imread(str(ROOT / "data/camera_check/cam0.jpg")), cv2.COLOR_BGR2RGB)
    shot = cv2.cvtColor(cv2.imread(str(ROOT / "docs/img/record-web-done.jpg")), cv2.COLOR_BGR2RGB)
    gun = cv2.resize(shot[72:1075, 18:1505], (640, 480), interpolation=cv2.INTER_AREA)
    mark = load_gun_mark()
    return {"front_camera": cv2.resize(rear, (640, 480)), "wrist_camera": draw_gun_crosshair(gun, *mark) if mark else gun}


def rank_all() -> None:
    """Every probed checkpoint: cell ordering + mean distance of its per-cell aims (pan, pitch) from the median of
    all probed checkpoints. Adjacent cells are ~5° apart, so lower agreement distance = safer; ordering must be ~1.0."""
    probes = {f.stem: json.loads(f.read_text()) for f in sorted((ROOT / "runs/probe").glob("*.json"))}
    cells = sorted(next(iter(probes.values()))["cells"])
    ends = {k: np.array([[v["cells"][c]["end"][0], v["cells"][c]["end"][3]] for c in cells]) for k, v in probes.items()}
    med = np.median(np.stack(list(ends.values())), axis=0)
    rows = sorted((float(np.abs(e - med).mean()), k) for k, e in ends.items())
    print(f"{'checkpoint':28} {'agree':>7} {'pan order':>9} {'pitch order':>11} {'pan spread':>10} {'pitch spread':>12}")
    for dev, k in rows:
        sm = probes[k]["summary"]
        print(f"{k:28} {dev:6.2f}° {sm['pan_column_rank_corr']:9.1f} {sm['pitch_row_rank_corr']:11.1f} "
              f"{sm['pan_spread_deg']:9.1f}° {sm['pitch_spread_deg']:11.1f}°")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("finetune", nargs="?"), p.add_argument("--checkpoint-step", type=int)
    p.add_argument("--rank", action="store_true", help="rank all probed checkpoints instead of probing")
    a = p.parse_args()
    if a.rank:
        return rank_all()
    if not a.finetune:
        p.error("give a fine-tune name, or --rank")
    url = gateway_url_for(a.finetune, a.checkpoint_step)
    policy = PiPolicy(url)
    imgs = frames()
    ends = {}
    for cell in CELLS:
        joints, trig = policy.infer(imgs, STATE, s.task_for(cell))
        ends[cell] = {"end": [round(float(v), 2) for v in joints[-1]], "trigger_max": round(float(trig.max()), 3)}
    policy.close()
    pan = {c: ends[c]["end"][0] for c in CELLS}
    pitch = {c: ends[c]["end"][3] for c in CELLS}
    col_pan = {col: float(np.mean([pan[c] for c in CELLS if c[0] == col])) for col in s.COLS}
    row_pitch = {row: float(np.mean([pitch[c] for c in CELLS if c[1] == row])) for row in s.ROWS}
    rank = lambda xs: np.argsort(np.argsort(xs))
    corr = lambda xs: float(np.corrcoef(rank(np.arange(len(xs))), rank(np.array(xs)))[0, 1])
    summary = {
        "url": url,
        "pan_by_column": {k: round(v, 2) for k, v in col_pan.items()},
        "pitch_by_row": {k: round(v, 2) for k, v in row_pitch.items()},
        "pan_spread_deg": round(max(col_pan.values()) - min(col_pan.values()), 2),
        "pitch_spread_deg": round(max(row_pitch.values()) - min(row_pitch.values()), 2),
        "pan_column_rank_corr": round(abs(corr(list(col_pan.values()))), 2),
        "pitch_row_rank_corr": round(abs(corr(list(row_pitch.values()))), 2),
        "trigger_max_any": max(e["trigger_max"] for e in ends.values()),
    }
    out = ROOT / "runs/probe" / f"{a.finetune}-{url.rsplit('ckpt-', 1)[-1].split('-')[0]}.json"
    out.write_text(json.dumps({"summary": summary, "cells": ends}, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
