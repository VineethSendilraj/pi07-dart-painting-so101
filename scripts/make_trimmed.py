"""Make trimmed copies of recorded takes (cut shortly after the trigger pulse) and upload them as a new dataset.

Why: every final-vector take kept recording 2 s after the shot. On target, "about to fire" and "already fired"
(dry) look identical, and ~2/3 of those frames say "don't fire", so the model can aim and never fire. PI's span
filter can't drop the tail (it only keeps episodes whose human span covers the whole take), so the fix is new,
shorter episodes. Recording times, frames, joints and labels are copied unchanged; only the tail is dropped.

  .venv/bin/python scripts/make_trimmed.py data/recordings/final-vector                 # write + report, no upload
  .venv/bin/python scripts/make_trimmed.py data/recordings/final-vector --upload        # + upload to so101-blaster/v2-trim

Output: data/recordings/<session>-trim/trim-<episode>/ (new episode IDs), plus gun_marks.json: the colour and
position of the plus burned into each take's gun image (red, cyan or none). The median red position is the value
to use for robot_server --crosshair / calibration/gun_mark.json.
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

import numpy as np
from mcap.reader import make_reader
from mcap_protobuf.decoder import DecoderFactory

sys.path.insert(0, str(Path(__file__).resolve().parent))
from autobot import schema as s
from autobot.episode import Step, find_episodes, load_meta, validate, write_episode
from inspect_episode import arrays

KEEP_AFTER_PULSE = 5  # frames kept after the 10-frame trigger pulse (so the pulse is never cut)
DATASET_PATH = "so101-blaster/v2-trim"
MARK_COLOURS = {"red": (248, 113, 113), "cyan": (34, 211, 238)}


def decode_camera(ep_dir: Path, cam: str) -> list[np.ndarray]:
    """All frames of one camera shard, in step order, as RGB arrays."""
    import av

    topic = s.CAMERA_TOPICS[cam]
    shard = next(p for p in ep_dir.glob("*.mcap") if cam in p.name)
    codec, frames = None, []
    with open(shard, "rb") as f:
        for _, _, _, msg in make_reader(f, decoder_factories=[DecoderFactory()]).iter_decoded_messages(topics=[topic]):
            if codec is None:
                codec = av.CodecContext.create("h264", "r")
                if msg.extradata:
                    codec.extradata = bytes(msg.extradata)
            for data in msg.packets:
                for frame in codec.decode(av.Packet(bytes(data))):
                    frames.append(frame.to_ndarray(format="rgb24"))
    if codec is not None:
        for frame in codec.decode(None):
            frames.append(frame.to_ndarray(format="rgb24"))
    return frames


def find_mark(img: np.ndarray, window=(0.25, 0.7, 0.5, 0.98)) -> dict:
    """Colour and centre (fractions) of the thin plus the recorder burned into the gun image, if any.
    H.264's half-resolution colour washes the 2-px lines out (red arms decode as ~(200,132,132)), so classify by
    dominant channel, only inside `window` (x0, x1, y0, y1 fractions) around the expected spot (~0.46, 0.76)."""
    h, w = img.shape[:2]
    x0, x1, y0, y1 = int(window[0] * w), int(window[1] * w), int(window[2] * h), int(window[3] * h)
    sub = img[y0:y1, x0:x1].astype(int)
    r, g, b = sub[..., 0], sub[..., 1], sub[..., 2]
    masks = {"red": (r > 150) & (r - np.maximum(g, b) > 45), "cyan": (g > 140) & (b > 140) & (np.minimum(g, b) - r > 60)}
    best = {"colour": "none", "pixels": 0}
    for name, mask in masks.items():
        n = int(mask.sum())
        if n < 30:
            continue
        rows, cols = mask.sum(axis=1), mask.sum(axis=0)
        cy, cx = int(rows.argmax()), int(cols.argmax())  # the plus's horizontal arm row and vertical arm column
        if rows[cy] >= 12 and cols[cx] >= 12 and n > best["pixels"]:
            best = {"colour": name, "pixels": n, "x": round((x0 + cx) / (w - 1), 4), "y": round((y0 + cy) / (h - 1), 4)}
    return best


def trim_one(ep_dir: Path, out_root: Path, dataset_path: str) -> dict:
    meta = load_meta(ep_dir)
    t, obs = arrays(ep_dir, s.OBS_JOINTS)
    _, act = arrays(ep_dir, s.ACT_JOINTS)
    _, trig = arrays(ep_dir, s.ACT_TRIGGER)
    ones = np.flatnonzero(trig[:, 0] > 0.5)
    if not ones.size:
        return {"episode": meta.episode_id, "skipped": "no trigger"}
    shot = int(ones[0])
    keep = min(len(t), int(ones[-1]) + 1 + KEEP_AFTER_PULSE)
    frames = {cam: decode_camera(ep_dir, cam) for cam in s.CAMERAS}
    if any(len(f) != len(t) for f in frames.values()):
        return {"episode": meta.episode_id, "skipped": f"frame count {[len(f) for f in frames.values()]} != steps {len(t)}"}
    mark = find_mark(frames["wrist_camera"][shot])
    steps = [Step(int(t[i]), {cam: frames[cam][i] for cam in s.CAMERAS}, obs[i], act[i], float(trig[i, 0])) for i in range(keep)]
    new_id = f"trim-{meta.episode_id}"
    out = out_root / new_id
    write_episode(out, steps, episode_id=new_id, cell=meta.cell, operator=meta.operator,
                  notes=f"{meta.notes} trimmed_from={meta.episode_id} steps={len(t)}->{keep} mark={mark['colour']}".strip(),
                  dataset_path=dataset_path, fake=meta.fake)
    validate(out)
    report = {"episode": meta.episode_id, "new": new_id, "steps": len(t), "kept": keep, "shot": shot,
              "dropped_s": round((len(t) - keep) / s.FPS, 2), "mark": mark}
    (out / "trim.json").write_text(json.dumps(report))  # reruns reuse it, so the summary stays complete
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("session", type=Path, help="e.g. data/recordings/final-vector")
    p.add_argument("--dataset-path", default=DATASET_PATH)
    p.add_argument("--upload", action="store_true", help="upload the trimmed episodes (idempotent; safe to rerun)")
    a = p.parse_args()
    out_root = a.session.parent / f"{a.session.name}-trim"
    report = []
    for ep in find_episodes(a.session):
        done = out_root / f"trim-{load_meta(ep).episode_id}"
        if (done / "trim.json").exists():
            if load_meta(done).dataset_path != a.dataset_path:
                raise SystemExit(f"{done} was made for {load_meta(done).dataset_path!r}, not {a.dataset_path!r}: delete {out_root} first")
            report.append(json.loads((done / "trim.json").read_text()))
            continue
        r = trim_one(ep, out_root, a.dataset_path)
        report.append(r)
        print(r, flush=True)
    marks = [r["mark"] for r in report if "mark" in r]
    red = [m for m in marks if m["colour"] == "red"]
    summary = {
        "episodes": len(report), "trimmed": sum("kept" in r for r in report), "skipped": [r for r in report if "skipped" in r],
        "mark_colours": {c: sum(m["colour"] == c for m in marks) for c in ("red", "cyan", "none")},
        "red_mark_median": {"x": statistics.median(m["x"] for m in red), "y": statistics.median(m["y"] for m in red)} if red else None,
        "not_red": [r["episode"] for r in report if "mark" in r and r["mark"]["colour"] != "red"],
    }
    (out_root / "gun_marks.json").write_text(json.dumps({"summary": summary, "episodes": report}, indent=1))
    print(json.dumps(summary, indent=1))
    if a.upload:
        from upload import upload_one

        data, _ = s.clients()
        contract_id = s.ensure_contract(data)
        for ep in find_episodes(out_root):
            if not (ep / "uploaded.json").exists():
                print("uploaded", upload_one(data, contract_id, ep, a.dataset_path), ep.name, flush=True)


if __name__ == "__main__":
    main()
