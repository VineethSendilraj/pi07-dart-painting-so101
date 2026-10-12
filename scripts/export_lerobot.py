"""Export recorded takes to a LeRobot v3.0 dataset (parquet + mp4), the format SO-101 datasets use on the Hugging Face Hub.

  .venv/bin/python scripts/export_lerobot.py data/recordings/final-vector --repo-id <hf-user>/pi07-dart-painting-so101
  .venv/bin/python scripts/export_lerobot.py ... --push        # also upload to the Hub with docs/dataset-card.md (needs `hf auth login`)

Per frame (30 Hz): observation.state = follower joints [5] (degrees), action = leader joints [5] (degrees) + trigger
(0, or 1 for the 10 steps after the fire button), observation.images.front / .wrist (640×480). Task = "shoot <cell>".
Per-episode extras (source episode ID, cell, board clutter, live dart, start pose) go to meta/blaster_episodes.jsonl.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import av
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from mcap.reader import make_reader
from mcap_protobuf.decoder import DecoderFactory

sys.path.insert(0, str(Path(__file__).resolve().parent))
from autobot import schema as s
from autobot.episode import find_episodes, load_meta
from inspect_episode import arrays

IMAGE_KEYS = {"front_camera": "observation.images.front", "wrist_camera": "observation.images.wrist"}
JOINTS = [f"{j}.pos" for j in s.JOINT_NAMES]


def features() -> dict:
    video = {"dtype": "video", "shape": (480, 640, 3), "names": ["height", "width", "channels"]}
    return {
        "observation.state": {"dtype": "float32", "shape": (5,), "names": JOINTS},
        "action": {"dtype": "float32", "shape": (6,), "names": [*JOINTS, "trigger"]},
        **{key: video for key in IMAGE_KEYS.values()},
    }


def frames(ep_dir: Path, cam: str) -> list[np.ndarray]:
    """Decode one camera shard (H.264, one frame per step) to RGB arrays in step order."""
    codec = av.CodecContext.create("h264", "r")
    out = []
    with open(ep_dir / f"{cam}.mcap", "rb") as f:
        for _, _, _, proto in make_reader(f, decoder_factories=[DecoderFactory()]).iter_decoded_messages():
            for packet in proto.packets:
                out += [fr.to_ndarray(format="rgb24") for fr in codec.decode(av.Packet(packet))]
    out += [fr.to_ndarray(format="rgb24") for fr in codec.decode(None)]
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("session", type=Path)
    p.add_argument("--repo-id", required=True)
    p.add_argument("--root", type=Path, help="output dir (default data/lerobot/<repo name>)")
    p.add_argument("--push", action="store_true")
    a = p.parse_args()

    root = a.root or Path("data/lerobot") / a.repo_id.split("/")[-1]
    if root.exists():
        shutil.rmtree(root)
    ds = LeRobotDataset.create(a.repo_id, s.FPS, features(), root=root, robot_type="so101_follower")
    schedule = {e["episode_id"]: e for e in json.loads((a.session / "schedule.json").read_text()) if e.get("episode_id")}
    extras = []

    for i, ep_dir in enumerate(find_episodes(a.session)):
        meta = load_meta(ep_dir)
        _, state = arrays(ep_dir, s.OBS_JOINTS)
        _, act = arrays(ep_dir, s.ACT_JOINTS)
        _, trig = arrays(ep_dir, s.ACT_TRIGGER)
        images = {cam: frames(ep_dir, cam) for cam in s.CAMERAS}
        n = meta.num_steps
        if not (len(state) == len(act) == len(trig) == n and all(len(v) == n for v in images.values())):
            sys.exit(f"{meta.episode_id}: length mismatch (steps {n}, state {len(state)}, "
                     f"cams {[len(v) for v in images.values()]})")
        for t in range(n):
            ds.add_frame({
                "observation.state": state[t],
                "action": np.concatenate([act[t], trig[t]]).astype(np.float32),
                **{IMAGE_KEYS[cam]: images[cam][t] for cam in s.CAMERAS},
                "task": meta.task,
            })
        ds.save_episode()
        sched = schedule.get(meta.episode_id, {})
        extras.append({"episode_index": i, "source_episode_id": meta.episode_id, "cell": meta.cell,
                       "live": sched.get("live"), "board": sched.get("board"), "start": sched.get("start")})
        print(f"{i + 1:3d} {meta.episode_id} {meta.task} {n} frames", flush=True)

    ds.finalize()
    (root / "meta/blaster_episodes.jsonl").write_text("".join(json.dumps(e) + "\n" for e in extras))
    print(f"wrote {ds.num_episodes} episodes, {ds.num_frames} frames to {root}")
    if a.push:
        ds.push_to_hub(tags=["so101", "pi0.7", "teleop"], license="cc-by-4.0",
                       dataset_description=Path("docs/dataset-card.md").read_text(),
                       citation_bibtex=Path("docs/citation.bib").read_text(),
                       url="https://github.com/VineethSendilraj/pi07-dart-painting-so101")


if __name__ == "__main__":
    main()
