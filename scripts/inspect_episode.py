"""Check recorded episodes: contract validation, rate, timing gaps, trigger label, upload state.

  .venv/bin/python scripts/inspect_episode.py data/recordings/<session>          # every episode in a session
  .venv/bin/python scripts/inspect_episode.py data/recordings/<session>/<episode>

A good episode: `valid`, ~30 Hz, max gap well under 100 ms, trigger = exactly 1 block of 10 steps.
"""

import json
import sys
from pathlib import Path

import numpy as np
from mcap.reader import make_reader
from mcap_protobuf.decoder import DecoderFactory

from autobot import schema as s
from autobot.episode import find_episodes, load_meta, validate


def arrays(ep_dir: Path, topic: str) -> tuple[np.ndarray, np.ndarray]:
    """(timestamps_ns, values) for one array topic in light.mcap."""
    times, values = [], []
    with open(ep_dir / "light.mcap", "rb") as f:
        for _, _, msg, proto in make_reader(f, decoder_factories=[DecoderFactory()]).iter_decoded_messages(topics=[topic]):
            times.append(msg.log_time)
            values.append(np.frombuffer(proto.data, dtype=np.float32))
    return np.array(times), np.array(values)


def inspect(ep_dir: Path) -> bool:
    meta = load_meta(ep_dir)
    try:
        validate(ep_dir)
        valid = "valid"
    except Exception as e:
        valid = f"INVALID ({type(e).__name__}: {e})"
    t, trig = arrays(ep_dir, s.ACT_TRIGGER)
    _, joints = arrays(ep_dir, s.OBS_JOINTS)
    dt = np.diff(t) / 1e9
    hz = (len(t) - 1) / max((t[-1] - t[0]) / 1e9, 1e-9)
    ones = np.flatnonzero(trig[:, 0] == 1)
    blocks = 0 if not ones.size else 1 + int((np.diff(ones) > 1).sum())
    uploaded = json.loads((ep_dir / "uploaded.json").read_text())["dataset_path"] if (ep_dir / "uploaded.json").exists() else "not uploaded"
    ok = valid == "valid" and hz >= 27 and blocks == 1 and ones.size == 10
    print(f"{'OK ' if ok else 'BAD'} {meta.episode_id}  {meta.task}  {meta.notes}\n"
          f"    {valid} · {len(t)} steps · {hz:.1f} Hz · max gap {dt.max() * 1e3:.0f} ms · "
          f"trigger {ones.size} steps in {blocks} block(s) · joints {joints.shape} · {uploaded}")
    return ok


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    root = Path(sys.argv[1])
    episodes = [root] if (root / "meta.json").exists() else find_episodes(root)
    results = [inspect(ep) for ep in episodes]
    print(f"\n{sum(results)}/{len(results)} episodes OK")


if __name__ == "__main__":
    main()
