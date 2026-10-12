"""Install this repo's SO-101 calibration into the local LeRobot cache.

  .venv/bin/python scripts/install_calibration.py

Writes:
  ~/.cache/huggingface/lerobot/calibration/robots/so_follower/follower_arm.json
  ~/.cache/huggingface/lerobot/calibration/teleoperators/so_leader/leader_arm.json

Then pass --follower-id follower_arm (and --leader-id leader_arm when recording).
These files are for THIS physical pair of arms. Recalibrate if you swap hardware.
"""

from __future__ import annotations

import shutil
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "calibration"
DST = Path.home() / ".cache/huggingface/lerobot/calibration"


def main() -> None:
    files = [
        Path("robots/so_follower/follower_arm.json"),
        Path("teleoperators/so_leader/leader_arm.json"),
    ]
    for rel in files:
        src = SRC / rel
        if not src.is_file():
            raise SystemExit(f"missing {src}")
        dest = DST / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        print(f"wrote {dest}")
    print("use --follower-id follower_arm  --leader-id leader_arm")


if __name__ == "__main__":
    main()
