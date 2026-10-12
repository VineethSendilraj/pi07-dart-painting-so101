"""Record teleop demos: leader arm aims, SHOOT button fires. Writes PI-format episodes locally.

  .venv/bin/python scripts/record.py --follower-port /dev/tty.usbmodemA --leader-port /dev/tty.usbmodemB \\
      --rigd-url http://localhost:8080 --front-cam 0 --wrist-cam 1 --per-cell 10
  .venv/bin/python scripts/record.py --fake --auto --per-cell 1     # no hardware, for testing

Each episode: reset the arm with the leader → SPACE → aim → press SHOOT → the episode ends
AFTER_SHOT_S later and is saved to data/recordings/<session>/<episode_id>/.
Keys while recording: e = end + keep, x = discard + redo, q = quit after this episode.
Upload later with scripts/upload.py.
"""

import argparse
import getpass
import random
import time
from datetime import datetime
from pathlib import Path

from autobot import schema as s
from autobot.episode import Step, validate, write_episode
from autobot.hardware import ESP32, FakeESP32, FakeRig, Rig, RigdESP32
from autobot.keys import keyboard

ROOT = Path(__file__).resolve().parents[1]
TRIGGER_HOLD_STEPS = 10  # label each SHOOT press as trigger=1 for ~0.33 s
AFTER_SHOT_S = 0.7  # keep recording this long after the shot, then end the episode
MAX_EPISODE_S = 15.0
MIN_HZ, MAX_GAP_MS = 27.0, 250  # same save rules as record_web.py (contract: 30 Hz ±10%)


def teleop_tick(rig) -> tuple:
    """Leader → follower, then read the follower. Returns (action_joints, images, joints)."""
    act_joints = rig.leader_action()
    rig.command(act_joints)
    images, joints = rig.observe()
    return act_joints, images, joints


def record_episode(rig, esp, key) -> tuple[list[Step] | None, str]:
    """Run one episode. Returns (steps, reason); steps is None when discarded."""
    steps: list[Step] = []
    hold, shot_step = 0, None
    period = 1 / s.FPS
    next_tick = time.perf_counter()
    while True:
        k = key()
        if k == "x":
            return None, "discarded"
        if k == "e":
            return (steps, "ended") if shot_step is not None else (None, "no shot yet, discarded")
        act_joints, images, joints = teleop_tick(rig)
        for ev in esp.events():
            if ev.startswith("SHOOT") and shot_step is None:  # one shot per episode: ignore double-clicks
                hold = TRIGGER_HOLD_STEPS
                shot_step = len(steps)
        steps.append(Step(time.time_ns(), images, joints, act_joints, 1.0 if hold > 0 else 0.0))
        hold = max(hold - 1, 0)
        if shot_step is not None and len(steps) - shot_step >= AFTER_SHOT_S * s.FPS:
            return steps, "shot"
        if len(steps) >= MAX_EPISODE_S * s.FPS:
            return None, f"no shot within {MAX_EPISODE_S:.0f} s, discarded"
        next_tick += period
        time.sleep(max(0.0, next_tick - time.perf_counter()))


def wait_to_start(rig, esp, key, auto: bool) -> str:
    """Keep teleop running (so the operator can reset the arm) until SPACE, s (skip) or q (quit)."""
    if auto:
        return " "
    while True:
        k = key()
        if k in (" ", "s", "q"):
            return k
        teleop_tick(rig)
        esp.events()  # drop button presses between episodes
        time.sleep(1 / s.FPS)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--session", default=datetime.now().strftime("s%Y%m%d-%H%M"))
    p.add_argument("--cells", nargs="+", default=list(s.CELLS), help="cells to record (default: all 25)")
    p.add_argument("--per-cell", type=int, default=10)
    p.add_argument("--operator", default=getpass.getuser())
    p.add_argument("--follower-port"), p.add_argument("--leader-port"), p.add_argument("--esp32-port")
    p.add_argument("--rigd-url", help="act-athon rigd instead of --esp32-port (button release = SHOOT)")
    p.add_argument("--follower-id", default="follower_arm"), p.add_argument("--leader-id", default="leader_arm")
    p.add_argument("--front-cam", default="0"), p.add_argument("--wrist-cam", default="1")
    p.add_argument("--fake", action="store_true", help="fake arm, cameras and ESP32")
    p.add_argument("--auto", action="store_true", help="start episodes without waiting for SPACE")
    p.add_argument("--fake-shot-every", type=float, default=2.5, help="--fake: seconds between fake SHOOT presses")
    a = p.parse_args()
    for cell in a.cells:
        s.task_for(cell)  # validates

    if a.fake:
        rig, esp = FakeRig(), FakeESP32(shoot_every=a.fake_shot_every)
    else:
        if not (a.follower_port and a.leader_port and (a.esp32_port or a.rigd_url)):
            p.error("--follower-port, --leader-port and --esp32-port or --rigd-url are required (or use --fake)")
        cam = lambda v: int(v) if v.isdigit() else v
        rig = Rig(a.follower_port, {"front_camera": cam(a.front_cam), "wrist_camera": cam(a.wrist_cam)},
                  leader_port=a.leader_port, follower_id=a.follower_id, leader_id=a.leader_id)
        esp = RigdESP32(a.rigd_url, mode="train") if a.rigd_url else ESP32(a.esp32_port)
    rig.connect()

    schedule = [c for c in a.cells for _ in range(a.per_cell)]
    random.shuffle(schedule)  # interleave cells so the model can't key on recording order
    session_dir = ROOT / "data" / "recordings" / a.session
    # next index = one past the highest existing one, so deleting a folder never reuses an ID
    taken = [int(p.name.rsplit("-", 1)[-1]) for p in session_dir.glob(f"{a.session}-*") if p.name.rsplit("-", 1)[-1].isdigit()] if session_dir.exists() else []
    n_done = max(taken, default=-1) + 1
    print(f"session {a.session}: {len(schedule)} episodes → {session_dir}")

    try:
        with keyboard() as key:
            i = 0
            while i < len(schedule):
                cell = schedule[i]
                start = "the rest pose" if random.random() < 0.5 else random.choice([c for c in s.CELLS if c != cell])
                print(f"\n[{i + 1}/{len(schedule)}] next: shoot {cell}. First point at {start}; SPACE = start, s = skip, q = quit")
                k = wait_to_start(rig, esp, key, a.auto)
                if k == "q":
                    break
                if k == "s":
                    i += 1
                    continue
                print(f"  recording shoot {cell}… (press SHOOT; e = end, x = redo)")
                steps, reason = record_episode(rig, esp, key)
                if steps is None:
                    print(f"  {reason}; redo {cell}")
                    continue
                hz = (len(steps) - 1) / max((steps[-1].time_ns - steps[0].time_ns) / 1e9, 1e-6)
                gap_ms = max((b.time_ns - q.time_ns) / 1e6 for q, b in zip(steps, steps[1:]))
                if hz < MIN_HZ or gap_ms > MAX_GAP_MS:
                    print(f"  NOT saved: {hz:.1f} Hz, max gap {gap_ms:.0f} ms (need ≥{MIN_HZ:.0f} Hz, ≤{MAX_GAP_MS} ms); redo {cell}")
                    continue
                episode_id = f"{a.session}-{n_done:04d}"
                ep_dir = session_dir / episode_id
                write_episode(ep_dir, steps, episode_id=episode_id, cell=cell, operator=a.operator,
                              notes=f"start={start}", dataset_path="test/record-fake" if a.fake else "", fake=a.fake)
                validate(ep_dir)
                n_done += 1
                print(f"  saved {episode_id}: {len(steps)} steps, {hz:.1f} Hz, max gap {gap_ms:.0f} ms ({reason})")
                i += 1
    finally:
        rig.disconnect()
        esp.close()
    print(f"\ndone: {n_done} episodes in {session_dir}. Upload: .venv/bin/python scripts/upload.py {session_dir}")


if __name__ == "__main__":
    main()
