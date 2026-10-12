"""Hit a hosted π0.7 checkpoint with synthetic frames: checks the connection and output format.

  .venv/bin/python scripts/infer_test.py test-v2-pipeline            # by fine-tune name
  .venv/bin/python scripts/infer_test.py --gateway-url wss://…         # or a URL directly
  .venv/bin/python scripts/infer_test.py full-v1 --checkpoint-step 2000

Also checks: same input twice (zero noise) → identical plan? random-noise requests → different plans?

Needs a checkpoint with a gateway_url (PI remote hosting). No robot needed.
"""

import argparse
import time

import numpy as np

from autobot import schema as s
from autobot.hardware import FakeRig
from autobot.policy import PiPolicy, gateway_url_for


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("finetune", nargs="?")
    p.add_argument("--gateway-url")
    p.add_argument("--checkpoint-step", type=int, help="default: the run's final checkpoint")
    p.add_argument("--cell", default="B2", choices=s.CELLS)
    p.add_argument("--calls", type=int, default=5)
    a = p.parse_args()
    url = a.gateway_url or gateway_url_for(a.finetune, a.checkpoint_step)

    t0 = time.perf_counter()
    print(f"connecting to {url} (first connect can take minutes)…")
    policy = PiPolicy(url)
    meta = policy.client.server_metadata
    print(f"connected in {time.perf_counter() - t0:.1f} s\nserver metadata: {meta}\ncamera map: {policy.cam_map}")

    images, joints = FakeRig(leader_port=None).observe()
    times = []
    for i in range(a.calls):
        t = time.perf_counter()
        act_joints, act_trigger = policy.infer(images, joints, s.task_for(a.cell))
        times.append(time.perf_counter() - t)
        if i == 0:
            print(f"joints  {act_joints.shape} {act_joints.dtype} range [{act_joints.min():.2f}, {act_joints.max():.2f}]")
            print(f"trigger {act_trigger.shape} {act_trigger.dtype} range [{act_trigger.min():.2f}, {act_trigger.max():.2f}]")
            print("first rows (L1–L5, trigger):")
            for row, trig in zip(act_joints[:3], act_trigger[:3, 0]):
                print("  ", np.round(row, 2), round(float(trig), 3))
    print(f"latency: first {times[0] * 1e3:.0f} ms, then median {np.median(times[1:]) * 1e3:.0f} ms" if len(times) > 1 else "")
    task = s.task_for(a.cell)
    z1, z2 = policy.infer(images, joints, task), policy.infer(images, joints, task)
    r1, r2 = policy.infer(images, joints, task, explore=True), policy.infer(images, joints, task, explore=True)
    diff = lambda x, y: float(np.abs(x[0] - y[0]).max())
    print(f"zero noise twice: joints differ by {diff(z1, z2):.3f}°  |  random noise twice: {diff(r1, r2):.3f}°  "
          f"|  zero vs random: {diff(z1, r1):.3f}°")
    print("trigger max (zero, zero, random, random):", [round(float(x[1].max()), 3) for x in (z1, z2, r1, r2)])
    policy.close()


if __name__ == "__main__":
    main()
