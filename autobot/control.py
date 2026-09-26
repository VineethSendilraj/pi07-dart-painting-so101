"""The shoot(cell) control loop plus the two safety rules between π0.7 and the hardware.

Speed limit: each joint moves at most MAX_STEP_DEG per 30 Hz step (~90°/s), so a bad model
output becomes slow, correctable motion instead of a lurch.

Fire rule: fire only when the trigger has been > 0.5 for FIRE_CONSECUTIVE steps in a row,
the arm has stopped moving, and this turn hasn't fired yet.
"""

import time
from collections import deque
from dataclasses import dataclass

import numpy as np

from autobot import schema as s

MAX_STEP_DEG = 3.0
FIRE_CONSECUTIVE = 3
SETTLED_DEG = 0.5  # max joint change per step to count as "stopped"
SHOOT_TIMEOUT_S = 10.0
FIRED_WAIT_S = 1.0  # how long to wait for the ESP32's FIRED after sending F


def speed_limit(target: np.ndarray, current: np.ndarray, max_step: float = MAX_STEP_DEG) -> np.ndarray:
    return current + np.clip(target - current, -max_step, max_step)


class FireRule:
    def __init__(self):
        self.streak = 0
        self.fired = False
        self.prev_joints = None

    def update(self, trigger: float, joints: np.ndarray) -> bool:
        settled = self.prev_joints is not None and float(np.abs(joints - self.prev_joints).max()) <= SETTLED_DEG
        self.prev_joints = joints
        self.streak = self.streak + 1 if trigger > 0.5 else 0
        if not self.fired and self.streak >= FIRE_CONSECUTIVE and settled:
            self.fired = True
            return True
        return False


@dataclass
class ShotResult:
    cell: str
    fired: bool
    reason: str
    steps: int
    seconds: float


def _tick(period: float, next_tick: float) -> float:
    next_tick += period
    time.sleep(max(0.0, next_tick - time.perf_counter()))
    return next_tick


def shoot(rig, esp, policy, cell: str, chunk: int = 10, log=print) -> ShotResult:
    """Run π0.7 on "shoot <cell>" until the ESP32 confirms FIRED (or timeout)."""
    task = s.task_for(cell)
    rule, queue = FireRule(), deque()
    period, t0 = 1 / s.FPS, time.perf_counter()
    next_tick, steps, fired_at = t0, 0, None
    images, joints = rig.observe()
    cmd = joints.copy()
    esp.events()  # drop stale button presses
    while True:
        esp.heartbeat()
        if not queue:
            act_joints, act_trigger = policy.infer(images, joints, task)  # the arm holds position meanwhile
            queue.extend(zip(act_joints[:chunk], act_trigger[:chunk, 0]))
            next_tick = time.perf_counter()
        target, trigger = queue.popleft()
        cmd = speed_limit(target, cmd)
        rig.command(cmd)
        images, joints = rig.observe()
        steps += 1
        if fired_at is None and rule.update(float(trigger), joints):
            esp.fire()
            fired_at = time.perf_counter()
            log(f"  fire at step {steps}")
        for ev in esp.events():
            if ev == "FIRED":
                return ShotResult(cell, True, "fired", steps, time.perf_counter() - t0)
            if ev in ("NO_HEARTBEAT", "COOLDOWN"):
                log(f"  ESP32: {ev}")
        if fired_at is not None and time.perf_counter() - fired_at > FIRED_WAIT_S:
            return ShotResult(cell, False, "sent F but no FIRED from ESP32", steps, time.perf_counter() - t0)
        if time.perf_counter() - t0 > SHOOT_TIMEOUT_S:
            return ShotResult(cell, False, "timeout: model never fired", steps, time.perf_counter() - t0)
        next_tick = _tick(period, next_tick)


def move_to(rig, esp, pose: np.ndarray, max_seconds: float = 4.0) -> None:
    """Scripted, speed-limited move (e.g. back to the rest pose between turns)."""
    period, next_tick, t0 = 1 / s.FPS, time.perf_counter(), time.perf_counter()
    _, joints = rig.observe()
    cmd = joints.copy()
    while float(np.abs(pose - cmd).max()) > 0.2 and time.perf_counter() - t0 < max_seconds:
        esp.heartbeat()
        cmd = speed_limit(pose, cmd)
        rig.command(cmd)
        next_tick = _tick(period, next_tick)
