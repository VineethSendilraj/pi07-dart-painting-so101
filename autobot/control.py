"""The shoot(cell) control loop plus the two safety rules between π0.7 and the hardware.

Speed limit: each joint moves at most MAX_STEP_DEG per 30 Hz step (~90°/s), so a bad model
output becomes slow, correctable motion instead of a lurch.

Fire rule: fire only when the arm has stopped moving (measured) and isn't about to move (the planned
next step is small), this turn hasn't fired yet, and the model asks for it: the trigger has been
> 0.5 for FIRE_CONSECUTIVE executed steps in a row, OR the rest of the current 50-step plan holds
this aim (within HOLD_DEG) and contains such a run (look-ahead: the recorded takes hold still ~0.5 s
before the shot, so the pulse is often planned past the 25 steps we execute before re-asking).

Stuck on target: hosted inference is deterministic (zero noise), so re-asking from the same still
scene returns the same plan. After STILL_EXPLORE_STEPS settled steps without a shot, every other request
uses random noise (a fresh sample of what the model would do here); that sample's motion is ignored (it
scatters the aim by several degrees) and the arm holds its aim while the sampled trigger is checked.

Settle fallback (off unless settle_fire_s is given): if the model has held the arm still on its own chosen aim
(settled, and its latest plain plan ends there) for settle_fire_s without firing, fire anyway. The aim stays
the model's; only the trigger timing becomes a rule. Logged as "settle fallback".
"""

import time
from collections import deque
from dataclasses import dataclass

import numpy as np

from autobot import schema as s

MAX_STEP_DEG = 3.0
FIRE_CONSECUTIVE = 3
FIRE_THRESHOLD = 0.5
HOLD_DEG = 1.0  # look-ahead fire: the rest of the plan must stay this close to the current aim
STILL_EXPLORE_STEPS = 10  # settled steps without a shot before the next request samples random noise
SETTLED_DEG = 0.5  # max joint change per step to count as "stopped"
MAX_SHOT_STEPS = 1800  # executed steps (60 s of motion at 30 Hz) before giving up; inference pauses don't count
MAX_SHOT_WALL_S = 150.0  # hard wall-clock cap, above the step budget plus inference time
FIRED_WAIT_S = 1.0  # how long to wait for FIRED after requesting a shot
HOLD_AFTER_FIRE_S = 1.0  # keep the aim while the 2-pulse burst pushes the dart (~0.6 s) before moving


def speed_limit(target: np.ndarray, current: np.ndarray, max_step: float | None = None) -> np.ndarray:
    step = MAX_STEP_DEG if max_step is None else max_step  # read at call time so --max-step-deg can change it
    return current + np.clip(target - current, -step, step)


class FireRule:
    def __init__(self):
        self.streak = 0
        self.fired = False
        self.prev_joints = None
        self.settled = False

    def reset_motion(self) -> None:
        """Call after a pause (e.g. waiting for inference): 'stopped' must be re-measured."""
        self.prev_joints = None

    def update(self, trigger: float, joints: np.ndarray, planned_step: float = 0.0, ahead: bool = False) -> bool:
        """planned_step: largest joint change the next command asks for (degrees).
        ahead: the rest of the model's current plan holds this aim and contains a trigger run (see pulse_ahead)."""
        self.settled = (self.prev_joints is not None
                        and float(np.abs(joints - self.prev_joints).max()) <= SETTLED_DEG
                        and planned_step <= SETTLED_DEG)
        self.prev_joints = joints
        self.streak = self.streak + 1 if trigger > FIRE_THRESHOLD else 0
        if not self.fired and self.settled and (self.streak >= FIRE_CONSECUTIVE or ahead):
            self.fired = True
            return True
        return False


def pulse_ahead(trigger: np.ndarray, plan: np.ndarray, cmd: np.ndarray) -> bool:
    """True if `plan` (the rest of the predicted joint targets) stays within HOLD_DEG of `cmd` and `trigger`
    (the rest of the predicted trigger) has FIRE_CONSECUTIVE steps in a row above FIRE_THRESHOLD."""
    if len(plan) and float(np.abs(np.asarray(plan) - cmd).max()) > HOLD_DEG:
        return False
    run = 0
    for v in np.asarray(trigger).reshape(-1):
        run = run + 1 if v > FIRE_THRESHOLD else 0
        if run >= FIRE_CONSECUTIVE:
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
    """Sleep to the next 30 Hz tick. After a stall (rigd hiccup, slow read), drop the lost time instead of
    running the following steps back-to-back, which would play the plan faster than the speed limit intends."""
    next_tick = max(next_tick + period, time.perf_counter())
    time.sleep(max(0.0, next_tick - time.perf_counter()))
    return next_tick


def _observe(rig, log, tries: int = 5):
    """rig.observe() with a few retries (USB camera / servo bus hiccup). None if it keeps failing."""
    for attempt in range(tries):
        try:
            return rig.observe()
        except Exception as e:
            log(f"  camera/arm read failed ({type(e).__name__}: {e}); retry {attempt + 1}/{tries}")
            time.sleep(0.1)
    return None


def _events(esp, log) -> list[str]:
    try:
        return esp.events()
    except OSError as e:  # rigd hiccup: keep the arm under control, report it
        log(f"  button/ESP32 read failed: {e}")
        return []


def hold(rig, cmd: np.ndarray, seconds: float) -> None:
    """Keep commanding the same pose (the aim) for a while."""
    period, next_tick, t0 = 1 / s.FPS, time.perf_counter(), time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        rig.command(cmd)
        next_tick = _tick(period, next_tick)


def _operator(halt) -> str | None:
    """'stop', 'pause', or None. halt() may return those words or True for stop."""
    if halt is None:
        return None
    got = halt()
    if got is True or got == "stop":
        return "stop"
    if got == "pause":
        return "pause"
    return None


def _hold_until(rig, cmd, halt, on_obs=None) -> str | None:
    """While the operator is paused, keep the current pose. Returns 'stop' if they hit stop."""
    period, next_tick = 1 / s.FPS, time.perf_counter()
    while True:
        op = _operator(halt)
        if op == "stop":
            return "stop"
        if op != "pause":
            return None
        rig.command(cmd)
        obs = _observe(rig, lambda *_: None)
        if obs is not None and on_obs is not None:
            on_obs(obs[0])
        next_tick = _tick(period, next_tick)


def shoot(rig, esp, policy, cell: str, chunk: int = 25, log=print, settle_fire_s: float | None = None, halt=None, on_obs=None,
          max_steps: int = MAX_SHOT_STEPS, max_wall_s: float = MAX_SHOT_WALL_S) -> ShotResult:
    """Run π0.7 on "shoot <cell>" until the rig confirms FIRED (or timeout), then hold the aim briefly."""
    task = s.task_for(cell)
    rule, queue = FireRule(), deque()
    period, t0 = 1 / s.FPS, time.perf_counter()
    next_tick, steps, fired_at, calls, still, infer_errors, on_aim = t0, 0, None, 0, 0, 0, 0
    plan_joints = plan_trigger = last_plain_end = None
    result = lambda fired, reason: ShotResult(cell, fired, reason, steps, time.perf_counter() - t0)
    obs = _observe(rig, log)
    if obs is None:
        return result(False, "camera/arm read failed before the shot")
    images, joints = obs
    cmd = joints.copy()
    _events(esp, log)  # drop stale button presses
    while True:
        if _hold_until(rig, cmd, halt, on_obs) == "stop" or _operator(halt) == "stop":
            return result(False, "stopped")
        esp.heartbeat()
        if not queue:
            # Explore only when stuck ON TARGET: settled for a while AND the last plain plan ends where the arm is
            # (not during a slow move); alternate so plain asks can still correct the aim.
            lag = 0.0 if plan_joints is None else float(np.abs(plan_joints[min(chunk, len(plan_joints)) - 1] - cmd).max())
            explore = (still >= STILL_EXPLORE_STEPS and calls % 2 == 1 and last_plain_end is not None
                       and float(np.abs(last_plain_end - cmd).max()) <= HOLD_DEG)
            try:
                plan_joints, plan_trigger = policy.infer(images, joints, task, explore=explore)  # arm holds meanwhile
                infer_errors = 0
            except Exception as e:  # hosted inference hiccup: keep the aim, retry a couple of times, then give up the shot
                infer_errors += 1
                log(f"  inference failed ({type(e).__name__}: {e}); attempt {infer_errors}/3")
                if infer_errors >= 3 or time.perf_counter() - t0 > max_wall_s:
                    return result(False, f"inference error: {type(e).__name__}: {e}")
                if _operator(halt) == "stop":
                    return result(False, "stopped")
                hold(rig, cmd, 1.0)
                next_tick = time.perf_counter()
                continue
            if explore:  # random noise also scatters the aim by several degrees: keep the aim, use only the trigger
                plan_joints = np.repeat(cmd[None], len(plan_joints), axis=0)
            else:
                last_plain_end = np.array(plan_joints[-1], dtype=np.float32)
            calls += 1
            trig = plan_trigger[:, 0]
            log(f"  call {calls}: trigger max {trig.max():.2f} at step {int(trig.argmax())}/{len(trig)}, "
                f"settled {still} steps, arm behind last plan {lag:.1f}°, noise {'random' if explore else 'zero'}")
            queue.extend(range(min(chunk, len(plan_joints))))
            rule.reset_motion()
            next_tick = time.perf_counter()
        i = queue.popleft()
        target, trigger = plan_joints[i], float(plan_trigger[i, 0])
        new_cmd = speed_limit(target, cmd)
        planned = float(np.abs(new_cmd - cmd).max())
        cmd = new_cmd
        rig.command(cmd)
        obs = _observe(rig, log)
        if obs is None:
            return result(False, "camera/arm read failed")
        images, joints = obs
        steps += 1
        ahead = pulse_ahead(plan_trigger[i:, 0], plan_joints[i + 1:], cmd)
        fire = fired_at is None and rule.update(trigger, joints, planned, ahead)
        still = still + 1 if rule.settled else 0
        holds_aim = last_plain_end is not None and float(np.abs(last_plain_end - cmd).max()) <= HOLD_DEG
        on_aim = on_aim + 1 if (rule.settled or on_aim) and holds_aim and planned <= SETTLED_DEG else 0
        fallback = (not fire and fired_at is None and settle_fire_s is not None and not rule.fired
                    and on_aim >= settle_fire_s * s.FPS)
        if fallback:
            rule.fired = True
            fire = True
        if _operator(halt) == "stop":
            return result(False, "stopped")
        if fire:
            try:
                esp.fire()
            except OSError as e:
                return result(False, f"fire request failed ({e}): dart state unknown, check the board")
            fired_at = time.perf_counter()
            how = "settle fallback" if fallback else ("look-ahead" if rule.streak < FIRE_CONSECUTIVE else "trigger run")
            log(f"  fire at step {steps} ({how})")
        for ev in _events(esp, log):
            if ev == "FIRED":
                hold(rig, cmd, HOLD_AFTER_FIRE_S)
                return result(True, "fired")
            if ev in ("NO_HEARTBEAT", "COOLDOWN", "BLOCKED", "DISARMED", "MODE_RESET"):
                log(f"  rig: {ev}")
        if fired_at is not None and time.perf_counter() - fired_at > FIRED_WAIT_S:
            return result(False, "requested a shot but the rig never confirmed FIRED")
        if steps >= max_steps or time.perf_counter() - t0 > max_wall_s:
            return result(False, "timeout: model never fired")
        next_tick = _tick(period, next_tick)


def move_to(rig, esp, pose: np.ndarray, max_seconds: float = 4.0) -> None:
    """Scripted, speed-limited move (e.g. back to the rest pose between turns)."""
    period, next_tick, t0 = 1 / s.FPS, time.perf_counter(), time.perf_counter()
    obs = _observe(rig, print)
    if obs is None:
        return  # can't read the arm: leave it where it is
    cmd = obs[1].copy()
    while float(np.abs(pose - cmd).max()) > 0.2 and time.perf_counter() - t0 < max_seconds:
        esp.heartbeat()
        cmd = speed_limit(pose, cmd)
        rig.command(cmd)
        next_tick = _tick(period, next_tick)
