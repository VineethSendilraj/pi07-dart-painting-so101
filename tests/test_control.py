"""Safety-rule tests. Run: .venv/bin/python -m pytest tests  (or: .venv/bin/python tests/test_control.py)"""

import numpy as np

from autobot.control import FIRE_CONSECUTIVE, HOLD_DEG, MAX_STEP_DEG, FireRule, pulse_ahead, speed_limit

STILL = np.zeros(5, dtype=np.float32)


def test_speed_limit_clamps_each_joint():
    out = speed_limit(np.array([40, -40, 1, 0, 0], dtype=np.float32), STILL)
    assert np.allclose(out, [MAX_STEP_DEG, -MAX_STEP_DEG, 1, 0, 0])


def test_single_flicker_does_not_fire():
    rule = FireRule()
    assert not any(rule.update(t, STILL) for t in [0, 1, 0, 1, 0, 0])


def test_fires_after_consecutive_steps_when_settled():
    rule = FireRule()
    fired = [rule.update(1.0, STILL) for _ in range(FIRE_CONSECUTIVE + 1)]
    assert fired.count(True) == 1


def test_no_fire_while_moving():
    rule = FireRule()
    assert not any(rule.update(1.0, np.full(5, i * 2.0, dtype=np.float32)) for i in range(10))


def test_no_fire_when_next_step_is_large():
    rule = FireRule()
    assert not any(rule.update(1.0, STILL, planned_step=3.0) for _ in range(10))


def test_no_fire_right_after_inference_pause():
    rule = FireRule()
    rule.update(1.0, STILL)
    rule.update(1.0, STILL)
    rule.reset_motion()  # paused for inference: 'stopped' must be re-measured
    assert not rule.update(1.0, STILL)
    assert rule.update(1.0, STILL)


def test_only_one_shot_per_turn():
    rule = FireRule()
    assert sum(rule.update(1.0, STILL) for _ in range(30)) == 1


def test_lookahead_fires_on_a_planned_pulse_when_settled():
    """The recorded takes hold still ~0.5 s before the shot, so the pulse is often planned past the executed steps."""
    plan = np.zeros((30, 5), dtype=np.float32)
    trigger = np.zeros(30, dtype=np.float32)
    trigger[20:30] = 1.0
    assert pulse_ahead(trigger, plan, STILL)
    rule = FireRule()
    rule.update(0.0, STILL)  # first step after a pause can't count as settled
    assert rule.update(0.0, STILL, ahead=True)


def test_lookahead_ignores_a_pulse_if_the_plan_moves_away():
    plan = np.zeros((30, 5), dtype=np.float32)
    plan[10:, 0] = HOLD_DEG + 2.0  # the model plans to re-aim before its pulse
    trigger = np.zeros(30, dtype=np.float32)
    trigger[20:30] = 1.0
    assert not pulse_ahead(trigger, plan, STILL)


def test_lookahead_needs_a_run_not_a_flicker():
    trigger = np.array([0, 1, 0, 1, 0, 1, 1, 0], dtype=np.float32)
    assert not pulse_ahead(trigger, np.zeros((8, 5), dtype=np.float32), STILL)


def test_lookahead_still_needs_the_arm_settled():
    rule = FireRule()
    assert not any(rule.update(0.0, np.full(5, i * 2.0, dtype=np.float32), ahead=True) for i in range(10))


class _SlowMovePolicy:
    """Plans a 5° pan move over 40 steps, holds, then pulses; random-noise samples put the pulse early."""
    goal = np.array([5.0, 0, 0, 0, 0], dtype=np.float32)

    def infer(self, images, joints, task, explore=False):
        steps = np.array([joints + (self.goal - joints) * min(1.0, (k + 1) / 40) for k in range(50)], dtype=np.float32)
        trig = np.zeros((50, 1), dtype=np.float32)
        if explore:
            trig[5:15] = 1.0  # a sample that would fire now (before the move is done)
        elif float(np.abs(self.goal - joints).max()) < 0.2:
            trig[20:30] = 1.0  # plain plan: fire once on target
        return steps, trig


def test_slow_move_fires_on_target_not_short():
    from autobot.control import shoot
    from autobot.hardware import FakeESP32, FakeRig
    rig = FakeRig(leader_port=None)
    r = shoot(rig, FakeESP32(shoot_every=None), _SlowMovePolicy(), "C3", log=lambda *_: None)
    assert r.fired
    assert abs(float(rig.cmd[0]) - 5.0) < 0.3, f"fired at pan {rig.cmd[0]:.2f}, target 5.0"


class _NeverFiresPolicy:
    """Aims at pan 5° and holds, never raises the trigger (the risk the settle fallback covers)."""
    goal = np.array([5.0, 0, 0, 0, 0], dtype=np.float32)

    def infer(self, images, joints, task, explore=False):
        steps = np.array([joints + (self.goal - joints) * min(1.0, (k + 1) / 20) for k in range(50)], dtype=np.float32)
        return steps, np.zeros((50, 1), dtype=np.float32)


def test_settle_fallback_fires_on_the_models_aim_only_when_enabled():
    from autobot.control import shoot
    from autobot.hardware import FakeESP32, FakeRig
    rig = FakeRig(leader_port=None)
    assert not shoot(rig, FakeESP32(shoot_every=None), _NeverFiresPolicy(), "C3", log=lambda *_: None, max_steps=40).fired
    rig = FakeRig(leader_port=None)
    r = shoot(rig, FakeESP32(shoot_every=None), _NeverFiresPolicy(), "C3", log=lambda *_: None, settle_fire_s=1.0)
    assert r.fired and abs(float(rig.cmd[0]) - 5.0) < 0.2


def test_stop_aborts_before_the_arm_moves():
    from autobot.control import shoot
    from autobot.hardware import FakeESP32, FakeRig

    class _Lurch:
        def infer(self, images, joints, task, explore=False):
            steps = np.repeat((joints + np.array([40, 0, 0, 0, 0], dtype=np.float32))[None], 50, axis=0)
            trig = np.ones((50, 1), dtype=np.float32)
            return steps, trig

    rig = FakeRig(leader_port=None)
    r = shoot(rig, FakeESP32(shoot_every=None), _Lurch(), "C3", log=lambda *_: None, halt=lambda: "stop")
    assert r.reason == "stopped" and not r.fired
    assert abs(float(rig.cmd[0])) < 1.0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
