"""Safety-rule tests. Run: .venv/bin/python -m pytest tests  (or: .venv/bin/python tests/test_control.py)"""

import numpy as np

from autobot.control import FIRE_CONSECUTIVE, MAX_STEP_DEG, FireRule, speed_limit

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


def test_only_one_shot_per_turn():
    rule = FireRule()
    assert sum(rule.update(1.0, STILL) for _ in range(30)) == 1


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
