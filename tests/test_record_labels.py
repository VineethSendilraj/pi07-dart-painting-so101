"""record_episode labels the shot at the SHOOT event (act-athon rig: the button release) for
TRIGGER_HOLD_STEPS steps and ends AFTER_SHOT_S later. Run: .venv/bin/python -m pytest tests"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import record  # noqa: E402
from autobot import schema as s  # noqa: E402
from autobot.hardware import FakeRig  # noqa: E402


class ScriptedESP32:
    """Emits SHOOT once, on the given call to events() (one call per recorded step)."""

    def __init__(self, shoot_at: int):
        self.calls, self.shoot_at = 0, shoot_at

    def events(self):
        self.calls += 1
        return ["SHOOT"] if self.calls == self.shoot_at + 1 else []

    def heartbeat(self): ...
    def fire(self): ...
    def close(self): ...


def test_trigger_starts_at_shoot_and_episode_ends_after_shot(monkeypatch):
    monkeypatch.setattr(record.time, "sleep", lambda _s: None)   # don't wait 30 Hz in the test
    steps, reason = record.record_episode(FakeRig(), ScriptedESP32(shoot_at=12), lambda: None)
    trig = [st.trigger for st in steps]
    assert reason == "shot"
    assert trig.index(1.0) == 12
    assert sum(trig) == record.TRIGGER_HOLD_STEPS
    assert len(steps) - 12 == round(record.AFTER_SHOT_S * s.FPS)
