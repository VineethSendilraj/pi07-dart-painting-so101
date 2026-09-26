"""RigdESP32 (act-athon rigd adapter) without a network: rigd's HTTP API is stubbed.
Run: .venv/bin/python -m pytest tests"""

from autobot.hardware import RigdESP32


class StubRigd(RigdESP32):
    """Answers rigd's endpoints from local state instead of HTTP."""

    def __init__(self, mode):
        self.calls, self.releases, self.fire_result = [], 0, {"result": "fired", "n": 1, "ms": 150}
        super().__init__("http://stub", mode=mode)

    def _get(self, path):
        assert path == "/fire_button"
        return {"down": False, "presses": self.releases, "releases": self.releases}

    def _post(self, path, body=None):
        self.calls.append((path, body))
        return self.fire_result if path == "/fire" else {"ok": True}


def test_train_mode_sets_button_fires_and_reports_shoot_on_release():
    esp = StubRigd("train")
    assert ("/raw", {"cmd": "cfg", "button_fires": True}) in esp.calls
    assert esp.events() == []
    esp.releases += 1
    assert esp.events() == ["SHOOT"]
    assert esp.events() == []


def test_test_mode_never_fires_and_reports_turn():
    esp = StubRigd("test")
    assert ("/raw", {"cmd": "cfg", "button_fires": False}) in esp.calls
    esp.releases += 2
    assert esp.events() == ["TURN", "TURN"]


def test_fire_maps_rigd_results():
    esp = StubRigd("test")
    esp.fire()
    assert esp.events() == ["FIRED"]
    esp.fire_result = {"result": "blocked", "reason": "cooldown"}
    esp.fire()
    assert esp.events() == ["COOLDOWN"]
