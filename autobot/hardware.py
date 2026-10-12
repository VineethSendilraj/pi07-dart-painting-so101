"""Hardware access: SO-101 arms + cameras (via LeRobot) and the ESP32 blaster controller.

Each device has a fake twin (same methods) so the pipeline runs without hardware:
`python scripts/record.py --fake`.

ESP32 serial protocol (115200 baud, newline-terminated) — Vineeth's sketch only:
  laptop → ESP32:  F  fire one pulse (only honored if a heartbeat arrived in the last ~500 ms)
                   H  heartbeat
  ESP32 → laptop:  SHOOT    SHOOT button pressed (ESP32 also pulses the relay itself)
                   TURN     DONE button pressed
                   FIRED    relay pulsed in response to F

The act-athon rig uses Yuvraj's fire_ctl 1.3.0 + rigd instead. Use RigdESP32
(record.py --rigd-url http://localhost:8080). fire_ctl has no heartbeat /lease;
one button release = one shot = a 2-pulse burst, one `fired` event.
"""

import json
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

from autobot import schema as s

REST_POSE_PATH = Path(__file__).resolve().parents[1] / "calibration" / "rest_pose.json"
GUN_MARK_PATH = Path(__file__).resolve().parents[1] / "calibration" / "gun_mark.json"
GUN_CAMERA = "wrist_camera"


def draw_gun_crosshair(image: np.ndarray, x: float = 0.5, y: float = 0.5) -> np.ndarray:
    """Short red plus at (x, y) fractions of the gun image. Same mark the page lets you place."""
    img = np.array(image, copy=True, order="C")
    if img.ndim != 3 or img.shape[2] != 3:
        return img
    h, w = img.shape[:2]
    color = np.array([248, 113, 113], dtype=img.dtype)
    t = max(2, w // 320)
    arm = max(10, w // 32)
    cx = int(np.clip(x, 0.0, 1.0) * (w - 1))
    cy = int(np.clip(y, 0.0, 1.0) * (h - 1))
    img[max(0, cy - arm): min(h, cy + arm + 1), max(0, cx - t // 2): cx - t // 2 + t] = color
    img[max(0, cy - t // 2): cy - t // 2 + t, max(0, cx - arm): min(w, cx + arm + 1)] = color
    return img


def load_rest_pose() -> np.ndarray | None:
    """The shared rest pose (5 joints, degrees): where every game turn starts, and the 'rest' start in demos."""
    if not REST_POSE_PATH.exists():
        return None
    return np.array(json.loads(REST_POSE_PATH.read_text())["joints"], dtype=np.float32)


def load_gun_mark() -> tuple[float, float] | None:
    """Where the red plus was burned into the recorded gun frames (fractions of width, height). Inference must draw
    the same mark: most training gun frames have it. Measured from the recordings by scripts/make_trimmed.py."""
    if not GUN_MARK_PATH.exists():
        return None
    d = json.loads(GUN_MARK_PATH.read_text())
    return float(d["x"]), float(d["y"])


def save_rest_pose(joints: np.ndarray) -> None:
    REST_POSE_PATH.parent.mkdir(parents=True, exist_ok=True)
    REST_POSE_PATH.write_text(json.dumps({"names": list(s.JOINT_NAMES), "joints": [round(float(v), 2) for v in joints]}, indent=2))


class Rig:
    """Follower arm + cameras, and optionally the leader arm (recording/DAgger only)."""

    def __init__(self, follower_port: str, cameras: dict[str, int | str], leader_port: str | None = None,
                 follower_id: str = "follower_arm", leader_id: str = "leader_arm", width: int = 640, height: int = 480):
        from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
        from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
        from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig

        cams = {
            # MJPG helps two cameras share a USB hub on Linux; macOS's AVFoundation backend rejects it.
            name: OpenCVCameraConfig(index_or_path=idx, fps=s.FPS, width=width, height=height,
                                     fourcc=None if sys.platform == "darwin" else "MJPG", warmup_s=2)
            for name, idx in cameras.items()
        }
        # Same as `lerobot-teleoperate`: send the leader angle as the goal. A 4°/tick cap
        # left wrist_flex (gun pitch) stuck when the leader moved up — the clamp never caught up.
        self.follower = SO101Follower(SO101FollowerConfig(port=follower_port, id=follower_id, cameras=cams,
                                                          use_degrees=True, max_relative_target=None))
        self.leader = SO101Leader(SO101LeaderConfig(port=leader_port, id=leader_id, use_degrees=True)) if leader_port else None
        self.crosshair: tuple[float, float] | None = None  # set from the page marker, or --crosshair at test time

    def connect(self) -> None:
        self.follower.connect()
        if self.leader:
            self.leader.connect()

    def disconnect(self) -> None:
        if self.leader:
            self.leader.disconnect()
        self.follower.disconnect()

    def observe(self) -> tuple[dict[str, np.ndarray], np.ndarray]:
        """Camera images and follower joints [5] (degrees). The gripper servo is ignored."""
        obs = self.follower.get_observation()
        images = {cam: obs[cam] for cam in s.CAMERAS}
        if self.crosshair and GUN_CAMERA in images:
            images[GUN_CAMERA] = draw_gun_crosshair(images[GUN_CAMERA], *self.crosshair)
        return images, np.array([obs[f"{m}.pos"] for m in s.JOINT_NAMES], dtype=np.float32)

    def leader_action(self) -> np.ndarray:
        act = self.leader.get_action()
        return np.array([act[f"{m}.pos"] for m in s.JOINT_NAMES], dtype=np.float32)

    def command(self, joints: np.ndarray) -> None:
        """Joint targets only; the gripper servo isn't commanded."""
        self.follower.send_action({f"{m}.pos": float(v) for m, v in zip(s.JOINT_NAMES, joints)})

    def torque(self, on: bool) -> None:
        """Pause should leave the follower limp. Torque off = you can move it by hand."""
        if on:
            self.follower.bus.enable_torque()
        else:
            self.follower.bus.disable_torque()


class ESP32:
    def __init__(self, port: str, baud: int = 115200):
        import serial

        self.ser = serial.Serial(port, baud, timeout=0)
        self._buf = b""

    def events(self) -> list[str]:
        """Non-blocking: all complete lines received since the last call."""
        self._buf += self.ser.read(self.ser.in_waiting or 0)
        *lines, self._buf = self._buf.split(b"\n")
        return [ln.decode(errors="replace").strip() for ln in lines if ln.strip()]

    def heartbeat(self) -> None:
        self.ser.write(b"H\n")

    def fire(self) -> None:
        self.ser.write(b"F\n")

    def close(self) -> None:
        self.ser.close()


class RigdESP32:
    """Same interface as `ESP32`, backed by act-athon's rigd (github.com/yuviji/act-athon).

    fire_ctl 1.3.0 has no heartbeat / arming / /lease. The board is READY from boot.
    One physical button (GPIO25):
      mode="train"  releasing it fires (2-pulse burst) and we report "SHOOT"   (recording)
      mode="test"   releasing it never fires; we report "TURN"                 (game: the DONE button)
    Burst does not double-label: one release → one SHOOT. The mode is re-asserted whenever rigd
    reports a different one (e.g. the ESP32 rebooted into its default), and "MODE_RESET" is reported.
    """

    BLOCKED = {"cooldown": "COOLDOWN", "no_hb": "NO_HEARTBEAT", "disarmed": "DISARMED"}
    FIRE_TIMEOUT = 1.5  # rigd waits up to ~0.5 s for the board's reply to /fire

    def __init__(self, url: str = "http://localhost:8080", mode: str = "train", timeout: float = 0.3):
        if mode not in ("train", "test"):
            raise ValueError("mode must be 'train' or 'test'")
        self.url, self.mode, self.timeout = url.rstrip("/"), mode, timeout
        self._pending: list[str] = []
        state = self._set_mode()
        self._releases = int(state["fire_button"].get("releases", 0))

    def _get(self, path: str) -> dict:
        with urllib.request.urlopen(self.url + path, timeout=self.timeout) as r:
            return json.loads(r.read())

    def _post(self, path: str, body: dict | None = None, timeout: float | None = None) -> dict:
        req = urllib.request.Request(
            self.url + path,
            data=json.dumps(body or {}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout or self.timeout) as r:
            return json.loads(r.read())

    def _mode_ok(self, state: dict) -> bool:
        cfg = state.get("cfg") or {}
        return cfg.get("button_fires") == (self.mode == "train")

    def _set_mode(self) -> dict:
        """POST the mode and confirm the board applied it (~2 s), else raise."""
        for _ in range(8):
            self._post("/raw", {"cmd": "cfg", "button_fires": self.mode == "train"})
            time.sleep(0.25)
            state = self._get("/state")
            if self._mode_ok(state):
                return state
        raise RuntimeError(f"rigd didn't confirm {self.mode} mode (is the ESP32 connected?)")

    def events(self) -> list[str]:
        out, self._pending = self._pending, []  # FIRED from fire() must reach the caller even if /state fails
        try:
            state = self._get("/state")
        except OSError:
            if out:
                return out
            raise
        if not self._mode_ok(state):
            self._post("/raw", {"cmd": "cfg", "button_fires": self.mode == "train"})
            out.append("MODE_RESET")
        releases = int(state["fire_button"].get("releases", 0))
        new, self._releases = max(releases - self._releases, 0), releases
        return out + ["SHOOT" if self.mode == "train" else "TURN"] * new

    def heartbeat(self) -> None:
        return

    def fire(self) -> None:
        r = self._post("/fire", {"ms": 150}, timeout=self.FIRE_TIMEOUT)
        self._pending.append(
            "FIRED" if r.get("result") == "fired" else self.BLOCKED.get(r.get("reason"), "BLOCKED")
        )

    def close(self) -> None: ...


class FakeRig:
    """Smooth fake motion and synthetic camera frames (a 5x5 grid with a moving aim dot)."""

    def __init__(self, *_, leader_port: str | None = "fake", **__):
        self.t0 = time.time()
        self.cmd = np.zeros(len(s.JOINT_NAMES), dtype=np.float32)
        self.has_leader = leader_port is not None
        self.crosshair: tuple[float, float] | None = None

    def connect(self) -> None: ...
    def disconnect(self) -> None: ...

    def _image(self, cam: str) -> np.ndarray:
        img = np.full((480, 640, 3), 235, dtype=np.uint8)
        for r in range(5):
            for col in range(5):
                shade = 120 + 25 * ((r + col) % 2)
                img[40 + r * 80: 115 + r * 80, 120 + col * 80: 195 + col * 80] = (shade, 110, shade)
        x = int(320 + 150 * np.tanh(self.cmd[0] / 30)) + (0 if cam == "front_camera" else 20)
        y = int(240 - 150 * np.tanh(self.cmd[1] / 30))
        img[max(y - 6, 0): y + 6, max(x - 6, 0): x + 6] = (0, 0, 0)
        return img

    def observe(self):
        images = {cam: self._image(cam) for cam in s.CAMERAS}
        if self.crosshair and GUN_CAMERA in images:
            images[GUN_CAMERA] = draw_gun_crosshair(images[GUN_CAMERA], *self.crosshair)
        return images, self.cmd.copy()

    def leader_action(self):
        t = time.time() - self.t0
        return np.array([25 * np.sin(0.4 * t + i) for i in range(len(s.JOINT_NAMES))], dtype=np.float32)

    def command(self, joints) -> None:
        self.cmd = np.asarray(joints, dtype=np.float32)

    def torque(self, on: bool) -> None:
        pass


class FakeESP32:
    """Presses SHOOT every `shoot_every` seconds (None = never) and answers F with FIRED."""

    def __init__(self, *_, shoot_every: float | None = 2.5):
        self.shoot_every = shoot_every
        self.next_shot = time.time() + shoot_every if shoot_every else float("inf")
        self.pending: list[str] = []

    def events(self) -> list[str]:
        out, self.pending = self.pending, []
        if time.time() >= self.next_shot:
            self.next_shot = time.time() + self.shoot_every
            out.append("SHOOT")
        return out

    def heartbeat(self) -> None: ...

    def fire(self) -> None:
        self.pending.append("FIRED")

    def close(self) -> None: ...
