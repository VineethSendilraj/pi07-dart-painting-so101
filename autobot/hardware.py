"""Hardware access: SO-101 arms + cameras (via LeRobot) and the ESP32 blaster controller.

Each device has a fake twin (same methods) so the pipeline runs without hardware:
`python scripts/record.py --fake`.

ESP32 serial protocol (115200 baud, newline-terminated):
  laptop → ESP32:  F  fire one pulse (only honored if a heartbeat arrived in the last ~500 ms)
                   H  heartbeat
  ESP32 → laptop:  SHOOT    SHOOT button pressed (ESP32 also pulses the relay itself)
                   TURN     DONE button pressed
                   FIRED    relay pulsed in response to F
"""

import time

import numpy as np

from autobot import schema as s

class Rig:
    """Follower arm + cameras, and optionally the leader arm (recording/DAgger only)."""

    def __init__(self, follower_port: str, cameras: dict[str, int | str], leader_port: str | None = None,
                 follower_id: str = "follower", leader_id: str = "leader", width: int = 640, height: int = 480):
        from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
        from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
        from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig

        cams = {
            name: OpenCVCameraConfig(index_or_path=idx, fps=s.FPS, width=width, height=height, fourcc="MJPG")
            for name, idx in cameras.items()
        }
        self.follower = SO101Follower(SO101FollowerConfig(port=follower_port, id=follower_id, cameras=cams, use_degrees=True))
        self.leader = SO101Leader(SO101LeaderConfig(port=leader_port, id=leader_id, use_degrees=True)) if leader_port else None

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
        return images, np.array([obs[f"{m}.pos"] for m in s.JOINT_NAMES], dtype=np.float32)

    def leader_action(self) -> np.ndarray:
        act = self.leader.get_action()
        return np.array([act[f"{m}.pos"] for m in s.JOINT_NAMES], dtype=np.float32)

    def command(self, joints: np.ndarray) -> None:
        """Joint targets only; the gripper servo isn't commanded."""
        self.follower.send_action({f"{m}.pos": float(v) for m, v in zip(s.JOINT_NAMES, joints)})


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


class FakeRig:
    """Smooth fake motion and synthetic camera frames (a 3x3 grid with a moving aim dot)."""

    def __init__(self, *_, leader_port: str | None = "fake", **__):
        self.t0 = time.time()
        self.cmd = np.zeros(len(s.JOINT_NAMES), dtype=np.float32)
        self.has_leader = leader_port is not None

    def connect(self) -> None: ...
    def disconnect(self) -> None: ...

    def _image(self, cam: str) -> np.ndarray:
        img = np.full((480, 640, 3), 235, dtype=np.uint8)
        colors = [(230, 80, 70), (240, 160, 50), (240, 220, 60), (90, 180, 90), (70, 150, 220),
                  (140, 100, 200), (220, 120, 180), (120, 120, 120), (40, 40, 40)]
        for i, c in enumerate(colors):
            r, col = divmod(i, 3)
            img[60 + r * 120: 170 + r * 120, 170 + col * 110: 270 + col * 110] = c
        x = int(320 + 150 * np.tanh(self.cmd[0] / 30)) + (0 if cam == "front_camera" else 20)
        y = int(240 - 150 * np.tanh(self.cmd[1] / 30))
        img[max(y - 6, 0): y + 6, max(x - 6, 0): x + 6] = (0, 0, 0)
        return img

    def observe(self):
        images = {cam: self._image(cam) for cam in s.CAMERAS}
        return images, self.cmd.copy()

    def leader_action(self):
        t = time.time() - self.t0
        return np.array([25 * np.sin(0.4 * t + i) for i in range(len(s.JOINT_NAMES))], dtype=np.float32)

    def command(self, joints) -> None:
        self.cmd = np.asarray(joints, dtype=np.float32)


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
