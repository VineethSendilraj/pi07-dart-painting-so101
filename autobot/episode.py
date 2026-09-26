"""One recorded episode on disk: data/recordings/<session>/<episode_id>/

  meta.json            episode_id, task, cell, recording window, step count, success
  light.mcap           system/step + all array topics
  front_camera.mcap    H.264 frames
  wrist_camera.mcap    H.264 frames
  uploaded.json        written by upload.py once PI has finalized the episode

Shards are written in PI's format at the end of each episode, so upload is just
register → upload → finalize.
"""

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from pi_sdk.mcap import v1
from pi_sdk.mcap.v1.validation import validate_mcap_shard

from autobot import schema as s


@dataclass
class Step:
    time_ns: int  # wall-clock UTC nanoseconds
    images: dict[str, np.ndarray]  # camera name -> uint8 RGB (H, W, 3)
    joints: np.ndarray  # follower joints [5], degrees
    act_joints: np.ndarray  # commanded joints [5] (leader position during teleop)
    trigger: float  # 0 or 1


@dataclass
class EpisodeMeta:
    episode_id: str
    task: str
    cell: str
    recording_start_time: str  # ISO UTC, inclusive
    recording_end_time: str  # ISO UTC, exclusive
    num_steps: int
    success: bool = True
    operator: str = ""
    notes: str = ""

    def start(self) -> datetime:
        return datetime.fromisoformat(self.recording_start_time)

    def end(self) -> datetime:
        return datetime.fromisoformat(self.recording_end_time)


def shard_paths(episode_dir: Path) -> list[Path]:
    return [episode_dir / "light.mcap", *(episode_dir / f"{cam}.mcap" for cam in s.CAMERAS)]


def _even(img: np.ndarray) -> np.ndarray:
    """H.264 needs even width and height."""
    h, w = img.shape[:2]
    return img[: h - h % 2, : w - w % 2]


def write_episode(episode_dir: Path, steps: list[Step], *, episode_id: str, cell: str, operator: str = "") -> EpisodeMeta:
    """Write one episode's shards and meta.json. Returns the metadata."""
    if not steps:
        raise ValueError("episode has no steps")
    episode_dir.mkdir(parents=True, exist_ok=True)
    contract = s.contract()
    light_path, *cam_paths = shard_paths(episode_dir)
    light = v1.Writer(light_path, contract=contract)
    cam_writers = {cam: v1.Writer(p, contract=contract) for cam, p in zip(s.CAMERAS, cam_paths)}
    encoders = {cam: v1.VideoEncoder(fps=s.FPS) for cam in s.CAMERAS}

    for seq, step in enumerate(steps):
        fields = []
        for cam in s.CAMERAS:
            topic = s.CAMERA_TOPICS[cam]
            for frame in encoders[cam].encode(_even(step.images[cam]), capture_time_ns=step.time_ns, sequence_id=seq):
                cam_writers[cam].write_video_frame(topic, frame)
            fields.append(v1.FieldRef(topic=topic, sequence_id=seq, capture_time_ns=step.time_ns))
        arrays = {
            s.OBS_JOINTS: step.joints,
            s.ACT_JOINTS: step.act_joints,
            s.ACT_TRIGGER: [step.trigger],
        }
        for topic, value in arrays.items():
            fields.append(
                light.write_ndarray(topic, np.asarray(value, dtype=np.float32), capture_time_ns=step.time_ns, sequence_id=seq)
            )
        light.write_step(fields, capture_time_ns=step.time_ns, sequence_id=seq)

    for cam in s.CAMERAS:
        for frame in encoders[cam].finalize():
            cam_writers[cam].write_video_frame(s.CAMERA_TOPICS[cam], frame)
    for w in (light, *cam_writers.values()):
        w.close()

    # Window aligned to whole microseconds; end is exclusive, so 1 µs after the last step.
    start = datetime.fromtimestamp(steps[0].time_ns // 1000 / 1e6, tz=UTC)
    end = datetime.fromtimestamp((steps[-1].time_ns // 1000 + 1) / 1e6, tz=UTC)
    meta = EpisodeMeta(
        episode_id=episode_id,
        task=s.task_for(cell),
        cell=cell,
        recording_start_time=start.isoformat(),
        recording_end_time=end.isoformat(),
        num_steps=len(steps),
        operator=operator,
    )
    save_meta(episode_dir, meta)
    return meta


def save_meta(episode_dir: Path, meta: EpisodeMeta) -> None:
    (episode_dir / "meta.json").write_text(json.dumps(asdict(meta), indent=2))


def load_meta(episode_dir: Path) -> EpisodeMeta:
    return EpisodeMeta(**json.loads((episode_dir / "meta.json").read_text()))


def validate(episode_dir: Path) -> None:
    """Check every shard against the contract locally (raises on mismatch)."""
    for p in shard_paths(episode_dir):
        validate_mcap_shard(p, s.contract())


def find_episodes(root: Path) -> list[Path]:
    """All episode directories under root (a session dir or data/recordings)."""
    return sorted(p.parent for p in root.rglob("meta.json"))
