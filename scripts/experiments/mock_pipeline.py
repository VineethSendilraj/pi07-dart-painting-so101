"""End-to-end smoke test of the PI pipeline with fake SO-101 + blaster data.

Each step is idempotent and saves IDs to runs/mock_state.json, so re-run any step.

  .venv/bin/python scripts/experiments/mock_pipeline.py contract   # register the SO-101 contract
  .venv/bin/python scripts/experiments/mock_pipeline.py upload     # write + upload fake episodes
  .venv/bin/python scripts/experiments/mock_pipeline.py finetune   # profile + filter + small fine-tune
  .venv/bin/python scripts/experiments/mock_pipeline.py status     # fine-tune progress + checkpoints
  .venv/bin/python scripts/experiments/mock_pipeline.py infer      # one inference call on the hosted checkpoint
"""

import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
from pi_sdk.data import EpisodeDeclaration, EpisodeMetadata, EpisodeShard, control_source
from pi_sdk.finetuning import CheckpointPolicy, DatasetFilterInput, EpisodeFilter
from pi_sdk.mcap import v1

from autobot import schema as s

ROOT = Path(__file__).resolve().parents[2]
STATE_PATH = ROOT / "runs" / "mock_state.json"
SHARD_DIR = ROOT / "data" / "shards" / "mock"

DATASET_PATH = "mock/v1"
TASK = "aim at the target and fire"
NUM_EPISODES = 3
EPISODE_SECONDS = 60  # 3 x 60 s x 30 Hz = 5,400 steps
IMAGE_HW = (240, 320)
FINETUNE_NAME = "mock-so101-smoke-v1"
BASE_MODEL = "pi07"
NUM_STEPS = 100


def load_state() -> dict:
    return json.loads(STATE_PATH.read_text()) if STATE_PATH.exists() else {}


def save_state(**updates) -> dict:
    state = load_state() | updates
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2))
    return state


def cmd_contract() -> None:
    data, _ = s.clients()
    result = data.ensure_platform_contract(
        s.CONTRACT_NAME, s.contract(), description="SO-101 (5 joints + gripper), front + wrist camera, blaster trigger, 30 Hz"
    )
    cid = result.platform_contract.platform_contract_id
    save_state(platform_contract_id=cid)
    print(f"contract {cid} newly_created={result.newly_created}")


def fake_image(cam_index: int, step: int) -> np.ndarray:
    """A moving square on a flat background: cheap to encode, not pure noise."""
    h, w = IMAGE_HW
    img = np.full((h, w, 3), 40 + 60 * cam_index, dtype=np.uint8)
    x = int((w - 40) * (0.5 + 0.5 * np.sin(step / 25 + cam_index)))
    y = int((h - 40) * (0.5 + 0.5 * np.cos(step / 40)))
    img[y : y + 40, x : x + 40] = (220, 60, 60)
    return img


def fake_arrays(step: int) -> dict[str, np.ndarray]:
    t = step / s.FPS
    joints = np.array([30 * np.sin(0.3 * t + i) for i in range(len(s.JOINT_NAMES))], dtype=np.float32)
    target = np.array([30 * np.sin(0.3 * (t + 0.1) + i) for i in range(len(s.JOINT_NAMES))], dtype=np.float32)
    gripper = np.float32(50 + 40 * np.sin(0.2 * t))
    trigger = np.float32(1.0 if step % 150 < 10 else 0.0)  # fire ~every 5 s, held 10 steps
    return {
        s.OBS_JOINTS: joints,
        s.OBS_GRIPPER: np.array([gripper], dtype=np.float32),
        s.ACT_JOINTS: target,
        s.ACT_GRIPPER: np.array([gripper], dtype=np.float32),
        s.ACT_TRIGGER: np.array([trigger], dtype=np.float32),
    }


def write_shards(contract, start: datetime, num_steps: int, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    light_path = out_dir / "light.mcap"
    cam_paths = {cam: out_dir / f"{cam}.mcap" for cam in s.CAMERAS}
    light = v1.Writer(light_path, contract=contract)
    cam_writers = {cam: v1.Writer(p, contract=contract) for cam, p in cam_paths.items()}
    encoders = {cam: v1.VideoEncoder(fps=s.FPS) for cam in s.CAMERAS}
    start_ns = int(start.timestamp() * 1e9)

    for seq in range(num_steps):
        time_ns = start_ns + round(seq * 1e9 / s.FPS)
        fields = []
        for i, cam in enumerate(s.CAMERAS):
            topic = s.CAMERA_TOPICS[cam]
            for frame in encoders[cam].encode(fake_image(i, seq), capture_time_ns=time_ns, sequence_id=seq):
                cam_writers[cam].write_video_frame(topic, frame)
            fields.append(v1.FieldRef(topic=topic, sequence_id=seq, capture_time_ns=time_ns))
        for topic, value in fake_arrays(seq).items():
            fields.append(light.write_ndarray(topic, value, capture_time_ns=time_ns, sequence_id=seq))
        light.write_step(fields, capture_time_ns=time_ns, sequence_id=seq)

    for cam in s.CAMERAS:
        for frame in encoders[cam].finalize():
            cam_writers[cam].write_video_frame(s.CAMERA_TOPICS[cam], frame)
    for w in (light, *cam_writers.values()):
        w.close()
    return [light_path, *cam_paths.values()]


def cmd_upload() -> None:
    data, _ = s.clients()
    cid = load_state()["platform_contract_id"]
    contract = data.get_platform_contract(cid)
    num_steps = EPISODE_SECONDS * s.FPS
    # Fixed start times keep episode windows and shard IDs stable across retries.
    base = datetime(2026, 9, 26, 0, 0, 0, tzinfo=UTC)

    for n in range(NUM_EPISODES):
        episode_id = f"mock-so101-{n:04d}"
        start = base + timedelta(minutes=5 * n)
        end = start + timedelta(seconds=EPISODE_SECONDS, microseconds=1)
        existing = _get_episode(data, episode_id)
        if existing is not None and existing.upload_completed_at is not None:
            print(f"{episode_id}: already finalized")
            continue

        paths = write_shards(contract, start, num_steps, SHARD_DIR / episode_id)
        if existing is None:
            data.create_episode(
                EpisodeDeclaration(
                    episode_id=episode_id,
                    robot_id="mock-so101",
                    robot_platform=s.CONTRACT_NAME,
                    recording_start_time=start,
                    recording_end_time=end,
                ),
                EpisodeMetadata(
                    dataset="mock", task=TASK, is_eval_episode=False, success=True, source_dataset_path=DATASET_PATH
                ),
                platform_contract_id=cid,
                spans=[control_source(start=start, end=end, value="human")],
            )
        for p in paths:
            shard = EpisodeShard.create(p, episode_id=episode_id, recording_start_time=start, recording_end_time=end)
            data.upload_shard(episode_id, shard)
        data.finalize_episode_upload(episode_id)
        done = data.get_episode(episode_id).upload_completed_at
        print(f"{episode_id}: {num_steps} steps, finalized={done is not None}")


def _get_episode(data, episode_id):
    try:
        return data.get_episode(episode_id)
    except Exception:
        return None


def cmd_finetune() -> None:
    _, ft = s.clients()
    state = load_state()
    cid = state["platform_contract_id"]
    profile = ft.create_training_profile("mock-so101-blaster-profile-v1", cid, s.training_profile())
    dfilter = ft.create_dataset_filter(
        "mock-so101-smoke-v1",
        EpisodeFilter(platform_contract_id=cid, source_dataset_path=DATASET_PATH, success=True),
    )
    run = ft.create_finetune(
        FINETUNE_NAME,
        [DatasetFilterInput(dfilter.dataset_filter_id)],
        base_model=BASE_MODEL,
        num_steps=NUM_STEPS,
        checkpoint_policy=CheckpointPolicy("final_only"),
        training_profile_id=profile.training_profile_id,
    )
    save_state(
        training_profile_id=profile.training_profile_id,
        dataset_filter_id=dfilter.dataset_filter_id,
        run_id=run.run_id,
    )
    print(f"profile {profile.training_profile_id}\nfilter {dfilter.dataset_filter_id}\nrun {run.run_id} state={run.state}")


def cmd_status() -> None:
    _, ft = s.clients()
    state = load_state()
    run = ft.get_finetune(state["run_id"])
    print(f"run {run.run_id}: {run.state} progress={run.progress} reason={run.terminal_reason}")
    for c in ft.list_checkpoints(run.run_id).checkpoints:
        print(f"  checkpoint {c.checkpoint_id} step={c.step} state={c.state} gateway_url={c.gateway_url}")
        if c.state == "available" and c.gateway_url:
            save_state(checkpoint_id=c.checkpoint_id, gateway_url=c.gateway_url)


def cmd_infer() -> None:
    from pi_sdk.inference import ClientConfig, InferenceInput, PolicyClient

    state = load_state()
    config = ClientConfig(
        gateway_url=state["gateway_url"],
        api_key=s.api_key(),
        robot_task_string=TASK,
        action_keys=list(s.ACTION_TOPICS),
        execute_chunk_size=10,
    )
    arrays = fake_arrays(0)
    with PolicyClient(config) as policy:
        meta = policy.server_metadata
        for field in ("camera_names", "input_spec", "output_spec", "action_keys", "action_horizon", "action_dim", "image_preprocessing"):
            print(f"server {field}: {getattr(meta, field)}")
        result = policy.infer(
            InferenceInput(
                images={cam: fake_image(i, 0) for i, cam in enumerate(meta.camera_names)},
                states={t: arrays[t] for t in s.STATE_TOPICS},
                raw_text=TASK,
            )
        )
    for topic, chunk in result.actions_by_key.items():
        chunk = np.asarray(chunk)
        print(f"{topic}: shape={chunk.shape} dtype={chunk.dtype} min={chunk.min():.3f} max={chunk.max():.3f}")
    print(f"metrics: {result.metrics}")


if __name__ == "__main__":
    commands = {"contract": cmd_contract, "upload": cmd_upload, "finetune": cmd_finetune, "status": cmd_status, "infer": cmd_infer}
    if len(sys.argv) != 2 or sys.argv[1] not in commands:
        sys.exit(__doc__)
    commands[sys.argv[1]]()
