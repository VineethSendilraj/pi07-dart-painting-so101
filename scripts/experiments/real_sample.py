"""PI pipeline on a small slice of real public SO-101 data (no blaster trigger).

Dataset: Ricky97/SO101_pick_dice_in_bowl (LeRobot v3.0, 30 fps, wrist + fixed "agent" camera,
6-dim state/action). pi-upload-lerobot maps each LeRobot feature to exactly one contract topic
and can't split a feature, so state and action are single [6] topics
(5 joints in JOINT_NAMES order + gripper last). IDs are saved to runs/real_state.json.

  .venv/bin/python scripts/experiments/real_sample.py contract   # register the no-trigger contract
  .venv/bin/python scripts/experiments/real_sample.py dryrun     # validate the selected episodes (no upload)
  .venv/bin/python scripts/experiments/real_sample.py upload     # convert + upload, then add control_source spans
  .venv/bin/python scripts/experiments/real_sample.py finetune   # profile + filter + fine-tune
  .venv/bin/python scripts/experiments/real_sample.py poll       # poll every 3 min until done (log: runs/real_poll.log)
  .venv/bin/python scripts/experiments/real_sample.py infer      # one inference call vs. a recorded step
"""

import csv
import json
import sys
import time
from datetime import UTC, datetime, timedelta

import numpy as np
from pi_sdk.data import McapContract, NdArrayContract, TopicContract, control_source
from pi_sdk.finetuning import ArraySlice, CheckpointPolicy, DatasetFilterInput, EpisodeFilter, TrainingProfileConfig

from autobot import schema as s

ROOT = s.Path(__file__).resolve().parents[2]
STATE_PATH = ROOT / "runs" / "real_state.json"
LOG_PATH = ROOT / "runs" / "real_poll.log"
WORK_DIR = ROOT / "data" / "shards" / "real_sample"

REPO_ID = "Ricky97/SO101_pick_dice_in_bowl"
EPISODES = (0, 1, 2, 3, 4)
DATASET_PATH = "hf-so101-sample/v1"
CONTRACT_NAME = "so101_v1"
PROFILE_NAME = "so101-real-profile-v1"
FILTER_NAME = "hf-so101-sample-v1"
FINETUNE_NAME = "hf-so101-sample-v1"
BASE_MODEL = "pi07"
NUM_STEPS = 200
FPS = 30

CAMERAS = ("agent", "wrist")  # agent = fixed third-person camera, wrist = gripper camera
CAMERA_TOPICS = {cam: f"observation/{cam}/rgb/image" for cam in CAMERAS}
OBS_ARM = "observation/arm/position"  # [6] = 5 joints + gripper
ACT_ARM = "action/arm/position"  # [6] = 5 joints + gripper
STATE_TOPICS = (OBS_ARM,)
ACTION_TOPICS = (ACT_ARM,)
TOPIC_MAP = {
    "observation.state": OBS_ARM,
    "action": ACT_ARM,
    **{f"observation.images.{cam}": topic for cam, topic in CAMERA_TOPICS.items()},
}


def load_state() -> dict:
    return json.loads(STATE_PATH.read_text()) if STATE_PATH.exists() else {}


def save_state(**updates) -> dict:
    state = load_state() | updates
    STATE_PATH.write_text(json.dumps(state, indent=2))
    return state


def log(msg: str) -> None:
    line = f"{datetime.now().isoformat(timespec='seconds')} {msg}"
    print(line, flush=True)
    with LOG_PATH.open("a") as f:
        f.write(line + "\n")


def episode_id(index: int) -> str:
    return f"hf-so101-dice-{index:04d}"


def contract() -> McapContract:
    def array_topic(topic: str, role: str) -> TopicContract:
        return TopicContract(
            topic=topic,
            role=role,
            schema_encoding="protobuf",
            schema_name="pi.data.v1.NdArray",
            expected_frequency_hz=0,
            frequency_tolerance=0,
            ndarray=NdArrayContract(shape=(6,), dtype="float32"),
            required_in_snapshots=("system/step",),
        )

    def camera_topic(topic: str) -> TopicContract:
        return TopicContract(
            topic=topic,
            role="observation",
            schema_encoding="protobuf",
            schema_name="pi.data.v1.VideoFrame",
            expected_frequency_hz=0,
            frequency_tolerance=0,
            required_in_snapshots=("system/step",),
        )

    step = TopicContract(
        topic="system/step",
        role="system",
        schema_encoding="protobuf",
        schema_name="pi.data.v1.SnapshotRef",
        expected_frequency_hz=FPS,
        frequency_tolerance=0.1,
    )
    topics = [step, array_topic(OBS_ARM, "observation"), *map(camera_topic, CAMERA_TOPICS.values()), array_topic(ACT_ARM, "action")]
    return McapContract(topics=tuple(topics))


def cmd_contract() -> None:
    data, _ = s.clients()
    result = data.ensure_platform_contract(
        CONTRACT_NAME, contract(), description="SO-101 arm [6] (5 joints + gripper), fixed + wrist camera, 30 Hz, no trigger"
    )
    cid = result.platform_contract.platform_contract_id
    save_state(platform_contract_id=cid)
    print(f"contract {cid} newly_created={result.newly_created}")


def write_inputs() -> s.Path:
    """custom_metadata.csv gives each episode a stable ID and a fixed absolute start time."""
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    path = WORK_DIR / "custom_metadata.csv"
    base = datetime(2026, 9, 26, 3, 0, 0, tzinfo=UTC)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["episode_index", "episode_id", "start_timestamp", "robot_id"])
        for i in EPISODES:
            w.writerow([i, episode_id(i), (base + timedelta(minutes=5 * i)).isoformat(), "hf-ricky97-so101"])
    (WORK_DIR / "topic_map.json").write_text(json.dumps(TOPIC_MAP, indent=2))
    return path


def run_upload(dry_run: bool):
    from pi_sdk.lerobot import upload_dataset

    data, _ = s.clients()
    return upload_dataset(
        data,
        repo_id=REPO_ID,
        platform_contract_id=load_state()["platform_contract_id"],
        topic_map=TOPIC_MAP,
        metadata_csv=write_inputs(),
        output_dir=WORK_DIR / "shards",
        source_dataset_path=DATASET_PATH,
        episode_indices=list(EPISODES),
        dry_run=dry_run,
        progress=True,
    )


def cmd_dryrun() -> None:
    for r in run_upload(dry_run=True):
        d, m = r.declaration, r.metadata
        print(f"episode {r.episode_index} -> {d.episode_id} {d.recording_start_time} .. {d.recording_end_time} task={m.task!r} success={m.success}")


def cmd_upload() -> None:
    t0 = time.monotonic()
    results = run_upload(dry_run=False)
    print(f"upload done in {time.monotonic() - t0:.0f} s")
    data, _ = s.clients()
    total_steps = 0
    for r in results:
        ep = data.get_episode(r.declaration.episode_id)
        steps = round((ep.episode_end - ep.episode_start).total_seconds() * FPS) + 1
        total_steps += steps
        # pi-upload-lerobot can't attach spans; add a full-length human control span afterward.
        data.replace_annotations(
            r.declaration.episode_id, spans=[control_source(start=ep.episode_start, end=ep.episode_end, value="human")]
        )
        print(f"{r.declaration.episode_id}: uploaded={r.uploaded} ~{steps} steps finalized={ep.upload_completed_at is not None} span added")
    save_state(episode_ids=[episode_id(i) for i in EPISODES], approx_steps=total_steps, task=results[0].metadata.task)
    print(f"total ~{total_steps} steps")


def training_profile() -> TrainingProfileConfig:
    return TrainingProfileConfig(
        base_camera_topics=(CAMERA_TOPICS["agent"],),
        wrist_camera_topics=(CAMERA_TOPICS["wrist"],),
        state_slices=tuple(ArraySlice(t) for t in STATE_TOPICS),
        action_topics=ACTION_TOPICS,
        action_space="absolute",
    )


def cmd_finetune() -> None:
    _, ft = s.clients()
    cid = load_state()["platform_contract_id"]
    profile = ft.create_training_profile(PROFILE_NAME, cid, training_profile())
    # success isn't set in this dataset, so no success filter.
    dfilter = ft.create_dataset_filter(FILTER_NAME, EpisodeFilter(platform_contract_id=cid, source_dataset_path=DATASET_PATH))
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
        finetune_created=datetime.now().isoformat(timespec="seconds"),
    )
    print(f"profile {profile.training_profile_id}\nfilter {dfilter.dataset_filter_id}\nrun {run.run_id} state={run.state} num_steps={run.num_steps} batch={run.batch_size}")


def cmd_poll() -> None:
    _, ft = s.clients()
    run_id = load_state()["run_id"]
    last, available_at = None, None
    while True:
        try:
            run = ft.get_finetune(run_id)
            ckpts = ft.list_checkpoints(run_id).checkpoints
        except Exception as e:  # transient API errors: log and retry next tick
            log(f"poll error: {type(e).__name__}: {e}")
            time.sleep(180)
            continue
        snapshot = (run.state, run.progress, tuple((c.checkpoint_id, c.state, c.gateway_url) for c in ckpts))
        if snapshot != last:
            log(f"run {run.state} progress={run.progress} reason={run.terminal_reason}")
            for c in ckpts:
                log(f"  checkpoint {c.checkpoint_id} step={c.step} state={c.state} gateway_url={c.gateway_url}")
            last = snapshot
        if run.state == "succeeded" and not load_state().get("evaluations"):
            evals = [(m.step, m.train_mae, m.validation_mae) for m in ft.list_finetune_evaluations(run_id)]
            for step, train, val in evals:
                log(f"  eval step={step} train_mae={train} validation_mae={val}")
            if evals:
                save_state(evaluations=evals)
        if run.state in {"failed", "cancelled"}:
            log("stopping: run did not succeed")
            return
        hosted = [c for c in ckpts if c.state == "available" and c.gateway_url]
        if hosted:
            save_state(checkpoint_id=hosted[0].checkpoint_id, gateway_url=hosted[0].gateway_url)
            log("stopping: gateway_url available")
            return
        if available_at is None and any(c.state == "available" for c in ckpts):
            available_at = time.monotonic()
            save_state(checkpoint_id=next(c.checkpoint_id for c in ckpts if c.state == "available"))
        if available_at is not None and time.monotonic() - available_at > 90 * 60:
            log("stopping: checkpoint available 90 min without gateway_url")
            return
        time.sleep(180)


def recorded_step(episode_index: int, step: int) -> dict:
    """Real images (uint8 HWC), state and action from the local LeRobot cache."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(REPO_ID, episodes=[episode_index], video_backend="pyav")
    item = ds[step]
    images = {cam: (item[f"observation.images.{cam}"].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8) for cam in CAMERAS}
    return {
        "images": images,
        "state": item["observation.state"].numpy().astype(np.float32),
        "actions": np.stack([ds[step + k]["action"].numpy() for k in range(10)]),
    }


def cmd_infer() -> None:
    from pi_sdk.inference import ClientConfig, InferenceInput, PolicyClient

    state = load_state()
    task = state["task"]
    step = 100
    rec = recorded_step(EPISODES[0], step)
    config = ClientConfig(
        gateway_url=state["gateway_url"],
        api_key=s.api_key(),
        robot_task_string=task,
        action_keys=list(ACTION_TOPICS),
        execute_chunk_size=10,
    )
    with PolicyClient(config) as policy:
        meta = policy.server_metadata
        for field in ("camera_names", "input_spec", "output_spec", "action_keys", "action_horizon", "action_dim", "image_preprocessing"):
            print(f"server {field}: {getattr(meta, field, None)}")
        images = {name: rec["images"][next(c for c in CAMERAS if c in name)] for name in meta.camera_names}
        result = policy.infer(InferenceInput(images=images, states={OBS_ARM: rec["state"]}, raw_text=task))
    for topic, chunk in result.actions_by_key.items():
        chunk = np.asarray(chunk)
        print(f"{topic}: shape={chunk.shape} dtype={chunk.dtype} min={chunk.min():.3f} max={chunk.max():.3f}")
    pred = np.asarray(result.actions_by_key[ACT_ARM])
    np.set_printoptions(precision=2, suppress=True)
    print(f"episode {EPISODES[0]} step {step}\n  state     {rec['state']}\n  recorded  {rec['actions'][0]}\n  predicted {pred[0]}")
    n = min(len(pred), len(rec["actions"]))
    print(f"  |pred - recorded| mean over first {n} actions per dim: {np.abs(pred[:n] - rec['actions'][:n]).mean(axis=0)}")
    print(f"metrics: {result.metrics}")


if __name__ == "__main__":
    commands = {
        "contract": cmd_contract,
        "dryrun": cmd_dryrun,
        "upload": cmd_upload,
        "finetune": cmd_finetune,
        "poll": cmd_poll,
        "infer": cmd_infer,
    }
    if len(sys.argv) != 2 or sys.argv[1] not in commands:
        sys.exit(__doc__)
    commands[sys.argv[1]]()
