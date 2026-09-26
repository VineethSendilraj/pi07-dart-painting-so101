"""Smaller variant of mock_pipeline.py: 3 x 20 s episodes and a very short fine-tune,
to find the minimum data the fine-tune accepts. Reuses the contract and profile from
runs/mock_state.json; saves its own IDs to runs/mock_short_state.json.

  .venv/bin/python scripts/experiments/mock_short.py upload
  .venv/bin/python scripts/experiments/mock_short.py finetune [num_steps]
  .venv/bin/python scripts/experiments/mock_short.py status
"""

import json
import sys
from datetime import UTC, datetime, timedelta

from pi_sdk.data import EpisodeDeclaration, EpisodeMetadata, EpisodeShard, control_source
from pi_sdk.finetuning import CheckpointPolicy, DatasetFilterInput, EpisodeFilter

import mock_pipeline as m
from autobot import schema as s

STATE_PATH = m.ROOT / "runs" / "mock_short_state.json"
DATASET_PATH = "mock-short/v1"
NUM_EPISODES = 3
EPISODE_SECONDS = 20  # 3 x 20 s x 30 Hz = 1,800 steps


def load_state() -> dict:
    return json.loads(STATE_PATH.read_text()) if STATE_PATH.exists() else {}


def save_state(**updates) -> None:
    STATE_PATH.write_text(json.dumps(load_state() | updates, indent=2))


def cmd_upload() -> None:
    data, _ = s.clients()
    cid = m.load_state()["platform_contract_id"]
    contract = data.get_platform_contract(cid)
    base = datetime(2026, 9, 26, 1, 0, 0, tzinfo=UTC)
    for n in range(NUM_EPISODES):
        episode_id = f"mock-short-{n:04d}"
        start = base + timedelta(minutes=5 * n)
        end = start + timedelta(seconds=EPISODE_SECONDS, microseconds=1)
        existing = m._get_episode(data, episode_id)
        if existing is not None and existing.upload_completed_at is not None:
            print(f"{episode_id}: already finalized")
            continue
        paths = m.write_shards(contract, start, EPISODE_SECONDS * s.FPS, m.SHARD_DIR / episode_id)
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
                    dataset="mock", task=m.TASK, is_eval_episode=False, success=True, source_dataset_path=DATASET_PATH
                ),
                platform_contract_id=cid,
                spans=[control_source(start=start, end=end, value="human")],
            )
        for p in paths:
            data.upload_shard(
                episode_id, EpisodeShard.create(p, episode_id=episode_id, recording_start_time=start, recording_end_time=end)
            )
        data.finalize_episode_upload(episode_id)
        print(f"{episode_id}: {EPISODE_SECONDS * s.FPS} steps, finalized={data.get_episode(episode_id).upload_completed_at is not None}")


def cmd_finetune(num_steps: int) -> None:
    _, ft = s.clients()
    cid = m.load_state()["platform_contract_id"]
    profile_id = m.load_state()["training_profile_id"]
    dfilter = ft.create_dataset_filter(
        "mock-so101-short-v1", EpisodeFilter(platform_contract_id=cid, source_dataset_path=DATASET_PATH, success=True)
    )
    run = ft.create_finetune(
        f"mock-so101-short-{num_steps}steps",
        [DatasetFilterInput(dfilter.dataset_filter_id)],
        base_model=m.BASE_MODEL,
        num_steps=num_steps,
        checkpoint_policy=CheckpointPolicy("final_only"),
        training_profile_id=profile_id,
    )
    save_state(dataset_filter_id=dfilter.dataset_filter_id, run_id=run.run_id, created=datetime.now().isoformat())
    print(f"filter {dfilter.dataset_filter_id}\nrun {run.run_id} state={run.state} num_steps={run.num_steps} batch={run.batch_size}")


def cmd_status() -> None:
    _, ft = s.clients()
    run = ft.get_finetune(load_state()["run_id"])
    print(f"run {run.run_id}: {run.state} progress={run.progress} reason={run.terminal_reason}")
    for c in ft.list_checkpoints(run.run_id).checkpoints:
        print(f"  checkpoint {c.checkpoint_id} step={c.step} state={c.state} gateway_url={c.gateway_url}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "upload":
        cmd_upload()
    elif cmd == "finetune":
        cmd_finetune(int(sys.argv[2]) if len(sys.argv) > 2 else 20)
    elif cmd == "status":
        cmd_status()
    else:
        sys.exit(__doc__)
