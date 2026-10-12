"""Upload recorded episodes to PI: validate → register → upload shards → finalize.

  .venv/bin/python scripts/upload.py data/recordings/s20260926-1405    # one session
  .venv/bin/python scripts/upload.py                                     # everything not yet uploaded
  .venv/bin/python scripts/upload.py --dry-run                           # validate locally only

Safe to re-run: finished episodes get an uploaded.json and are skipped; PI's calls are idempotent.
"""

import argparse
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from pi_sdk.data import APIError, EpisodeDeclaration, EpisodeMetadata, EpisodeShard, control_source

from autobot import schema as s
from autobot.episode import find_episodes, load_meta, shard_paths, validate

ROOT = Path(__file__).resolve().parents[1]


def get_episode(data, episode_id):
    try:
        return data.get_episode(episode_id)
    except APIError as e:
        if "not_found" in str(e).lower() or "not found" in str(e).lower():
            return None
        raise


def target_path(meta, dataset_path: str) -> str:
    """The episode's own dataset path wins (set at record time); dataset_path is for older episodes.
    Fake episodes may only ever go to test/..."""
    path = meta.dataset_path or dataset_path
    if meta.fake and not path.startswith("test/"):
        raise ValueError(f"{meta.episode_id} was recorded with --fake: refusing to upload it to {path!r}")
    return path


def upload_one(data, contract_id: str, ep_dir: Path, dataset_path: str) -> str:
    """Register + upload + finalize one episode (idempotent). Returns the dataset path used."""
    if not contract_id:
        raise ValueError("no platform_contract_id")
    meta = load_meta(ep_dir)
    dataset_path = target_path(meta, dataset_path)
    start, end = meta.start(), meta.end()
    existing = get_episode(data, meta.episode_id)
    if existing is not None and (existing.episode_start != start or existing.episode.metadata.task != meta.task):
        raise ValueError(f"episode ID {meta.episode_id} already exists on PI with a different recording "
                         f"({existing.episode.metadata.task}, {existing.episode_start}): rename this session")
    if existing is None:
        data.create_episode(
            EpisodeDeclaration(
                episode_id=meta.episode_id,
                robot_id="so101-blaster",
                robot_platform=s.CONTRACT_NAME,
                recording_start_time=start,
                recording_end_time=end,
            ),
            EpisodeMetadata(
                dataset="so101-blaster",
                task=meta.task,
                is_eval_episode=False,
                success=meta.success,
                operator_id=meta.operator,
                source_dataset_path=dataset_path,
            ),
            platform_contract_id=contract_id,
            spans=[control_source(start=start, end=end, value="human")],
        )
    if existing is None or existing.upload_completed_at is None:
        for path in shard_paths(ep_dir):
            shard = EpisodeShard.create(path, episode_id=meta.episode_id, recording_start_time=start, recording_end_time=end)
            data.upload_shard(meta.episode_id, shard)
        data.finalize_episode_upload(meta.episode_id)
        if data.get_episode(meta.episode_id).upload_completed_at is None:
            raise RuntimeError(f"{meta.episode_id} did not finalize")
    (ep_dir / "uploaded.json").write_text(
        json.dumps({"uploaded_at": datetime.now(UTC).isoformat(), "dataset_path": dataset_path, "contract_id": contract_id}, indent=2)
    )
    return dataset_path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("paths", nargs="*", type=Path, default=[ROOT / "data" / "recordings"])
    p.add_argument("--dataset-path", default=s.DATASET_PATH,
                   help=f"for episodes that don't record their own path (default {s.DATASET_PATH})")
    p.add_argument("--dry-run", action="store_true", help="validate locally, upload nothing")
    a = p.parse_args()

    episodes = [ep for path in a.paths for ep in find_episodes(path)]
    todo = [ep for ep in episodes if not (ep / "uploaded.json").exists()]
    print(f"{len(episodes)} episodes found, {len(todo)} to upload → dataset {a.dataset_path}")
    if not todo:
        return

    data, contract_id = None, None
    if not a.dry_run:
        data, _ = s.clients()
        contract_id = s.ensure_contract(data)
        print(f"contract {s.CONTRACT_NAME}: {contract_id}")

    cells, failed = Counter(), []
    for ep in todo:
        try:
            meta = load_meta(ep)
            path = target_path(meta, a.dataset_path)
            validate(ep)
            if not a.dry_run:
                upload_one(data, contract_id, ep, a.dataset_path)
            cells[meta.cell] += 1
            print(f"  {'ok' if a.dry_run else 'uploaded'} {meta.episode_id} ({meta.task}, {meta.num_steps} steps) → {path}")
        except Exception as e:  # keep going; report at the end
            failed.append(ep.name)
            print(f"  FAILED {ep.name}: {type(e).__name__}: {e}")
    print(f"\n{sum(cells.values())} {'validated' if a.dry_run else 'uploaded'}, {len(failed)} failed. Per cell: "
          + " ".join(f"{c}={cells[c]}" for c in s.CELLS if cells[c]))


if __name__ == "__main__":
    main()
