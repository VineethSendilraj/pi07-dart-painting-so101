"""Contract, training profile, fine-tunes and their status: one CLI for the PI side.

  .venv/bin/python scripts/train.py contract                      # register the contract (idempotent)
  .venv/bin/python scripts/train.py profile                       # create the training profile (idempotent)
  .venv/bin/python scripts/train.py finetune pilot-v1 --steps 1000
  .venv/bin/python scripts/train.py finetune full-v2 --continue-from <checkpoint_id>
  .venv/bin/python scripts/train.py status                        # all fine-tunes + checkpoints + gateway_url
  .venv/bin/python scripts/train.py evals pilot-v1                # train/validation MAE
  .venv/bin/python scripts/train.py watch pilot-v1                # poll until trained (or --until-hosted)

Fine-tune IDs are saved to runs/finetunes/<name>.json.
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

from pi_sdk.finetuning import CheckpointPolicy, DatasetFilterInput, EpisodeFilter, StringSelection

from autobot import schema as s

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs" / "finetunes"


def _find(items, name):
    return next((x for x in items if x.name == name), None)


def ensure_profile(ft, contract_id: str) -> str:
    existing = _find(ft.list_training_profiles(page_size=100).training_profiles, s.PROFILE_NAME)
    if existing:
        return existing.training_profile_id
    return ft.create_training_profile(s.PROFILE_NAME, contract_id, s.training_profile()).training_profile_id


def run_record(name: str, ft) -> dict:
    """Local record if this laptop started the run, otherwise look it up by name (org-wide)."""
    path = RUNS / f"{name}.json"
    if path.exists():
        return json.loads(path.read_text())
    run = _find(ft.list_finetunes(page_size=100).finetunes, name)
    if run is None:
        raise SystemExit(f"no fine-tune named {name!r} in the org (see: train.py status)")
    return {"name": name, "run_id": run.run_id}


def cmd_contract(a, data, ft):
    print(f"{s.CONTRACT_NAME}: {s.ensure_contract(data)}")


def cmd_profile(a, data, ft):
    print(f"{s.PROFILE_NAME}: {ensure_profile(ft, s.ensure_contract(data))}")


def cmd_finetune(a, data, ft):
    contract_id = s.ensure_contract(data)
    if a.profile:  # an existing profile by name, e.g. so101-blaster-v2 (keep_idle_actions=False)
        found = _find(ft.list_training_profiles(page_size=100).training_profiles, a.profile)
        if found is None:
            raise SystemExit(f"no training profile named {a.profile!r}")
        profile_id = found.training_profile_id
    else:
        profile_id = ensure_profile(ft, contract_id)
    filters = []
    after = None
    if a.from_episode:  # recording times increase with the episode index, so "from this episode on" = recorded after it started - 1 s
        from datetime import timedelta
        after = data.get_episode(a.from_episode).episode_start - timedelta(seconds=1)
    for path in a.dataset_path:
        name = f"{a.name}-{path.replace('/', '-')}"
        f = ft.create_dataset_filter(
            name, EpisodeFilter(platform_contract_id=contract_id, source_dataset_path=path, success=True,
                                dagger_data_filter=a.human_only, recorded_after=after),
        )
        filters.append(DatasetFilterInput(f.dataset_filter_id))
    if a.focus_cells:  # a second filter over just these cells, sampled as --focus-share of every batch
        f = ft.create_dataset_filter(f"{a.name}-focus", EpisodeFilter(
            platform_contract_id=contract_id, source_dataset_path=a.dataset_path[0], success=True, dagger_data_filter=a.human_only,
            recorded_after=after,
            tasks=StringSelection(include=[s.task_for(c) for c in a.focus_cells])))
        filters = [DatasetFilterInput(x.dataset_filter_id, weight=(1 - a.focus_share) / len(filters)) for x in filters]
        filters.append(DatasetFilterInput(f.dataset_filter_id, weight=a.focus_share))
    policy = CheckpointPolicy("every_n_steps", every_n_steps=a.checkpoint_every, limit=3) if a.checkpoint_every else CheckpointPolicy("final_only")
    run = ft.create_finetune(
        a.name,
        filters,
        base_model=None if a.continue_from else a.base,
        continue_from_checkpoint_id=a.continue_from,
        num_steps=a.steps,
        batch_size=a.batch,
        checkpoint_policy=policy,
        training_profile_id=None if a.continue_from else profile_id,
    )
    RUNS.mkdir(parents=True, exist_ok=True)
    record = {"name": a.name, "run_id": run.run_id, "created": datetime.now().isoformat(), "dataset_paths": a.dataset_path,
              "base_model": a.base, "continue_from": a.continue_from, "num_steps": run.num_steps, "batch_size": run.batch_size,
              "contract_id": contract_id, "training_profile_id": profile_id,
              "human_only": a.human_only, "from_episode": a.from_episode, "focus_cells": a.focus_cells, "focus_share": a.focus_share if a.focus_cells else None}
    (RUNS / f"{a.name}.json").write_text(json.dumps(record, indent=2))
    print(f"fine-tune {a.name}: run {run.run_id} {run.state}, steps={run.num_steps}, batch={run.batch_size}")


def cmd_status(a, data, ft):
    runs = ft.list_finetunes(page_size=100).finetunes
    if a.name:
        runs = [r for r in runs if r.name in a.name]
    for r in runs:
        prog = f" {r.progress:.0%}" if r.progress is not None else ""
        print(f"{r.name:32} {r.state}{prog}  steps={r.num_steps}  run={r.run_id}" + (f"  reason={r.terminal_reason}" if r.terminal_reason else ""))
        for c in ft.list_checkpoints(r.run_id).checkpoints:
            print(f"    checkpoint {c.checkpoint_id} step={c.step} {c.state} gateway_url={c.gateway_url}")


def cmd_evals(a, data, ft):
    for m in ft.list_finetune_evaluations(run_record(a.name, ft)["run_id"]):
        val = f"{m.validation_mae:.3f}" if m.validation_mae is not None else "-"
        print(f"step {m.step:>6}  train_mae={m.train_mae:.3f}  validation_mae={val}")


def cmd_watch(a, data, ft):
    run_id, last = run_record(a.name, ft)["run_id"], None
    while True:
        r = ft.get_finetune(run_id)
        cps = ft.list_checkpoints(run_id).checkpoints
        now = (r.state, r.progress, [(c.state, c.gateway_url) for c in cps])
        if now != last:
            print(f"[{datetime.now():%H:%M:%S}] {r.state} progress={r.progress} checkpoints={now[2]}", flush=True)
            last = now
        if r.state in ("failed", "cancelled"):
            raise SystemExit(f"{a.name}: {r.state} {r.terminal_reason or ''}")
        final = max(cps, key=lambda c: c.step) if cps else None
        if r.state == "succeeded" and final is not None and final.gateway_url:  # the FINAL checkpoint is hosted
            print(f"final checkpoint {final.step} hosted: {final.gateway_url}")
            return
        if r.state == "succeeded" and cps and all(c.state == "available" for c in cps) and not a.until_hosted:
            print("trained; checkpoint available (no gateway_url yet: use --until-hosted to keep waiting)")
            return
        time.sleep(a.interval)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("contract")
    sub.add_parser("profile")
    f = sub.add_parser("finetune")
    f.add_argument("name", help="unique lowercase kebab-case, e.g. pilot-v1")
    f.add_argument("--dataset-path", action="append", help=f"source_dataset_path to train on (repeatable; default {s.DATASET_PATH})")
    f.add_argument("--base", default="pi07", help="base model: pi07 or pi07_cosmic")
    f.add_argument("--continue-from", help="checkpoint_id to continue training from (instead of --base)")
    f.add_argument("--steps", type=int, help="optimizer steps (omit for the service default)")
    f.add_argument("--batch", type=int, choices=[256, 512, 1024, 2048, 4096])
    f.add_argument("--checkpoint-every", type=int, help="keep up to 3 checkpoints, one every N steps")
    f.add_argument("--human-only", action="store_true",
                   help="dagger_data_filter: human-controlled steps only. NOTE: PI keeps only episodes whose human span covers "
                        "the whole take, so this can't trim tails; use the trimmed dataset (make_trimmed.py) instead")
    f.add_argument("--profile", help=f"existing training profile name (default {s.PROFILE_NAME}); "
                                     "so101-blaster-v2 = keep_idle_actions=False (skips still stretches, e.g. the post-shot tail)")
    f.add_argument("--from-episode", help="only episodes recorded from this one on, e.g. final-vector-0015 (red gun mark)")
    f.add_argument("--focus-cells", nargs="+", help="upweight these cells, e.g. the π logo: A2 B2 C2 D2 E2 B3 B4 B5 D3 D4 D5")
    f.add_argument("--focus-share", type=float, default=0.3, help="fraction of each batch drawn from --focus-cells")
    st = sub.add_parser("status")
    st.add_argument("name", nargs="*")
    sub.add_parser("evals").add_argument("name")
    w = sub.add_parser("watch")
    w.add_argument("name")
    w.add_argument("--interval", type=int, default=120)
    w.add_argument("--until-hosted", action="store_true", help="keep waiting after training until gateway_url appears")
    a = p.parse_args()
    if a.cmd == "finetune" and not a.dataset_path:
        a.dataset_path = [s.DATASET_PATH]

    data, ft = s.clients()
    globals()[f"cmd_{a.cmd}"](a, data, ft)


if __name__ == "__main__":
    main()
