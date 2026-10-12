"""Mark each take's post-shot tail as control_source "none", so a fine-tune with `train.py finetune --human-only`
(dagger_data_filter=True) trains on aim -> shot only, never on the still "already fired" frames after it.

  .venv/bin/python scripts/annotate_tail.py                          # dry run: show what would change
  .venv/bin/python scripts/annotate_tail.py --apply                  # back up current annotations, then patch
  .venv/bin/python scripts/annotate_tail.py --revert runs/annotations/<backup>.json

Takes end a fixed time after the fire-button release (record_web AFTER_SHOT_S, 2.0 s for final-vector) and the
trigger is 1 for the first 10 steps of that. Cutting at end - TAIL_S keeps the whole pulse at any rate the contract
allows (27-33 Hz) plus a few frames, and drops the rest. Only control_source spans change: fine-tunes without
dagger_data_filter ignore them, so runs already training are unaffected.
"""

import argparse
import json
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

from pi_sdk.data import EpisodeAnnotations, control_source

from autobot import schema as s

ROOT = Path(__file__).resolve().parents[1]
BACKUPS = ROOT / "runs" / "annotations"
TAIL_S = 1.5  # dropped from the end of every take (the tail after the pulse is ~1.67 s at 30 Hz)


def training_episodes(data, prefix: str) -> list[str]:
    ids, cur = [], None
    while True:
        r = data.list_episodes(recorded_after=datetime.now(UTC) - timedelta(days=14), cursor=cur)
        for e in r.episodes or []:
            m = e.episode.metadata
            if m.source_dataset_path == s.DATASET_PATH and m.success and e.episode_id.split(":")[-1].startswith(prefix):
                ids.append(e.episode_id)
        cur = r.next_cursor
        if not cur or not r.episodes:
            return sorted(ids)


def span_json(span) -> dict:
    return {"start": span.start_time.isoformat(), "end": span.end_time.isoformat(),
            "kind": type(span.kind).__name__, "value": getattr(span.kind, "value", None)}


def plan(data, ids: list[str]) -> tuple[list[dict], list[dict], list[str]]:
    """(backup rows, bulk items, skipped reasons). Refuses anything but one full-window human span."""
    backup, items, skipped = [], [], []
    for eid in ids:
        g = data.get_episode(eid)
        a = g.annotations
        spans = list(a.spans) if a else []
        name = a.episode_id if a else eid.split(":")[-1]
        backup.append({"episode_id": name, "attributes": {}, "spans": [span_json(x) for x in spans]})
        ok = (len(spans) == 1 and type(spans[0].kind).__name__ == "ControlSource" and spans[0].kind.value == "human"
              and spans[0].start_time == g.episode_start and spans[0].end_time == g.episode_end and not (a and a.attributes))
        dur = (g.episode_end - g.episode_start).total_seconds()
        if not ok:
            skipped.append(f"{name}: annotations are not one full-window human span, left alone")
            continue
        if dur < TAIL_S + 1.0:
            skipped.append(f"{name}: only {dur:.1f} s long, left alone")
            continue
        cut = g.episode_end - timedelta(seconds=TAIL_S)
        items.append({"episode_id": name, "annotations": EpisodeAnnotations(episode_id=name, attributes={}, spans=[
            control_source(start=g.episode_start, end=cut, value="human"),
            control_source(start=cut, end=g.episode_end, value="none"),
        ])})
    return backup, items, skipped


def send(data, items: list[dict]) -> list[str]:
    failed = []
    for i in range(0, len(items), 100):
        for r in data.bulk_update(items[i:i + 100]):
            if r.code not in ("ok", "OK", "success", ""):
                failed.append(f"{r.episode_id}: {r.code} {r.message or ''}")
    return failed


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--session", default="final-vector", help="episode id prefix")
    p.add_argument("--apply", action="store_true")
    p.add_argument("--revert", type=Path, help="restore the annotations saved in this backup file")
    a = p.parse_args()
    data, _ = s.clients()

    if a.revert:
        rows = json.loads(a.revert.read_text())
        items = [{"episode_id": r["episode_id"], "annotations": EpisodeAnnotations(episode_id=r["episode_id"], attributes={}, spans=[
            control_source(start=datetime.fromisoformat(x["start"]), end=datetime.fromisoformat(x["end"]), value=x["value"])
            for x in r["spans"]])} for r in rows]
        failed = send(data, items)
        print(f"reverted {len(items) - len(failed)}/{len(items)}", *failed, sep="\n")
        return

    ids = training_episodes(data, a.session)
    backup, items, skipped = plan(data, ids)
    print(f"{len(ids)} training episodes ({s.DATASET_PATH}, success, {a.session}*): {len(items)} to patch, {len(skipped)} skipped")
    print(*skipped, sep="\n")
    for it in items[:3]:
        print(" ", it["episode_id"], [(x.start_time.time(), x.end_time.time(), x.kind.value) for x in it["annotations"].spans])
    if not a.apply:
        print("dry run: pass --apply to back up and patch")
        return
    BACKUPS.mkdir(parents=True, exist_ok=True)
    path = BACKUPS / f"{a.session}-{datetime.now():%Y%m%d-%H%M%S}-backup.json"
    path.write_text(json.dumps(backup, indent=1))
    print(f"backup: {path}")
    failed = send(data, items)
    print(f"patched {len(items) - len(failed)}/{len(items)}", *failed, sep="\n")
    time.sleep(2)
    bad = [it["episode_id"] for it in items
           if [x.kind.value for x in data.get_episode(it["episode_id"]).annotations.spans] != ["human", "none"]]
    print("verified: all patched episodes read back as [human, none]" if not bad else f"NOT as expected: {bad}")


if __name__ == "__main__":
    main()
