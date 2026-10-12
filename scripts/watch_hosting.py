"""Poll every fine-tune's checkpoints until one gets a gateway_url, then connect to it.

Logs changes to runs/hosting_watch.log. On the first gateway_url it prints the server's
metadata (the exact inference format) and exits.

  .venv/bin/python -u scripts/watch_hosting.py [interval_s] [max_hours]
"""

import sys
import time
from datetime import datetime

from autobot import schema as s

interval = int(sys.argv[1]) if len(sys.argv) > 1 else 180
deadline = time.time() + 3600 * float(sys.argv[2] if len(sys.argv) > 2 else 6)
_, ft = s.clients()


def snapshot() -> dict:
    out = {}
    for r in ft.list_finetunes().finetunes:
        for c in ft.list_checkpoints(r.run_id).checkpoints:
            out[c.checkpoint_id] = (r.name, r.state, c.state, c.gateway_url)
    return out


def connect(name: str, url: str) -> None:
    from pi_sdk.inference import ClientConfig, PolicyClient

    print(f"connecting to {name} at {url} (first connect can take minutes)")
    with PolicyClient(ClientConfig(gateway_url=url, api_key=s.api_key())) as policy:
        print(f"server metadata: {policy.server_metadata}")


prev = {}
while time.time() < deadline:
    now = datetime.now().strftime("%H:%M:%S")
    try:
        cur = snapshot()
    except Exception as e:
        print(f"[{now}] poll error: {e}")
        time.sleep(interval)
        continue
    for cid, v in cur.items():
        if prev.get(cid) != v:
            print(f"[{now}] {v[0]}: run={v[1]} checkpoint={cid[:8]} {v[2]} gateway_url={v[3]}")
    prev = cur
    hosted = [(v[0], v[3]) for v in cur.values() if v[3]]
    if hosted:
        print(f"[{now}] HOSTED: {hosted}")
        try:
            connect(*hosted[0])
        except Exception as e:
            print(f"connect failed: {type(e).__name__}: {e}")
        break
    time.sleep(interval)
else:
    print(f"[{datetime.now():%H:%M:%S}] gave up: no gateway_url before deadline")
