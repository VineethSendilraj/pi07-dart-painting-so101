# autobot-act

An SO-101 arm with a blaster plays tic-tac-toe. Claude picks the move; a fine-tuned π0.7 (Physical Intelligence) aims and fires. Act-athon, Sep 25–27, 2026.

**Status:** record → upload → fine-tune works end to end. Inference is blocked until PI enables hosting (no checkpoint gets a `gateway_url`).

![How the VLA works](docs/img/vla.svg)

![Wiring](docs/img/wiring.svg)

## Docs

| Doc | For |
| --- | --- |
| [docs/plan.md](docs/plan.md) | **the detailed plan**: board, cameras, wiring, data format, recording, game loop |
| [docs/pi-sdk-self-serve-guide.md](docs/pi-sdk-self-serve-guide.md) | **using PI's API**: contracts, episodes, fine-tunes, hosted inference |
| [docs/notes.md](docs/notes.md) | status, IDs, open questions, multi-laptop tips |
| [docs/concepts.md](docs/concepts.md) · [docs/hardware.md](docs/hardware.md) | how π0.7 works · arm, blaster, electronics |
| [AGENTS.md](AGENTS.md) | handoff for coding agents (code map, rules) |

## Setup

Get two files from Vineeth over a private channel (never commit them):

| File | Put it at |
| --- | --- |
| API key (`sk_…`, one line) | `~/.config/pi/api_key` |
| Registry service-account JSON (only for installing `pi-sdk`) | `~/.config/pi/service-account.json` |

```bash
python3.13 -m venv .venv
.venv/bin/pip install keyring keyrings.google-artifactregistry-auth
GOOGLE_APPLICATION_CREDENTIALS=~/.config/pi/service-account.json .venv/bin/pip install "pi-sdk[video,inference,lerobot]>=0.3.2" \
  --index-url https://us-east5-python.pkg.dev/pi-external-partners/pi-sdk/simple/ --extra-index-url https://pypi.org/simple
.venv/bin/pip install -e .
.venv/bin/python scripts/smoke_test.py     # all OK = key works
```

## Workflow

```bash
.venv/bin/python scripts/record.py --follower-port … --leader-port … --esp32-port … --cells A1 B2 C3 --per-cell 10
.venv/bin/python scripts/upload.py data/recordings/<session>
.venv/bin/python scripts/train.py finetune pilot-v1 --steps 1000 && .venv/bin/python scripts/train.py watch pilot-v1
.venv/bin/python scripts/robot_server.py game --finetune pilot-v1 --brain claude --follower-port … --esp32-port …
```
Add `--fake` to `record.py` or `robot_server.py` to run without hardware. Details and options: [docs/plan.md](docs/plan.md), `--help` on each script.
