# autobot-act

An SO-101 arm with a blaster plays tic-tac-toe. Claude picks the move; a fine-tuned π0.7 (Physical Intelligence) aims and fires. Built for the act-athon, Sep 25–27, 2026.

Start with [docs/plan.md](docs/plan.md).

![Wiring](docs/img/wiring.svg)

## Status

- Pipeline works end to end up to a checkpoint: contract → episodes → training profile → fine-tune (`pi07`). 4 test runs succeeded (fake and real SO-101 data).
- **Blocked on inference:** every checkpoint is `available` but has no `gateway_url`; PI needs to enable remote hosting for org `actathon_Ledger`. Details: [docs/notes.md](docs/notes.md).

## Setup (each teammate)

You need **two credential files** from Vineeth (send them over a private channel, never commit them):

| File | Put it at | Used for |
| --- | --- | --- |
| Org API key (one line, starts with `sk_`) | `~/.config/pi/api_key` | every PI API call: upload, training, inference. The SDK adds the `Authorization` header itself. |
| Registry service-account JSON | `~/.config/pi/service-account.json` | only `pip install pi-sdk` from PI's private registry |

```bash
mkdir -p ~/.config/pi && chmod 700 ~/.config/pi
# save the two files there, then:
chmod 600 ~/.config/pi/api_key ~/.config/pi/service-account.json
```
(Alternatively `export PI_API_KEY=sk_...` instead of the api_key file.)

Install:
```bash
python3.13 -m venv .venv
.venv/bin/pip install keyring keyrings.google-artifactregistry-auth
GOOGLE_APPLICATION_CREDENTIALS=~/.config/pi/service-account.json .venv/bin/pip install "pi-sdk[video,inference,lerobot]>=0.3.2" \
  --index-url https://us-east5-python.pkg.dev/pi-external-partners/pi-sdk/simple/ --extra-index-url https://pypi.org/simple
.venv/bin/pip install -e .
.venv/bin/python scripts/smoke_test.py     # all OK = key works
```
Use plain pip, not `uv` (uv can't use the registry's keyring login by default).

## Workflow

```bash
# 1. Record (teleop with the leader arm, SHOOT button fires). Saves to data/recordings/<session>/
.venv/bin/python scripts/record.py --follower-port /dev/tty.usbmodemA --leader-port /dev/tty.usbmodemB \
    --esp32-port /dev/cu.usbserial-0001 --front-cam 0 --wrist-cam 1 --cells A1 B2 C3 --per-cell 10
#    no hardware? add --fake --auto

# 2. Upload (validates, registers, uploads, finalizes; safe to re-run)
.venv/bin/python scripts/upload.py data/recordings/<session>

# 3. Train and follow it
.venv/bin/python scripts/train.py finetune pilot-v1 --steps 1000
.venv/bin/python scripts/train.py watch pilot-v1
.venv/bin/python scripts/train.py status
```
Find ports with `lerobot-find-port`; calibrate both arms first with `lerobot-calibrate` (ids `follower` / `leader`).

Working from several laptops:
- Record on the laptop the arm is plugged into. Its calibration lives in `~/.cache/huggingface/lerobot/calibration/`; copy that folder if you switch laptops, or recorded angles won't match.
- Give each session a distinct name: `--session <yourname>-0926a`.
- Recordings stay local (`data/` isn't in git) until `upload.py`; after that anyone can train on them.
- Fine-tunes are org-wide: `status`, `evals NAME`, `watch NAME` work from any laptop.

## Layout

```
autobot/                 shared library
  schema.py              contract, training profile, topic names, cells, PI clients
  episode.py             episode → MCAP shards on disk, validation
  hardware.py            SO-101 arms + cameras (LeRobot), ESP32 serial, fake versions
scripts/
  record.py              teleop recording
  upload.py              push episodes to PI
  train.py               contract / profile / finetune / status / evals / watch
  smoke_test.py          check the API key
  watch_hosting.py       poll every checkpoint for a gateway_url
firmware/blaster_esp32/  ESP32 sketch: 1 relay, SHOOT + DONE buttons
data/                    recordings, shards, downloads (git-ignored)
runs/                    IDs of past test runs and poll logs
docs/                    plan, notes/status, concepts, hardware, PI SDK guide
```
