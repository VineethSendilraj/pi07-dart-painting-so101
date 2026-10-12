# Data collection: recording blaster episodes

Everything an agent (or person) needs to record teleop episodes for the 5×5 blaster, verify them, get them into PI, and train. Read this before touching the rig. Background: [hardware.md](hardware.md), [../AGENTS.md](../AGENTS.md).

## What we're collecting

- **Task:** the arm aims the blaster (on its tripod) at one cell of the 5×5 board and fires once. Prompt = `"shoot <cell>"`, cells `A1`…`E5` (columns A–E left→right, rows 1–5 top→bottom). The same prompt string is used at inference.
- **Contract:** `so101_blaster_v2` (ID `01a0de1a-01ee-75e2-bdc1-99d2b695b9c7`), 30 Hz. One row per step:
  - observation: `front_camera` (rear camera, sees blaster + whole board), `wrist_camera` (on the blaster, above the barrels), follower joints F1–F5 (degrees)
  - action: leader joints L1–L5 (degrees), trigger (0, or 1 for 10 steps from the fire-button press)
  - no gripper. Don't change topics/FPS/camera names: that needs a new contract and invalidates recorded data.
- **Training profile:** `so101-blaster-v2-keepidle` (state = F1–F5, actions = L1–L5 + trigger; `keep_idle_actions=True` so the still frames around the shot aren't dropped).
- **Dataset path on PI:** `so101-blaster/v2` for real data only (test data goes to `test/...`).
- **One episode = one take:** start pose → aim → fire once → it ends by itself 0.7 s after the button release. ~3–8 s.

## How much, and what variety

| Phase | Cells | Per cell | Episodes |
| --- | --- | --- | --- |
| Pilot | `A1 E1 C3 A5 E5` | 10 | 50 |
| Rest | the other 20 | 10 | 200 |
| DAgger fixes | worst ~5 cells | 5–10 | ~40 |

Every cell ends with 10 (pilot cells aren't recorded twice). The app plans **passes**: one pass = every cell once under the same condition, cells shuffled, so the board and master switch only change between passes and stopping early stays balanced. Per cell (10): board **1 clean / 3 light (1–3 darts or O's on other cells) / 4 heavy (4–8) / 2 late game (9–14)**, **2 live** (master ON), **7 rest-pose starts / 3 from another cell**. Passes run clean → late.

**Board/darts:** the app counts live darts per cell and blocks Record with a red box: *COLLECT DARTS* after a random 3–7 live shots, *too many darts* for the pass's clutter level, *clean board needed*, or *pull the dart out of the target cell*. Each box has **Cleaned ✓** and **Ignore, leave as is**; which one you pressed is saved in the next take's notes (`board_action=cleaned|ignored|none`, plus `tracked_darts`).

## Hardware

| Device | Port / id |
| --- | --- |
| Follower arm | `/dev/tty.usbmodem5B7A0000841`, calibration id `follower_arm` |
| Leader arm | `/dev/tty.usbmodem5B7A0001561`, calibration id `leader_arm` |
| ESP32 (fire_ctl 1.3.0) | `/dev/cu.usbserial-0001`, owned by `rigd`; relay IN = GPIO 13; fire button NO → GPIO 25, C → GND |
| Rear camera | → `front_camera`. OpenCV index **changes on replug** (seen as 0 and as 1) |
| Gun camera | → `wrist_camera`, the other index |

The arm serials are fixed per device, so those ports don't change; the camera indices do. Firmware: one button **release** = one shot (a burst of 2 relay pulses) → rigd counts one release. The same button is the human's **DONE** button at game time (rigd test mode).

## One-time setup per laptop

```bash
# 1. this repo (pi07-dart-painting-so101): README Setup covers the venv, pi-sdk and the two credential files
cd ~/dev/Actathon
.venv/bin/pip install 'lerobot[feetech]==0.4.4'     # servo driver for the arms
.venv/bin/python scripts/install_calibration.py    # arm calibration → follower_arm / leader_arm
.venv/bin/python scripts/smoke_test.py             # PI API key works (all OK)

# 2. rigd, from Yuvraj's repo (owns the ESP32)
git clone https://github.com/yuviji/act-athon ~/dev/act-athon
cd ~/dev/act-athon && python3 -m venv .venv && .venv/bin/pip install -e ".[rigd]"
```
Camera access only works from **Terminal.app** (macOS won't grant it to the Claude app): run everything that opens cameras there, and click Allow the first time.

## Before every session (5 min)

1. `cd ~/dev/Actathon && git pull` (teammates' fixes; calibration changes → rerun `install_calibration.py`).
2. **rigd** (Terminal tab 1, leave running; scoreboard http://localhost:8080):
   ```bash
   cd ~/dev/act-athon && .venv/bin/rigd --port /dev/cu.usbserial-0001
   ```
   **Button test:** `curl localhost:8080/fire_button`, press + release the button once (master switch OFF), repeat: `releases` must go up by exactly 1.
3. **Cameras** (Terminal.app): `.venv/bin/python scripts/camera_check.py` → look at `data/camera_check/cam*.jpg` and note which index is the rear camera and which is the gun camera.
4. **Framing** (with the blaster aimed at C3):
   - rear camera: the whole board visible and clear of the blaster; no people/hands in frame.
   - gun camera: barrel tips in the bottom ~15–20% of the image; the aimed-at cell just above the tips, a bit below the image center; neighbors visible.
   - Once recording starts, don't move either camera, the tripod or the board (tape their positions).
5. Board labels readable, lights in the same place as last session, laptop plugged in.

## Recording (the web app)

Terminal.app, from the repo root (swap `--front-cam/--wrist-cam` if step 3 said so):
```bash
cd ~/dev/Actathon && caffeinate -i .venv/bin/python scripts/record_web.py \
  --follower-port /dev/tty.usbmodem5B7A0000841 --leader-port /dev/tty.usbmodem5B7A0001561 \
  --rigd-url http://localhost:8080 --front-cam 0 --wrist-cam 1 \
  --session <yourname>-pilot --cells A1 E1 C3 A5 E5 --per-cell 10 --auto-upload
```
Opens http://127.0.0.1:7870. Rest run: `--cells A2 A3 A4 B1 B2 B3 B4 B5 C1 C2 C4 C5 D1 D2 D3 D4 D5 E2 E3 E4 --per-cell 10`. `caffeinate` keeps the Mac awake. When it starts, the follower goes straight to the leader's pose like `lerobot-teleoperate` (no per-step cap): keep clear.

**First time:** check both previews and press **Cameras look right ✓** (`front_camera` = rear view). Put the arm where every game turn should start and press **Set rest pose = current** (writes `calibration/rest_pose.json`: commit + push it; the game returns there between turns).

**Per episode** (the page tells you each step):
1. Board as shown; master switch as the banner says (**red MASTER ON = live dart**, otherwise OFF = dry: relay clicks, label still recorded).
2. Start pose: rest (the page shows ✓ when within 3°) or the named cell.
3. **● Record** (`R`) → aim at the target → hold still ½ s → **press + release the fire button once** → keep still. The take **ends by itself 2 s later** (change it under More options → End take; saved per session) and saves (validated against the contract).
4. A **review panel** opens: the take read back from the saved file, both cameras side by side with a scrubber/Play, and a table per step of F1–F5 (follower = observation), L1–L5 (leader = action) and trigger (trigger rows highlighted). **✓ Good** (`G`) accepts + uploads; **✗ Bad** (`B`) deletes it, same cell again (e.g. a live dart missed). Pressing Record for the next take counts as Good.

Other buttons: **Discard take** (`X`) during a take; **Skip cell**; **Upload** retries anything not uploaded; **⏸ Pause teleop** stops the arms (page + previews stay up), Resume continues.

**Not saved (same cell again):** Discard, a second press in one take, < 27 Hz or a gap > 250 ms, a servo/camera/rigd error during the take. A take with no shot keeps recording until you fire or discard.

**Saved state / resume:** progress (`schedule.json`), dart tracking (`board.json`) and uploads (`uploaded.json` per episode) are written after every take. Ctrl-C and rerun **the same command** to continue where you stopped (the default session name is `<user>-<cells>x<per-cell>`; different arguments on an existing session are refused). A take still waiting for Good/Bad when you quit counts as Good. Uploads retry 5× (30 s apart); leftovers upload on the next start.

## Verify the data (structure + sync)

```bash
# structure: contract validation, rate, gaps, trigger = exactly 1 block of 10 steps, upload state
.venv/bin/python scripts/inspect_episode.py data/recordings/<session>
# anything not uploaded yet (e.g. after a crash or network drop): idempotent, safe to rerun
.venv/bin/python scripts/upload.py data/recordings/<session>
```
Every line should read `OK … valid · ~30 Hz · trigger 10 steps in 1 block(s) · so101-blaster/v2`. PI side: partner.pi-fleet.com → Episodes shows each upload (video, task, Pass). Recordings stay local in `data/` (git-ignored) until uploaded; once on PI, anyone on the team can train on them.

## Train

```bash
.venv/bin/python scripts/train.py finetune pilot-v1 --steps 1500    # profile + dataset filter (success=True, so101-blaster/v2) + fine-tune
.venv/bin/python scripts/train.py watch pilot-v1                    # until trained (~25 min, mostly data prep)
.venv/bin/python scripts/train.py evals pilot-v1                    # train MAE
.venv/bin/python scripts/train.py status                            # every fine-tune in the org + checkpoint + gateway_url
.venv/bin/python scripts/infer_test.py pilot-v1                     # hosted endpoint answers, output shapes, latency
```
Full run: `train.py finetune full-v1 --continue-from <pilot checkpoint_id>` (IDs from `train.py status`). Judge models by hits per cell on the robot, not MAE. Keep the laptop awake for the first inference connection (10–60 s).

**Game:** `robot_server.py --rigd-url http://localhost:8080 --follower-port … --finetune NAME game --brain claude`: rigd goes to test mode (the button = DONE, never fires), π0.7 fires via `/fire`, and the arm returns to `calibration/rest_pose.json` between turns.

## Don'ts

- Don't run `lerobot-teleoperate`, `camera_check.py` or a serial monitor while recording: one program per device (rigd owns the ESP32, record_web owns the arms + cameras).
- Don't upload fake/test data to `so101-blaster/v2` (`--fake` forces `test/record-web-fake`).
- Don't recalibrate the arms mid-collection (recorded angles would stop matching).
- Don't move the cameras, tripod or board between sessions.
- Don't delete PI resources or change the contract/profile without asking.

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `waiting for rigd…` then exit | start rigd first; `curl localhost:8080/fire_button` |
| Fire button does nothing (take never shows shot ✓) | `releases` in `curl localhost:8080/fire_button` stays 0 → the button isn't reaching the ESP32: NO → GPIO 25, C → GND |
| Cameras `not authorized` / black previews | run from Terminal.app and allow Camera; check indices with `camera_check.py` |
| Previews under the wrong labels | swap `--front-cam` / `--wrist-cam` |
| A follower joint lags the leader | same check as `lerobot-teleoperate`: blocked by the blaster/tripod, or the servo overload-tripped (power-cycle the arm). Recording no longer clamps goals to 4°/step. |
| `NOT saved: … Hz` | loop too slow: fewer devices on the hub, close camera apps |
| `arm/camera read failed` | reseat the arm USB/power; the take is dropped automatically |
| `UPLOAD FAILED` | network/API; press Upload or run `upload.py` later (idempotent) |
| Test without hardware | `--fake` (fake arm/cameras), optionally with `rigd --fake` + `--rigd-url`; `POST /test/shoot` simulates the button |
