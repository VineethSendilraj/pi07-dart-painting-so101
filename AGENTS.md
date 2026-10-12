# AGENTS.md: context for coding agents

Read this first; details live in `docs/`. Recording: [docs/data-collection.md](docs/data-collection.md). Training runs and the demo model: [docs/training-runs.md](docs/training-runs.md). Inference: [docs/inference.md](docs/inference.md).

## What this is

Built at the Act-athon (Actor Labs × Physical Intelligence, Mountain View, Sep 25–27 2026; 3rd place). An **SO-101 arm aims a motorized dart blaster** at a 5×5 board and draws pictures with darts:
- **Claude picks the cells** from camera snapshots of the board (e.g. "draw the PI logo") and retries shots that miss.
- **π0.7 (Physical Intelligence), fine-tuned on 116 teleop demos, aims and fires** at that cell at 30 Hz.

Docs: `docs/concepts.md` (how π0.7 and PI's data format work) · `docs/hardware.md` (arm, teleop, blaster, wiring) · `docs/img/wiring.svg`, `docs/img/vla.svg`.

## The task and data

- Board: **5×5** grid on foam, cells labeled inside A1…E5 (columns A–E, rows 1–5). Blaster on a tripod; the arm moves its handle (no gripper). Robot's move = a dart stuck in the foam; human's move = an O marker.
- Cameras: `wrist_camera` on the blaster parallel to the barrel; `front_camera` overhead top-right (never move it after recording starts). 640×480, 30 fps.
- Prompt per episode/inference: `"shoot A1"` … `"shoot E5"`. One model covers all 25 cells; no subtasks. Claude plays 5×5 tic-tac-toe, 4 in a row wins (`WIN_LENGTH` in `robot_server.py`).
- **One row per step at 30 Hz** (contract `so101_blaster_v2`, ID `01a0de1a-01ee-75e2-bdc1-99d2b695b9c7`; no gripper, the blaster sits on a stand):
  - observation: both camera frames, follower joints [5]
  - action: leader joints [5], trigger [1]
- **Training profile** `so101-blaster-v2-keepidle`: state = joints [5]; actions = joints [5] + trigger [1]. Order is fixed. (`so101_blaster_v1`, which had gripper topics, is deprecated; its test episodes and runs stay in the org.)
- **Trigger label:** 0 on every step, 1 for ~10 steps after each SHOOT press (intent, even when the blaster is unpowered). Keep trigger=1 frames well above 1% of all frames.
- Episodes: ~5–17 s (aim → shoot → end 2 s later). 116 takes in `data/recordings/final-vector` (Git LFS); public copy in LeRobot format via `scripts/export_lerobot.py`.
- Limits: ≤32 state numbers, ≤32 action numbers, 1–16 cameras per row.

## Hardware and wiring (see docs/img/wiring.svg)

- Laptop → powered USB hub → follower arm (Seeed servo board), ESP32, wrist cam, front cam, leader arm (recording only).
- Blaster: master switch (≥5 A) in battery +; white joined to red (flywheels always on while powered). **The relay replaces the fire microswitch: blue → COM, red → NO, black → NC. Never wire a relay across the microswitch: that shorts the battery.**
- ESP32 (act-athon fire_ctl 1.3.0): relay **GPIO13**, operator button **GPIO25**. Train image: **release** fires a 2-pulse burst, one `fired` event. No heartbeat. `rigd` owns the serial port; `record.py --rigd-url http://localhost:8080`.
- Vineeth's unused sketch (`firmware/blaster_esp32`, `ESP32` serial `F`/`H`) is not what is plugged in. Do not pass `--esp32-port` for this rig.
- Recording without darts: master switch off (relay clicks, nothing fires).

## Code map and commands

```
autobot/schema.py    contract, profile, topic names, cells, PI clients, ensure_contract()
autobot/episode.py   episode → MCAP shards + meta.json, local validation
autobot/hardware.py  Rig (LeRobot SO101 follower/leader + cameras), RigdESP32, leftover ESP32 serial
autobot/policy.py    PiPolicy (hosted checkpoint via PolicyClient), FakePolicy
autobot/control.py   shoot(cell) loop, speed limit (≤3°/step), fire rule, move_to
autobot/keys.py      single-key terminal input
scripts/record_web.py  recording web app (next cell, board setup, Ready → record → auto-save, Upload) :7870
scripts/record.py    teleop recording, terminal version → data/recordings/<session>/<episode_id>/
scripts/upload.py    validate → register → upload → finalize (idempotent)
scripts/train.py     contract | profile | finetune NAME | status | evals NAME | watch NAME
scripts/robot_server.py  draw PROMPT | shoot CELL | game --brain claude|random
scripts/install_calibration.py  installs calibration/ (ids follower_arm / leader_arm, the defaults)
scripts/inspect_episode.py  verify recorded episodes (contract, Hz, gaps, trigger block, upload state)
scripts/export_lerobot.py   recordings → LeRobot v3.0 dataset (parquet + mp4) for the Hugging Face Hub
scripts/camera_check.py     one frame per camera index (run in Terminal.app)
scripts/infer_test.py   hit a hosted checkpoint with synthetic frames (connection + output format)
scripts/smoke_test.py, scripts/watch_hosting.py
firmware/blaster_esp32/  ESP32 sketch
tests/test_control.py    safety-rule tests
```
Everything runs from the repo root with `.venv/bin/python …`; add `--fake` to run without hardware. Full commands: `README.md`.

## Rules for agents

- **Credentials never enter the repo or logs.** API key: `~/.config/pi/api_key` (or `PI_API_KEY`); registry service-account JSON: `~/.config/pi/service-account.json` (only for `pip install pi-sdk`). Scan for `sk_` / `BEGIN PRIVATE KEY` before committing.
- **Ask before creating PI resources** (contracts, profiles, filters, fine-tunes, episodes); they're permanent in the org. Contracts and profiles are immutable: to change one, bump the name (`_v2`) in `autobot/schema.py`.
- Never upload fake/test episodes to the real dataset path (`so101-blaster/v2`); delete test recordings under `data/recordings/` after tests.
- Don't change topic names, order, or `FPS` without a new contract + profile; recorded data would stop matching.
- Keep the safety rules (speed limit, fire rule, heartbeat) between π0.7 and the hardware.
- Install with plain pip from PI's registry (see README); `uv` can't use the registry login. Ask before adding dependencies.
- Don't commit or push unless asked. Small, surgical changes that match the existing style.
- Verify with `--fake` runs and `tests/test_control.py` after changes; say plainly what was and wasn't tested on hardware.
