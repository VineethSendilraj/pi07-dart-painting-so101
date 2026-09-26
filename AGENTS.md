# AGENTS.md: handoff for coding agents

Read this first. It's the complete working context for this repo; details live in `docs/`.

## What we're building

Act-athon hackathon (actor labs, Mountain View, Sep 25–27 2026; **submissions close Sunday Sep 27, 9:00 am PT**; live demo on the real machine). An **SO-101 arm with an X-Shot Dartstorm blaster taped on plays tic-tac-toe** against a person:
- **Claude (`claude-opus-5-5`) picks the move** once per turn from an overhead photo and calls a `shoot(cell)` tool.
- **π0.7 (Physical Intelligence), fine-tuned on our teleop demos, aims and fires** at that cell at 30 Hz.

Judging: autonomy 30 (the model must drive the behavior, not a script), hardware hack 25, reliability 20, ambition 15, style 10.

Docs: `docs/plan.md` (the plan, start here) · `docs/notes.md` (status, IDs, open questions) · `docs/concepts.md` (how π0.7 and PI's data format work) · `docs/hardware.md` (arm, blaster, wiring) · `docs/pi-sdk-self-serve-guide.md` (PI's official guide) · `docs/img/wiring.svg`, `docs/img/vla.svg`.

## Current status (2026-09-26)

- **Works:** contract registered; episode format validates; fake-data and real-data fine-tunes succeed (~25 min each, mostly data prep); `record.py`, `upload.py --dry-run`, `robot_server.py` all pass in `--fake` mode; safety-rule tests pass.
- **Not yet run on hardware:** LeRobot arm/camera code, ESP32 firmware, real upload of our own recordings.
- **Blocker:** no fine-tune checkpoint ever gets a `gateway_url`, so **inference can't run**. PI must enable remote hosting for org `actathon_Ledger` (`3cda890e-6cf4-40b9-a400-b9fb3763902d`). Don't try to work around this by guessing endpoints. Fallback if it never comes: self-hosted π0.5 via LeRobot on a GPU box.
- **Claude brain untested:** the `anthropic` package isn't installed yet (ask the user before installing), needs `ANTHROPIC_API_KEY`.

## The task and data

- Board: 3×3 styrofoam grid, ~20 cm cells, columns **A B C**, rows **1 2 3**, each cell its own color. Robot's move = suction dart stuck in the cell; human's move = an O marker.
- Cameras: `wrist_camera` on the blaster parallel to the barrel; `front_camera` overhead top-right (never move it after recording starts). 640×480, 30 fps.
- Prompt per episode/inference: `"shoot A1"` … `"shoot C3"`. One model covers all 9 cells; no subtasks.
- **One row per step at 30 Hz** (contract `so101_blaster_v2`, ID `01a0de1a-01ee-75e2-bdc1-99d2b695b9c7`; no gripper, the blaster sits on a stand):
  - observation: both camera frames, follower joints [5]
  - action: leader joints [5], trigger [1]
- **Training profile** `so101-blaster-v2`: state = joints [5]; actions = joints [5] + trigger [1]. Order is fixed. (`so101_blaster_v1`, which had gripper topics, is deprecated; its test episodes and runs stay in the org.)
- **Trigger label:** 0 on every step, 1 for ~10 steps after each SHOOT press (intent, even when the blaster is unpowered). Keep trigger=1 frames well above 1% of all frames.
- Episodes: ~5–8 s (aim → shoot → end 0.7 s later), ~12–15 per cell (~110–135 total). Pilot first: A1, B2, C3 × 10.
- Limits: ≤32 state numbers, ≤32 action numbers, 1–16 cameras per row.

## Hardware and wiring (see docs/img/wiring.svg)

- Laptop → powered USB hub → follower arm (Seeed servo board), ESP32, wrist cam, front cam, leader arm (recording only).
- Blaster: master switch (≥5 A) in battery +; white joined to red (flywheels always on while powered). **The relay replaces the fire microswitch: blue → COM, red → NO, black → NC. Never wire a relay across the microswitch: that shorts the battery.**
- ESP32: pin 26 → relay IN, 25 ← SHOOT button, 33 ← DONE button (buttons pin → GND). No ARM switch, no LED.
- Serial (115200): laptop → `F` (fire), `H` (heartbeat). ESP32 → `SHOOT`, `TURN` (DONE), `FIRED`, `NO_HEARTBEAT`, `COOLDOWN`. Remote fire needs a heartbeat within 500 ms.
- Recording without darts: master switch off (relay clicks, nothing fires).

## Code map and commands

```
autobot/schema.py    contract, profile, topic names, cells, PI clients, ensure_contract()
autobot/episode.py   episode → MCAP shards + meta.json, local validation
autobot/hardware.py  Rig (LeRobot SO101 follower/leader + cameras), ESP32 serial, FakeRig/FakeESP32
autobot/policy.py    PiPolicy (hosted checkpoint via PolicyClient), FakePolicy
autobot/control.py   shoot(cell) loop, speed limit (≤3°/step), fire rule, move_to
autobot/keys.py      single-key terminal input
scripts/record.py    teleop recording → data/recordings/<session>/<episode_id>/
scripts/upload.py    validate → register → upload → finalize (idempotent)
scripts/train.py     contract | profile | finetune NAME | status | evals NAME | watch NAME
scripts/robot_server.py  shoot CELL | game --brain claude|random
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
