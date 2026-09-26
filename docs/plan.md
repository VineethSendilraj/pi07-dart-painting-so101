# Plan: tic-tac-toe blaster

Blaster on the SO-101 plays tic-tac-toe against a person. Claude picks the move; π0.7 aims and fires. Background: [concepts.md](concepts.md), [hardware.md](hardware.md), status: [notes.md](notes.md).

## Board

- 3×3 grid on styrofoam, ~20 cm cells. **A B C** across the top, **1 2 3** down the left.
- Each cell a different solid color (colored printer paper), big labels.
- Mark the board's spot with **black tape**. Shift it a few cm between some episodes so the model has to find cells visually.
- Robot's move = its **suction dart** stuck in the cell. Human's move = an **O marker**.
- Record some episodes with darts and O's already on the board (what the model sees mid-game).

## Cameras

| Name | Where |
| --- | --- |
| `wrist_camera` | on the blaster, parallel to the barrel |
| `front_camera` | overhead, top right, clamped: sees arm, blaster, board |

640×480, 30 fps, MJPEG. Never move `front_camera` after recording starts.

## Wiring

```
laptop ──USB-C── powered USB hub
          ├─ 1: follower arm (Seeed servo board)
          ├─ 2: ESP32 (1 relay + DONE and SHOOT buttons)
          ├─ 3: wrist_camera
          ├─ 4: front_camera
          └─ 5: leader arm (plug in only for recording)
```
![Wiring](img/wiring.svg)

- **Blaster circuit** (measured, see [hardware.md](hardware.md#automating-the-trigger-correct-wiring)):
  - **Master switch (≥5 A) in series** with battery +. Off = everything dead, including the flywheels. It's both the rev switch and the kill switch.
  - White joined to red: flywheels always spin while powered.
  - **The relay replaces the fire microswitch** (not parallel, not in series): blue → COM, red → NO, black → NC. Relay off = pusher braked; relay click = one dart. ⚠️ Wiring it *across* the microswitch shorts the battery.
- **Buttons only connect to the ESP32** (pin → GND), never to the blaster circuit. The ESP32 decides whether to click the relay.
- ESP32 pins: **26 → relay IN, 25 ← SHOOT button, 33 ← DONE button** (buttons go pin → GND). One relay. No ARM switch, no LED.
- Serial: laptop → `F` (fire), heartbeat. ESP32 → `SHOOT` (button pressed), `FIRED`, `TURN` (DONE pressed). Firmware stops firing if no heartbeat for ~500 ms. Teammate's `firmware/fire_ctl/` (yuviji/act-athon) is a good base.
- Laptop needs internet for π0.7 (hotspot as backup).

## Model (contract `so101_blaster_v1`)

- **No gripper:** the blaster is taped to the arm permanently. The contract still has gripper topics (record it, or a constant); the **training profile leaves it out**: state = joints [5], actions = joints [5] + trigger [1]. No new contract needed.
- In: joints [5] + both cameras. Out per step: joints → arm, trigger → ESP32.
- Task string: `"shoot A1"` … `"shoot C3"`. One model, no subtasks.

## How the VLA works

![How the VLA works](img/vla.svg)

- **30 Hz = 30 steps per second.** Each step is one row of data; one **episode** is one teleop recording (both camera videos + the arm/trigger rows), ~5–8 s ≈ 150–240 rows.
- **One row (the contract, `so101_blaster_v1`):**

  | Column | Size | Role | Used by the model? |
  | --- | --- | --- | --- |
  | `system/step` (timestamp) | — | row header, links everything in the row | — |
  | `front_camera`, `wrist_camera` frames | 2 images | observation | yes (input) |
  | follower joints (measured) | 5 | observation | yes (state input) |
  | follower gripper (measured) | 1 | observation | no (blaster taped on) |
  | leader joints (commanded) | 5 | action | yes (output) |
  | leader gripper (commanded) | 1 | action | no |
  | trigger (SHOOT button, 0/1) | 1 | action | yes (output) |

  There's no "observed gun" column: the blaster has no sensor, so the trigger is action-only.
- **Limits:** up to **32 state numbers** and, separately, up to **32 action numbers** per row, plus 1–16 cameras. We use 5 in, 6 out.
- **Output:** each call returns an **action chunk**: rows = future time steps (π0.7 plans ~50; we play 10, then ask again), columns = joints [5] → arm USB, trigger [1] → ESP32.
- **Prompt:** `"shoot B2"` (overall task and subtask are the same string).

## Fine-tuning API (PI)

Two services plus a hosted model:
1. **Data** (`api.pi-fleet.com/v1/data`): register the contract, then per episode: register → upload shards → finalize (PI validates it against the contract). `scripts/upload.py`.
2. **Fine-tuning** (`/v1/finetuning`): training profile (which columns the model uses) → dataset filter (which episodes, `success=True`) → fine-tune → poll status / MAE → checkpoint. `scripts/train.py`.
3. **Inference:** a checkpoint's `gateway_url` (websocket). `scripts/robot_server.py`. Currently blocked: no checkpoint gets a `gateway_url`.

Fine-tunes can run **in parallel**: e.g. a pilot on 20–50 episodes while recording the rest, then a full run (or `--continue-from` the pilot). A run takes ~25 min, mostly data prep.

## Recording (`scripts/record.py` → `scripts/upload.py` → `scripts/train.py`)

- Aim with the leader arm, press SHOOT (off camera). Every press is **logged as a shot** (trigger = 1 for ~10 steps) and clicks the relay.
  - **Master switch off (most demos):** the relay clicks with no power behind it, so no dart leaves. Nothing to collect.
  - **Master switch on (a few demos):** real shots, so recordings include real darts on the board.
- Episode: reset with the leader → SPACE → aim → SHOOT → auto-ends 0.7 s later (~5–8 s total). Keys: `e` end, `x` redo, `q` quit.
- ~12–15 per cell across the **9 cells** → ~110–135 episodes (~70 min). Upload with `success=True` + full-length `control_source="human"` span.
- Pilot first: A1, B2, C3 × 10 → short fine-tune → check.

## Game loop (`scripts/robot_server.py` + Claude)

```
human places O → presses DONE → Claude (claude-opus-5-5) sees board → shoot(cell)
shoot(cell): π0.7 loop at 30 Hz, "shoot B2"
   joints → speed limit (≤3° per step) → arm
   fire rule: trigger > 0.5 for 3 steps in a row + arm stopped + not fired yet this turn → 'F'
   ends on FIRED (or 10 s timeout) → scripted return to rest
Claude checks the dart landed → screen shows "your turn"
```
- Claude runs **once per turn** (one overhead photo), not continuously: it's too slow for the 30 Hz loop, and the board only changes between turns. π0.7 does the continuous part.
- One Python process owns all USB devices. Spacebar = backup done button.
- **Safety rules** (our code, between π0.7 and the hardware): the fire rule stops a one-step flicker or a mid-swing shot from firing, and allows one dart per turn; the speed limit turns a bad model output into slow, correctable motion instead of a lurch.

## Open

- **Hosting (blocker):** 4 fine-tunes succeeded, all checkpoints `available`, none has a `gateway_url` (oldest since 01:22 PT). PI must enable remote hosting for `actathon_Ledger`. Check with the API one-liner in [notes.md](notes.md).
- Blaster weight at full reach (teammate assumed too heavy and planned a tripod + arm-pushes-handle mount).
- Fallback if hosting never comes: self-hosted π0.5 via LeRobot (`lerobot/pi05_base`) on a GPU box. Needs LeRobot v3 data.
- Check the 9 cell colors are still distinguishable at the model's ~224 px input size.
- Cameras on site?
