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
laptop ── powered USB hub
          ├─ 1: follower arm (Seeed servo board)
          ├─ 2: ESP32 (relay, buttons, LED)
          ├─ 3: wrist_camera
          ├─ 4: front_camera
          └─ 5: leader arm (plug in only for recording)
```
- Blaster: arm toggle switch across the rev strips (rev always on when armed). Relay COM/NO across the fire switch. Check which wires are which with a multimeter first.
- ESP32 pins: 26 relay, 27 arm switch, 25 fire button (recording), 33 done button, 32 turn LED.
- Serial: laptop → `F` (fire), `L1`/`L0` (LED). ESP32 → `FIRED`, `DISARMED`, `TURN`.
- Laptop needs internet for π0.7 (hotspot as backup).

## Model (contract `so101_blaster_v1`)

- **No gripper:** the blaster is taped to the arm permanently. The contract still has gripper topics (record it, or a constant); the **training profile leaves it out**: state = joints [5], actions = joints [5] + trigger [1]. No new contract needed.
- In: joints [5] + both cameras. Out per step: joints → arm, trigger → ESP32.
- Task string: `"shoot A1"` … `"shoot C3"`. One model, no subtasks.

## Recording (`scripts/record.py`)

- Aim with the leader arm, fire with the button (off camera). ESP32 fires and sends `FIRED`; trigger = 1 for ~10 steps.
- Episode: varied start pose → aim → fire → end ~0.5 s later (~5–8 s total).
- ~12–15 per cell → ~110–135 episodes (~70 min). Upload with `success=True` + full-length `control_source="human"` span.
- Pilot first: A1, B2, C3 × 10 → short fine-tune → check.

## Game loop (`scripts/robot_server.py` + Claude)

```
human places O → presses DONE → Claude (claude-opus-5-5) sees board → shoot(cell)
shoot(cell): π0.7 loop at 30 Hz, "shoot B2"
   joints → arm ; trigger > 0.5 (edge, armed, cooldown) → 'F'
   ends on FIRED (or 10 s timeout) → scripted return to rest
Claude checks the dart landed → LED on: human's turn
```
- One Python process owns all USB devices. Spacebar = backup done button.

## Open

- **Hosting:** no checkpoint has a `gateway_url` yet. PI must enable it for `actathon_Ledger`.
- Blaster weight at full reach.
- Cameras on site?
