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
Diagram: [img/wiring.svg](img/wiring.svg).

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

## Recording (`scripts/record.py`)

- Aim with the leader arm, press SHOOT (off camera). Every press is **logged as a shot** (trigger = 1 for ~10 steps) and clicks the relay.
  - **Master switch off (most demos):** the relay clicks with no power behind it, so no dart leaves. Nothing to collect.
  - **Master switch on (a few demos):** real shots, so recordings include real darts on the board.
- Episode: varied start pose → aim → fire → end ~0.5 s later (~5–8 s total).
- ~12–15 per cell → ~110–135 episodes (~70 min). Upload with `success=True` + full-length `control_source="human"` span.
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
- One Python process owns all USB devices. Spacebar = backup done button.
- **Safety rules** (our code, between π0.7 and the hardware): the fire rule stops a one-step flicker or a mid-swing shot from firing, and allows one dart per turn; the speed limit turns a bad model output into slow, correctable motion instead of a lurch.

## Open

- **Hosting:** no checkpoint has a `gateway_url` yet. PI must enable it for `actathon_Ledger`.
- Blaster weight at full reach (teammate assumed too heavy and planned a tripod + arm-pushes-handle mount).
- Fallback if hosting never comes: self-hosted π0.5 via LeRobot (`lerobot/pi05_base`) on a GPU box. Needs LeRobot v3 data.
- Check the 9 cell colors are still distinguishable at the model's ~224 px input size.
- Cameras on site?
