# Hardware: SO-101, teleop, cameras, blaster

Status and plans live in [notes.md](notes.md); model and data concepts in [concepts.md](concepts.md).

## SO-101 (Seeed kit)

- **Follower:** 6 × Feetech STS3215 at 1:345 gearing (Pro kit 12 V, standard kit 5 V). Joints: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper.
- **Leader:** mixed gearing (1:191 / 1:345 / 1:147) so it's light to move by hand; ends in a handle + trigger instead of a claw.
- Each arm has its own USB controller board. **The arms aren't connected to each other**: both plug into the laptop, which relays every movement. Place them side by side (USB reach, operator can see the follower), with the leader and operator out of camera view.
- STS3215 servos report their own angle (0–4095 ticks = 360°) and have a built-in position controller.
- Setup: `lerobot-find-port` → `lerobot-setup-motors` (if new) → `lerobot-calibrate` both arms → `lerobot-teleoperate` ~10 min → ~5 test recordings.
- **LeRobot 0.4.x notes** (teammate verified vs 0.4.4 source): SO-101 classes moved to `lerobot.robots.so_follower.SOFollower` / `lerobot.teleoperators.so_leader.SOLeader` (calibration folders `so_follower/`, `so_leader/`). Pin `transformers>=4.57.1,<5` (5.x breaks). Datasets are format v3.0 (openpi's pinned LeRobot only reads v2.x). Packages named `lerobot_teleoperator_*` auto-load as plugins, which is how a custom teleoperator can add a fire-button action to stock `lerobot-record`.
- **Back up calibration** (`~/.cache/huggingface/lerobot/calibration/`): recalibrating invalidates recorded data.
- `lerobot-record` controls: → end episode early, ← redo it, Esc stop and save; reset window between episodes; `--display_data=true` live viewer. macOS terminal needs Accessibility permission. Check `--help` for flag names.

## How teleop works

- **Leader: torque off.** You move it; its servos act only as angle sensors. **Follower: torque on**, copying the leader's angles. Same servo type; torque on/off is the only difference (which is what makes the DAgger handoff trick work).
- Recording: `you move leader → laptop (lerobot-record) → follower copies`, and the laptop saves cameras + follower readings + leader positions each step. Keyboard only for session control.
- After training: `cameras + follower readings → laptop → π0.7 → action chunk → laptop → follower`. The leader is unused (except for DAgger takeovers).
- Two arms for recording, **one (the follower) for running the model**.
- **The leader is the controller, shaped like the robot**: bend its elbow and the follower's elbow bends; no mapping to learn.

### Why not move the follower by hand (kinesthetic teaching)?

Possible, but your hands end up in the camera images (the model never sees hands at test time), the 1:345 follower is stiff to back-drive, you can't work the gripper easily, there's no clean command signal, and the arm sags when released. Kinesthetic teaching is mostly used for camera-free "teach and repeat"; for vision models, people mask hands out or hide them. Leader-follower avoids the problem entirely.

### Other teleop methods (industry)

| Method | Used by |
| --- | --- |
| Leader-follower (puppeteering) | ALOHA, GELLO, SO-100/101 |
| VR controllers (hand pose → IK) | DROID (Meta Quest + Franka), humanoids |
| VR / motion-capture suits and gloves | humanoids (e.g. reported for Tesla Optimus) |
| Handheld gripper tool, no robot | UMI (Stanford) |
| Kinesthetic (hand-guide mode) | cobots (Franka, UR), simple motions |
| 3D mouse (SpaceMouse) / gamepad | common in research labs for single arms |
| Human video | pretraining; not enough alone for precise control |

Controllers command the hand position (IK solves joints) and usually velocity, not position, which makes demos slower and stop-start. The SO-101 has only 5 joints + gripper, so IK can't reach every pose. The leader arm is the right tool here.

### Two arms at test time?

- The leader *can* be driven as an arm, but it has no real gripper, weaker servos, and using it removes your recording/DAgger device. Not worth it.
- A single-arm model outputs 6 numbers; it can't drive two arms. Real two-arm tasks need a two-arm contract/profile and two-arm demos from **2 leaders + 2 followers** (~1.5–2× the episodes).
- Cheaper alternative: the same single-arm checkpoint running independently on two identical followers (no coordination).

## Cameras

- **None in the kit.** Mounts in TheRobotStudio/SO-ARM100 `Optional/`: 32×32 UVC wrist mounts (SO-101 "Hex Nut" variant, integrated, plug-on), RealSense D405/D435, Vinmooog webcam, overhead mounts for a webcam or 32×32 module.
- **Plan:** wrist 32×32 UVC + a rigid front/top webcam, 640×480 at 30 fps. Never move the fixed camera; keep names and order identical everywhere (a LeRobot issue reports worse results when camera order changes).
- The event hardware list doesn't include cameras. Ask the organizers or bring two USB webcams.

## Blaster: X-Shot Insanity Motorized Dartstorm

### What's inside (Blaster Hub teardown review)

- Two dual-shaft 280 motors spin flywheels. The **rev trigger just pushes two metal strips together** (a leaf switch powering the flywheels while held).
- The **fire trigger** is a real switch plus an orange ratchet lever; it runs the motorized drum feed that pushes darts into the flywheels.
- 8 × AA (~12 V) through a 3 A PTC fuse; battery tray on a small 2-pin plug. Two 40-dart drums, ships with a tripod.
- No public wiring diagram was found.

### The 4 wires (measured by teammate, yuviji/act-athon `docs/02-hardware/wiring.md`)

All wires meet on a junction board (`36918 PCBAX-003R`) with **no controller**: the trigger switches carry motor current directly.

| Wire | What it is |
| --- | --- |
| **red** | BAT+ (after PTC fuse F1, `JK 300`, ~3 A hold) + the rev blade + fire microswitch pin 3 **NO** |
| **white** | rev spring leaf: touching red spins the flywheels |
| **blue** | fire microswitch pin 1 **C** → pusher motor (2–5 Ω to BAT− when pressed) |
| **black** | fire microswitch pin 2 **NC** = brake (~0 Ω to BAT−): released, C–NC shorts the pusher so it stops |

The fire trigger is an **SPDT microswitch with a motor brake**, not a simple on/off switch. To tell C from NC: pack unplugged, switch held pressed or disconnected; NC reads ~0 Ω to battery −, C reads a few Ω (motor winding). Released and wired in, both read ~0 Ω.

### Automating the trigger (correct wiring)

> ⚠️ **Never put a relay in parallel with the fire microswitch.** That joins red (BAT+) to blue (C), and while the switch is released C is tied to NC (BAT−): closing the relay **shorts the battery**. (Our earlier plan had this bug.)

- **Replace the microswitch with the relay's changeover contacts:** blue → relay **COM**, red → relay **NO**, black → relay **NC**. Relay off = pusher braked (like a released trigger); relay on = pusher runs. Swapping blue/black puts BAT+ straight onto BAT−; double-check before powering.
- Find COM on the relay: relay off, the two screws that beep together are COM–NC; hold the relay on and probe NO against each: the one that beeps is COM.
- **Rev always on:** join white to red permanently.
- **Master power switch in the BAT+ line** (≥5 A rocker, e.g. KCD1 10 A; a 2.5 A toggle was too small for flywheel start-up spikes). Off = nothing powered = hardware interlock. A separate arm input to the ESP32 is the software interlock (a DPDT switch can do both).
- Relay module: bench-test that it **releases at 3.3 V** (active-low 5 V boards may not; use the high-level-trigger jumper and a 10 kΩ pull-down on IN). Power relay VCC from its own 5 V if it has many coils (teammate's 8-channel board draws ~0.6–0.7 A; an MB102 breadboard supply is fine for ≤300 ms pulses).
- ESP32: keep relay pins off boot-glitch/strapping pins (avoid 0, 1, 3, 5, 6–11, 12, 14, 15; 34–39 are input-only). Set the relay OFF before `pinMode` at boot. Pulse ~150 ms (50–300), cooldown ~500 ms; tune to one dart.
- ESP32 dev boards with a CP2102 need the Silicon Labs CP210x driver on macOS; appears as `/dev/cu.usbserial-*`. Many USB-C cables are charge-only. If upload hangs: hold BOOT, tap EN.

### Using act-athon's rig (one button, rigd)

The built rig runs act-athon's `fire_ctl` firmware ([github.com/yuviji/act-athon](https://github.com/yuviji/act-athon), `docs/02-hardware/wiring.md`) behind its `rigd` HTTP daemon. Pins: relay IN on **GPIO13** (high-level trigger), one operator button on **GPIO25**; no arming input (the blaster's rocker switch, power + rev, is the only switch). **Don't flash this repo's `firmware/` sketch onto that rig**: its pin map (relay on GPIO26, DONE on 33) doesn't match that rig's wiring.

- Start rigd (it owns the ESP32 port): `rigd --port /dev/cu.usbserial-0001` (from act-athon's venv).
- `record.py --rigd-url http://localhost:8080 …` instead of `--esp32-port`: releasing the button fires (firmware) and is recorded as `SHOOT`.
- `robot_server.py … --rigd-url http://localhost:8080`: the same button's release means `TURN` (test mode never fires); `fire()` goes through rigd with heartbeat and cooldown checks.
- `autobot/hardware.py: RigdESP32` sets the button mode on connect (the board remembers it in flash) and renews rigd's heartbeat lease from `events()`/`heartbeat()`, so the board is only live while the script runs.
- **Heartbeat:** firmware drops to SAFE if the laptop sends nothing for ~500 ms, so a crashed laptop can't leave it live. Teammate's `firmware/fire_ctl/` already implements SAFE → READY → FIRING → COOLDOWN with this.

### How the trigger fits the model

- One action column `action/blaster/trigger` `[1]`: 0 every step, 1 while firing. It's in contract `so101_blaster_v1`.
- **Hold the 1 for ~10 steps (~0.3 s)** per shot so the model can learn it (a single 1 in 600 zeros is invisible).
- **Fire gate at inference** (teammate's `src/actathon/fire_gate.py`): fire only when trigger > 0.5 for **3 consecutive steps**, the arm is **settled** (max joint change ≤ 0.5°/step), and the shot budget allows it (1 shot per `shoot(cell)`). Fallback "settle" mode fires after the arm is still ~9 steps (≥30 steps in), if the learned trigger is flaky.
- **Safety filter** on π0.7 output before the servos: clamp each joint move to ≤3° per step (~90°/s) plus joint limits.
- Recording needs a **fire button/pedal** read by our own recording loop (stock `lerobot-record` won't capture it). **Label intent, not outcome:** the label comes from the button press, even when disarmed, so most episodes can be recorded without firing darts. Count presses (a 30 ms tap between frames isn't lost) and widen each to ≥10 frames.
- Keep trigger=1 frames well above 1% of all frames. With quantile normalization a rarer action can blow up (teammate saw q01 == q99 → fire=1 normalized to ~2e8). 10 steps per 150–240-step episode ≈ 4–7%: fine.
- Optional later: separate `rev` and `fire` columns (model can pre-rev while aiming), or an observation `observation/blaster/loaded` if the ESP32 can sense it.

### Weight: check first

The Dartstorm is large (two drums, tripod) and almost certainly too heavy for the SO-101 to hold and aim. Weigh it and test at full reach. Options: strip it to one flywheel cage + feed + small battery on a light printed mount, or leave it on its tripod pivot and have the arm push a printed aiming handle ("the arm is the gunner").

### Task framing

- Imitation learning can do "aim at the target and fire" (move the target around; the model learns to point where it sees it). A dart portrait needs more precision than darts have.
- Hybrid: Claude picks and marks spots on a canvas; π0.7 aims and fires at each marked spot.
- Safety: aim only at the canvas/target, operator behind the blaster, eye protection, hardware arm/disarm switch.

## Electronics notes

- **A MOSFET is an electronic switch**: a small gate voltage from a microcontroller switches a large current. No moving parts, microsecond switching, PWM-capable. An ESP32 pin supplies only ~20–40 mA at 3.3 V; motors need amps.
- Gotchas: buy **logic-level** parts (IRLZ44N, AO3400; not the IRF520 in many kits); **flyback diode** (1N4007/1N5819) across every motor/solenoid/coil; **shared ground**; **10 kΩ gate pull-down** so loads stay off during boot; MOSFETs switch one direction only, so use an H-bridge (DRV8871, TB6612) for forward/reverse.
- **Relay vs MOSFET:** a relay is a coil-driven mechanical switch (~10 ms, clicks, wears) that isolates the two circuits; a MOSFET is faster and silent but shares ground.
- The SO-101 servos don't need any of this; they have their own board and supply.

## Drawing-robot alternative (earlier idea)

A spring-loaded pen holder for the gripper (most important part), thick felt markers not paint, taped paper frame, top-down + wrist camera. Claude picks stroke primitives; π0.7 draws each one. ~8–10 primitives × 20–40 episodes.
