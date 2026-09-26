# Hardware: SO-101, teleop, cameras, blaster

Status and plans live in [notes.md](notes.md); model and data concepts in [concepts.md](concepts.md).

## SO-101 (Seeed kit)

- **Follower:** 6 × Feetech STS3215 at 1:345 gearing (Pro kit 12 V, standard kit 5 V). Joints: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper.
- **Leader:** mixed gearing (1:191 / 1:345 / 1:147) so it's light to move by hand; ends in a handle + trigger instead of a claw.
- Each arm has its own USB controller board. **The arms aren't connected to each other**: both plug into the laptop, which relays every movement. Place them side by side (USB reach, operator can see the follower), with the leader and operator out of camera view.
- STS3215 servos report their own angle (0–4095 ticks = 360°) and have a built-in position controller.
- Setup: `lerobot-find-port` → `lerobot-setup-motors` (if new) → `lerobot-calibrate` both arms → `lerobot-teleoperate` ~10 min → ~5 test recordings.
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

### The 4 wires (red, black, white, blue): likely but unverified

- Red/black: probably battery + / −. White/blue: probably the rev strips' leads (or the fire switch).
- Verify: batteries out, multimeter continuity on each pair while pressing rev, then fire. Then batteries in, **no darts**, measure each switch lead vs black: ~0 V = switch on the ground side of the motor; ~12 V = on the + side.

### Automating the triggers

- Put a switch **in parallel** with each trigger switch; closing it acts like pressing the trigger. Original triggers keep working.
- **2-channel relay module (recommended):** isolated from the blaster, works whichever side the switch is on, no shared ground. Needs a 3.3 V-triggerable/optocoupler input and contacts rated ≥5 A. The ~10 ms click is fine.
- **Logic-level N-channel MOSFET:** only if the switch is on the ground side. Needs a shared ground and a flyback diode.
- ESP32 sequence on FIRE: rev relay on → wait ~400 ms for spin-up (measure it) → fire relay on ~150 ms (≈ one dart; tune) → off. Don't fire before the flywheels are up to speed (weak shots, jams).
- Power the ESP32 separately (USB) or from 12 V via a buck converter. A **physical arm/disarm switch** in series with the relay contacts: nothing fires unless armed.

### How the trigger fits the model

- One action column `action/blaster/trigger` `[1]`: 0 every step, 1 while firing. It's in contract `so101_blaster_v1`.
- **Hold the 1 for ~10 steps (~0.3 s)** per shot so the model can learn it (a single 1 in 600 zeros is invisible).
- The laptop fires on the **0→1 edge** of `trigger > 0.5` with a ~1 s cooldown, so noisy outputs can't machine-gun.
- Recording needs a **fire button/pedal on the leader handle** read by our own recording loop (stock `lerobot-record` won't capture it).
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
