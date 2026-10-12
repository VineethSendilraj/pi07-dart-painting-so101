"""Draw with darts (Claude or a preset picks cells, π0.7 aims and fires), or single shots / tic-tac-toe.

  # the demo: Claude sees both cameras after every shot and picks the next cell until the picture is done
  .venv/bin/python scripts/robot_server.py --finetune full-v1 --follower-port … --rigd-url http://localhost:8080 \
      draw "the PI logo"
  .venv/bin/python scripts/robot_server.py … draw pi --brain preset          # no Claude: fixed π order
  .venv/bin/python scripts/robot_server.py --fake draw pi --brain preset     # no hardware

  # one shot (test the policy + fire rule)
  .venv/bin/python scripts/robot_server.py shoot B2 --finetune pilot-v1 --follower-port … --esp32-port …
  .venv/bin/python scripts/robot_server.py shoot B2 --fake                 # no hardware, fake policy

  # full game: human places O + presses DONE (or SPACE); Claude picks a cell; the arm shoots
  .venv/bin/python scripts/robot_server.py game --finetune pilot-v1 --brain claude …
  .venv/bin/python scripts/robot_server.py game --fake --brain random

One process owns every device. The red gun mark from calibration/gun_mark.json is drawn into the gun image
(training frames have it); --finetune uses the run's final checkpoint unless --checkpoint-step is given.
"""

import argparse
import base64
import json
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from autobot import schema as s
from autobot import control
from autobot.artist import Action, ClaudeArtist, PresetArtist, anthropic_client
from autobot.control import ShotResult, move_to, shoot
from autobot.hardware import ESP32, FakeESP32, FakeRig, Rig, RigdESP32, load_gun_mark, load_rest_pose
from autobot.keys import keyboard
from autobot.policy import FakePolicy, PiPolicy, gateway_url_for

CLAUDE_MODEL = "claude-opus-5-5"
SYSTEM_PROMPT = """You play tic-tac-toe as X against a human (O) by shooting suction darts at a 5×5 board.
Every cell is labeled with its name (A1 top-left … E5 bottom-right): columns A–E left to right, rows 1–5 top
to bottom. WIN_RULE Your moves are the darts stuck in cells;
the human's moves are O markers. You'll get a photo of the board and the list of cells you've already shot
(use it if a dart fell off). Pick the best empty cell and call the shoot tool with it. If the game is over
(someone has won, or the board is full), don't call the tool; reply with one line saying who won."""
# lift / elbow / roll barely moved while recording; the models return these values for every cell (probe_cells.py)
TRAINING_POSE = {"shoulder_lift": -110.0, "elbow_flex": 96.8, "wrist_roll": -92.8}
WIN_LENGTH = 4  # in a row (row, column or diagonal) to win on 5×5
SYSTEM_PROMPT = SYSTEM_PROMPT.replace("WIN_RULE", f"The first to get {WIN_LENGTH} in a row (any direction) wins.")
SHOOT_TOOL = {
    "name": "shoot",
    "description": "Aim the blaster at one empty cell of the tic-tac-toe board and fire one dart.",
    "input_schema": {
        "type": "object",
        "properties": {"cell": {"type": "string", "enum": list(s.CELLS)}},
        "required": ["cell"],
        "additionalProperties": False,
    },
    "strict": True,
}


class RandomBrain:
    def pick(self, image: np.ndarray, shots: list[str]) -> tuple[str | None, str]:
        free = [c for c in s.CELLS if c not in shots]
        return (random.choice(free), "random move") if free else (None, "no free cells")


class ClaudeBrain:
    def __init__(self):
        self.client = anthropic_client()  # only needed for --brain claude

    def pick(self, image: np.ndarray, shots: list[str]) -> tuple[str | None, str]:
        import cv2

        ok, jpg = cv2.imencode(".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        response = self.client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=16000,
            output_config={"effort": "medium"},
            system=SYSTEM_PROMPT,
            tools=[SHOOT_TOOL],
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                             "data": base64.standard_b64encode(jpg.tobytes()).decode()}},
                {"type": "text", "text": f"Cells you've already shot: {', '.join(shots) or 'none'}. Your move."},
            ]}],
        )
        if response.stop_reason == "refusal":
            return None, f"Claude declined: {response.stop_details}"
        for block in response.content:
            if block.type == "tool_use" and block.name == "shoot" and block.input.get("cell") in s.CELLS:
                return block.input["cell"], "claude"
        text = " ".join(b.text for b in response.content if b.type == "text").strip()
        return None, text or f"no move (stop_reason={response.stop_reason})"


def connect(a):
    if a.fake_rig:  # fake arm/cameras/ESP32, real hosted model: exercises the inference + fire-rule code paths
        rig, esp = FakeRig(leader_port=None), FakeESP32(shoot_every=None)
        url = a.gateway_url or gateway_url_for(a.finetune, a.checkpoint_step)
        print(f"connecting to π0.7 at {url}…")
        policy = PiPolicy(url)
    elif a.fake:
        rig, policy = FakeRig(leader_port=None), FakePolicy()
        esp = RigdESP32(a.rigd_url, mode="test") if a.rigd_url else FakeESP32(shoot_every=None)
    else:
        cam = lambda v: int(v) if v.isdigit() else v
        rig = Rig(a.follower_port, {"front_camera": cam(a.front_cam), "wrist_camera": cam(a.wrist_cam)}, follower_id=a.follower_id)
        # rigd in TEST mode: the one physical button is the human's DONE button and never fires;
        # the model fires through POST /fire.
        esp = RigdESP32(a.rigd_url, mode="test") if a.rigd_url else ESP32(a.esp32_port)
        url = a.gateway_url or gateway_url_for(a.finetune, a.checkpoint_step)
        print(f"connecting to π0.7 at {url} (first connect can take minutes)…")
        policy = PiPolicy(url)
    rig.connect()
    mark = None if a.no_crosshair else (tuple(a.crosshair) if a.crosshair else load_gun_mark())
    rig.crosshair = mark
    print(f"red gun mark at x={mark[0]:.3f} y={mark[1]:.3f}" if mark else
          "⚠ no red gun mark: most training gun frames have one (calibration/gun_mark.json or --crosshair X Y)")
    return rig, esp, policy


def cmd_shoot(a, rig, esp, policy, rest):
    r = shoot(rig, esp, policy, a.cell, chunk=a.chunk, settle_fire_s=a.settle_fire)
    move_to(rig, esp, rest)
    print(f"{r.cell}: {'FIRED' if r.fired else 'NOT fired'} ({r.reason}) after {r.steps} steps, {r.seconds:.1f} s")


def cmd_game(a, rig, esp, policy, rest):
    brain = ClaudeBrain() if a.brain == "claude" else RandomBrain()
    shots: list[str] = []
    with keyboard() as key:
        while True:
            print("\nYour turn: place an O, then press DONE (or SPACE). q = quit.")
            while True:
                k = key()
                if k == "q":
                    return
                try:
                    done = "TURN" in esp.events()
                except OSError as e:  # rigd hiccup: SPACE still works
                    done = False
                    print(f"  (DONE button read failed: {e})")
                if k == " " or done or a.fake or a.fake_rig:
                    break
                time.sleep(0.05)
            images, _ = rig.observe()
            cell, note = brain.pick(images["front_camera"], shots)
            if cell is None:
                print(f"game over: {note}")
                return
            print(f"robot shoots {cell} ({note})")
            r = shoot(rig, esp, policy, cell, chunk=a.chunk, settle_fire_s=a.settle_fire)
            move_to(rig, esp, rest)
            if r.fired:
                shots.append(cell)
            print(f"  {'fired' if r.fired else 'did NOT fire'}: {r.reason} ({r.seconds:.1f} s)")


def cmd_draw(a, rig, esp, policy, rest):
    """Claude (or a preset) picks each cell from one frame of each camera; π0.7 aims and fires; repeat until done.
    No return to rest between shots: the recorded takes started from the previous aim. A crashed shot counts
    as not fired and the picture continues. A Claude failure stops it."""
    goal = " ".join(a.goal)
    log_dir = Path("data/draw") / datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir.mkdir(parents=True, exist_ok=True)

    def save(n, images, record):
        import cv2
        for cam, tag in (("front_camera", "rear"), ("wrist_camera", "gun")):
            cv2.imwrite(str(log_dir / f"{n:02d}_{tag}.jpg"), cv2.cvtColor(images[cam], cv2.COLOR_RGB2BGR))
        with open(log_dir / "log.jsonl", "a") as f:
            f.write(json.dumps(record) + "\n")

    def stop(why: str) -> Action:
        print(f"⚠ {why}; stopping")
        return Action("finish", note=why)

    artist = ClaudeArtist(goal, a.max_shots, dry=a.dry) if a.brain == "claude" else PresetArtist(goal)
    fired_cells: list[str] = []

    images, _ = rig.observe()
    try:
        action = artist.start(images)
    except Exception as e:  # Claude unreachable (key, network)
        action = stop(f"Claude failed: {type(e).__name__}: {e}")
    save(0, images, {"t": time.time(), "goal": goal, "action": action.__dict__})
    shots = 0
    with keyboard() as key:
        while action.kind == "shoot" and shots < a.max_shots:
            print(f"\nshot {shots + 1}: {action.cell}  {action.note}")
            if a.confirm:
                print("  press DONE (or SPACE) to fire, q to stop")
                try:  # ignore presses made while Claude was thinking
                    esp.events()
                except OSError:
                    pass
                while key() is not None:
                    pass
                while True:
                    k = key()
                    if k == "q":
                        return
                    try:
                        done = "TURN" in esp.events()
                    except OSError:
                        done = False
                    if k == " " or done or a.fake or a.fake_rig:
                        break
                    time.sleep(0.05)
            try:
                r = shoot(rig, esp, policy, action.cell, chunk=a.chunk, settle_fire_s=a.settle_fire)
            except Exception as e:  # never let one shot end the picture
                r = ShotResult(action.cell, False, f"shot crashed: {type(e).__name__}: {e}", 0, 0.0)
            shots += 1
            if r.fired:
                fired_cells.append(action.cell)
            if a.rest_between:
                move_to(rig, esp, rest)
            time.sleep(0.5)  # dart settles
            try:
                images, _ = rig.observe()
            except Exception as e:
                print(f"  camera read failed after the shot: {e}")
            result = (f"{'Fired' if r.fired else 'Did NOT fire'} at {action.cell} ({r.reason}, {r.seconds:.1f} s). "
                      f"Shots used: {shots}/{a.max_shots}.")
            print(f"  {result}")
            try:
                action = artist.after(action, result, r.fired, images)
            except Exception as e:
                action = stop(f"Claude failed mid-picture: {type(e).__name__}: {e}")
            save(shots, images, {"t": time.time(), "result": result, "next": action.__dict__})
    print(f"\ndone after {shots} shots ({', '.join(fired_cells) or 'none fired'}): {action.note}\nframes + log: {log_dir}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fake", action="store_true", help="fake arm, cameras, ESP32 and policy")
    p.add_argument("--fake-rig", action="store_true", help="fake arm, cameras and ESP32 with the real hosted policy")
    p.add_argument("--finetune", help="fine-tune name whose hosted checkpoint to use")
    p.add_argument("--gateway-url", help="use this checkpoint URL directly")
    p.add_argument("--chunk", type=int, default=25, help="actions played per π0.7 call (horizon 50; ~0.5 s per call)")
    p.add_argument("--follower-port"), p.add_argument("--esp32-port"), p.add_argument("--follower-id", default="follower_arm")
    p.add_argument("--rigd-url", help="act-athon rigd (e.g. http://localhost:8080): fires via the rig, button = DONE")
    p.add_argument("--front-cam", default="0"), p.add_argument("--wrist-cam", default="1")
    p.add_argument("--checkpoint-step", type=int, help="with --finetune: this checkpoint (default: the run's final one)")
    p.add_argument("--crosshair", nargs=2, type=float, metavar=("X", "Y"),
                   help="red plus on the gun image at these fractions of width/height (default: calibration/gun_mark.json)")
    p.add_argument("--no-crosshair", action="store_true", help="don't draw the red gun mark")
    p.add_argument("--fire-threshold", type=float, help=f"trigger level that counts as 'fire' (default {control.FIRE_THRESHOLD})")
    p.add_argument("--settled-deg", type=float, help=f"max joint change per step that counts as still (default {control.SETTLED_DEG})")
    p.add_argument("--max-step-deg", type=float, help=f"speed limit per 30 Hz step (default {control.MAX_STEP_DEG}° = 90°/s)")
    p.add_argument("--hold-deg", type=float, help=f"look-ahead: rest of the plan must stay this close to the aim (default {control.HOLD_DEG})")
    p.add_argument("--settle-fire", type=float, metavar="SECONDS",
                   help="fallback: fire if the model has held the arm on its own aim this long without firing (off by default)")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("shoot").add_argument("cell", choices=s.CELLS)
    g = sub.add_parser("game")
    g.add_argument("--brain", choices=["claude", "random"], default="claude")
    d = sub.add_parser("draw", help="draw a picture with darts")
    d.add_argument("goal", nargs="+", help='what to draw, e.g. "the PI logo", smiley, or cells: B2 C2 D2')
    d.add_argument("--brain", choices=["claude", "preset"], default="claude")
    d.add_argument("--max-shots", type=int, default=16)
    d.add_argument("--confirm", action="store_true", help="wait for DONE/SPACE before each shot (reload, check)")
    d.add_argument("--rest-between", action="store_true", help="return to the rest pose between shots")
    d.add_argument("--dry", action="store_true", help="master switch OFF: tell Claude no darts will appear")
    a = p.parse_args()
    if not (a.fake or a.fake_rig) and not (a.follower_port and (a.esp32_port or a.rigd_url) and (a.finetune or a.gateway_url)):
        p.error("need --follower-port, --rigd-url (or --esp32-port) and --finetune/--gateway-url (or --fake)")

    for flag, name in (("fire_threshold", "FIRE_THRESHOLD"), ("settled_deg", "SETTLED_DEG"), ("hold_deg", "HOLD_DEG"),
                       ("max_step_deg", "MAX_STEP_DEG")):
        if getattr(a, flag) is not None:
            setattr(control, name, getattr(a, flag))
    print(f"fire rule: trigger > {control.FIRE_THRESHOLD} for {control.FIRE_CONSECUTIVE} steps, still = ≤{control.SETTLED_DEG}°/step, "
          f"look-ahead hold ≤{control.HOLD_DEG}°, speed ≤{control.MAX_STEP_DEG}°/step, chunk {a.chunk}/50" + (f", settle-fire {a.settle_fire} s" if a.settle_fire else ""))
    rig, esp, policy = connect(a)
    if not (a.fake or a.fake_rig):
        _, now = rig.observe()
        off = {n: round(float(now[s.JOINT_NAMES.index(n)] - v), 1) for n, v in TRAINING_POSE.items()
               if abs(float(now[s.JOINT_NAMES.index(n)]) - v) > 8.0}
        if off:
            print(f"⚠ start pose unlike the training data (degrees off: {off}). Move lift/elbow/roll back "
                  f"(≈{TRAINING_POSE}) and aim at the board before shooting.")
    rest = load_rest_pose()  # calibration/rest_pose.json: the same rest pose the demos started from
    if rest is None:
        _, rest = rig.observe()
        print("⚠ no calibration/rest_pose.json: using the current pose as rest (set one in record_web.py)")
    else:
        move_to(rig, esp, rest)
    try:
        {"shoot": cmd_shoot, "game": cmd_game, "draw": cmd_draw}[a.cmd](a, rig, esp, policy, rest)
    finally:
        move_to(rig, esp, rest)
        policy.close()
        esp.close()
        rig.disconnect()


if __name__ == "__main__":
    main()
