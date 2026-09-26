"""Play: Claude picks the move, π0.7 aims and fires.

  # one shot (test the policy + fire rule)
  .venv/bin/python scripts/robot_server.py shoot B2 --finetune pilot-v1 --follower-port … --esp32-port …
  .venv/bin/python scripts/robot_server.py shoot B2 --fake                 # no hardware, fake policy

  # full game: human places O + presses DONE (or SPACE); Claude picks a cell; the arm shoots
  .venv/bin/python scripts/robot_server.py game --finetune pilot-v1 --brain claude …
  .venv/bin/python scripts/robot_server.py game --fake --brain random

One process owns every device. The arm returns to its starting pose after each shot.
"""

import argparse
import base64
import random

import numpy as np

from autobot import schema as s
from autobot.control import move_to, shoot
from autobot.hardware import ESP32, FakeESP32, FakeRig, Rig
from autobot.keys import keyboard
from autobot.policy import FakePolicy, PiPolicy, gateway_url_for

CLAUDE_MODEL = "claude-opus-5-5"
SYSTEM_PROMPT = """You play tic-tac-toe as X against a human (O) by shooting suction darts at a 3×3 board.
Columns are A B C left to right; rows are 1 2 3 top to bottom. Your moves are the darts stuck in cells;
the human's moves are O markers. You'll get a photo of the board and the list of cells you've already shot
(use it if a dart fell off). Pick the best empty cell and call the shoot tool with it. If the game is over
(someone has three in a row, or the board is full), don't call the tool; reply with one line saying who won."""
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
        import anthropic  # only needed for --brain claude

        self.client = anthropic.Anthropic()  # ANTHROPIC_API_KEY or an `ant auth login` profile

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
    if a.fake:
        rig, esp, policy = FakeRig(leader_port=None), FakeESP32(shoot_every=None), FakePolicy(a.chunk)
    else:
        cam = lambda v: int(v) if v.isdigit() else v
        rig = Rig(a.follower_port, {"front_camera": cam(a.front_cam), "wrist_camera": cam(a.wrist_cam)}, follower_id=a.follower_id)
        esp = ESP32(a.esp32_port)
        url = a.gateway_url or gateway_url_for(a.finetune)
        print(f"connecting to π0.7 at {url} (first connect can take minutes)…")
        policy = PiPolicy(url, a.chunk)
    rig.connect()
    return rig, esp, policy


def cmd_shoot(a, rig, esp, policy, rest):
    r = shoot(rig, esp, policy, a.cell, chunk=a.chunk)
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
                if k == " " or "TURN" in esp.events() or a.fake:
                    break
            images, _, _ = rig.observe()
            cell, note = brain.pick(images["front_camera"], shots)
            if cell is None:
                print(f"game over: {note}")
                return
            print(f"robot shoots {cell} ({note})")
            r = shoot(rig, esp, policy, cell, chunk=a.chunk)
            move_to(rig, esp, rest)
            if r.fired:
                shots.append(cell)
            print(f"  {'hit' if r.fired else 'missed the shot'}: {r.reason} ({r.seconds:.1f} s)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fake", action="store_true", help="fake arm, cameras, ESP32 and policy")
    p.add_argument("--finetune", help="fine-tune name whose hosted checkpoint to use")
    p.add_argument("--gateway-url", help="use this checkpoint URL directly")
    p.add_argument("--chunk", type=int, default=10, help="actions played per π0.7 call")
    p.add_argument("--follower-port"), p.add_argument("--esp32-port"), p.add_argument("--follower-id", default="follower")
    p.add_argument("--front-cam", default="0"), p.add_argument("--wrist-cam", default="1")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("shoot").add_argument("cell", choices=s.CELLS)
    g = sub.add_parser("game")
    g.add_argument("--brain", choices=["claude", "random"], default="claude")
    a = p.parse_args()
    if not a.fake and not (a.follower_port and a.esp32_port and (a.finetune or a.gateway_url)):
        p.error("need --follower-port, --esp32-port and --finetune/--gateway-url (or --fake)")

    rig, esp, policy = connect(a)
    _, rest, _ = rig.observe()  # the starting pose is the rest pose
    try:
        (cmd_shoot if a.cmd == "shoot" else cmd_game)(a, rig, esp, policy, rest)
    finally:
        move_to(rig, esp, rest)
        policy.close()
        esp.close()
        rig.disconnect()


if __name__ == "__main__":
    main()
