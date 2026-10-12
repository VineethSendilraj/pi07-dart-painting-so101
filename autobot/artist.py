"""Who picks the next cell when drawing with darts: Claude (looks at both cameras after every shot) or a preset.

The loop (scripts/robot_server.py draw): artist.start(frames) -> Action; for each "shoot": π0.7 aims and fires
(control.shoot), then artist.after(action, result, frames) -> next Action; stops on "finish".
"""

import base64
import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from autobot import schema as s

CLAUDE_MODEL = "claude-opus-5-5"
NO_DATA = ("A5", "E5")  # never recorded: the aiming model has no idea where these are
WEAK = ("A1", "B1", "C1", "D1", "E1", "A3", "E3")  # 2 demos each
STRONG = ("A2", "B2", "C2", "D2", "E2", "B3", "B4", "B5", "D3", "D4", "D5")  # 7 demos each (the π logo)
ALLOWED = tuple(c for c in s.CELLS if c not in NO_DATA)
SHAPES = {
    # top bar left→right, then the left leg top→bottom, then the right leg: the order the demos were recorded in
    "pi": ["A2", "B2", "C2", "D2", "E2", "B3", "B4", "B5", "D3", "D4", "D5"],
    "smiley": ["B2", "D2", "A4", "B5", "C5", "D5", "E4"],
}

SYSTEM_PROMPT = f"""You are drawing pictures with foam darts on a 5x5 board, using a robot: an SO-101 arm aims a
motorized dart blaster that sits on a tripod, and a learned policy (π0.7) does the aiming and firing. Your job is to
plan the picture and pick the next cell after every shot.

The board: 25 square cells labeled inside, A1 (top-left) to E5 (bottom-right). Columns A-E run left to right, rows
1-5 top to bottom. A dart that hits sticks in the foam, so every shot leaves a mark. One dart per cell.

Each turn you get two photos taken right now:
- REAR camera: behind the blaster, sees the whole board. Use it to see which cells already have darts.
- GUN camera: mounted on the blaster above the barrels. The red plus is drawn at a FIXED spot in this image (it marks
  roughly where the barrel points), so it never moves in the picture; the board moves under it as the blaster turns.
  Read which cell is under the plus.

Tools: call `shoot` with one cell to aim and fire one dart there; call `finish` when the picture is done (or if you
must stop). Call exactly one tool per turn. After each shot you get the result (fired or not) and new photos.

Rules for picking cells (they come from how much training data the aiming policy has):
- Never use {', '.join(NO_DATA)}: the policy was never trained on them.
- Avoid {', '.join(WEAK)} unless the picture needs them: only 2 training demos each, so they miss more often.
- Best cells: {', '.join(STRONG)} (7 demos each). Also fine: C3, C4, C5, A4, E4.
- Keep pictures simple and recognizable: about 5-12 darts. Prefer shapes built from the best cells.
- Plan the whole picture first, then shoot in an order where each next cell is next to or near the previous one; the
  policy is best at short moves.
- If the user asks for the PI logo (π), use exactly these 11 cells in exactly this order: top bar A2, B2, C2, D2, E2;
  then the left leg B3, B4, B5; then the right leg D3, D4, D5.
- If the user asks for a smiley: eyes B2, D2; mouth A4, B5, C5, D5, E4 (in that order).
- After each shot, check the REAR photo: did the dart land in the cell you asked for? If it landed in a neighbor that
  your picture also needs, count it and skip that cell later. If it missed the picture, you may re-shoot the intended
  cell once; don't chase the same cell more than twice. If the blaster did not fire, retry that cell once.
- Never shoot a cell that already has a dart. Stop when the picture is recognizable or you run out of shots.
Keep your text short: one line on what you see and why you pick the next cell."""

TOOLS = [
    {
        "name": "shoot",
        "description": "Aim the blaster at one cell and fire one dart. Returns whether it fired, plus new photos.",
        "input_schema": {"type": "object", "properties": {"cell": {"type": "string", "enum": list(ALLOWED)}},
                         "required": ["cell"], "additionalProperties": False},
        "strict": True,
    },
    {
        "name": "finish",
        "description": "The picture is done (or you must stop). Say what was drawn and anything that went wrong.",
        "input_schema": {"type": "object", "properties": {"summary": {"type": "string"}},
                         "required": ["summary"], "additionalProperties": False},
        "strict": True,
    },
]


@dataclass
class Action:
    kind: str  # "shoot" | "finish"
    cell: str | None = None
    note: str = ""
    tool_use_id: str | None = None


def shape_for(goal: str) -> list[str] | None:
    """A preset shape named in the goal as a whole word ("pi", "π", "smiley"), else None."""
    words = set(re.findall(r"[a-z]+", goal.lower().replace("π", " pi ")))
    return next((cells for name, cells in SHAPES.items() if name in words), None)


def jpeg_b64(rgb: np.ndarray, quality: int = 85) -> str:
    import cv2

    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return base64.standard_b64encode(buf.tobytes()).decode()


def frame_blocks(images: dict[str, np.ndarray]) -> list[dict]:
    """One frame from each camera, labeled."""
    blocks = []
    for cam, label in (("front_camera", "REAR camera (whole board)"), ("wrist_camera", "GUN camera (red plus = aim)")):
        blocks += [{"type": "text", "text": label},
                   {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": jpeg_b64(images[cam])}}]
    return blocks


class PresetArtist:
    """A fixed cell list (a shape name from SHAPES or explicit cells). Retries a shot that didn't fire once."""

    def __init__(self, goal: str):
        cells = shape_for(goal) or [c.upper() for c in goal.replace(",", " ").split()]
        bad = [c for c in cells if c not in ALLOWED]
        if bad or not cells:
            raise SystemExit(f"preset: unknown shape or bad cells {bad or goal!r} (shapes: {', '.join(SHAPES)}; A5/E5 not allowed)")
        self.todo, self.retried = list(cells), set()

    def start(self, images) -> Action:
        return self._next("preset start")

    def after(self, action: Action, result: str, fired: bool, images) -> Action:
        if not fired and action.cell not in self.retried:
            self.retried.add(action.cell)
            self.todo.insert(0, action.cell)
        return self._next(result)

    def _next(self, note: str) -> Action:
        return Action("shoot", self.todo.pop(0), "preset") if self.todo else Action("finish", note="preset shape done")


def anthropic_client(**kwargs):
    """Claude client. Key from $ANTHROPIC_API_KEY, else ~/.config/anthropic/api_key."""
    import anthropic

    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not key:
        path = Path.home() / ".config" / "anthropic" / "api_key"
        if path.is_file():
            key = path.read_text().strip()
    if not key:
        raise SystemExit("Claude needs ANTHROPIC_API_KEY or ~/.config/anthropic/api_key")
    kwargs.setdefault("api_key", key)
    # Keys not scoped to a workspace also need ANTHROPIC_WORKSPACE_ID (Console → Settings → Workspaces).
    ws = os.environ.get("ANTHROPIC_WORKSPACE_ID")
    if ws and "default_headers" not in kwargs:
        kwargs["default_headers"] = {"anthropic-workspace-id": ws}
    return anthropic.Anthropic(**kwargs)


class ClaudeArtist:
    """Claude plans the picture and picks every next cell from one frame of each camera, sent after every shot."""

    def __init__(self, goal: str, max_shots: int, client=None, dry: bool = False):
        if client is None:
            client = anthropic_client(timeout=90)
        self.client, self.goal, self.max_shots, self.dry = client, goal, max_shots, dry
        self.messages: list[dict] = []

    def start(self, images) -> Action:
        self.messages.append({"role": "user", "content": [
            {"type": "text", "text": f"Draw: {self.goal}. You have at most {self.max_shots} shots. Here is the board now."
             + (" DRY RUN: the blaster is unloaded, so no darts will appear. Treat every shot reported as fired as done, "
                "don't re-shoot because of missing darts, and use the GUN camera only to see where it aimed." if self.dry else "")},
            *frame_blocks(images),
        ]})
        return self._ask()

    def after(self, action: Action, result: str, fired: bool, images) -> Action:
        self.messages.append({"role": "user", "content": [{
            "type": "tool_result", "tool_use_id": action.tool_use_id,
            "content": [{"type": "text", "text": result}, *frame_blocks(images)],
        }]})
        return self._ask()

    def _ask(self, nudged: bool = False) -> Action:
        # History is append-only: editing earlier turns would invalidate the thinking blocks Opus returns.
        # Photos are ~400 tokens each, so a whole 16-shot picture stays far inside the context window.
        response = self.client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=16000,
            thinking={"type": "adaptive"},
            output_config={"effort": "medium"},
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            tool_choice={"type": "auto", "disable_parallel_tool_use": True},
            cache_control={"type": "ephemeral"},
            messages=self.messages,
        )
        self.messages.append({"role": "assistant", "content": response.content})
        text = " ".join(b.text for b in response.content if b.type == "text").strip()
        if response.stop_reason == "refusal":
            return Action("finish", note=f"Claude declined: {response.stop_details}")
        for b in response.content:
            if b.type == "tool_use" and b.name == "shoot" and b.input.get("cell") in ALLOWED:
                return Action("shoot", b.input["cell"], text, b.id)
            if b.type == "tool_use" and b.name == "finish":
                return Action("finish", note=b.input.get("summary", text), tool_use_id=b.id)
        if not nudged:  # replied without a tool call: ask once more, append-only
            self.messages.append({"role": "user", "content": "Call shoot or finish now."})
            return self._ask(nudged=True)
        return Action("finish", note=text or f"no tool call (stop_reason={response.stop_reason})")
