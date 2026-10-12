"""Dashboard for the dart-drawing loop: prompt, Claude's frames and replies, policy as a black box, cell path.

  # real rig (stop record_web first; it holds the follower). rigd must be running.
  caffeinate -i .venv/bin/python scripts/draw_web.py \\
      --follower-port /dev/tty.usbmodem5B7A0000841 --rigd-url http://localhost:8080 \\
      --front-cam 0 --wrist-cam 1 --finetune full-v2 --checkpoint-step 3799

  .venv/bin/python scripts/draw_web.py --fake          # no hardware, preset or claude still drives the page

Opens http://127.0.0.1:7871. Type what to draw, Start. Pause holds the loop between turns
(after a shot, before Claude decides the next cell, and before the next shot) so you can clear a jam.
Resume recaptures both cameras and continues. P toggles pause.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import numpy as np

from autobot import schema as s
from autobot.artist import ClaudeArtist, PresetArtist
from autobot.control import shoot
from autobot.hardware import ESP32, FakeESP32, FakeRig, Rig, RigdESP32, load_gun_mark
from autobot.policy import FakePolicy, PiPolicy, gateway_url_for

ROOT = Path(__file__).resolve().parents[1]
PORT = 7871


class DrawState:
    def __init__(self):
        self.lock = threading.Lock()
        self.phase = "idle"  # idle | thinking | shooting | paused | done | error
        self.goal = ""
        self.brain = "preset"
        self.note = ""
        self.err = ""
        self.policy = ""  # black-box line, no joints
        self.path: list[dict] = []  # {cell, n, status: next|now|hit|miss}
        self.log: list[dict] = []
        self.frames: dict | None = None
        self.pause = False
        self.stop = False
        self.busy = False
        self.limp = False
        self.want_limp = False


STATE = DrawState()
RIG_LOCK = threading.Lock()
READY = {"rig": None, "esp": None, "policy": None}
PREVIEW_STOP = threading.Event()


def _observe(rig):
    with RIG_LOCK:
        return rig.observe()


def _preview(rig) -> None:
    """Keep the page on the live cameras. A shot holds RIG_LOCK, so this waits it out."""
    while not PREVIEW_STOP.is_set():
        if not RIG_LOCK.acquire(blocking=False):
            PREVIEW_STOP.wait(0.05)
            continue
        try:
            images, _ = rig.observe()
            err = None
        except Exception as e:
            images, err = None, e
        finally:
            RIG_LOCK.release()
        with STATE.lock:
            if images is not None:
                STATE.frames = images
                if STATE.err.startswith("camera:"):
                    STATE.err = ""
            elif err is not None and not STATE.busy:
                STATE.err = f"camera: {err}"
        PREVIEW_STOP.wait(0.08)


def jpeg(rgb: np.ndarray | None) -> bytes | None:
    if rgb is None:
        return None
    import cv2

    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR),
                           [int(cv2.IMWRITE_JPEG_QUALITY), 70])
    return buf.tobytes() if ok else None


def _log(text: str, kind: str, cell: str | None = None, fired: bool | None = None) -> None:
    with STATE.lock:
        STATE.log.append({"t": time.time(), "kind": kind, "text": text, "cell": cell, "fired": fired})
        del STATE.log[:-24]


def _set_path_status(cell: str | None, status: str) -> None:
    if not cell:
        return
    for step in reversed(STATE.path):
        if step["cell"] == cell and step["status"] in ("next", "now", "miss"):
            step["status"] = status
            return


PAGE = r"""<!doctype html>
<meta charset="utf-8">
<title>draw</title>
<style>
  :root { color-scheme: dark; --bg:#0e0e0e; --card:#1b1b1b; --line:#2e2e2e; --mute:#9aa; }
  html, body { height: 100%; }
  body { font: 15px/1.4 ui-sans-serif, system-ui; margin: 0; background: var(--bg); color: #eee; overflow: hidden; }
  main { height: 100vh; box-sizing: border-box; display: grid; grid-template-columns: minmax(0,1fr) 340px; gap: 10px; padding: 10px; }
  header { grid-column: 1 / -1; display: flex; align-items: baseline; gap: 12px; }
  h1 { font-size: 22px; margin: 0; }
  #phase { color: var(--mute); }
  .stage { min-height: 0; display: grid; grid-template-columns: minmax(0,1fr) 200px; grid-template-rows: auto minmax(0,1fr); gap: 8px; }
  figure { margin: 0; background: #000; border: 1px solid var(--line); border-radius: 10px; overflow: hidden; display: flex; flex-direction: column; min-height: 0; }
  #gun { grid-column: 1; grid-row: 1 / span 2; }
  #rear { grid-column: 2; grid-row: 1; }
  figure img { width: 100%; flex: 1; min-height: 0; object-fit: contain; background: #000; }
  #rear img { max-height: 140px; flex: none; }
  figcaption { font-size: 12px; color: var(--mute); padding: 4px 8px; }
  .side { min-height: 0; overflow: hidden; display: flex; flex-direction: column; gap: 8px; }
  .card { background: var(--card); border: 1px solid var(--line); border-radius: 12px; padding: 10px; }
  .card h3 { margin: 0 0 8px; font-size: 12px; letter-spacing: .08em; text-transform: uppercase; color: var(--mute); }
  input[type=text] { width: 100%; box-sizing: border-box; font: inherit; padding: 8px; border-radius: 8px; border: 1px solid #444; background: #111; color: #eee; }
  .row { display: flex; gap: 8px; margin-top: 8px; }
  button { font: inherit; padding: 8px 12px; border: 0; border-radius: 8px; cursor: pointer; color: #fff; background: #333a44; }
  button:disabled { opacity: .35; cursor: default; }
  #go { background: #b91c1c; flex: 1; }
  #pause.on { background: #92400e; }
  #limp { flex: 1; }
  #limp.on { background: #1d4ed8; }
  #policy { min-height: 2.6em; }
  .card.grow { flex: 1; min-height: 160px; display: flex; flex-direction: column; overflow: hidden; }
  #log { flex: 1; min-height: 0; overflow: auto; font-size: 13px; overflow-wrap: anywhere; }
  .turn { padding: 6px 0; border-bottom: 1px solid var(--line); }
  .turn b { color: #93c5fd; }
  .turn.policy b { color: #fbbf24; }
  .turn.cmd b { color: #86efac; }
  table { border-collapse: collapse; width: 100%; }
  td { border: 1px solid var(--line); text-align: center; padding: 4px 1px; font-size: 12px; font-weight: 650; }
  td.now { outline: 2px solid #22c55e; }
  td.hit { background: #14532d; color: #bbf7d0; }
  td.next { color: #fde68a; }
  td.miss { color: #fca5a5; }
  .n { display: block; font-size: 10px; color: #86efac; }
</style>
<main>
  <header><h1 id="title">draw</h1><span id="phase">idle</span></header>
  <div class="stage">
    <figure id="gun"><img id="gun-img" alt="gun"><figcaption>GUN — live (red plus = aim)</figcaption></figure>
    <figure id="rear"><img id="rear-img" alt="rear"><figcaption>REAR — live</figcaption></figure>
    <div></div>
  </div>
  <div class="side">
    <div class="card">
      <h3>What to draw</h3>
      <input id="goal" type="text" placeholder="the PI logo, a smiley, or B2 D2" autocomplete="off">
      <div class="row">
        <button id="go">Start</button>
        <button id="pause">Pause agent (P)</button>
        <button id="stop">Stop</button>
      </div>
      <div class="row">
        <button id="limp">Limp</button>
      </div>
      <div class="row">
        <label class="small"><input id="brain" type="checkbox"> Claude (unchecked = preset list)</label>
      </div>
    </div>
    <div class="card"><h3>Policy</h3><div id="policy">idle</div></div>
    <div class="card grow"><h3>Claude / preset</h3><div id="log"></div></div>
    <div class="card"><h3>Path</h3><table id="grid"></table></div>
  </div>
</main>
<script>
const $ = id => document.getElementById(id);
let logHtml = '';
function post(url, body) {
  return fetch(url, {method:'POST', headers:{'content-type':'application/json'}, body: body ? JSON.stringify(body) : '{}'});
}
$('go').onclick = async () => {
  const r = await post('/start', {goal: $('goal').value, brain: $('brain').checked ? 'claude' : 'preset'});
  let d = {};
  try { d = await r.json(); } catch (e) {}
  if (!r.ok) $('phase').textContent = d.err || 'start failed';
};
$('pause').onclick = () => post('/pause');
$('stop').onclick = () => post('/stop');
$('limp').onclick = async () => {
  const r = await post('/limp');
  let d = {};
  try { d = await r.json(); } catch (e) {}
  if (!r.ok) $('phase').textContent = d.err || 'could not change torque';
};
document.addEventListener('keydown', e => {
  if (e.target.tagName === 'INPUT') return;
  if (e.key === 'p' || e.key === 'P') post('/pause');
});
async function tick() {
  try {
    const s = await (await fetch('/status')).json();
    $('title').textContent = s.goal ? s.goal : 'draw';
    $('phase').textContent = s.phase + (s.note ? ' — ' + s.note : '') + (s.err ? ' — ' + s.err : '');
    $('policy').textContent = s.policy || 'idle';
    $('go').disabled = s.busy || !s.ready;
    $('pause').textContent = s.pause ? 'Resume agent (P)' : 'Pause agent (P)';
    $('pause').className = s.pause ? 'on' : '';
    $('limp').textContent = s.limp ? 'Hold arm' : 'Limp';
    $('limp').className = s.limp ? 'on' : '';
    const labels = {cmd: 'to policy'};
    const html = (s.log || []).map(t => `<div class="turn ${t.kind}"><b>${labels[t.kind] || t.kind}</b> ${t.cell ? t.cell + ' — ' : ''}${t.text || ''}</div>`).join('') || '<span style="color:#9aa">nothing yet</span>';
    const log = $('log');
    if (html !== logHtml) {
      const top = log.scrollTop;
      const follow = log.scrollHeight - top - log.clientHeight < 32;
      logHtml = html;
      log.innerHTML = html;
      log.scrollTop = follow ? log.scrollHeight : top;
    }
    const path = {};
    (s.path || []).forEach(p => { path[p.cell] = p; });
    const rows = [];
    for (const r of '12345') {
      rows.push('<tr>' + [...'ABCDE'].map(c => {
        const k = c + r, p = path[k];
        const cls = p ? p.status : '';
        const n = p ? `<span class="n">${p.n}</span>` : '';
        return `<td class="${cls}">${k}${n}</td>`;
      }).join('') + '</tr>');
    }
    $('grid').innerHTML = rows.join('');
    const ts = Date.now();
    $('gun-img').src = '/cam/wrist_camera?t=' + ts;
    $('rear-img').src = '/cam/front_camera?t=' + ts;
  } catch (e) { $('phase').textContent = 'page lost the server'; }
  setTimeout(tick, 200);
}
tick();
</script>
"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        return

    def _send(self, code: int, body: bytes, ctype: str = "application/json") -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            body = PAGE.encode()
            self._send(200, body, "text/html; charset=utf-8")
            return
        if path == "/status":
            with STATE.lock:
                body = {
                    "phase": STATE.phase, "goal": STATE.goal, "brain": STATE.brain, "note": STATE.note,
                    "err": STATE.err, "policy": STATE.policy, "path": STATE.path, "log": STATE.log[-24:],
                    "pause": STATE.pause, "busy": STATE.busy, "limp": STATE.limp,
                    "ready": READY["policy"] is not None,
                }
            self._send(200, json.dumps(body).encode())
            return
        if path.startswith("/cam/"):
            name = path.split("/", 2)[-1]
            with STATE.lock:
                frame = None if STATE.frames is None else STATE.frames.get(name)
                frame = None if frame is None else frame.copy()
            blob = jpeg(frame)
            if blob is None:
                self._send(204, b"", "image/jpeg")
                return
            self._send(200, blob, "image/jpeg")
            return
        self._send(404, b"not found", "text/plain")

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n) if n else b"{}"
        path = urlparse(self.path).path
        if path == "/start":
            try:
                body = json.loads(raw or b"{}")
            except json.JSONDecodeError:
                self._send(400, b'{"ok":false}')
                return
            goal = str(body.get("goal") or "").strip()
            brain = body.get("brain") or "preset"
            if brain not in ("claude", "preset") or not goal:
                self._send(400, b'{"ok":false,"err":"need a goal"}')
                return
            if READY["policy"] is None or READY["rig"] is None:
                self._send(409, b'{"ok":false,"err":"still connecting"}')
                return
            if brain == "preset":
                try:
                    PresetArtist(goal)
                except SystemExit as e:
                    self._send(400, json.dumps({"ok": False, "err": str(e)}).encode())
                    return
            with STATE.lock:
                if STATE.busy:
                    self._send(409, b'{"ok":false,"err":"already running"}')
                    return
                STATE.busy = True
                STATE.stop = False
                STATE.pause = False
                STATE.goal = goal
                STATE.brain = brain
                STATE.phase = "thinking"
                STATE.err = ""
                STATE.log = []
                STATE.path = []
                STATE.policy = ""
                STATE.note = "starting"
            threading.Thread(target=RUNNER, args=(goal, brain), daemon=True).start()
            self._send(200, b'{"ok":true}')
            return
        if path == "/pause":
            with STATE.lock:
                if STATE.busy:
                    STATE.pause = not STATE.pause
                    if STATE.pause:
                        STATE.phase = "paused"
                        STATE.note = "paused — arm holding"
            self._send(200, b'{"ok":true}')
            return
        if path == "/limp":
            with STATE.lock:
                hold = STATE.limp
                busy = STATE.busy
                if busy and not hold:
                    STATE.stop = True
                    STATE.pause = False
                    STATE.want_limp = True
                    STATE.phase = "stopping"
                    STATE.note = "stopping — then the arm goes limp"
                    self._send(200, b'{"ok":true}')
                    return
            err = _set_torque(hold)
            if err:
                self._send(409, json.dumps({"ok": False, "err": err}).encode())
                return
            self._send(200, b'{"ok":true}')
            return
        if path == "/stop":
            with STATE.lock:
                STATE.stop = True
                STATE.pause = False
                if STATE.busy:
                    STATE.phase = "stopping"
                    STATE.note = "stopping"
            self._send(200, b'{"ok":true}')
            return
        self._send(404, b"not found", "text/plain")


RUNNER = None  # set in main to the bound loop


def _set_torque(on: bool) -> str | None:
    """Torque on locks the pose the arm is in right now. Torque off lets you move it."""
    rig = READY["rig"]
    if rig is None:
        return "arm not connected"
    if not RIG_LOCK.acquire(timeout=3):
        return "arm is busy"
    try:
        if on:
            _, joints = rig.observe()
            rig.command(joints)
        rig.torque(on)
    except Exception as e:
        return f"{type(e).__name__}: {e}"
    finally:
        RIG_LOCK.release()
    with STATE.lock:
        STATE.limp = not on
        if not STATE.busy:
            STATE.phase = "idle"
            STATE.note = "limp — move the gun, then Hold" if STATE.limp else "holding this pose"
    return None


def _apply_pending_limp() -> None:
    with STATE.lock:
        want = STATE.want_limp
        STATE.want_limp = False
    if want:
        _set_torque(False)


def _operator_gate() -> str | None:
    """Read by the shot loop. Stop aborts it; pause holds the current pose until resume."""
    with STATE.lock:
        if STATE.stop:
            return "stop"
        if STATE.pause:
            STATE.phase = "paused"
            STATE.note = "paused — arm holding"
            return "pause"
        if STATE.phase == "paused":
            STATE.phase = "shooting"
            STATE.note = ""
    return None


def _show_frames(images) -> None:
    with STATE.lock:
        STATE.frames = images


def _wait_between_turns(rig) -> dict | None:
    """Hold after a shot and before the next decision or the next shot. Recapture when released."""
    paused = False
    while True:
        with STATE.lock:
            if STATE.stop:
                return None
            want = STATE.pause
        if not want:
            break
        paused = True
        with STATE.lock:
            STATE.phase = "paused"
            STATE.note = "paused between turns — fix the gun, then Resume"
        time.sleep(0.05)
    if not paused:
        return None
    images, _ = _observe(rig)
    with STATE.lock:
        STATE.frames = images
        STATE.phase = "thinking"
        STATE.note = "resumed — Claude sees a fresh frame"
    return images


def _publish_plan(plan: list[str], current: str | None, hits: set[str], misses: set[str]) -> None:
    path = []
    for i, cell in enumerate(plan, start=1):
        if cell in misses and cell == current:
            status = "now"
        elif cell in hits and cell != current:
            status = "hit"
        elif cell == current:
            status = "now"
        elif cell in misses:
            status = "miss"
        else:
            status = "next"
        path.append({"cell": cell, "n": i, "status": status})
    with STATE.lock:
        STATE.path = path


def run_draw(rig, esp, policy, goal: str, brain: str, chunk: int, settle_fire_s: float | None, max_shots: int) -> None:
    try:
        if STATE.limp:
            err = _set_torque(True)
            if err:
                with STATE.lock:
                    STATE.phase = "error"
                    STATE.err = err
                    STATE.busy = False
                    STATE.note = "could not hold the arm"
                return
        if brain == "claude":
            artist = ClaudeArtist(goal, max_shots)
            plan: list[str] = []
        else:
            artist = PresetArtist(goal)
            plan = list(artist.todo)
        images, _ = _observe(rig)
        with STATE.lock:
            STATE.frames = images
            STATE.phase = "thinking"
            STATE.note = "looking at the board"
        _publish_plan(plan, None, set(), set())
        action = artist.start(images)
        if brain == "preset" and action.cell:
            _log(action.note or "preset", "preset", action.cell)
        else:
            _log(action.note or action.kind, "claude", action.cell)
        hits: set[str] = set()
        misses: set[str] = set()
        shots = 0
        while action.kind == "shoot" and shots < max_shots:
            if STATE.stop:
                break
            fresh = _wait_between_turns(rig)
            if STATE.stop:
                break
            if fresh is not None:
                images = fresh
            command = s.task_for(action.cell)
            _log(command, "cmd")
            with STATE.lock:
                STATE.phase = "shooting"
                STATE.note = action.cell or ""
                STATE.policy = command
            if action.cell and action.cell not in plan:
                plan.append(action.cell)
            _publish_plan(plan, action.cell, hits, misses)
            try:
                with RIG_LOCK:
                    result = shoot(rig, esp, policy, action.cell, chunk=chunk, settle_fire_s=settle_fire_s,
                                   log=lambda m: None, halt=_operator_gate, on_obs=_show_frames)
            except Exception as e:
                from autobot.control import ShotResult
                result = ShotResult(action.cell or "", False, f"shot crashed: {type(e).__name__}: {e}", 0, 0.0)
            if STATE.stop or result.reason == "stopped":
                _log("stopped", "policy", result.cell, False)
                break
            shots += 1
            if result.fired:
                hits.add(result.cell)
                misses.discard(result.cell)
            else:
                misses.add(result.cell)
            _publish_plan(plan, None, hits, misses)
            line = f"{'fired' if result.fired else 'did not fire'} ({result.reason}, {result.seconds:.1f}s)"
            _log(line, "policy", result.cell, result.fired)
            with STATE.lock:
                STATE.policy = f"{result.cell}: {line}"
            time.sleep(0.4)
            try:
                images, _ = _observe(rig)
            except Exception as e:
                _log(f"camera read failed: {e}", "policy")
            with STATE.lock:
                STATE.frames = images
                STATE.phase = "thinking"
                STATE.note = "Claude looking at the new frame"
            fresh = _wait_between_turns(rig)
            if STATE.stop:
                break
            if fresh is not None:
                images = fresh
                note = f"{'Fired' if result.fired else 'Did NOT fire'} at {result.cell}. Operator paused and resumed; photos are after the pause."
            else:
                note = f"{'Fired' if result.fired else 'Did NOT fire'} at {result.cell} ({result.reason})."
            try:
                action = artist.after(action, note, result.fired, images)
            except Exception as e:
                _log(f"planner failed: {type(e).__name__}: {e}", "claude")
                with STATE.lock:
                    STATE.phase = "error"
                    STATE.err = str(e)
                    STATE.busy = False
                _apply_pending_limp()
                return
            _log(action.note or action.kind, "claude" if brain == "claude" else "preset", action.cell)
            if action.cell and action.cell not in plan:
                plan.append(action.cell)
            _publish_plan(plan, action.cell if action.kind == "shoot" else None, hits, misses)
        with STATE.lock:
            if STATE.stop:
                STATE.phase = "idle"
                STATE.note = "stopped"
                STATE.policy = "idle"
                STATE.pause = False
            else:
                STATE.phase = "done"
                STATE.note = action.note or "done"
                STATE.policy = "idle"
            STATE.busy = False
        _apply_pending_limp()
    except BaseException as e:
        _log(f"{type(e).__name__}: {e}", "claude")
        with STATE.lock:
            STATE.phase = "error"
            STATE.err = str(e)
            STATE.busy = False
            STATE.policy = "idle"
            STATE.note = "stopped on an error"
        _apply_pending_limp()


def main() -> None:
    global RUNNER
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fake", action="store_true")
    p.add_argument("--follower-port")
    p.add_argument("--follower-id", default="follower_arm")
    p.add_argument("--rigd-url", default="http://localhost:8080")
    p.add_argument("--esp32-port")
    p.add_argument("--front-cam", default="0")
    p.add_argument("--wrist-cam", default="1")
    p.add_argument("--finetune", default="full-v2")
    p.add_argument("--checkpoint-step", type=int, default=3799)
    p.add_argument("--gateway-url")
    p.add_argument("--chunk", type=int, default=25)
    p.add_argument("--settle-fire", type=float, default=None)
    p.add_argument("--max-shots", type=int, default=16)
    p.add_argument("--port", type=int, default=PORT)
    p.add_argument("--no-browser", action="store_true")
    p.add_argument("--no-crosshair", action="store_true")
    a = p.parse_args()

    if not a.fake and not a.follower_port:
        raise SystemExit("need --follower-port (or --fake)")

    def runner(goal: str, brain: str) -> None:
        run_draw(READY["rig"], READY["esp"], READY["policy"], goal, brain, a.chunk, a.settle_fire, a.max_shots)

    def boot() -> None:
        rig = esp = policy = None
        try:
            if a.fake:
                rig, esp, policy = FakeRig(leader_port=None), FakeESP32(shoot_every=None), FakePolicy()
                rig.connect()
            else:
                cam = lambda v: int(v) if str(v).isdigit() else v
                rig = Rig(a.follower_port, {"front_camera": cam(a.front_cam), "wrist_camera": cam(a.wrist_cam)},
                          follower_id=a.follower_id)
                print("connecting cameras and arm…", flush=True)
                rig.connect()
            if not a.no_crosshair:
                mark = load_gun_mark()
                rig.crosshair = mark
                print(f"red gun mark at {mark}" if mark else "no gun mark file", flush=True)
            READY["rig"] = rig
            threading.Thread(target=_preview, args=(rig,), daemon=True).start()
            with STATE.lock:
                STATE.note = "cameras live" if a.fake else "cameras live — connecting the policy"
                STATE.phase = "idle"
            print("cameras live", flush=True)
            if a.fake:
                READY["esp"] = esp
                READY["policy"] = policy
                return
            esp = RigdESP32(a.rigd_url, mode="test") if a.rigd_url else ESP32(a.esp32_port)
            READY["esp"] = esp
            gateway = a.gateway_url or gateway_url_for(a.finetune, a.checkpoint_step)
            print(f"connecting to {gateway}…", flush=True)
            policy = PiPolicy(gateway)
            READY["policy"] = policy
            with STATE.lock:
                if not STATE.busy:
                    STATE.note = "cameras live"
            print("policy ready", flush=True)
        except Exception as e:
            print(f"startup failed: {type(e).__name__}: {e}", flush=True)
            with STATE.lock:
                STATE.phase = "error"
                STATE.err = str(e)
                STATE.note = "startup failed"
            if policy is not None and READY["policy"] is None:
                try:
                    policy.close()
                except Exception:
                    pass

    RUNNER = runner
    httpd = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    url = f"http://127.0.0.1:{a.port}/"
    print(f"draw UI: {url}", flush=True)
    threading.Thread(target=boot, daemon=True).start()
    if not a.no_browser:
        webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down")
    finally:
        PREVIEW_STOP.set()
        with STATE.lock:
            STATE.stop = True
        httpd.shutdown()
        for closer, obj in (
            ("disconnect", READY["rig"]),
            ("close", READY["policy"]),
            ("close", READY["esp"]),
        ):
            if obj is None:
                continue
            try:
                getattr(obj, closer)()
            except Exception:
                pass


if __name__ == "__main__":
    main()
