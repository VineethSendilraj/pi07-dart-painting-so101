"""Recording web app: shows which cell to aim at and how to set up; Record → aim → fire button → auto-save.

  # with the act-athon rig (rigd running on :8080; see docs/data-collection.md). Run from Terminal.app:
  caffeinate -i .venv/bin/python scripts/record_web.py --follower-port /dev/tty.usbmodem5B7A0000841 \\
      --leader-port /dev/tty.usbmodem5B7A0001561 --rigd-url http://localhost:8080 --front-cam 0 --wrist-cam 1 \\
      --cells A1 E1 C3 A5 E5 --per-cell 10 --auto-upload
  .venv/bin/python scripts/record_web.py --fake        # no hardware

Opens http://127.0.0.1:7870. Teleop (leader → follower) runs the whole time. Per episode: set up the board
as shown → point the arm where it says → Record → aim, hold still → press + release the fire button →
the take ends by itself 0.7 s later, saves, and shows the next cell. Then Good (G) accepts it (and uploads
with --auto-upload) and Bad (B) deletes it (e.g. a live dart missed); pressing Record counts as Good.
Progress, dart tracking and uploads are saved on disk: rerun the same command to continue where you stopped.
"""

from __future__ import annotations

import argparse
import getpass
import json
import queue
import random
import shutil
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from autobot import schema as s
from autobot.episode import Step, find_episodes, load_meta, validate, write_episode
from autobot.hardware import ESP32, FakeESP32, FakeRig, Rig, RigdESP32, load_rest_pose, save_rest_pose

ROOT = Path(__file__).resolve().parents[1]
TRIGGER_HOLD_STEPS = 10  # trigger=1 for 10 steps from the button release
AFTER_SHOT_S = 2.0  # default: the take ends this long after the shot (More options on the page, saved per session)
# A take runs until the fire button or Discard. No idle timeout.
MIN_HZ = 27.0  # contract: system/step 30 Hz ±10%
MAX_GAP_MS = 250  # any longer stall between two steps → not saved
COLLECT_RANGE = (3, 7)  # ask to collect darts after a random number of live shots in this range
REST_TOL_DEG = 3.0

# Per cell, one entry per episode (board clutter, live dart?, start pose). Ordered so every prefix stays
# balanced: 5 → 2 light / 2 heavy / 1 late; 8 → 1 clean / 2 light / 3 heavy / 2 late, 2 live, 6 rest starts;
# 10 → 1 clean / 3 light / 4 heavy / 2 late, 2 live, 7 rest. Game boards are never empty, so clean is rare.
SPREAD = [("light", True, "rest"), ("heavy", False, "rest"), ("late", False, "other"), ("light", False, "rest"),
          ("heavy", False, "other"), ("clean", False, "rest"), ("heavy", True, "rest"), ("late", False, "rest"),
          ("light", False, "other"), ("heavy", False, "rest")]
BOARD_LEVEL = {"clean": 0, "light": 1, "heavy": 2, "late": 3}
BOARD_MAX_DARTS = {"clean": 0, "light": 3, "heavy": 8, "late": 14}
BOARD_HELP = {
    "clean": "Clean board: pull every dart and O off.",
    "light": "Light clutter: 1–3 darts or O's on OTHER cells.",
    "heavy": "Heavy clutter: 4–8 darts or O's on OTHER cells.",
    "late": "Late game: 9–14 darts or O's on OTHER cells.",
}

PAGE = r"""<!doctype html>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>record</title>
<style>
  :root { color-scheme: dark; --card: #181818; --line: #2e2e2e; --mute: #8f9a9a; }
  html, body { height: 100%; }
  body { font: 15px/1.45 ui-sans-serif, system-ui; margin: 0; background: #0e0e0e; color: #eee; overflow: hidden; }
  main { height: 100vh; box-sizing: border-box; max-width: none; margin: 0; padding: 8px 12px; display: flex; flex-direction: column; }
  header { display: flex; align-items: baseline; gap: 14px; flex-wrap: wrap; margin-bottom: 6px; flex: 0 0 auto; }
  #target { font-size: 32px; font-weight: 700; line-height: 1; }
  #sub { color: var(--mute); font-size: 16px; }
  .chips { margin-left: auto; display: flex; gap: 6px; flex-wrap: wrap; }
  .chip { font-size: 12px; padding: 4px 9px; border-radius: 999px; background: var(--card); border: 1px solid var(--line); color: var(--mute); }
  .chip b { color: #eee; } .chip.bad b { color: #f87171; } .chip.good b { color: #86efac; }
  .layout { display: grid; grid-template-columns: 1fr 300px; gap: 16px; align-items: start; }
  .card { background: var(--card); border: 1px solid var(--line); border-radius: 12px; padding: 14px; margin-bottom: 14px; }
  .card h3 { margin: 0 0 10px; font-size: 12px; letter-spacing: .08em; text-transform: uppercase; color: var(--mute); }
  .line { padding: 8px 10px; border-radius: 8px; background: #111; border: 1px solid var(--line); margin-bottom: 8px; }
  .master-on { background: #7f1d1d; border-color: #ef4444; color: #fff; font-weight: 700; font-size: 17px; }
  .todo { background: #450a0a; border: 1px solid #b91c1c; color: #fecaca; font-weight: 600; padding: 10px; border-radius: 8px; margin-bottom: 8px; }
  #state { font-size: 18px; font-weight: 600; margin: 12px 0 10px; }
  .REC { color: #f87171; } .IDLE { color: #86efac; } .SAVING, .BLOCKED { color: #fbbf24; }
  .row { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }
  button { font: inherit; padding: 9px 14px; border: 0; border-radius: 8px; cursor: pointer; color: #fff; background: #333a44; }
  button:disabled { opacity: .35; cursor: default; }
  .ok { background: #15803d; } .warnbtn { background: #92400e; } .blue { background: #2563eb; }
  #rec-slot { display: flex; flex-direction: column; gap: 8px; }
  #main-btn { width: 100%; font-size: 16px; padding: 14px 8px; background: #b91c1c; position: relative; z-index: 2; }
  #main-btn.stop { background: #374151; }
  .small { font-size: 13px; color: var(--mute); }
  #msg { color: var(--mute); font-size: 13px; margin-top: 8px; min-height: 18px; } #err { color: #f87171; font-size: 13px; }
  table { border-collapse: collapse; width: 100%; }
  #grid-card { margin: 0; padding: 8px; }
  #grid-head { display: flex; align-items: baseline; justify-content: space-between; gap: 8px; }
  #grid-head h3 { margin: 0; }
  #left { font-size: 13px; font-weight: 700; color: #bbf7d0; white-space: nowrap; }
  details.more { margin-top: 8px; font-size: 12px; color: var(--mute); }
  details.more summary { cursor: pointer; }
  #grid td { border: 1px solid var(--line); text-align: center; padding: 6px 1px; font-size: 13px; font-weight: 650; }
  #grid td.cur { outline: 2px solid #22c55e; }
  #grid td.done { background: #14532d; color: #bbf7d0; }
  #grid td.done.dart { color: #bbf7d0; }
  #grid td.dart { color: #fca5a5; }
  .cams { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
  .stage { flex: 1; min-height: 0; display: grid; grid-template-columns: minmax(0, 1fr) 240px; grid-template-rows: auto minmax(0, 1fr); gap: 8px; margin: 0 0 8px; align-items: stretch; }
  .stage.rear-big { grid-template-columns: 240px minmax(0, 1fr); }
  #wrist-view, #front-view { min-height: 0; display: flex; flex-direction: column; }
  #wrist-view .frame { flex: 1; min-height: 0; }
  #wrist-view img, .stage.rear-big #front-view img { flex: 1; min-height: 0; height: 100%; aspect-ratio: auto; }
  .stage:not(.rear-big) #front-view img { max-height: 120px; }
  .stage.rear-big #wrist-view img { max-height: 120px; aspect-ratio: auto; }
  #rec-slot { min-height: 0; overflow: auto; }
  .layout { display: none; }
  #review.layout-review { display: none; }
  .stage #wrist-view { grid-column: 1; grid-row: 1 / span 2; } .stage #front-view { grid-column: 2; grid-row: 1; }
  .stage #rec-slot { grid-column: 2; grid-row: 2; }
  .stage.rear-big #front-view { grid-column: 2; grid-row: 1 / span 2; }
  .stage.rear-big #wrist-view { grid-column: 1; grid-row: 1; }
  .stage.rear-big #rec-slot { grid-column: 1; }
  figure { margin: 0; background: #000; border: 1px solid var(--line); border-radius: 10px; overflow: hidden; }
  figure img { display: block; width: 100%; object-fit: contain; background: #000; }
  .frame { position: relative; cursor: crosshair; }
  .marker { position: absolute; pointer-events: none; width: 22px; height: 22px; margin: -11px 0 0 -11px; border: 2px solid #f87171; border-radius: 50%; box-shadow: 0 0 0 1px #000; }
  .marker::before, .marker::after { content: ""; position: absolute; background: #f87171; }
  .marker::before { left: 50%; top: -6px; bottom: -6px; width: 2px; margin-left: -1px; }
  .marker::after { top: 50%; left: -6px; right: -6px; height: 2px; margin-top: -1px; }
  .stage .pip { cursor: pointer; }
  figcaption { font-size: 12px; color: var(--mute); padding: 6px 10px; background: var(--card); }
  #review { display: none; position: fixed; z-index: 30; inset: 16px; overflow: auto; margin: 0;
    border: 2px solid #b45309; box-shadow: 0 0 0 100vmax rgba(0,0,0,.72); }
  #review .cams { margin: 10px 0; }
  #rv-table { max-height: 300px; overflow: auto; margin-top: 10px; }
  #rv-table table { font: 12px ui-monospace, monospace; } #rv-table td, #rv-table th { padding: 2px 6px; border: 1px solid var(--line); text-align: right; }
  #rv-table th { position: sticky; top: 0; background: #222; } #rv-table tr.trig td { background: #422006; } #rv-table tr.now td { outline: 1px solid #22c55e; }
  .rv-bar input { flex: 1; }
  input[type=number] { font: inherit; width: 64px; background: #111; color: #eee; border: 1px solid var(--line); border-radius: 6px; padding: 6px; }
  @media (max-width: 800px) { .layout, .cams { grid-template-columns: 1fr; } #target { font-size: 42px; } .chips { margin-left: 0; } }
</style>
<main>
  <header><div id="target">—</div><div id="sub"></div><div class="chips" id="chips"></div></header>
  <div class="stage" id="stage">
    <figure id="wrist-view">
      <div class="frame" id="barrel-frame">
        <img id="wrist" alt="gun camera">
        <div class="marker" id="barrel-marker" hidden></div>
      </div>
      <figcaption>GUN / barrel — click to set the red mark saved into the video <button type="button" id="clear-mark" style="margin-left:8px;padding:2px 8px;font-size:12px">clear</button></figcaption>
    </figure>
    <figure class="pip" id="front-view">
      <img id="front" alt="rear camera">
      <figcaption id="rear-cap">REAR — click to enlarge</figcaption>
    </figure>
    <div id="rec-slot">
      <button id="main-btn">● Record (R)</button>
      <button id="pause">⏸ Pause teleop</button>
      <div id="state">…</div>
      <div id="err"></div><div id="msg"></div>
      <div class="card" id="grid-card"><div id="grid-head"><h3>Shoot this cell</h3><span id="left">—</span></div><div id="todo"></div><table id="grid"></table><div class="small" id="prog" style="margin-top:8px"></div></div>
      <details class="more"><summary>More options</summary>
        <div class="row" style="margin-top:8px"><span class="small">End take</span><input id="after" type="number" min="0.3" max="5" step="0.1"><span class="small">s after the shot</span><button id="after-save">Set</button></div>
      </details>
    </div>
  </div>
  <div class="layout">
    <div>
      <div class="card">
        <h3>This take</h3>
        <div class="line" id="fire"></div>
        <div class="line" id="board"></div>
        <div class="line" id="steps"></div>
      </div>
      <div class="card">
        <h3>Board</h3>
        <div class="row"><span class="small" id="board-stat"></span><button id="collected" style="margin-left:auto">Darts collected ✓</button></div>
      </div>
    </div>
    <div>
      <div class="card"><div class="row"><button id="skip">Skip cell</button><button id="upload" class="blue">Upload now</button></div></div>
      <div class="card"><h3>Setup</h3>
        <div class="row" style="margin-bottom:8px"><button id="cams" class="ok">Cameras look right ✓</button></div>
        <div class="row" style="margin-bottom:8px"><button id="rest">Set rest pose = current</button></div>
      </div>
    </div>
  </div>
  <div class="card" id="review">
    <h3 id="rv-title"></h3>
    <div class="row"><button id="rv-good" class="ok">✓ Good (G): keep + upload</button><button id="rv-bad" class="warnbtn">✗ Bad (B): delete, redo</button>
      <span class="small">Record also counts as Good</span></div>
    <div class="cams">
      <figure><img id="rv-front"><figcaption>saved front_camera (rear)</figcaption></figure>
      <figure><img id="rv-wrist"><figcaption>saved wrist_camera (gun)</figcaption></figure>
    </div>
    <div class="row rv-bar"><button id="rv-play">▶ Play</button><input type="range" id="rv-slider" min="0" value="0"><span id="rv-step" class="small"></span></div>
    <div class="small" id="rv-info" style="margin-top:6px"></div>
    <div id="rv-table"></div>
  </div>
</main>
<script>
const post = (p) => fetch(p, {method: 'POST'});
const $ = (id) => document.getElementById(id);
let S = {};
for (const [id, path] of [['collected','/collected'],['skip','/skip'],['upload','/upload'],['cams','/cams_ok'],['rest','/set_rest'],['pause','/pause']]) $(id).onclick = () => post(path);
$('main-btn').onclick = () => post(S.state === 'REC' ? '/discard' : '/record');
document.onkeydown = (e) => { if (e.target.tagName === 'INPUT') return; const k = e.key.toLowerCase();
  if (k === 'r') post('/record'); if (k === 'x') post('/discard'); if (k === 'b') post('/bad'); if (k === 'g') post('/good');
  if (k === 'p') post('/pause'); };
async function tick() {
  try {
    const s = S = await (await fetch('/status')).json();
    $('target').textContent = s.cell ? `shoot ${s.cell}` : 'all done ✓';
    $('sub').textContent = s.cell ? `take ${s.index + 1} / ${s.total} · pass ${s.pass}` : '';
    const u = s.uploads;
    $('chips').innerHTML = `<span class="chip ${s.loop_hz >= 27 ? 'good' : 'bad'}">loop <b>${s.loop_hz} Hz</b></span>`
      + `<span class="chip">saved <b>${s.saved}</b></span><span class="chip">uploaded <b>${s.uploaded}</b></span>`
      + (u.failed ? `<span class="chip bad">upload failed <b>${u.failed}</b></span>` : '')
      + (s.fake ? '<span class="chip bad"><b>FAKE</b></span>' : '');
    $('fire').className = 'line ' + (s.live ? 'master-on' : '');
    $('fire').textContent = !s.cell ? '' : s.live ? 'MASTER SWITCH ON: live dart' : 'Master switch OFF: dry (label only)';
    $('board').textContent = s.board_help || '';
    $('steps').innerHTML = !s.cell ? '' : `Start wherever the arm is now. Record → aim at <b>${s.cell}</b>, hold still ½ s → press + release the fire button once → keep still; it saves ${s.after_shot_s} s later`;
    $('state').className = s.state === 'IDLE' && s.blocked ? 'BLOCKED' : s.state;
    $('state').textContent = s.state === 'REC' ? `● recording ${s.rec_s.toFixed(1)} s ${s.shot ? '· shot ✓ finishing…' : '· aim, then fire'}` :
      s.state === 'SAVING' ? 'saving…' : (s.blocked || 'ready: Record');
    const mb = $('main-btn');
    mb.className = s.state === 'REC' ? 'stop' : ''; mb.textContent = s.state === 'REC' ? '■ Discard this take (X)' : '● Record (R)';
    mb.disabled = s.state === 'SAVING' || (s.state === 'IDLE' && !!s.blocked);
    $('todo').innerHTML = '';
    $('board-stat').textContent = `darts tracked ≈ ${s.darts_total} · live shots since collect ${s.live_since_collect}/${s.collect_at}`;
    $('pause').textContent = s.paused ? '▶ Resume teleop (P)' : '⏸ Pause teleop (P)';
    $('cams').parentElement.style.display = s.cams_ok ? 'none' : '';
    if (document.activeElement !== $('after')) $('after').value = s.after_shot_s;
    $('err').textContent = s.arm_error || s.esp_error || '';
    $('msg').textContent = s.msg || '';
    const rows = [];
    for (const r of '12345') { rows.push('<tr>' + [...'ABCDE'].map(c => { const k = c + r, d = s.done[k] || 0, t = s.target[k] || 0, darts = s.darts[k] || 0;
      return `<td class="${k === s.cell ? 'cur' : ''} ${t && d >= t ? 'done' : ''} ${darts ? 'dart' : ''}">${k}${darts ? ' •' + darts : ''}<br>${t ? d + '/' + t : '·'}</td>`; }).join('') + '</tr>'); }
    $('grid').innerHTML = rows.join('');
    let targetN = 0, doneN = 0;
    for (const k in s.target) { targetN += s.target[k]; doneN += s.done[k] || 0; }
    $('left').textContent = (targetN - doneN) + ' left';
    $('prog').innerHTML = `→ ${s.dataset_path}<br>` + (s.auto_upload ? `uploads: ${u.ok} ok · ${u.pending} pending · ${u.failed} failed<br>${u.last}` : 'auto-upload off');
  } catch (e) { $('err').textContent = 'app not reachable: ' + e; }
  const t = Date.now(); $('front').src = '/cam/front_camera?t=' + t; $('wrist').src = '/cam/wrist_camera?t=' + t;
  setTimeout(tick, 250);
}
function showMarker(x, y) {
  const m = $('barrel-marker');
  m.hidden = false; m.style.left = (x * 100) + '%'; m.style.top = (y * 100) + '%';
}
function placeMarker(e) {
  const frame = $('barrel-frame');
  const r = frame.getBoundingClientRect();
  const x = Math.min(1, Math.max(0, (e.clientX - r.left) / r.width));
  const y = Math.min(1, Math.max(0, (e.clientY - r.top) / r.height));
  showMarker(x, y);
  localStorage.setItem('barrel-marker', JSON.stringify({x, y}));
  post('/marker?x=' + x + '&y=' + y);
}
(function () {
  const saved = localStorage.getItem('barrel-marker');
  if (!saved) return;
  try {
    const p = JSON.parse(saved);
    showMarker(p.x, p.y);
    post('/marker?x=' + p.x + '&y=' + p.y);
  } catch (e) {}
})();
$('barrel-frame').onclick = (e) => {
  if ($('stage').classList.contains('rear-big')) { $('stage').classList.remove('rear-big'); $('rear-cap').textContent = 'REAR — click to enlarge'; return; }
  placeMarker(e);
};
$('front-view').onclick = () => {
  if ($('stage').classList.contains('rear-big')) return;
  $('stage').classList.add('rear-big');
  $('rear-cap').textContent = 'REAR — click the gun thumb to go back';
};
$('clear-mark').onclick = (e) => { e.stopPropagation(); $('barrel-marker').hidden = true; localStorage.removeItem('barrel-marker'); post('/marker/clear'); };
let rv = null, rvTimer = null;
$('after-save').onclick = () => post('/settings?after_shot_s=' + encodeURIComponent($('after').value));
$('rv-good').onclick = () => post('/good'); $('rv-bad').onclick = () => post('/bad');
function rvShow(i) {
  if (!rv) return; i = Math.max(0, Math.min(rv.rows.length - 1, i));
  $('rv-slider').value = i; const r = rv.rows[i];
  $('rv-step').textContent = `step ${i} · ${r[0].toFixed(2)} s${r[11] ? ' · TRIGGER' : ''}`;
  $('rv-front').src = `/review/frame/front_camera/${i}?e=${rv.episode}`; $('rv-wrist').src = `/review/frame/wrist_camera/${i}?e=${rv.episode}`;
  document.querySelectorAll('#rv-table tr.now').forEach(x => x.classList.remove('now'));
  const row = document.getElementById('rv-r' + i); if (row) { row.classList.add('now'); row.scrollIntoView({block: 'nearest'}); }
}
$('rv-slider').oninput = (e) => { clearInterval(rvTimer); rvTimer = null; $('rv-play').textContent = '▶ Play'; rvShow(+e.target.value); };
$('rv-play').onclick = () => {
  if (rvTimer) { clearInterval(rvTimer); rvTimer = null; $('rv-play').textContent = '▶ Play'; return; }
  $('rv-play').textContent = '⏸ Pause';
  rvTimer = setInterval(() => { const i = +$('rv-slider').value + 1; if (i >= rv.rows.length) { clearInterval(rvTimer); rvTimer = null; $('rv-play').textContent = '▶ Play'; } else rvShow(i); }, 1000 / 15);
};
async function rvLoad(name) {
  if (!name) { rv = null; $('review').style.display = 'none'; return; }
  if (rv && rv.episode === name) return;
  rv = {episode: name, rows: []};
  const d = await (await fetch('/review')).json();
  if (!d.episode) return;
  rv = d; $('review').style.display = 'block';
  $('rv-title').textContent = `Review last take: ${d.episode} · "${d.task}"`;
  $('rv-info').textContent = `read back from the saved file: ${d.rows.length} steps · ${d.hz} Hz · trigger ${d.trigger_steps} steps from step ${d.trigger_start} · F = follower (observation), L = leader (action)`;
  $('rv-slider').max = d.rows.length - 1;
  const head = '<tr><th>step</th><th>t (s)</th>' + d.joints.map((j, k) => `<th title="${j}">F${k + 1}</th>`).join('')
    + d.joints.map((j, k) => `<th title="${j}">L${k + 1}</th>`).join('') + '<th>trigger</th></tr>';
  $('rv-table').innerHTML = '<table>' + head + d.rows.map((r, i) => `<tr id="rv-r${i}" class="${r[11] ? 'trig' : ''}" onclick="rvShow(${i})"><td>${i}</td>`
    + r.map((v, k) => `<td>${k === 11 ? v : v.toFixed(k ? 1 : 2)}</td>`).join('') + '</tr>').join('') + '</table>';
  $('rv-table').insertAdjacentHTML('afterbegin', `<div style="color:#9aa;font-size:12px;margin-bottom:4px">${d.joints.map((j, k) => `${k + 1} = ${j}`).join(' · ')}</div>`);
  rvShow(Math.max(0, d.trigger_start));
}
setInterval(async () => { try { const s = await (await fetch('/status')).json(); rvLoad(s.pending); } catch (e) {} }, 500);
tick();
</script>
"""


def jpeg(rgb: np.ndarray | None) -> bytes | None:
    if rgb is None:
        return None
    import cv2

    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 70])
    return buf.tobytes() if ok else None


class Session:
    """Schedule + progress, persisted to schedule.json.

    The schedule is in passes: pass k shoots every cell once under condition SPREAD[k] (shuffled cells),
    so the board and master switch change only between passes and stopping early stays balanced.
    Passes run from least to most clutter, so live darts can accumulate naturally."""

    def __init__(self, session_dir: Path, cells: list[str], per_cell: int, operator: str):
        self.dir, self.operator = session_dir, operator
        self.dir.mkdir(parents=True, exist_ok=True)
        path = self.dir / "schedule.json"
        if path.exists():
            self.items = json.loads(path.read_text())
            return
        passes = sorted(range(per_cell), key=lambda k: BOARD_LEVEL[SPREAD[k % len(SPREAD)][0]])
        self.items = []
        for n, k in enumerate(passes, 1):
            board, live, start = SPREAD[k % len(SPREAD)]
            order = list(cells)  # CLI order: inner / easy cells first, corners last
            for cell in order:
                other = random.choice([c for c in s.CELLS if c != cell])
                self.items.append({"cell": cell, "board": board, "live": live, "pass": n,
                                   "start": "rest" if start == "rest" else other, "episode_id": None})
        self.save()

    def save(self) -> None:
        (self.dir / "schedule.json").write_text(json.dumps(self.items, indent=1))

    def current(self) -> tuple[int, dict | None]:
        for i, it in enumerate(self.items):
            if not it["episode_id"] and not it.get("skipped"):
                return i, it
        return len(self.items), None

    def next_episode_id(self) -> str:
        """<session>-NNNN. Session is final-vector, so ids are final-vector-0000. Task text stays `shoot C5`."""
        prefix = f"{self.dir.name}-"
        taken = [int(p.name[len(prefix):]) for p in self.dir.iterdir()
                 if p.is_dir() and p.name.startswith(prefix) and p.name[len(prefix):].isdigit()]
        return f"{prefix}{max(taken, default=-1) + 1:04d}"


class Board:
    """Darts the app knows are on the board (from live shots), persisted to board.json.
    It can't see darts/O's you place by hand, or where a missed dart landed (it's counted on the target cell).
    Each board job can be done (Cleaned ✓) or ignored; that choice is saved in the next take's notes."""

    def __init__(self, session_dir: Path):
        self.path = session_dir / "board.json"
        d = json.loads(self.path.read_text()) if self.path.exists() else {}
        self.darts: dict[str, int] = d.get("darts", {})
        self.live_since_collect: int = d.get("live_since_collect", 0)
        self.collect_at: int = d.get("collect_at") or random.randint(*COLLECT_RANGE)
        self.ignored: dict = d.get("ignored", {"idx": -1, "kinds": []})  # jobs ignored for one schedule item
        self.action: str = d.get("action", "none")  # what the operator did since the last take: none | cleaned | ignored

    def save(self) -> None:
        self.path.write_text(json.dumps({"darts": self.darts, "live_since_collect": self.live_since_collect,
                                         "collect_at": self.collect_at, "ignored": self.ignored, "action": self.action}))

    def add_shot(self, cell: str) -> None:
        self.darts[cell] = self.darts.get(cell, 0) + 1
        self.live_since_collect += 1
        self.save()

    def collected(self) -> None:
        self.darts, self.live_since_collect, self.action = {}, 0, "cleaned"
        self.collect_at = random.randint(*COLLECT_RANGE)
        self.save()

    def pulled(self, cell: str) -> None:
        self.darts.pop(cell, None)
        self.action = "cleaned"
        self.save()

    def ignore(self, kind: str, idx: int) -> None:
        if kind == "collect":  # keep the darts, start a new count
            self.live_since_collect, self.collect_at = 0, random.randint(*COLLECT_RANGE)
        else:
            if self.ignored["idx"] != idx:
                self.ignored = {"idx": idx, "kinds": []}
            self.ignored["kinds"].append(kind)
        self.action = "ignored"
        self.save()

    def took(self) -> str:
        """Called when a take is saved: returns and resets what was done to the board before it."""
        action, self.action = self.action, "none"
        self.save()
        return action

    def total(self) -> int:
        return sum(self.darts.values())

    def todo(self, item: dict | None, idx: int = -1) -> list[dict]:
        """Blocking board jobs. Each has a Cleaned button (fix) and an Ignore button (kind)."""
        out, total = [], self.total()
        skip = self.ignored.get("kinds", []) if self.ignored.get("idx") == idx else []
        where = ", ".join(f"{c}×{n}" if n > 1 else c for c, n in sorted(self.darts.items()))
        job = lambda kind, fix, text: kind not in skip and out.append({"kind": kind, "fix": fix, "text": text})
        tail = ": master switch OFF, pull the darts, reload if the drum is low"
        if self.live_since_collect >= self.collect_at:
            job("collect", "/collected", f"COLLECT DARTS: {self.live_since_collect} live shots since the last collection ({where}){tail}")
        if item and item["board"] == "clean" and total:
            job("clean", "/collected", f"Clean board needed: {total} dart(s) on it ({where}){tail}")
        elif item and total > BOARD_MAX_DARTS[item["board"]]:
            job("toomany", "/collected", f"Too many darts for {item['board']} clutter ({total} > {BOARD_MAX_DARTS[item['board']]}: {where}){tail}")
        if item and self.darts.get(item["cell"]):
            job("target", "/pulled", f"Pull the dart out of {item['cell']} (the target cell)")
        return out[:1]


class Uploader:
    """One background worker for all uploads (auto and the Upload button): no double uploads,
    retries with backoff, and leftovers are re-queued on restart."""

    MAX_ATTEMPTS = 5
    RETRY_S = 30

    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        self.q: queue.Queue = queue.Queue()
        self.in_flight: set[str] = set()
        self.stats = {"ok": 0, "failed": 0, "pending": 0, "last": ""}
        threading.Thread(target=self._run, daemon=True).start()

    def enqueue(self, ep_dir: Path) -> None:
        if str(ep_dir) in self.in_flight or (ep_dir / "uploaded.json").exists():
            return
        self.in_flight.add(str(ep_dir))
        self.stats["pending"] += 1
        self.q.put((ep_dir, 1))

    def _run(self) -> None:
        from upload import upload_one

        data, contract_id = None, None
        while True:
            ep_dir, attempt = self.q.get()
            try:
                if contract_id is None:
                    d, _ = s.clients()
                    contract_id, data = s.ensure_contract(d), d
                t0 = time.perf_counter()
                validate(ep_dir)
                path = upload_one(data, contract_id, ep_dir, self.dataset_path)
                result, done, ok = f"uploaded {ep_dir.name} in {time.perf_counter() - t0:.1f} s → {path} (finalized)", True, True
            except Exception as e:
                ok = False
                done = attempt >= self.MAX_ATTEMPTS or not ep_dir.exists()
                result = (f"UPLOAD FAILED {ep_dir.name}: {type(e).__name__}: {e}"
                          + ("" if done else f" (retry {attempt + 1}/{self.MAX_ATTEMPTS} in {self.RETRY_S} s)"))
                if not done:
                    threading.Timer(self.RETRY_S, self.q.put, args=((ep_dir, attempt + 1),)).start()
            with STATE.lock:
                if done:
                    self.in_flight.discard(str(ep_dir))
                    self.stats["pending"] -= 1
                    self.stats["ok" if ok else "failed"] += 1
                self.stats["last"] = result
            print(result, flush=True)


class State:
    def __init__(self):
        self.lock = threading.Lock()
        self.mode = "IDLE"  # IDLE | REC | SAVING
        self.steps: list[Step] = []
        self.shot_step: int | None = None
        self.hold = 0
        self.msg = ""
        self.esp_error: str | None = None
        self.frames: dict[str, np.ndarray | None] = {cam: None for cam in s.CAMERAS}
        self.joints: np.ndarray | None = None
        self.stop = threading.Event()
        self.test_shot = False  # --fake only: POST /test/shoot simulates the fire button
        self.loop_hz = 0.0
        self.pending: Path | None = None  # last saved take, still rejectable with Bad take
        self.review_steps: list[Step] = []  # its frames, for the review panel
        self.cams_ok = True  # don't block Record on a hidden setup button; the previews are already on screen
        self.paused = True  # start limp; the operator presses Resume teleop to follow the leader
        self.arm_error: str | None = None  # set while the follower/leader bus is down; the page stays up
        self.limp = False  # follower torque released while paused
        self.marker: tuple[float, float] | None = None  # cyan plus burned into the gun frame


STATE = State()
SESSION: Session | None = None
BOARD: Board | None = None
UPLOADER: Uploader | None = None
CFG: dict = {}  # dataset_path, fake, auto_upload, cams


def finish(keep: bool, reason: str) -> None:
    """End the current take (called with STATE.lock held)."""
    steps, STATE.steps = STATE.steps, []
    STATE.shot_step, STATE.hold = None, 0
    if not keep or not steps:
        STATE.mode, STATE.msg = "IDLE", f"not saved ({reason}); same cell again"
        return
    span = max((steps[-1].time_ns - steps[0].time_ns) / 1e9, 1e-6)
    hz = (len(steps) - 1) / span
    gap_ms = max((b.time_ns - a.time_ns) / 1e6 for a, b in zip(steps, steps[1:])) if len(steps) > 1 else 0
    if hz < MIN_HZ or gap_ms > MAX_GAP_MS:
        STATE.mode, STATE.msg = "IDLE", f"NOT saved: {hz:.1f} Hz, max gap {gap_ms:.0f} ms (need ≥{MIN_HZ:.0f} Hz, ≤{MAX_GAP_MS} ms); same cell again"
        return
    STATE.mode = "SAVING"
    threading.Thread(target=save_episode, args=(steps, hz, gap_ms, reason), daemon=True).start()


def save_episode(steps: list[Step], hz: float, gap_ms: float, reason: str) -> None:
    i, item = SESSION.current()
    episode_id = SESSION.next_episode_id()
    ep_dir = SESSION.dir / episode_id
    try:
        notes = (f"board={item['board']} live={int(item['live'])} start={item['start']} cams={CFG['cams']} "
                 f"after_shot_s={CFG['after_shot_s']} tracked_darts={BOARD.total()} board_action={BOARD.took()}")
        write_episode(ep_dir, steps, episode_id=episode_id, cell=item["cell"], operator=SESSION.operator,
                      notes=notes, dataset_path=CFG["dataset_path"], fake=CFG["fake"])
        validate(ep_dir)
        item["episode_id"] = episode_id
        SESSION.save()
        msg = (f"saved {episode_id}: shoot {item['cell']}, {len(steps)} steps, {hz:.1f} Hz, max gap {gap_ms:.0f} ms. "
               f"Good (G) or Bad (B)? e.g. Bad if a live dart missed. Pressing Record counts it as Good")
        pending = ep_dir
    except Exception as e:
        shutil.rmtree(ep_dir, ignore_errors=True)  # no half-written episode left behind
        msg, pending = f"save FAILED (same cell again): {type(e).__name__}: {e}", None
    with STATE.lock:
        STATE.mode, STATE.msg, STATE.pending = "IDLE", msg, pending
        STATE.review_steps = steps if pending else []
    print(msg, flush=True)


def accept_pending() -> None:
    """The last take is accepted once the operator moves on (lock held)."""
    if STATE.pending is not None:
        if CFG["auto_upload"]:
            UPLOADER.enqueue(STATE.pending)
        STATE.pending, STATE.review_steps = None, []


def blocked_reason(item: dict | None) -> str | None:
    """Why Record is disabled right now (lock held)."""
    if STATE.arm_error:
        return "arm not connected — the page stays up and retries. Power-cycle the follower 12V."
    if item is None:
        return "all episodes in this session are done"
    if STATE.paused:
        return "paused: press Resume teleop"
    if STATE.esp_error:
        return "fire button / ESP32 not reachable (see error)"
    return None


def control_loop(rig, esp) -> None:
    period, next_tick, ticks = 1 / s.FPS, time.perf_counter(), []
    while not STATE.stop.is_set():
        try:
            rig.crosshair = STATE.marker
            if STATE.paused:
                if not STATE.limp:
                    rig.torque(False)
                    STATE.limp = True
                images, joints = rig.observe()
                with STATE.lock:
                    STATE.frames, STATE.joints, STATE.loop_hz = images, joints, 0.0
                esp.events()  # drop presses while paused
                time.sleep(0.1)
                next_tick = time.perf_counter()
                continue
            if STATE.limp:
                rig.torque(True)
                STATE.limp = False
            act_joints = rig.leader_action()
            rig.command(act_joints)
            images, joints = rig.observe()
        except Exception as e:  # servo bus / camera hiccup: drop the take, keep the loop alive
            with STATE.lock:
                if STATE.mode == "REC":
                    finish(False, f"arm/camera error: {type(e).__name__}: {e}")
                STATE.msg = f"arm/camera read failed: {type(e).__name__}: {e}"
            time.sleep(0.2)
            next_tick = time.perf_counter()
            continue
        now = time.perf_counter()
        ticks = [t for t in ticks if now - t < 1.0] + [now]
        try:
            events, esp_error = esp.events(), None
        except Exception as e:  # rigd restarting / timeout: keep teleop running, block recording
            events, esp_error = [], f"fire button / ESP32 read failed: {type(e).__name__}: {e}"
        with STATE.lock:
            STATE.esp_error = esp_error
            STATE.loop_hz = len(ticks) - 1 if len(ticks) > 1 else 0.0
            STATE.frames, STATE.joints = images, joints
            if "MODE_RESET" in events:
                STATE.msg = "the ESP32 reported a different mode (rebooted?): re-set to recording mode"
            n = sum(ev.startswith("SHOOT") for ev in events) + STATE.test_shot
            STATE.test_shot = False
            _, item = SESSION.current()
            if n and item and item["live"]:
                for _ in range(n):  # count darts at the event, so discarded takes still count
                    BOARD.add_shot(item["cell"] if STATE.mode == "REC" else "?")
            if n and STATE.mode != "REC":
                STATE.msg = "fire button pressed while not recording" + (": a live dart was fired, marked as '?'" if item and item["live"] else "")
            if STATE.mode == "REC":
                if esp_error:
                    finish(False, "lost the fire button / ESP32 during the take")
                elif n and STATE.shot_step is not None:
                    finish(False, "second shot in one take (a live second dart isn't labeled)")
                else:
                    if n:
                        STATE.hold, STATE.shot_step = TRIGGER_HOLD_STEPS, len(STATE.steps)
                    STATE.steps.append(Step(time.time_ns(), images, joints, act_joints, 1.0 if STATE.hold > 0 else 0.0))
                    STATE.hold = max(STATE.hold - 1, 0)
                    if STATE.shot_step is not None and len(STATE.steps) - STATE.shot_step >= round(CFG["after_shot_s"] * s.FPS):
                        finish(True, "shot")
        next_tick += period
        time.sleep(max(0.0, next_tick - time.perf_counter()))


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        return

    def _send(self, code: int, body: bytes, ctype: str = "application/json") -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            return self._send(200, PAGE.encode(), "text/html; charset=utf-8")
        if path == "/status":
            return self._send(200, json.dumps(self.status()).encode())
        if path == "/review":
            return self._send(200, json.dumps(self.review()).encode())
        if path.startswith("/review/frame/"):
            _, _, _, cam, i = path.split("/")
            with STATE.lock:
                steps = STATE.review_steps
                frame = steps[int(i)].images.get(cam) if i.isdigit() and int(i) < len(steps) else None
            blob = jpeg(frame)
            return self._send(200, blob, "image/jpeg") if blob else self._send(204, b"", "image/jpeg")
        if path.startswith("/cam/"):
            with STATE.lock:
                frame = STATE.frames.get(path.rsplit("/", 1)[-1])
            blob = jpeg(frame)
            return self._send(200, blob, "image/jpeg") if blob else self._send(204, b"", "image/jpeg")
        self._send(404, b"not found", "text/plain")

    def review(self) -> dict:
        """The pending take as saved on disk (light.mcap read back): one row per step."""
        from inspect_episode import arrays

        with STATE.lock:
            ep = STATE.pending
        if ep is None:
            return {}
        t, obs = arrays(ep, s.OBS_JOINTS)
        _, act = arrays(ep, s.ACT_JOINTS)
        _, trig = arrays(ep, s.ACT_TRIGGER)
        secs = (t - t[0]) / 1e9
        on = np.flatnonzero(trig[:, 0] > 0.5)
        rows = [[round(float(secs[i]), 3), *map(float, obs[i]), *map(float, act[i]), int(trig[i, 0] > 0.5)] for i in range(len(t))]
        return {"episode": ep.name, "task": load_meta(ep).task, "joints": list(s.JOINT_NAMES), "rows": rows,
                "hz": round((len(t) - 1) / max(float(secs[-1]), 1e-6), 1), "trigger_steps": int(on.size),
                "trigger_start": int(on[0]) if on.size else -1}

    def status(self) -> dict:
        i, item = SESSION.current()
        target, done = {}, {}
        for it in SESSION.items:
            if it.get("skipped"):
                continue
            target[it["cell"]] = target.get(it["cell"], 0) + 1
            done[it["cell"]] = done.get(it["cell"], 0) + bool(it["episode_id"])
        episodes = find_episodes(SESSION.dir)
        rest = load_rest_pose()
        with STATE.lock:
            rest_dist = None if rest is None or STATE.joints is None else float(np.abs(STATE.joints - rest).max())
            return {"state": STATE.mode, "msg": STATE.msg, "esp_error": STATE.esp_error, "blocked": blocked_reason(item),
                    "index": i, "total": len(SESSION.items), "cell": item and item["cell"], "pass": item and item["pass"],
                    "live": bool(item and item["live"]), "start": item and item["start"],
                    "board_help": item and BOARD_HELP[item["board"]] + " Nothing in the target cell.",
                    "todo": BOARD.todo(item, i), "darts": {c: n for c, n in BOARD.darts.items()}, "darts_total": BOARD.total(),
                    "live_since_collect": BOARD.live_since_collect, "collect_at": BOARD.collect_at, "after_shot_s": CFG["after_shot_s"],
                    "rec_s": len(STATE.steps) / s.FPS, "shot": STATE.shot_step is not None,
                    "pending": STATE.pending.name if STATE.pending else None, "cams_ok": STATE.cams_ok, "paused": STATE.paused,
                    "arm_error": STATE.arm_error,
                    "rest_dist": rest_dist, "rest_tol": REST_TOL_DEG,
                    "target": target, "done": done, "saved": len(episodes),
                    "uploaded": sum((ep / "uploaded.json").exists() for ep in episodes),
                    "loop_hz": round(STATE.loop_hz, 1), "auto_upload": CFG["auto_upload"],
                    "uploads": dict(UPLOADER.stats), "dataset_path": CFG["dataset_path"], "fake": CFG["fake"]}

    def do_POST(self):
        from urllib.parse import parse_qs, urlsplit

        url = urlsplit(self.path)
        with STATE.lock:
            idx, item = SESSION.current()
            if url.path == "/marker":
                try:
                    q = parse_qs(url.query)
                    STATE.marker = (float(q["x"][0]), float(q["y"][0]))
                    _save_marker()
                    STATE.msg = f"red mark at x={STATE.marker[0]:.2f} y={STATE.marker[1]:.2f} (saved into the gun video)"
                except (KeyError, ValueError):
                    STATE.msg = "mark position must be x and y between 0 and 1"
                return self._send(200, b'{"ok":true}')
            if url.path == "/marker/clear":
                STATE.marker = None
                _save_marker()
                STATE.msg = "red mark cleared; gun video has no mark"
                return self._send(200, b'{"ok":true}')
            if url.path == "/settings" and STATE.mode == "IDLE":
                try:
                    v = float(parse_qs(url.query)["after_shot_s"][0])
                    CFG["after_shot_s"] = round(min(max(v, 0.3), 5.0), 2)
                    path = SESSION.dir / "settings.json"  # keep the other keys (the gun mark)
                    saved = json.loads(path.read_text()) if path.exists() else {}
                    path.write_text(json.dumps({**saved, "after_shot_s": CFG["after_shot_s"]}))
                    STATE.msg = f"takes now end {CFG['after_shot_s']} s after the shot (saved for this session)"
                except (KeyError, ValueError):
                    STATE.msg = "after-shot time must be a number of seconds (0.3–2)"
                return self._send(200, b'{"ok":true}')
            if url.path.startswith("/ignore/") and STATE.mode == "IDLE":
                kind = url.path.rsplit("/", 1)[-1]
                if kind in ("collect", "clean", "toomany", "target"):
                    BOARD.ignore(kind, idx)
                    STATE.msg = f"board job ignored ({kind}); noted in the next take"
                return self._send(200, b'{"ok":true}')
            if self.path == "/record" and STATE.mode == "IDLE":
                reason = blocked_reason(item)
                if reason:
                    STATE.msg = f"can't record: {reason}"
                else:
                    accept_pending()
                    STATE.mode, STATE.steps, STATE.shot_step, STATE.hold, STATE.msg = "REC", [], None, 0, ""
            elif self.path == "/discard" and STATE.mode == "REC":
                finish(False, "discarded by hand")
            elif self.path == "/good" and STATE.mode == "IDLE" and STATE.pending is not None:
                name = STATE.pending.name
                accept_pending()
                STATE.msg = f"{name} marked good" + (": uploading" if CFG["auto_upload"] else "")
            elif self.path == "/bad" and STATE.mode == "IDLE" and STATE.pending is not None:
                ep, STATE.pending, STATE.review_steps = STATE.pending, None, []
                for it in SESSION.items:
                    if it["episode_id"] == ep.name:
                        it["episode_id"] = None
                SESSION.save()
                shutil.rmtree(ep, ignore_errors=True)
                STATE.msg = f"deleted bad take {ep.name}; same cell again"
            elif self.path == "/collected" and STATE.mode == "IDLE":
                BOARD.collected()
                STATE.msg = "darts collected: board tracking reset"
            elif self.path == "/pulled" and STATE.mode == "IDLE" and item:
                BOARD.pulled(item["cell"])
                STATE.msg = f"dart pulled from {item['cell']}"
            elif self.path == "/skip" and STATE.mode == "IDLE" and item:
                accept_pending()
                item["skipped"] = True
                SESSION.save()
                STATE.msg = f"skipped {item['cell']}"
            elif self.path == "/upload" and STATE.mode == "IDLE":
                accept_pending()
                for ep in find_episodes(SESSION.dir):
                    UPLOADER.enqueue(ep)
                STATE.msg = "uploading everything not yet uploaded"
            elif self.path == "/pause" and STATE.mode == "IDLE":
                STATE.paused = not STATE.paused
                STATE.msg = ("teleop paused: follower servos are limp, move the arm by hand" if STATE.paused else
                             "teleop resumed: the follower goes straight to the leader, keep clear")
            elif self.path == "/cams_ok":
                STATE.cams_ok = True
            elif self.path == "/set_rest" and STATE.mode == "IDLE" and STATE.joints is not None:
                save_rest_pose(STATE.joints)
                STATE.msg = "rest pose saved to calibration/rest_pose.json: commit + push it so the game uses the same one"
            elif self.path == "/test/shoot" and CFG["fake"]:
                STATE.test_shot = True
        self._send(200, b'{"ok":true}')


def connect_rigd(url: str) -> RigdESP32:
    """rigd may still be starting: retry for ~15 s before giving up."""
    for attempt in range(30):
        try:
            return RigdESP32(url, mode="train")
        except (OSError, RuntimeError) as e:
            if attempt == 0:
                print(f"waiting for rigd at {url} ({e})…", flush=True)
            time.sleep(0.5)
    raise RuntimeError(f"rigd not reachable at {url}: start it first (see docs/data-collection.md)")


def _release_arm(rig) -> None:
    """Close a half-open bus. A failed handshake leaves the serial port open."""
    for dev in (getattr(rig, "follower", None), getattr(rig, "leader", None)):
        bus = getattr(dev, "bus", None)
        if bus is not None and getattr(bus, "is_connected", False):
            try:
                bus.port_handler.closePort()
            except Exception:
                pass


def hardware_thread(rig, esp_factory) -> None:
    """Connect the arms, then run teleop. Failures stay on the page; the HTTP server keeps running."""
    esp = None
    while not STATE.stop.is_set():
        try:
            if esp is None:
                esp = esp_factory()
            rig.connect()
            rig.torque(False)
            with STATE.lock:
                STATE.arm_error = None
                STATE.paused = True
                STATE.limp = True
                STATE.msg = "arm connected, teleop paused. Press Resume teleop to follow the leader."
            print("arm connected, teleop paused", flush=True)
            control_loop(rig, esp)
            break
        except Exception as e:
            text = str(e).strip()
            with STATE.lock:
                STATE.arm_error = text[:800]
                STATE.msg = "website is up. arm connect failed — power-cycle the follower 12V. Retrying."
            print(text, flush=True)
            _release_arm(rig)
            for _ in range(8):
                if STATE.stop.is_set():
                    break
                time.sleep(0.5)
    if esp is not None:
        try:
            esp.close()
        except Exception:
            pass


def main() -> None:
    global SESSION, BOARD, UPLOADER
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--session", help="default <user>-<cells>x<per-cell>, so rerunning the same command resumes")
    p.add_argument("--cells", nargs="+", default=list(s.CELLS), help="default: all 25")
    p.add_argument("--per-cell", type=int, default=10)
    p.add_argument("--operator", default=getpass.getuser())
    p.add_argument("--dataset-path", default=s.DATASET_PATH, help="where uploads go (--fake forces test/...)")
    p.add_argument("--auto-upload", action="store_true", help="upload each accepted take to PI in the background")
    p.add_argument("--follower-port"), p.add_argument("--leader-port"), p.add_argument("--esp32-port")
    p.add_argument("--rigd-url", help="act-athon rigd (e.g. http://localhost:8080) instead of --esp32-port")
    p.add_argument("--follower-id", default="follower_arm"), p.add_argument("--leader-id", default="leader_arm")
    p.add_argument("--front-cam", default="0"), p.add_argument("--wrist-cam", default="1")
    p.add_argument("--port", type=int, default=7870)
    p.add_argument("--fake", action="store_true", help="fake arm + cameras; fake ESP32 unless --rigd-url (POST /test/shoot = button)")
    p.add_argument("--no-browser", action="store_true")
    a = p.parse_args()
    for cell in a.cells:
        s.task_for(cell)
    a.session = a.session or f"{'fake-' if a.fake else ''}{getpass.getuser()}-{len(a.cells)}x{a.per_cell}"
    if a.fake and not a.dataset_path.startswith("test/"):
        a.dataset_path = "test/record-web-fake"  # fake data never goes to a real dataset
    CFG.update(dataset_path=a.dataset_path, fake=a.fake, auto_upload=a.auto_upload,
               cams=f"front:{a.front_cam},wrist:{a.wrist_cam}")

    if a.fake:
        rig = FakeRig()
        esp = connect_rigd(a.rigd_url) if a.rigd_url else FakeESP32(shoot_every=None)
    else:
        if not (a.follower_port and a.leader_port and (a.esp32_port or a.rigd_url)):
            p.error("--follower-port, --leader-port and --esp32-port or --rigd-url are required (or use --fake)")
        cam = lambda v: int(v) if v.isdigit() else v
        rig = Rig(a.follower_port, {"front_camera": cam(a.front_cam), "wrist_camera": cam(a.wrist_cam)},
                  leader_port=a.leader_port, follower_id=a.follower_id, leader_id=a.leader_id)
        esp_factory = (lambda: connect_rigd(a.rigd_url)) if a.rigd_url else (lambda: ESP32(a.esp32_port))
    SESSION = Session(ROOT / "data" / "recordings" / a.session, a.cells, a.per_cell, a.operator)
    BOARD = Board(SESSION.dir)
    settings = SESSION.dir / "settings.json"

    def _save_marker() -> None:
        data = json.loads(settings.read_text()) if settings.exists() else {}
        data["after_shot_s"] = CFG["after_shot_s"]
        if STATE.marker is None:
            data.pop("marker", None)
        else:
            data["marker"] = [round(STATE.marker[0], 4), round(STATE.marker[1], 4)]
        settings.write_text(json.dumps(data))

    globals()["_save_marker"] = _save_marker
    if settings.exists():
        saved = json.loads(settings.read_text())
        CFG["after_shot_s"] = saved.get("after_shot_s", AFTER_SHOT_S)
        if saved.get("marker"):
            STATE.marker = (float(saved["marker"][0]), float(saved["marker"][1]))
    else:
        CFG["after_shot_s"] = AFTER_SHOT_S
    UPLOADER = Uploader(a.dataset_path)
    if a.auto_upload:  # leftovers from a previous run of this session
        for ep in find_episodes(SESSION.dir):
            UPLOADER.enqueue(ep)

    if a.fake:
        worker = threading.Thread(target=control_loop, args=(rig, esp), daemon=True)
    else:
        worker = threading.Thread(target=hardware_thread, args=(rig, esp_factory), daemon=True)
    worker.start()
    httpd = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    url = f"http://127.0.0.1:{a.port}/"
    print(f"session {a.session}: {len(SESSION.items)} episodes planned → {SESSION.dir}\n"
          f"dataset {a.dataset_path}{' (auto-upload)' if a.auto_upload else ''}\nrecording UI: {url}", flush=True)
    if not a.no_browser:
        webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down", flush=True)
    finally:
        STATE.stop.set()
        worker.join(timeout=2)
        httpd.shutdown()
        try:
            rig.disconnect()
        except Exception:
            _release_arm(rig)


if __name__ == "__main__":
    main()
