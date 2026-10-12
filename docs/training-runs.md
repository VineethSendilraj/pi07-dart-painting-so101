# Training runs (overnight 2026-09-26 → 27)

The lab notebook from the night before the demo: every fine-tune, how we ranked checkpoints, and what we tried for firing. **How to run the demo model: [Run it on the rig](#run-it-on-the-rig).**

**TL;DR (08:15)**
- **Demo model: `full-v2` checkpoint 3799** (pi07, all 116 takes, schedule ending at ~33.5 passes): best validation error of the comparable runs (0.287), prompt probe perfectly ordered, close to the other models. **Close second: `full-v1` checkpoint 2999.** **Cosmic alternative: `logo-cosmic-v1` checkpoint 1999** (π-logo focus: lowest validation error 0.281, ordered probe) or `full-cosmic-v1` checkpoint 2000.
- Every one of the 24 probed checkpoints listens to the prompt (aims ordered by column and row): no "memorized routes".
- **Firing is the risk.** Offline (real rig frames, arm at the model's own aim, 40 random samples each), the pi07 models and plain cosmic never produced a usable trigger pulse; `logo-cosmic-v1` did once (peak 0.92) and `noidle-cosmic-v1` peaked at 0.48 (it also fired once in a closed-loop test). The frames are stale, so this is a hint, not proof. **Morning decision path:**
  1. Dry test `full-v2` @3799 on C2 (master switch OFF). Fires → use it.
  2. Aims but `trigger max` stays ~0 → dry test `logo-cosmic-v1` @1999 (most willing to fire). Fires → use it. **Aims E onto D or misses D5** → dry test `redmark-v1` @4599 (most evenly spaced columns, no D5 offset).
  3. Neither fires → keep the best-aiming one and add `--settle-fire 1.5` (fires on the model's own aim after 1.5 s still; tested).
- Demo: `robot_server.py … --finetune full-v2 --checkpoint-step 3799 draw "the PI logo"` (Claude, both cameras after each shot) or `draw pi --brain preset`.

## Use a model from your own script

Test in this order: **1. `full-v2` @3799 → 2. `full-v1` @2999 → 3. `logo-cosmic-v1` @1999** (if the first two aim but won't fire).

| Order | Run @ checkpoint | gateway_url |
|---|---|---|
| 1 | `full-v2` @3799 (pi07) | `wss://api.pi-fleet.com/v1/models/ckpt-3799-full-v2` |
| 2 | `full-v1` @2999 (pi07) | `wss://api.pi-fleet.com/v1/models/ckpt-2999-full-v1` |
| 3 | `logo-cosmic-v1` @1999 (cosmic) | `wss://api.pi-fleet.com/v1/models/ckpt-1999-logo-cosmic-v1` |

Needs the PI API key in `~/.config/pi/api_key` (or `PI_API_KEY`). Run from the repo root:
```python
from autobot.policy import PiPolicy, gateway_url_for

policy = PiPolicy(gateway_url_for("full-v2", 3799))    # or PiPolicy("wss://api.pi-fleet.com/v1/models/ckpt-3799-full-v2")
# images: {"front_camera": rear RGB uint8 HxWx3, "wrist_camera": gun RGB with the red plus drawn (Rig.observe does this)}
# joints: follower joints [5] in degrees (shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll)
plan_joints, plan_trigger = policy.infer(images, joints, "shoot C3")   # (50, 5) leader targets, (50, 1) trigger
policy.close()
```
The full robot loop (speed limit, fire rule, rigd firing) is `autobot.control.shoot(rig, esp, policy, "C3")`, which is what `robot_server.py` runs. First connect takes ~1 min; then ~0.2 s per request.

## Status

_Last update: 08:49 PDT (final)_

| Run | Base | Data | Steps (passes) | State | Checkpoints |
|---|---|---|---|---|---|
| `full-v1` | pi07 | all 116, as recorded | 3000 (26.5) | **succeeded**, all hosted | 1000 / 2000 / **2999** |
| `full-cosmic-v1` | pi07_cosmic | all 116, as recorded | 3000 (26.5) | **succeeded**, all hosted | 1000 / **2000** / 2999 |
| `redmark-v1` | pi07 | 0015–0115 (red gun mark only), tail kept | 4600 (~47) | **succeeded**, all hosted | 1534 / **3068** / 4599 |
| `redmark-cosmic-v1` | pi07_cosmic | same | 4600 (~47) | **succeeded**, all hosted | **1534** / 3068 / 4599 |
| `trim-*` | | **trimmed copies** (`so101-blaster/v2-trim`) | | **not run**: the trimmed upload needs the recording laptop (below) | |
| `full-long-v1` | pi07 | all 116, as recorded | 5500 (~48.5) | **succeeded**, all hosted | 1834 / **3668** / 5499 |
| `full-long-b512-v1` | pi07 | all 116, batch 512 | 2750 (~48.5) | **succeeded**, all hosted (val 0.334: batch 512 not better) | 917 / 1834 / 2749 |
| `full-v2` | pi07 | all 116, as recorded | 3800 (~33.5, pi07's sweet spot) | **succeeded**, hosted (val **0.287**, best) | 1267 / 2534 / **3799** |
| `full-cosmic-2k-v1` | pi07_cosmic | all 116, as recorded | 2000 (~17.6, cosmic's sweet spot) | **succeeded**, hosted (val 0.294) | 667 / 1334 / **1999** |
| `logo-v2` | pi07 | all 116, **50% of every batch from the 11 π-logo cells** | 3800 | **succeeded**, hosted (val 0.333: focus didn't help pi07) | 1267 / 2534 / 3799 |
| `logo-cosmic-v1` | pi07_cosmic | same focus | 2000 | **succeeded**, hosted (val **0.281**, probe ordered) | 667 / 1334 / **1999** |
| `noidle-cosmic-v1` | pi07_cosmic | all 116, profile `so101-blaster-v2` (**keep_idle_actions=False**: PI skips still stretches, i.e. the 2 s post-shot tail) | 2000 | **succeeded**, hosted (val 0.375: validates on moving frames only; probe ordered; trigger peaks 0.48) | 667 / 1334 / 1999 |
| `noidle-v1` | pi07 | same, keep_idle_actions=False | 3000 | **succeeded**, hosted (val 0.421, moving frames only; probe ordered; no firing gain on pi07) | 1000 / 2000 / 2999 |
| `logo-noidle-cosmic-v1` | pi07_cosmic | π-logo focus 50% + keep_idle_actions=False | 2000 | **succeeded**, hosted (probe ordered, 1.3° from consensus; no firing gain) | 667 / 1334 / 1999 |

PI allows **2 fine-tunes at a time** per org (tried a 3rd: `resource_exhausted: organization has reached its outstanding finetune limit`). Each run is ~1 h + ~20 min hosting. Cancelled: `step1-v1`, `step1-cosmic-v1` (started before the last uploads).

**Needs the recording laptop (one command):** the tail fix has to be new, trimmed episodes (see Data cleanups 1):
```bash
git pull && .venv/bin/python scripts/make_trimmed.py data/recordings/final-vector --upload
```
It writes `data/recordings/final-vector-trim/` (116 trimmed takes + `gun_marks.json`) and uploads them to `so101-blaster/v2-trim`. The printed summary gives the exact `--crosshair` value (put it in `calibration/gun_mark.json`) and which takes aren't red. Then, ≥5 min after the upload: `.venv/bin/python scripts/train.py finetune trim-cosmic-v1 --base pi07_cosmic --dataset-path so101-blaster/v2-trim --steps 2000 --checkpoint-every 667` (~40 min + ~20 min hosting).

## Data cleanups

1. **Post-shot tail: trimmed copies (the only way that works on PI).** Every take kept recording 2.0 s after the fire-button release: 10 frames with trigger=1, then ~50 frames of the arm sitting still on target with trigger=0. Dry takes look the same before and after the shot, so on target only about a third of training frames say "fire soon", and hosted inference is deterministic. The model could aim and then never fire.
   - Tried first: marking the tail `control_source=none` on PI and training with `dagger_data_filter`. **PI's span filter only keeps episodes whose human span covers the whole take**; probes showed that trimming even 0.1 s off the end, or trimming the start, selects zero steps (`dataset filters select no steps after span filtering`). All 116 episodes were **reverted** to their original annotations (verified 116/116).
   - Fix: `scripts/make_trimmed.py` decodes each take, keeps it up to 5 frames after the trigger pulse, re-encodes, validates and uploads as `trim-<episode>` to `so101-blaster/v2-trim`. Times, frames, joints and labels are unchanged. Tested on synthetic takes (120 → 75 steps, trigger block intact, contract-valid). It also measures the gun mark per take from the pixels (red / cyan / none + position).
2. **Crosshair consistency (variant run, not the default).** 102 of 116 takes have the red plus burned into the gun image at ≈(0.46, 0.76). 0000–0013 have none, 0006/0007 have a centred plus from uncommitted code, and 0014 was recorded seconds before the colour changed cyan → red. `redmark-v1` / `redmark-cosmic-v1` drop 0000–0014 so every training gun frame matches what inference draws (cost: one take on each of the 14 cells of the first recording cycle). **Result: not better** (validation error no lower, aims drift 2.4–3.9° from the other models): more data beats a consistent mark.
3. **Not changed:** thin cells (row 1, A3, E3: 2 takes each; A5/E5: 0). The drawings avoid them, so no upweighting. The π-logo cells are already 64% of training frames (≈1.35× an even split).

## Training changes

- Follow-up runs favour **pi07** (7/7 successful runs in this org; `pi07_cosmic` is undocumented publicly); cosmic twins run where a comparison is useful.
- Step counts are in passes over the data: PI caps runs at **50 passes** (Chandra, Discord) and trims longer requests. openpi's own recipe for a same-size dataset (123 episodes, 12 min) is 17–35 passes; ours are 17.6–49.5.
- **Sweet spots** (validation error): pi07 ~30–35 passes (`full-v2`, schedule ending at 33.5, is its best); cosmic ~18 passes. Beyond that both overfit.
- **Batch 512** (same passes, half the updates): worse (`full-long-b512-v1`).
- **π-logo focus** (50% of every batch from the 11 logo cells, which are already 64% of the data): lowest validation error on cosmic (`logo-cosmic-v1`, 0.281, and the most willing to fire), no gain on pi07 (`logo-v2`, 0.333).
- **keep_idle_actions=False** (PI skips still stretches, which may drop the post-shot tail): no firing gain on pi07 (`noidle-v1`); a small hint on cosmic (`noidle-cosmic-v1`, trigger peaks 0.48).

## Inference changes (no retraining)

See `autobot/control.py`, `autobot/policy.py`, `scripts/robot_server.py`:
- **Fire rule looks at the whole 50-step plan**, not only the 25 executed steps: fire when the arm is settled, the rest of the plan stays on this aim, and the plan has ≥3 consecutive trigger steps > 0.5.
- **Randomness only when stuck**: requests use PI's default zero noise (deterministic); once the arm has sat still on the model's own aim for ~0.3 s without a shot, every other request uses random noise (trigger only; the aim is held), so re-asking gives a new sample instead of the same "don't fire".
- **`--settle-fire SECONDS`** (off by default): fire once the model has held the arm on its own aim that long without firing. Aim = model, trigger timing = rule.
- **Per-request log**: max trigger over the plan and where it is, settled or not, noise mode.
- **Red gun mark on by default** at `calibration/gun_mark.json` (the recorded spot); `--no-crosshair` to turn off.
- **Checkpoint pick is explicit**: `--finetune NAME` uses the run's **final** checkpoint (or `--checkpoint-step 2000`), never "whichever got hosted first".
- **Drawing mode**: no return to rest between shots (the recorded takes started from the previous aim).
- **Hardened by an overnight code review (11 confirmed findings fixed):** Claude history is append-only (editing earlier turns would have made the 3rd Opus call fail); random-noise re-asks only when the last plain plan ends where the arm already is (no early shot during slow adjacent-cell moves; regression test); inference / camera / servo hiccups retry instead of ending the shot; a crashed shot counts as not fired and the picture continues; a Claude outage stops the picture; FIRED from rigd is never lost; the 30 Hz loop drops lost time instead of catching up; `--confirm` ignores presses made while Claude was thinking; `train.py watch --until-hosted` waits for the final checkpoint.
- `--human-only` does nothing useful on this data (PI keeps only full-window human spans); the tail fix is the trimmed dataset.

## Model ranking (before robot tests)

Test in this order; the robot decides (hits per cell), not train MAE.

1. **`full-v2` @3799** (pi07): best validation error (0.287), perfectly ordered probe, 0.93° from consensus.
2. **`full-v1` @2999** (pi07): 0.298, perfectly ordered, closest to consensus (0.81°). If full-v2 disappoints on the robot.
3. **`logo-cosmic-v1` @1999** (cosmic, π-logo focus): lowest validation error (0.281), ordered probe, the only model that produced a usable trigger pulse offline. First choice if the pi07 models won't fire, and a good π-drawing candidate.
4. `full-cosmic-v1` @2000 / `full-cosmic-2k-v1` @1999 (plain cosmic, 0.294, wider aims than pi07): if both pi07 models aim short of the outer columns/rows.
5. Backups: `full-long-v1` @3668, `noidle-cosmic-v1` @1999, `logo-v2` @3799, `noidle-v1` @2999, red-mark runs, and `logo-noidle-cosmic-v1` @1999 (π focus + no idle: ordered, no firing gain).

**PI's train / validation error** (PI holds out part of the data by itself; lower is better, robot hits still decide). Evaluated at PI's own steps, which don't always match the checkpoints:

| Run | step | train MAE | validation MAE |
|---|---|---|---|
| full-v1 | 2000 / 2999 | 0.425 / 0.254 | 0.476 / **0.298** (still improving) |
| full-cosmic-v1 | 2000 / 2999 | 0.237 / 0.166 | **0.294** / 0.307 (cosmic peaks ~18 passes) |
| redmark-v1 | 2000 / 4000 / 4599 | 0.446 / 0.188 / 0.171 | 0.497 / **0.325** / 0.343 (pi07 best ~40 passes) |
| redmark-cosmic-v1 | 2000 / 4000 / 4599 | 0.236 / 0.142 / 0.133 | **0.294** / 0.336 / 0.340 (cosmic best ~20 passes) |
| full-long-b512-v1 | 2000 / 2749 | 0.416 / 0.259 | 0.511 / 0.334 (batch 512: fewer updates, not better) |
| full-long-v1 | 2000 / 4000 / 5499 | 0.415 / 0.195 / 0.154 | 0.493 / **0.303** / 0.324 (pi07 peaks ~30-35 passes, then overfits) |
| **full-v2** | 2000 / 3799 | 0.412 / 0.198 | 0.479 / **0.287** (best pi07: schedule ends at ~33.5 passes) |
| full-cosmic-2k-v1 | 1999 | 0.226 | **0.294** (same as full-cosmic-v1 @2000) |
| logo-cosmic-v1 | 1999 | 0.213 | **0.281** (lowest; its validation set likely leans to the logo cells) |
| logo-v2 | 2000 / 3799 | 0.431 / 0.187 | 0.423 / 0.333 (π focus didn't help pi07) |
| logo-noidle-cosmic-v1 | 1999 | 0.238 | 0.444 (moving frames only; not comparable) |

Red-mark runs validate on a different subset, so compare trends, not absolute values, with the full runs.

**Prompt probe** (`scripts/probe_cells.py RUN --checkpoint-step N`, no robot): the same real rig frames and joint state with every `shoot <cell>` prompt; where does each plan end? A model that listens to the prompt spreads pan across columns A→E and wrist pitch across rows 1→5 (rank order 1.0 = perfectly ordered). Frames are from the recording night, so this checks cell discrimination, not absolute aim.

| Checkpoint | pan spread A→E | pan order | pitch spread 1→5 | pitch order | notes |
|---|---|---|---|---|---|
| full-v1 @1000 | 6.9° | 0.7 | 21.4° | 1.0 | columns not yet separated |
| full-v1 @2000 | 14.9° | 1.0 | 23.6° | 1.0 | |
| **full-v1 @2999** | **20.9°** | **1.0** | **24.5°** | **1.0** | pi07 still sharpening with training |
| full-cosmic-v1 @1000 | 21.0° | 0.9 | 21.2° | 1.0 | cosmic learns fast |
| **full-cosmic-v1 @2000** | **21.4°** | **1.0** | **25.9°** | **1.0** | cosmic's best (also its best validation MAE) |
| full-cosmic-v1 @2999 | 18.3° | 0.9 | 22.3° | 1.0 | slight overfit |
| redmark-v1 @1534 | 17.8° | 1.0 | 22.6° | 1.0 | |
| redmark-v1 @3068 | 23.7° | 1.0 | 17.4° | 1.0 | |
| redmark-v1 @4599 | 23.7° | 1.0 | 16.5° | 1.0 | |
| redmark-cosmic-v1 @1534 | 29.7° | 1.0 | 27.8° | 1.0 | widest spreads |
| redmark-cosmic-v1 @3068 | 24.1° | 0.9 | 21.7° | 1.0 | |
| redmark-cosmic-v1 @4599 | 22.1° | 0.9 | 18.7° | 1.0 | |
| full-long-b512-v1 @2749 | 18.5° | 0.9 | 23.2° | 1.0 | batch 512: not better |
| full-long-v1 @1834 | 14.6° | 1.0 | 23.1° | 1.0 | |
| **full-long-v1 @3668** | **22.8°** | **1.0** | **23.7°** | **1.0** | pi07 at ~32 passes (its validation sweet spot) |
| full-long-v1 @5499 | 22.4° | 1.0 | 23.6° | 1.0 | |
| full-v2 @2534 | 18.4° | 1.0 | 23.4° | 1.0 | |
| **full-v2 @3799** | **22.2°** | **1.0** | **23.8°** | **1.0** | best validation error of all runs |
| full-cosmic-2k-v1 @1999 | 23.4° | 1.0 | 28.4° | 1.0 | cosmic spreads wider than pi07 |
| logo-cosmic-v1 @1999 | 24.4° | 1.0 | 27.4° | 1.0 | trigger max 0.036 (others ≤0.03) |
| logo-v2 @3799 | 23.5° | 1.0 | 25.9° | 1.0 | 1.31° from consensus |
| noidle-cosmic-v1 @1999 | 20.1° | 1.0 | 25.2° | 1.0 | fired once in a closed-loop test |
| noidle-v1 @2999 | 24.1° | 1.0 | 21.2° | 1.0 | |
| logo-noidle-cosmic-v1 @1999 | 22.2° | 1.0 | 26.2° | 1.0 | 1.32° from consensus |

**Agreement with the other models** (`scripts/probe_cells.py --rank` prints the up-to-date table): mean |pan, pitch − the median of all probed checkpoints| over the 23 cells; adjacent cells are ~5° apart, so lower = safer. At 08:40 (23 checkpoints): full-v1 @2999 **0.89°**, full-v2 @3799 **0.96°**, full-v2 @2534 1.04°, full-long-b512-v1 @2749 1.06° (pan order 0.9), full-long-v1 @3668 1.30°, noidle-v1 @2999 1.34°, logo-v2 @3799 1.35°, full-cosmic-v1 @2000 1.58°, cosmic variants 1.6–2.9°, redmark-* 2.4–4.4°. The pool leans on pi07 runs (cosmic aims ~15% wider), so this favours pi07 somewhat. **Data finding: dropping the 15 inconsistent-mark takes didn't help (more data beats a consistent mark).**

Every checkpoint listens to the prompt (no "memorized routes"), but the per-cell map shows limits the averages hide. For `full-v2` @3799: **columns D and E barely differ in pan** (~−3° vs −2 to −4°; E1 even plans slightly left of D1), and **D5 is an outlier** (+7.3° pan, near column C). Rows separate well (≈ +4, −1, −4, −11, −19°). `full-v1` @1000 barely separates columns B–E at all. So on the rig, dry-test **D2, E2 and D5** first; if E lands on D, try a cosmic model (wider aims). All models hold lift ≈ −110°, elbow ≈ 97°, roll ≈ −93° for every cell: that's where the arm sat while recording, and robot_server warns if a start pose is >8° off.

**Column / row spacing** (evenly spaced board → pan gaps between neighbouring columns and pitch gaps between rows should be roughly equal):

| Family | Column pan gaps A-B / B-C / C-D / D-E | Rows | D5 |
|---|---|---|---|
| pi07, all data (`full-v2`, `full-v1`, `full-long`) | even until **D ≈ E** (gap 1–2°), same at 48 passes | well spaced | off by 10–12° |
| cosmic (all cosmic runs) | **C ≈ D** instead (gap −1 to 2.7°), D/E fine; gets worse with training | well spaced | off by 5–8° |
| **`redmark-v1` @4599** (pi07, red-mark takes) | **most even: 6.4 / 7.1 / 6.5 / 3.6°** | squeezed (pitch spread 17° vs ~24°) | fine (0.3°) |

No model is clean here, and more epochs don't fix it (the most-trained runs look the same or worse). The probe uses one stale gun frame and one start pose, so part of this may be the test; the rig decides. **If `full-v2` puts E on D (or misses D5) on the rig, dry-test `redmark-v1` @4599.** More cosmic epochs are not recommended: its validation error and C/D confusion both get worse past ~18 passes.

**Fire propensity** (arm at each model's own aim for C3, B2, D4, A2; 10 random-noise requests each; "usable" = 3+ consecutive trigger steps > 0.5): full-v2 @3799 0/40 (peak 0.01), full-cosmic-v1 @2000 0/40 (0.02), noidle-cosmic-v1 @1999 0/40 (0.48), **logo-cosmic-v1 @1999 1/40 (0.92)**, noidle-v1 @2999 0/40 (0.02), logo-noidle-cosmic-v1 @1999 0/40 (0.01): dropping still stretches didn't make the models more willing to fire. Real frames but from earlier in the night and static, so only relative.

Inference check per hosted checkpoint (`scripts/infer_test.py RUN --checkpoint-step N`): ckpt-1000-full-cosmic-v1 connected in 53 s, 50-step plans, ~200 ms/request after the first, zero noise = identical plans, random noise varies them (~7°).

## Run it on the rig

**Before plugging in**
```bash
cd ~/dev/Actathon && git pull
.venv/bin/pip install anthropic            # Claude drawing loop (installed on Vineeth's laptop)
export ANTHROPIC_API_KEY=...               # for --brain claude; the preset needs nothing
export ANTHROPIC_WORKSPACE_ID=wrkspc_...    # only if the key isn't scoped to a workspace (400: 'not scoped to a workspace')
```
Only the follower arm, the ESP32 and the two cameras are needed (no leader).

**1. rigd** (Terminal tab 1, leave running): `cd ~/dev/act-athon && .venv/bin/rigd --port /dev/cu.usbserial-0001`. robot_server puts it in test mode (the button = DONE, never fires; the model fires through `/fire`, the same 2-pulse burst as a button shot).

**2. Cameras** (Terminal.app): `.venv/bin/python scripts/camera_check.py`. Pass the rear camera as `--front-cam` and the gun camera as `--wrist-cam`. A swap fails silently.

**3. Gun mark**: `calibration/gun_mark.json` holds ≈(0.46, 0.76). If the recording laptop ran `make_trimmed.py`, put its `red_mark_median` there (or the `marker` from `data/recordings/final-vector/settings.json`). robot_server draws it by default.

**4. Pick and warm a checkpoint** (first connect takes ~1 min). All of these run without the rig:
```bash
.venv/bin/python scripts/train.py status                                   # every run, checkpoint and gateway_url
.venv/bin/python scripts/infer_test.py full-v2 --checkpoint-step 3799      # connects, shapes, latency, noise check
.venv/bin/python scripts/probe_cells.py full-v2 --checkpoint-step 3799     # aims differ per cell? (see the probe table)
.venv/bin/python scripts/robot_server.py --fake-rig --finetune full-v2 --checkpoint-step 3799 shoot C3   # full loop, fake arm
```

**5. Start pose**: aim the blaster at the board by hand (like the start of a recorded take, e.g. pointing at C3), with lift, elbow and roll where they sat while recording. Never start from a limp or folded pose: the arm returns there between game turns (drawing doesn't return between shots).

**6. Dry test (master switch OFF)**: one cell, then prompt sensitivity (same start, two cells; the arm must go to different places), then **D2, E2, D5** (the offline probe shows the model compresses columns D/E and plans D5 oddly):
```bash
R="--finetune full-v2 --checkpoint-step 3799 --follower-port /dev/tty.usbmodem5B7A0000841 --rigd-url http://localhost:8080 --front-cam 0 --wrist-cam 1"
.venv/bin/python scripts/robot_server.py $R shoot C2
.venv/bin/python scripts/robot_server.py $R shoot B2 && .venv/bin/python scripts/robot_server.py $R shoot D4
```
Each request prints `call N: trigger max … at step …/50`. A shot prints `fire at step … (look-ahead | trigger run | settle fallback)`. **If the arm settles on the cell but `trigger max` stays near 0 and it times out, rerun with `--settle-fire 1.5`** (add it to `$R`).

**7. Score checkpoints**: 2 dry shots on 4-6 logo cells (e.g. A2 C2 E2 B4 D4) per checkpoint, in the ranking order. Judge by the gun camera's red plus vs the target cell. Then 1 live shot each for the top two.

**8. The demo (live)**:
```bash
.venv/bin/python scripts/robot_server.py $R draw "the PI logo"                 # Claude: both cameras after every shot
.venv/bin/python scripts/robot_server.py $R draw pi --brain preset            # no Claude: A2..E2, B3..B5, D3..D5
.venv/bin/python scripts/robot_server.py $R draw "a smiley" --confirm         # wait for DONE/SPACE before each shot
.venv/bin/python scripts/robot_server.py $R draw "the PI logo" --dry --max-shots 4   # master OFF rehearsal: Claude won't look for darts
```
Frames sent to Claude and a log land in `data/draw/<time>/`. Claude is told: never A5/E5, avoid row 1/A3/E3, keep pictures simple (5-12 darts), π order = top bar A2→E2, left leg B3→B5, right leg D3→D5, check each dart in the rear photo, re-shoot a miss at most once. If Claude can't be reached, the picture stops; `--brain preset` is a separate run.

**Troubleshooting**
| Symptom | Fix |
|---|---|
| Aims, never fires ("timeout: model never fired") | Read the `call N: trigger max` lines. Max stays ~0: add `--settle-fire 1.5` (fires on the model's own aim after 1.5 s still). Max 0.3-0.5: lower `FIRE_THRESHOLD` in `autobot/control.py` to 0.3. Settled never true: joint jitter > 0.5° → raise `SETTLED_DEG` to 0.8 |
| `checkpoint N isn't hosted yet` | wait (15-25 min after the run succeeds) or `--checkpoint-step` an earlier hosted one; ask Chandra if >30 min |
| Goes to the wrong cell | prompt-sensitivity test; start from a board-aimed pose; try the other base / checkpoint |
| First request slow | normal (~1 min to connect); `infer_test.py` first to warm it |
| Claude errors | check `ANTHROPIC_API_KEY`; use `--brain preset` |
