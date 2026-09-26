# Act-athon notes: π0.7 on the SO-101

Status, plans and open questions for the act-athon (Sep 25–27, 2026, actor labs, Mountain View).

Knowledge base:
- [plan.md](plan.md): **the final plan**: tic-tac-toe blaster (board, cameras, wiring, recording, Claude + π0.7 inference).
- [concepts.md](concepts.md): how π0.7 works (chunks, flow matching, memory), the PI data format (topics, steps, shards, contracts, profiles), Hz math, data quality, DAgger, how the APIs are built.
- [hardware.md](hardware.md): SO-101, how teleop works, cameras, the Dartstorm blaster and its wiring, electronics (MOSFET vs relay).
- [pi-sdk-self-serve-guide.md](pi-sdk-self-serve-guide.md): PI's official guide (source: their Notion page, needs login).

## Status

- **2026-09-25:** `pi-sdk` 0.3.2 installed in `.venv` from PI's private registry. Org API key works (`scripts/smoke_test.py`); the org started empty.
- **There's no hosted base model to call directly.** Inference only goes through a fine-tuned checkpoint's `gateway_url`.
- **2026-09-26 mock pipeline** (`scripts/experiments/mock_pipeline.py`, schema in `autobot/schema.py`, IDs in `runs/mock_state.json`):
  - Contract `so101_blaster_v1` registered: `01a0dcb7-53fc-7c20-aeda-7dfddc56cfb8`. This is the real schema (front + wrist cameras, 5 joints + gripper, blaster trigger, 30 Hz), so real data reuses it.
  - 3 fake 60 s episodes (5,400 steps) uploaded and finalized in ~40 s.
  - Profile `mock-so101-blaster-profile-v1`, filter `mock-so101-smoke-v1`, fine-tune `mock-so101-smoke-v1` (run `28e28048-8cbb-4232-90de-b564ac2bfeea`, `pi07`, 100 steps). `pi07` is enabled for our org.
  - Timeline (PDT, from API timestamps): created 00:57:27 → started 00:58:00 (33 s queue) → ~18 min data prep (`progress=None`) → training 01:16–01:18 → succeeded + checkpoint `available` 01:22:36. **~25 min total for 100 steps.** Final train MAE 13.06 (raw units; one eval point at step 99, no validation split).
  - As of 02:10 no `gateway_url`. The partner portal's Models page ("shared model registry") is empty for org `actathon_Ledger`, so hosting may not be enabled yet; asking PI.
- **2026-09-26 real-data run** (`scripts/experiments/real_sample.py`, IDs in `runs/real_state.json`, log `runs/real_poll.log`): no-trigger contract `so101_v1` (`01a0dcf9-3718-7140-890f-3b91eac9ee18`), 5 public SO-101 episodes (~2,000 steps, task "pick the dice and put it in the white bowl"), `pi07`, 200 steps. Created 02:11:29 → running 02:14:34 → training 02:29–02:32 → succeeded + checkpoint `available` 02:38:44 (**~27 min**). Train MAE 13.2 (raw degrees), no validation. No `gateway_url`.
- **Hosting blocker:** none of the three checkpoints (100-step fake, 20-step fake, 200-step real) has received a `gateway_url`; the first has been `available` since 01:22. Likely hosting isn't enabled for the org. Only PI can fix this.
- Note: `pi-upload-lerobot` maps each LeRobot feature to exactly one topic and can't split a feature, so the real-sample contract `so101_v1` uses single [6] state/action topics. Our own recorder writes MCAP directly, so the blaster contract keeps separate topics.
- Real data must use a different `source_dataset_path` (not `mock/v1`) so the fake episodes are never trained on.

Check hosting yourself (from the repo root):
```bash
.venv/bin/python -c "from autobot import schema as s; _,ft=s.clients(); [print(r.name, r.state, [(c.state, c.gateway_url) for c in ft.list_checkpoints(r.run_id).checkpoints]) for r in ft.list_finetunes().finetunes]"
```
`gateway_url=None` everywhere = nothing hosted. The partner portal (partner.pi-fleet.com) has no fine-tune/checkpoint page; its Models page ("shared model registry") is empty, possibly unrelated.

The test-run code (mock_pipeline.py, mock_short.py, real_sample.py) was removed on 2026-09-26; it's in git history (commit ddaaf44). Its IDs stay in `runs/*_state.json` and its data in `data/`. The real workflow is `scripts/record.py` → `upload.py` → `train.py` (see README).

## Credentials (kept out of this folder)

| What | Where | Used for |
| --- | --- | --- |
| Org API key (`sk_…`) | `~/.config/pi/api_key` (chmod 600) | every API call (`PI_API_KEY`) |
| Service-account JSON | `~/.config/pi/service-account.json` (chmod 600) | `pip install` from PI's registry only |

Both keys were pasted into a Claude chat, so rotate them after the event (Partner Portal: partner.pi-fleet.com).

Reinstall the SDK (plain pip; `uv` can't use the registry keyring by default):
```bash
python3.13 -m venv .venv
.venv/bin/pip install keyring keyrings.google-artifactregistry-auth
GOOGLE_APPLICATION_CREDENTIALS=~/.config/pi/service-account.json .venv/bin/pip install "pi-sdk[video,inference,lerobot]>=0.3.2" \
  --index-url https://us-east5-python.pkg.dev/pi-external-partners/pi-sdk/simple/ --extra-index-url https://pypi.org/simple
```

## Event rubric

| Category | Weight | What it means for us |
| --- | --- | --- |
| Autonomy | 30 | π0.7 must actually drive the behavior, not a scripted routine |
| Hardware hack | 25 | something physical that didn't exist on Friday |
| Reliability | 20 | a modest demo that works beats a grand one that doesn't |
| Ambition | 15 | how hard the task is for this robot |
| Style | 10 | did the room gasp |

Submissions close **Sunday 9:00 am**. Demos are live on the real machine. Fine-tuning compute is provided by PI. On site: SO-101 arms, a Bambu A1 print farm, Pi 5s, ESP32s, servos, electronics kits.

## Current plan: SO-101 + blaster

**Task direction:** "aim at the target and fire". The model sees where the target is and points at it (a learnable imitation task). Optional hybrid: Claude marks spots on a canvas, π0.7 aims and fires at each. See [hardware.md](hardware.md#blaster-x-shot-insanity-motorized-dartstorm) for the blaster mod and the weight concern.

**Schema (`autobot/schema.py`, contract `so101_blaster_v1`, 30 Hz):**
- Observations: `observation/arm/joints/position` [5], `observation/arm/gripper/position` [1], `observation/front_camera/rgb/image`, `observation/wrist_camera/rgb/image`
- Actions: `action/arm/joints/position` [5], `action/arm/gripper/position` [1], `action/blaster/trigger` [1]
- Profile: base = front camera, wrist = wrist camera, state = joints + gripper (6), actions = joints + gripper + trigger (7), `absolute`.

**Pipeline:**
1. Build and calibrate the arms; mount the cameras; mod the blaster (relays + ESP32 + arm/disarm switch); fire button on the leader handle.
2. Record with a custom loop (LeRobot Python API + fire button), since stock `lerobot-record` won't capture the trigger. Pure-arm tasks can use `lerobot-record` + `pi-upload-lerobot`.
3. Pilot: ~30–50 episodes → short fine-tune → check it works end to end.
4. Final: ~100–200 episodes of the one task; then 1–2 DAgger rounds with `continue_from_checkpoint_id`.
5. Run loop: `PolicyClient` / `AsyncRTCPolicyRunner` → servo targets + edge-detected FIRE to the ESP32.

**Episodes:** failures out (`success=False`), recoveries in, every teleop demo gets a full-length `control_source="human"` span.

## Tooling

Built (2026-09-26): `autobot/{schema,episode,hardware}.py`, `scripts/record.py`, `scripts/upload.py`, `scripts/train.py`, `firmware/blaster_esp32/`. Tested end to end with `record.py --fake` → local contract validation → `upload.py --dry-run`; not yet on real hardware.

Also built: `scripts/robot_server.py` (`shoot CELL` / `game`), `autobot/policy.py`, `autobot/control.py`, `tests/test_control.py`. Tested with `--fake` (fake arm + fake policy + random brain) and the safety-rule tests. The Claude brain (`claude-opus-5-5`, strict `shoot` tool, effort `medium`) needs `pip install anthropic` and an Anthropic API key; untested. The real `PiPolicy` path needs a hosted `gateway_url`.

Still to build:
- `scripts/dagger_record.py`: policy drives, a key hands control to the leader arm; logs policy/human `control_source` spans.

## Open questions for PI or the organizers

- Are there USB cameras or 32×32 UVC modules on site? (Not on the hardware list.)
- Does PI resample 30 Hz data to π0.7's native 50 Hz?
- Is `pi07_cosmic` also enabled for us, and what's the difference from `pi07`?
- Is there a limit on simultaneous fine-tunes?
- Could we get a second SO-101 set (for a two-arm task)?

Answered: `pi07` is enabled. A 100-step fine-tune takes ~25 min to a checkpoint (mostly data prep; queue was 33 s). Time to a hosted endpoint is still unknown.

Partner portal: https://partner.pi-fleet.com (org `actathon_Ledger`): Episodes (video viewer, pass/fail), API keys, Models (shared model registry), Admins, Settings.
