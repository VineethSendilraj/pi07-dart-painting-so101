# Concepts: how π0.7 and the PI pipeline work

Background explanations gathered during setup. Status and plans live in [notes.md](notes.md); hardware in [hardware.md](hardware.md); PI's own guide in [pi-sdk-self-serve-guide.md](pi-sdk-self-serve-guide.md).

## The model

- **What it is:** a vision-language-action (VLA) model. A vision-language backbone reads the camera images, the robot's current state and a text prompt; a smaller "action expert" turns that into motor commands.
- **π0:** PaliGemma (~3B) + 300M action expert, flow matching, 10 denoising steps. **π0.5:** same idea, trained on more varied data; can predict a text subtask before acting. **π0.7** (arXiv 2604.15483): ~5B total (4B Gemma3 backbone incl. 400M vision encoder + 860M flow-matching action expert), 50-action chunks, 5 denoising steps, built for 50 Hz robots and up to ~240 ms of latency ("real-time chunking"), short video + state history, prompts can carry subtask, metadata (speed, quality, mistake flag), control mode, and visual subgoals. Weights aren't public.
- **Inputs, every call:** camera images (base + wrist), current joint state, overall task (`robot_task_string`), current subtask (`raw_text`).

## Output: the action chunk

- The model outputs a **chunk**: ~50 moments × up to 32 numbers.
  - **Rows = moments** (time steps into the future). **Columns = actuators** (one set-point per joint, gripper, trigger…).
  - Each row is one snapshot of every actuator's target, not "one motor per step".
- **32 columns** is a fixed maximum width so one model can drive many robots. Smaller robots pad the rest with zeros. Column meaning isn't global; the training profile defines it for our robot.
- **Units are whatever you recorded** (servo target angle, velocity, gripper %, trigger 0/1). The model only sees normalized numbers (~−1…1); PI converts back using statistics from your data.
- **The model never drives a motor directly.** It outputs set-points; each actuator's own controller (the servo's internal PID, a motor driver, an ESP32) does the low-level work.
- **Absolute vs delta:** "be at 60°" vs "move +2° from where you are". We use `absolute`.

## Flow matching (how the chunk is generated)

- Training: blend a real action chunk with random noise at a random mix level `t`; the model learns to predict the direction from noise to the real chunk.
- Inference: start from pure noise the size of the whole chunk and take a few refinement steps ("drafts"): 10 for π0/π0.5, 5 for π0.7, configurable with `sample_args={"num_denoising_steps": N}`.
- Every draft refines **the whole 50 × 32 plan at once**. There's no explicit goal or path planner; the trajectory is what the model learned from demonstrations.
- Why not plain regression: when demos disagree (sometimes left around an obstacle, sometimes right), regression averages them into a straight line through the obstacle. Flow matching commits to one real option. It's a cousin of diffusion with straighter paths, so it needs far fewer steps.
- The 10 (or 5) "steps" are drafts of one plan, **not time**. The 50 "steps" in a chunk are time.

## Action chunking and the control loop

- Predict a sequence, play the first N moments (`execute_chunk_size`), then call again with fresh images. The rest of the chunk is thrown away ("receding horizon", like GPS rerouting).
- Benefits: smooth motion, commits to one strategy for the whole chunk, fewer compounding errors, hides model latency. Idea from ACT/ALOHA (Zhao et al., 2023).
- Trade-off: the robot is "blind" during the rows it plays without re-checking, so play only part of the chunk (~8–25 of 50).
- `AsyncRTCPolicyRunner` computes the next chunk while the current one plays.
- A long task is just many back-to-back chunks (a 10-minute task ≈ 600+ one-second plans). Nothing "runs out".

## Memory and "done"

- π0/π0.5: no memory beyond the current frame; the same prompt is sent every call and the scene itself shows progress. Weak spot: states the camera can't distinguish.
- π0.7: short video + state history, and accepts subtask instructions from outside.
- **The model never says it's finished.** Our code decides: timeout, sensor check, a vision-language model judging a camera frame ("is the cup lifted? yes/no"), or an orchestrator that steps through subtasks:
  ```
  for step in ["move to the bag", "grasp the handle", "lift the bag"]:
      while not vlm_says_done(frame(), step) and not timed_out():
          run one π0.7 chunk with raw_text = step
  ```

## Why there's no generic endpoint

- Text models share one input/output format, so one endpoint serves everyone. Robot models output joint commands for a specific body; π0.7 can't know what our 6 columns mean until it has trained on our recordings.
- Zero-shot only works on robots in the training mix (e.g. Trossen ran π0 on ALOHA unmodified). New robots need fine-tuning: PI cites 1–20 hours of data for π0; Google cites 50–100 demos for Gemini Robotics On-Device.
- So PI's API is **hosted fine-tuning + hosted inference**: contract → episodes → profile → fine-tune → checkpoint `gateway_url` → `PolicyClient`.

## The PI data format

Think of one episode as a spreadsheet: **topics are columns, steps are rows**. (MCAP, the open file format underneath, calls topics "channels".)

- **Topic:** one named stream, e.g. `observation/arm/joints/position`. Each message in it has a fixed shape, e.g. 5 float32 numbers. Servos are **positions inside an array**, not separate topics.
- **Role:** what the column is for. `observation` = what the robot sensed (model input), `action` = what it was told to do (model output), `system` = bookkeeping (only `system/step`).
- **Wire schema:** the message format. `pi.data.v1.NdArray` (numbers), `pi.data.v1.VideoFrame` (H.264 camera frames), `pi.data.v1.SnapshotRef` (the step header). Encoding is always `protobuf`.
- **`system/step`:** the row header. Topics are stored as separate streams with slightly different timestamps (cameras and servos never capture at the same instant). Each step message is a list of pointers: "step #412 = joints msg #412 + front frame #412 + wrist frame #412 + action msg #412". It points at the data; the data doesn't point at it.
- **`required_in_snapshots=("system/step",)`** on a topic means every step header must include a pointer to that topic, so every row is complete. That's why a trigger records 0 on every step rather than only when firing.
- **`expected_frequency_hz`:** a validation rule, written once in the contract, never in the data. Set 30 (±10% via `frequency_tolerance=0.1`) on `system/step` only; 0 ("don't check") elsewhere, since every topic is tied to the steps anyway. PI measures the real rate from step timestamps at upload and rejects stalled recordings.
- **Shape/dtype:** 1-D float32 arrays for state/actions (`[5]`, not `[5,3]`).
- **`TopicContract`** is the SDK object for one column's rule. `array_topic()` / `camera_topic()` in the guide are just local helper functions that return one.
- **Contract** = the list of column rules. Permanent (topics can't change; make `_v2`). No stated limit on topic count, so you can log extra sensors.
- **Training profile** = which contract topics feed the model, in what order: 1–16 cameras (base vs wrist), ≤32 state numbers total, ≤32 action numbers total. The action topics are concatenated in order into the model's output columns and split back by name at inference (`result.actions_by_key`). Several profiles can share one contract. State and actions don't need to match in `absolute` mode (e.g. a trigger action with no sensor, or a sensor that's input-only), but always include readings of the joints you control.

## Episodes, shards and upload

- **Episode** = one full recording of doing the task once (e.g. 20 s = 600 steps at 30 Hz) plus labels (task, success, spans). Not a single row.
- **Shard** = one MCAP file. An episode is split into a few: `light.mcap` (step headers + all numeric topics, tens of KB) and one file per camera (MBs). Split for fast, cheap retries. Any file split is valid.
- **Upload flow:** write shards locally (`v1.Writer`) → register the episode (`create_episode`) → upload each shard (`upload_shard`: get a URL, HTTP PUT, confirm) → finalize (`finalize_episode_upload`). Finalize validates the whole episode against the contract; only then is it trainable. Every call is safe to retry.
- **Video encoder lag:** the H.264 encoder returns frames a few images late, each tagged with its original `sequence_id`/timestamp, so step pointers stay correct. Call `finalize()` after the last image or the last frames are lost and the episode is rejected.
- **Action = the command sent that step** (the leader's position during teleop, the 0/1 sent to the ESP32), not where the follower ended up. Observation = the follower's measured position, slightly behind.

## Hz, steps, episodes

- Hz = steps per second. 30 Hz → one step every ~33 ms.
- Steps per episode = seconds × Hz. 20 s × 30 = 600. 50 episodes ≈ 30k steps. Guide floor for a test fine-tune: "a few thousand steps".
- Count episodes, not steps: neighboring steps are nearly identical; variety comes from more episodes.
- A 50-step chunk at 30 Hz ≈ 1.67 s; `execute_chunk_size=10` at 30 Hz ≈ 0.33 s.

## Training data quality

- Imitation learning copies everything, mistakes included. Keep pure failures out (`success=False` + `EpisodeFilter(success=True)`).
- Recoveries (slip → fix → finish) are valuable; mark `success=True`.
- Consistency beats volume: same camera placement, similar start pose, varied object positions.

## DAgger

- A loop after the first model works: the model drives, a human takes over just before a failure, only the correction is trained on. Each round continues training (`continue_from_checkpoint_id`) on original demos **plus** all corrections, never corrections alone.
- `control_source` spans: `"human"`, `"policy"`, `"none"`. `dagger_data_filter=True` keeps only human steps and truncates chunks at the end of each human span.
- Give every normal teleop demo a full-length `control_source="human"` span, or the DAgger filter drops them.
- When to take over (cup example): reach heading off to the side or wobbling; approach height will knock it over; fingers closing on air or only on the rim; cup tilting on lift; same small motion repeating for 2–3 s. Take over before it's unrecoverable; fix only the immediate problem; don't correct motion that's just ugly.
- Detection is human judgment with a foot pedal; automatic failure detection is out of scope.

## How the APIs are built

- `api.pi-fleet.com/v1/data` (`EpisodeService`) and `/v1/finetuning` (`FinetuningService`) are **ConnectRPC** services (protobuf over HTTP). The SDK's `Client` classes wrap vendored stubs.
- Shard uploads: `InitiateShardUpload` returns an `upload_url`, the file goes there by HTTP PUT, then `CompleteShardUpload`.
- Inference: a websocket to the checkpoint's `gateway_url`, numpy arrays packed with msgpack (same design as openpi's policy server), `Api-Key` auth.
- Python SDK only; no Node client. The protos ship in the package if another language were ever needed.
- **Install auth ≠ API auth:** the service-account JSON is used only by the `keyrings.google-artifactregistry-auth` plugin, which mints a short-lived Google token that pip sends as basic auth (`oauth2accesstoken:<token>`). The `sk_…` key goes in an `Authorization` header for every API call.
