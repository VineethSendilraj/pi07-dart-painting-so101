# Running inference on a hosted π0.7 checkpoint

PI hosts every fine-tune's checkpoints behind a websocket endpoint (`gateway_url`). Connecting needs the PI org **API key** (`sk_…`, in `~/.config/pi/api_key` or `PI_API_KEY`); the URL alone won't connect.

## Endpoints

| Model | Endpoint |
| --- | --- |
| Demo model: `full-v2` @3799 (all 116 takes) | `wss://api.pi-fleet.com/v1/models/ckpt-3799-full-v2` |
| Runner-up: `full-v1` @2999 | `wss://api.pi-fleet.com/v1/models/ckpt-2999-full-v1` |

Format: state `observation/arm/joints/position` [5] → actions `action/arm/joints/position` [50, 5] + `action/blaster/trigger` [50, 1]; cameras `observation/front_camera/rgb/image`, `observation/wrist_camera/rgb/image`; task `"shoot <cell>"`. Every run and checkpoint: `.venv/bin/python scripts/train.py status`. Ranking: [training-runs.md](training-runs.md).

## Run it from this repo (no robot needed)

```bash
# connection, server format, output shapes, latency
.venv/bin/python scripts/infer_test.py full-v2 --checkpoint-step 3799
```

## Your own code

```python
import numpy as np
from autobot.policy import PiPolicy, gateway_url_for

policy = PiPolicy(gateway_url_for("full-v2", 3799))
joints, trigger = policy.infer(
    images={"front_camera": front_rgb, "wrist_camera": wrist_rgb},  # uint8 H×W×3 RGB, any size
    joints=follower_deg,                                            # float32 [5], degrees
    task="shoot C3",
)
# joints (50, 5): next 50 steps at 30 Hz; trigger (50, 1): fire when it crosses the threshold
```

## What to expect

- **First connection:** 10–60 s while the model wakes up; keep the laptop awake (`caffeinate -i …` on macOS), since sleep breaks the connection.
- **Each call:** ~0.5 s (first ~1 s). One chunk is 50 steps = 1.67 s of motion at 30 Hz; play part of it, then call again with fresh images. To avoid pausing between calls, play 15–25 rows per call or use `pi_sdk.inference.AsyncRTCPolicyRunner`.
- **Images** are resized/padded to 448×448 on the server; send any size.
- **Units:** joints in degrees (LeRobot `use_degrees=True`), same as training.
- On the real arm for **blaster** checkpoints, use `scripts/robot_server.py` (speed limit ≤3°/step + fire rule). Don't send raw model output straight to the servos.
