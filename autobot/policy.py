"""π0.7 policy wrapper (hosted checkpoint via PI's gateway) and a fake stand-in.

Both return the model's whole plan (50 steps for π0.7) in the training profile's order: joints [k, 5] and
trigger [k, 1]. The caller executes only the first part (control.shoot) but reads the rest to decide firing.
"""

import numpy as np

from autobot import schema as s


def gateway_url_for(finetune_name: str, step: int | None = None) -> str:
    """The hosted URL of one checkpoint of a fine-tune: `step`, or the run's final checkpoint.
    Never silently falls back to an earlier checkpoint that happened to get hosted first."""
    _, ft = s.clients()
    run = next((r for r in ft.list_finetunes(page_size=100).finetunes if r.name == finetune_name), None)
    if run is None:
        raise SystemExit(f"no fine-tune named {finetune_name!r}")
    cps = ft.list_checkpoints(run.run_id).checkpoints
    if not cps:
        raise SystemExit(f"{finetune_name}: no checkpoints yet ({run.state})")
    want = step if step is not None else max(c.step for c in cps)
    cp = next((c for c in cps if c.step == want), None)
    hosted = sorted(c.step for c in cps if c.gateway_url)
    if cp is None:
        raise SystemExit(f"{finetune_name}: no checkpoint at step {want} (steps: {sorted(c.step for c in cps)})")
    if not cp.gateway_url:
        raise SystemExit(f"{finetune_name}: checkpoint {want} isn't hosted yet (hosted: {hosted or 'none'}); "
                         f"wait, or pick one with --checkpoint-step")
    return cp.gateway_url


class PiPolicy:
    HORIZON = 50  # π0.7 plans 50 steps; ask for all of them

    def __init__(self, gateway_url: str, seed: int = 0):
        from pi_sdk.inference import ClientConfig, PolicyClient

        self.client = PolicyClient(ClientConfig(
            gateway_url=gateway_url,
            api_key=s.api_key(),
            action_keys=list(s.MODEL_ACTION_TOPICS),
            execute_chunk_size=self.HORIZON,
        ))
        self.rng = np.random.default_rng(seed)
        self.client.connect()  # first connect can take minutes while the model starts
        names = self.client.server_metadata.camera_names
        # Map our camera names onto the server's (it may use topic names or short names).
        self.cam_map = {cam: next((n for n in names if cam in n), cam) for cam in s.CAMERAS}

    def infer(self, images: dict[str, np.ndarray], joints: np.ndarray, task: str,
              explore: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """explore=False: PI's default request (zero initial noise, deterministic). explore=True: random
        initial noise, i.e. a fresh sample of the model's action distribution for this observation."""
        from pi_sdk.inference import InferenceInput

        obs = InferenceInput(
            images={self.cam_map[cam]: img for cam, img in images.items()},
            states={s.OBS_JOINTS: np.asarray(joints, dtype=np.float32)},
            robot_task_string=task,
            raw_text=task,
        )
        meta = self.client.server_metadata
        if explore and meta.action_horizon and meta.action_dim:
            noise = self.rng.standard_normal((meta.action_horizon, meta.action_dim)).astype(np.float32)
            result = self.client.infer_realtime(obs, initial_noise=noise, prefix_info=self.client._default_prefix_info(),
                                                prefix_actions=None, sample_args=None, action_limit=self.HORIZON)
        else:
            result = self.client.infer(obs)
        acts = result.actions_by_key
        return np.asarray(acts[s.ACT_JOINTS], dtype=np.float32), np.asarray(acts[s.ACT_TRIGGER], dtype=np.float32)

    def close(self) -> None:
        self.client.close()


class FakePolicy:
    """Moves each joint toward a per-cell pose; raises the trigger once close."""

    def __init__(self, chunk: int = 50):
        self.chunk = chunk
        self.calls = 0

    @staticmethod
    def target(cell: str) -> np.ndarray:
        col, row = s.COLS.index(cell[0]), s.ROWS.index(cell[1])
        return np.array([(col - 2) * 12.0, (row - 2) * -10.0, 30.0, -10.0, 0.0], dtype=np.float32)

    def infer(self, images, joints, task, explore=False):
        """Reaches the goal in the first half of the plan, then holds; the trigger is planned only once
        the arm is close (like the recorded takes: settle, then fire)."""
        self.calls += 1
        goal = self.target(task.split()[-1])
        half = self.chunk // 2
        steps = np.concatenate([np.linspace(joints, goal, half + 1)[1:], np.repeat(goal[None], self.chunk - half, 0)])
        trigger = np.zeros((self.chunk, 1), dtype=np.float32)
        if float(np.abs(goal - joints).max()) < 1.0:
            trigger[self.chunk - 15: self.chunk - 5] = 1.0  # pulse late in the plan: only the look-ahead sees it
        return steps.astype(np.float32), trigger

    def close(self) -> None: ...
