"""π0.7 policy wrapper (hosted checkpoint via PI's gateway) and a fake stand-in.

Both return action chunks in the training profile's order: joints [k, 5] and trigger [k, 1].
"""

import numpy as np

from autobot import schema as s


def gateway_url_for(finetune_name: str) -> str:
    """The hosted checkpoint URL of a fine-tune (latest checkpoint that has one)."""
    _, ft = s.clients()
    run = next((r for r in ft.list_finetunes(page_size=100).finetunes if r.name == finetune_name), None)
    if run is None:
        raise SystemExit(f"no fine-tune named {finetune_name!r}")
    hosted = [c for c in ft.list_checkpoints(run.run_id).checkpoints if c.gateway_url]
    if not hosted:
        raise SystemExit(f"{finetune_name}: no checkpoint has a gateway_url yet (hosting not enabled/finished)")
    return max(hosted, key=lambda c: c.step).gateway_url


class PiPolicy:
    def __init__(self, gateway_url: str, chunk: int = 10):
        from pi_sdk.inference import ClientConfig, PolicyClient

        self.client = PolicyClient(ClientConfig(
            gateway_url=gateway_url,
            api_key=s.api_key(),
            action_keys=list(s.MODEL_ACTION_TOPICS),
            execute_chunk_size=chunk,
        ))
        self.client.connect()  # first connect can take minutes while the model starts
        names = self.client.server_metadata.camera_names
        # Map our camera names onto the server's (it may use topic names or short names).
        self.cam_map = {cam: next((n for n in names if cam in n), cam) for cam in s.CAMERAS}

    def infer(self, images: dict[str, np.ndarray], joints: np.ndarray, task: str) -> tuple[np.ndarray, np.ndarray]:
        from pi_sdk.inference import InferenceInput

        result = self.client.infer(InferenceInput(
            images={self.cam_map[cam]: img for cam, img in images.items()},
            states={s.OBS_JOINTS: np.asarray(joints, dtype=np.float32)},
            robot_task_string=task,
            raw_text=task,
        ))
        acts = result.actions_by_key
        return np.asarray(acts[s.ACT_JOINTS], dtype=np.float32), np.asarray(acts[s.ACT_TRIGGER], dtype=np.float32)

    def close(self) -> None:
        self.client.close()


class FakePolicy:
    """Moves each joint toward a per-cell pose; raises the trigger once close."""

    def __init__(self, chunk: int = 10):
        self.chunk = chunk

    @staticmethod
    def target(cell: str) -> np.ndarray:
        col, row = "ABC".index(cell[0]), "123".index(cell[1])
        return np.array([(col - 1) * 25.0, (row - 1) * -20.0, 30.0, -10.0, 0.0], dtype=np.float32)

    def infer(self, images, joints, task):
        goal = self.target(task.split()[-1])
        steps = np.linspace(joints, goal, self.chunk + 1)[1:].astype(np.float32)
        close = float(np.abs(goal - joints).max()) < 1.0
        return steps, np.full((self.chunk, 1), 1.0 if close else 0.0, dtype=np.float32)

    def close(self) -> None: ...
