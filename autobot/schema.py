"""SO-101 + blaster schema shared by the recorder, uploader, and inference loop.

One place defines topic names, shapes, and order. Change it here, then create a new
contract/profile (both are immutable on PI's side).
"""

import os
from pathlib import Path

import pi_sdk
from pi_sdk.data import McapContract, NdArrayContract, TopicContract
from pi_sdk.finetuning import ArraySlice, TrainingProfileConfig

FPS = 30
CONTRACT_NAME = "so101_blaster_v1"

JOINT_NAMES = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
CAMERAS = ("front_camera", "wrist_camera")

CAMERA_TOPICS = {cam: f"observation/{cam}/rgb/image" for cam in CAMERAS}
OBS_JOINTS = "observation/arm/joints/position"
OBS_GRIPPER = "observation/arm/gripper/position"
ACT_JOINTS = "action/arm/joints/position"
ACT_GRIPPER = "action/arm/gripper/position"
ACT_TRIGGER = "action/blaster/trigger"

ARRAY_SHAPES = {
    OBS_JOINTS: (len(JOINT_NAMES),),
    OBS_GRIPPER: (1,),
    ACT_JOINTS: (len(JOINT_NAMES),),
    ACT_GRIPPER: (1,),
    ACT_TRIGGER: (1,),
}
STATE_TOPICS = (OBS_JOINTS, OBS_GRIPPER)
ACTION_TOPICS = (ACT_JOINTS, ACT_GRIPPER, ACT_TRIGGER)


def api_key() -> str:
    return os.environ.get("PI_API_KEY") or (Path.home() / ".config/pi/api_key").read_text().strip()


def clients() -> tuple[pi_sdk.data.Client, pi_sdk.finetuning.Client]:
    key = api_key()
    return pi_sdk.data.Client(api_key=key), pi_sdk.finetuning.Client(api_key=key)


def contract() -> McapContract:
    def array_topic(topic: str, role: str) -> TopicContract:
        return TopicContract(
            topic=topic,
            role=role,
            schema_encoding="protobuf",
            schema_name="pi.data.v1.NdArray",
            expected_frequency_hz=0,
            frequency_tolerance=0,
            ndarray=NdArrayContract(shape=ARRAY_SHAPES[topic], dtype="float32"),
            required_in_snapshots=("system/step",),
        )

    def camera_topic(topic: str) -> TopicContract:
        return TopicContract(
            topic=topic,
            role="observation",
            schema_encoding="protobuf",
            schema_name="pi.data.v1.VideoFrame",
            expected_frequency_hz=0,
            frequency_tolerance=0,
            required_in_snapshots=("system/step",),
        )

    topics = [
        TopicContract(
            topic="system/step",
            role="system",
            schema_encoding="protobuf",
            schema_name="pi.data.v1.SnapshotRef",
            expected_frequency_hz=FPS,
            frequency_tolerance=0.1,
        ),
        *(array_topic(t, "observation") for t in STATE_TOPICS),
        *(camera_topic(t) for t in CAMERA_TOPICS.values()),
        *(array_topic(t, "action") for t in ACTION_TOPICS),
    ]
    return McapContract(topics=tuple(topics))


def training_profile() -> TrainingProfileConfig:
    return TrainingProfileConfig(
        base_camera_topics=(CAMERA_TOPICS["front_camera"],),
        wrist_camera_topics=(CAMERA_TOPICS["wrist_camera"],),
        state_slices=tuple(ArraySlice(t) for t in STATE_TOPICS),
        action_topics=ACTION_TOPICS,
        action_space="absolute",
    )
