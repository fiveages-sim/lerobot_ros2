#!/usr/bin/env python3
"""
ROS2 Grasp Recording Demo using LeRobot.

This script combines the grasp routine from demo_grasp.py with the dataset
capture utilities from demo_record.py. It performs the four-step grasp sequence
(approach, descend, close gripper, lift) while recording a fine-grained dataset
that follows the LeRobot format. Each recorded frame stores the current
observation (state + images) and an action defined as the NEXT frame's
end-effector pose plus gripper state (the final frame reuses its own pose).
"""

from __future__ import annotations

import signal
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import rclpy
from geometry_msgs.msg import Pose
from rclpy.executors import SingleThreadedExecutor
from tf2_msgs.msg import TFMessage

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot_robot_ros2 import (
    ControlType,
    ROS2Robot,
    ROS2RobotConfig,
    ROS2RobotInterfaceConfig,
)
from lerobot_camera_ros2 import ROS2CameraConfig


# ----------------------- Configuration --------------------------------------
FPS = 30
STAGE_DURATION = 2          # seconds to sample after each command 2s

CAMERA_TOPIC = "/global_camera/rgb"
DEPTH_TOPIC = "/global_camera/depth"
OBJECT_TF_TOPIC = "/isaac/tf"
TARGET_FRAME_ID = "apple"
USE_OBJECT_ORIENTATION = False
APPROACH_CLEARANCE = 0.12
GRASP_CLEARANCE = -0.03
GRIPPER_OPEN = 1.0
GRIPPER_CLOSED = 0.0
POSE_TIMEOUT = 10.0
TASK_NAME = "grasp_apples"
# ----------------------------------------------------------------------------


@dataclass
class PoseSample:
    x: float
    y: float
    z: float
    qx: float
    qy: float
    qz: float
    qw: float
    timestamp: float


class ObjectPoseListener:
    """Subscribe to TF and keep the latest transform for the target frame."""

    def __init__(self, topic: str, target_frame: str) -> None:
        self.topic = topic
        self.target_frame = target_frame
        self._latest: Optional[PoseSample] = None
        self._lock = threading.Lock()
        self._update = threading.Event()

        if not rclpy.ok():
            rclpy.init()

        self._node = rclpy.create_node("grasp_record_pose_listener")
        self._sub = self._node.create_subscription(
            TFMessage, self.topic, self._tf_callback, 10
        )
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)
        self._thread = threading.Thread(
            target=self._executor.spin, daemon=True
        )
        self._thread.start()

    def _tf_callback(self, msg: TFMessage) -> None:
        for transform in msg.transforms:
            if transform.child_frame_id != self.target_frame:
                continue
            trans = transform.transform.translation
            rot = transform.transform.rotation
            sample = PoseSample(
                x=trans.x,
                y=trans.y,
                z=trans.z,
                qx=rot.x,
                qy=rot.y,
                qz=rot.z,
                qw=rot.w,
                timestamp=time.time(),
            )
            with self._lock:
                self._latest = sample
                self._update.set()
            break

    def wait_for_pose(self, timeout: float) -> Optional[PoseSample]:
        if not self._update.wait(timeout):
            return None
        return self.get_latest()

    def get_latest(self) -> Optional[PoseSample]:
        with self._lock:
            return self._latest

    def shutdown(self) -> None:
        if self._executor is not None:
            self._executor.shutdown()
            self._executor = None
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._sub is not None:
            self._sub.destroy()
            self._sub = None
        if self._node is not None:
            self._node.destroy_node()
            self._node = None


def build_robot_config() -> ROS2RobotConfig:
    camera_cfg = {
        "global_camera": ROS2CameraConfig(
            topic_name=CAMERA_TOPIC,
            node_name="lerobot_global_camera",
            width=1280,
            height=720,
            fps=FPS,
            encoding="bgr8",
            depth_topic_name=DEPTH_TOPIC,
            depth_encoding="32FC1",
        )
    }
    ros2_interface = ROS2RobotInterfaceConfig(
        joint_states_topic="/joint_states",
        end_effector_pose_topic="/left_current_pose",
        end_effector_target_topic="/left_target",
        control_type=ControlType.CARTESIAN_POSE,
        joint_names=[
            "joint1",
            "joint2",
            "joint3",
            "joint4",
            "joint5",
            "joint6",
        ],
        max_linear_velocity=0.1,
        max_angular_velocity=0.5,
        gripper_enabled=True,
        gripper_joint_name="gripper_joint",
        gripper_min_position=0.0,
        gripper_max_position=1.0,
        gripper_command_topic="gripper_joint/position_command",
    )
    return ROS2RobotConfig(
        id="ros2_grasp_robot",
        cameras=camera_cfg,
        ros2_interface=ros2_interface,
    )


def pose_from_sample(sample: PoseSample) -> Pose:
    pose = Pose()
    pose.position.x = sample.x
    pose.position.y = sample.y
    pose.position.z = sample.z
    pose.orientation.x = sample.qx
    pose.orientation.y = sample.qy
    pose.orientation.z = sample.qz
    pose.orientation.w = sample.qw
    return pose


def action_from_pose(pose: Pose, gripper: float) -> dict[str, float]:
    return {
        "end_effector.position.x": pose.position.x,
        "end_effector.position.y": pose.position.y,
        "end_effector.position.z": pose.position.z,
        "end_effector.orientation.x": pose.orientation.x,
        "end_effector.orientation.y": pose.orientation.y,
        "end_effector.orientation.z": pose.orientation.z,
        "end_effector.orientation.w": pose.orientation.w,
        "gripper.position": float(np.clip(gripper, 0.0, 1.0)),
    }


def pose_from_observation(obs: dict[str, float]) -> Pose:
    pose = Pose()
    pose.position.x = obs.get("end_effector.position.x", 0.0)
    pose.position.y = obs.get("end_effector.position.y", 0.0)
    pose.position.z = obs.get("end_effector.position.z", 0.0)
    pose.orientation.x = obs.get("end_effector.orientation.x", 0.0)
    pose.orientation.y = obs.get("end_effector.orientation.y", 0.0)
    pose.orientation.z = obs.get("end_effector.orientation.z", 0.0)
    pose.orientation.w = obs.get("end_effector.orientation.w", 1.0)
    return pose


def action_from_observation(obs: dict[str, float], gripper: float) -> dict[str, float]:
    pose = pose_from_observation(obs)
    return action_from_pose(pose, gripper)


def build_dataset(robot: ROS2Robot) -> tuple[LeRobotDataset, Path]:
    features = {}
    features.update(hw_to_dataset_features(robot.action_features, "action"))
    features.update(hw_to_dataset_features(robot.observation_features, "observation"))
    dataset_root = Path.cwd() / "dataset"
    dataset_root.mkdir(parents=True, exist_ok=True)
    dataset_dir = dataset_root / f"grasp_dataset_{int(time.time())}"
    for key, info in features.items():
        if key.endswith(".depth") and info.get("dtype") == "video":
            h, w, _ = info["shape"]
            info["dtype"] = "image"
            info["shape"] = (h, w, 1)
            info["names"] = ["height", "width", "channels"]
    dataset = LeRobotDataset.create(
        repo_id=str(dataset_dir),
        fps=FPS,
        features=features,
        robot_type=robot.name,
        use_videos=True,
    )
    return dataset, dataset_dir


def extract_observation_frame(
    robot: ROS2Robot,
    obs: dict[str, float],
) -> tuple[dict, dict, dict, float]:
    """Return frame payload plus ee pose + gripper for action construction."""
    frame = {"task": TASK_NAME}
    obs_state = []
    cfg = robot.config.ros2_interface

    for joint in cfg.joint_names:
        obs_state.append(obs.get(f"{joint}.pos", 0.0))
    for joint in cfg.joint_names:
        obs_state.append(obs.get(f"{joint}.vel", 0.0))
    for joint in cfg.joint_names:
        obs_state.append(obs.get(f"{joint}.effort", 0.0))

    gripper_pos = 0.0
    if cfg.gripper_enabled:
        gripper_pos = obs.get(f"{cfg.gripper_joint_name}.pos", 0.0)
        obs_state.append(gripper_pos)

    obs_state.extend(
        [
            obs.get("end_effector.position.x", 0.0),
            obs.get("end_effector.position.y", 0.0),
            obs.get("end_effector.position.z", 0.0),
            obs.get("end_effector.orientation.x", 0.0),
            obs.get("end_effector.orientation.y", 0.0),
            obs.get("end_effector.orientation.z", 0.0),
            obs.get("end_effector.orientation.w", 1.0),
        ]
    )
    #状态暂存
    ee_pose = {
        "x": obs.get("end_effector.position.x", 0.0),
        "y": obs.get("end_effector.position.y", 0.0),
        "z": obs.get("end_effector.position.z", 0.0),
        "qx": obs.get("end_effector.orientation.x", 0.0),
        "qy": obs.get("end_effector.orientation.y", 0.0),
        "qz": obs.get("end_effector.orientation.z", 0.0),
        "qw": obs.get("end_effector.orientation.w", 1.0),
    }

    frame["observation.state"] = np.array(obs_state, dtype=np.float32)

    for cam_name in robot.config.cameras.keys():
        rgb_key = f"{cam_name}.rgb"
        depth_key = f"{cam_name}.depth"
        if rgb_key in obs:
            frame[f"observation.images.{cam_name}.rgb"] = obs[rgb_key]
        elif cam_name in obs:
            frame[f"observation.images.{cam_name}"] = obs[cam_name]
        if depth_key in obs:
            depth_img = obs[depth_key]
            if depth_img.ndim == 2:
                depth_img = depth_img[:, :, None]
            frame[f"observation.images.{cam_name}.depth"] = depth_img

    return frame, ee_pose, gripper_pos


def main() -> None:
    print("ROS2 Grasp Recording Demo (LeRobot)")
    print("=" * 70)

    loops = 1
    try:
        loops = max(1, int(input("How many grasp episodes to record? ")))
    except Exception:
        print("[info] invalid input, defaulting to 1 episode")

    robot_config = build_robot_config()
    robot = ROS2Robot(robot_config)
    pose_listener = ObjectPoseListener(OBJECT_TF_TOPIC, TARGET_FRAME_ID)

    def shutdown_handler(sig, frame) -> None:  # type: ignore[override]
        print("\nInterrupt received. Cleaning up...")
        try:
            pose_listener.shutdown()
            if robot.is_connected:
                robot.disconnect()
        finally:
            sys.exit(0)

    signal.signal(signal.SIGINT, shutdown_handler)

    try:
        print("Connecting robot...")
        robot.connect()
        print("[OK] Robot connected")

        dataset, dataset_path = build_dataset(robot)
        # # 修改视频编码方法以使用H.265编码器并保留图
        def _encode_temporary_episode_video_h265(self, video_key: str, episode_index: int) -> Path:
            """
            Use the H.265 encoder (HEVC) to convert PNG frames into MP4 videos
            and keep original images on disk for further inspection.
            """
            import tempfile
            from pathlib import Path as _Path

            from lerobot.datasets.video_utils import encode_video_frames

            temp_dir = tempfile.mkdtemp(dir=self.root)
            temp_path = _Path(temp_dir) / f"{video_key}_{episode_index:03d}.mp4"
            img_dir = self._get_image_file_dir(episode_index, video_key)

            encode_video_frames(
                img_dir,
                temp_path,
                self.fps,
                vcodec="hevc",
                crf=23,
                overwrite=True,
            )

            # NOTE: we intentionally keep img_dir to preserve original PNG frames
            return temp_path

        dataset._encode_temporary_episode_video = _encode_temporary_episode_video_h265.__get__(
            dataset, LeRobotDataset
        )
        print(f"[OK] Dataset initialized at {dataset_path}")

        initial_obs = robot.get_observation()
        initial_pose = pose_from_observation(initial_obs)
        initial_action_open = action_from_pose(initial_pose, GRIPPER_OPEN)
        initial_action_closed = action_from_pose(initial_pose, GRIPPER_CLOSED)

        for episode in range(loops):
            print(f"=== Recording episode {episode + 1}/{loops} ===")
            print("Waiting for object pose TF...")
            sample = pose_listener.wait_for_pose(timeout=POSE_TIMEOUT)
            if sample is None:
                raise RuntimeError(
                    f"No TF data for frame '{TARGET_FRAME_ID}' within {POSE_TIMEOUT:.1f}s"
                )
            target_pose = pose_from_sample(sample)

            current_obs = robot.get_observation()
            current_orientation = (
                current_obs["end_effector.orientation.x"],
                current_obs["end_effector.orientation.y"],
                current_obs["end_effector.orientation.z"],
                current_obs["end_effector.orientation.w"],
            )

            orientation_vec = np.array(
                [
                    target_pose.orientation.x,
                    target_pose.orientation.y,
                    target_pose.orientation.z,
                    target_pose.orientation.w,
                ]
            )
            if not USE_OBJECT_ORIENTATION or np.linalg.norm(orientation_vec) < 1e-3:
                (
                    target_pose.orientation.x,
                    target_pose.orientation.y,
                    target_pose.orientation.z,
                    target_pose.orientation.w,
                ) = current_orientation

            approach_pose = Pose()
            approach_pose.position.x = target_pose.position.x
            approach_pose.position.y = target_pose.position.y
            approach_pose.position.z = target_pose.position.z + APPROACH_CLEARANCE
            approach_pose.orientation = target_pose.orientation

            descend_pose = Pose()
            descend_pose.position.x = target_pose.position.x
            descend_pose.position.y = target_pose.position.y
            descend_pose.position.z = target_pose.position.z + GRASP_CLEARANCE
            descend_pose.orientation = target_pose.orientation

            lift_pose = Pose()
            lift_pose.position.x = target_pose.position.x
            lift_pose.position.y = target_pose.position.y
            lift_pose.position.z = target_pose.position.z + APPROACH_CLEARANCE
            lift_pose.orientation = target_pose.orientation

            sequence = [
                ("Approach", action_from_pose(approach_pose, GRIPPER_OPEN)),
                ("Descend", action_from_pose(descend_pose, GRIPPER_OPEN)),
                ("Close", action_from_pose(descend_pose, GRIPPER_CLOSED)),
                ("Lift", action_from_pose(lift_pose, GRIPPER_CLOSED)),
            ]

            recorded_frames: list[dict] = []

            for step_name, action in sequence:
                print(f"[Step] {step_name}")
                robot.send_action(action)
                start = time.time()
                while (time.time() - start) < STAGE_DURATION:
                    obs = robot.get_observation()
                    frame, ee_pose, gripper = extract_observation_frame(robot, obs)
                    frame["_ee_pose"] = ee_pose
                    frame["_gripper"] = gripper

                    recorded_frames.append(frame)
                    time.sleep(max(0.0, (1.0 / FPS)))

            if not recorded_frames:
                raise RuntimeError("No frames captured during grasp sequence.")

            for idx, frame in enumerate(recorded_frames):
                if idx < len(recorded_frames) - 1:
                    ref = recorded_frames[idx + 1]
                else:
                    ref = recorded_frames[idx]

                pose = ref["_ee_pose"]
                act = [
                    pose["x"],
                    pose["y"],
                    pose["z"],
                    pose["qx"],
                    pose["qy"],
                    pose["qz"],
                    pose["qw"],
                ]
                if robot.config.ros2_interface.gripper_enabled:
                    act.append(ref["_gripper"])

                frame["action"] = np.array(act, dtype=np.float32)
                del frame["_ee_pose"]
                del frame["_gripper"]
                dataset.add_frame(frame)

            dataset.save_episode()
            print(f"[OK] Episode saved with {len(recorded_frames)} frames.")

            if robot.config.ros2_interface.gripper_enabled:
                print("Opening gripper before returning to initial pose...")
                current_obs = robot.get_observation()
                open_current = action_from_observation(current_obs, GRIPPER_OPEN)
                robot.send_action(open_current)
                time.sleep(STAGE_DURATION)

            print("Returning to initial pose...")
            robot.send_action(initial_action_open)
            time.sleep(STAGE_DURATION)

            if robot.config.ros2_interface.gripper_enabled:
                print("Closing gripper after return...")
                robot.send_action(initial_action_closed)
                time.sleep(STAGE_DURATION)

        print(f"[OK] Dataset available at: {dataset_path}")

    except Exception as exc:
        print(f"[ERROR] Recording failed: {exc}")
    finally:
        pose_listener.shutdown()
        if robot.is_connected:
            robot.disconnect()
            print("Robot disconnected")


if __name__ == "__main__":
    main()
