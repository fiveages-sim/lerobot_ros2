#!/usr/bin/env python3
"""
Replay LeRobot end-effector trajectories as ROS2 target poses (single-file version).

Usage:
  python examples/Tianji_dual_XVLA/lerobot_absolute_replay.py \
    --dataset-root dataset/handover_lerobot_dataset_50trajs \
    --episode-index 0 \
    --feature-key action\
    --loop
"""

from __future__ import annotations

import argparse
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import rclpy
from geometry_msgs.msg import Pose
from rclpy.node import Node
from rclpy.publisher import Publisher
from std_msgs.msg import Int32
from visualization_msgs.msg import Marker

try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
except ImportError as exc:  # pragma: no cover - import guard
    raise RuntimeError(
        "The 'lerobot' python package is required to use this script."
    ) from exc

try:
    import torch
    from pytorch3d.transforms import matrix_to_quaternion, rotation_6d_to_matrix
except ImportError as exc:  # pragma: no cover - import guard
    raise RuntimeError(
        "The 'torch' and 'pytorch3d' python packages are required for orientation conversion."
    ) from exc

ARM_SLICES = {
    "left": {"pos": slice(0, 3), "rot6d": slice(3, 9), "gripper": 9},
    "right": {"pos": slice(10, 13), "rot6d": slice(13, 19), "gripper": 19},
}


@dataclass(slots=True)
class ArmTrajectory:
    positions: np.ndarray
    quaternions_xyzw: np.ndarray
    gripper: np.ndarray


@dataclass(slots=True)
class EpisodeTrajectory:
    timestamps: np.ndarray
    left: Optional[ArmTrajectory]
    right: Optional[ArmTrajectory]


@dataclass(slots=True)
class ReplayConfig:
    dataset_root: Path
    episode_index: int = 0
    feature_key: str = "observation.state"
    repo_id: Optional[str] = None
    publish_rate: float = 30.0
    frame_id: str = "arm_base"
    publish_left: bool = True
    publish_right: bool = True
    loop: bool = False
    left_topic: str = "left_target"
    right_topic: str = "right_target"
    left_gripper_controller: str = "left_gripper_controller"
    right_gripper_controller: str = "right_gripper_controller"
    gripper_close_threshold: float = 0.5
    publish_markers: bool = True
    reset_before_replay: bool = True
    require_confirm: bool = True
    initial_left_pose: Optional["PoseSpec"] = None
    initial_right_pose: Optional["PoseSpec"] = None


@dataclass(slots=True)
class PoseSpec:
    position: np.ndarray
    quaternion: np.ndarray


def _stack_float_arrays(values) -> np.ndarray:
    stacked = []
    for item in values:
        if hasattr(item, "detach"):
            stacked.append(item.detach().cpu().numpy())
        else:
            stacked.append(np.asarray(item, dtype=np.float32))
    return np.stack(stacked).astype(np.float32)


def _scalar_array(values, dtype=float) -> np.ndarray:
    scalars = []
    for item in values:
        if hasattr(item, "detach"):
            scalars.append(float(item.detach().cpu().item()))
        else:
            scalars.append(float(item))
    return np.asarray(scalars, dtype=dtype)


def rotation6d_to_quaternion(rot6d_batch: np.ndarray) -> np.ndarray:
    rot6d = torch.as_tensor(np.atleast_2d(rot6d_batch), dtype=torch.float32)
    rot_mats = rotation_6d_to_matrix(rot6d)
    quats_xyzw = matrix_to_quaternion(rot_mats)
    return quats_xyzw.detach().cpu().numpy().astype(np.float32)


def load_episode_trajectory(
    dataset_root: Path,
    episode_index: int,
    feature_key: str,
    repo_id: Optional[str] = None,
) -> EpisodeTrajectory:
    repo = repo_id if repo_id else dataset_root.name
    dataset = LeRobotDataset(
        repo_id=repo,
        root=str(dataset_root),
        episodes=[episode_index],
        download_videos=False,
    )

    subset = dataset.hf_dataset
    if "episode_index" in subset.column_names:
        ep_indices = _scalar_array(subset["episode_index"], dtype=int)
        mask = np.where(ep_indices == int(episode_index))[0]
        if mask.size == 0:
            raise ValueError(f"Episode {episode_index} not found in dataset.")
        subset = subset.select(mask.tolist())
    if feature_key not in subset.column_names:
        raise KeyError(f"Feature '{feature_key}' not available. Columns: {subset.column_names}")

    samples = _stack_float_arrays(subset[feature_key])
    if "timestamp" in subset.column_names:
        timestamps = _scalar_array(subset["timestamp"], dtype=float)
    else:
        timestamps = np.arange(samples.shape[0], dtype=np.float64) / float(dataset.fps)

    arm_data: Dict[str, ArmTrajectory] = {}
    for arm_name, slices in ARM_SLICES.items():
        pos = samples[:, slices["pos"]]
        rot6d = samples[:, slices["rot6d"]]
        quat = rotation6d_to_quaternion(rot6d)
        grip_idx = slices.get("gripper")
        gripper = samples[:, grip_idx] if grip_idx is not None else np.zeros(len(samples), dtype=np.float32)
        arm_data[arm_name] = ArmTrajectory(pos, quat, gripper)

    return EpisodeTrajectory(timestamps=timestamps, left=arm_data.get("left"), right=arm_data.get("right"))


class LeRobotAbsoluteReplayNode(Node):
    def __init__(self, config: ReplayConfig) -> None:
        super().__init__("lerobot_absolute_replay")

        self.publish_rate = float(config.publish_rate)
        self.frame_id = config.frame_id
        self.dataset_root = config.dataset_root
        self.episode_index = int(config.episode_index)
        self.feature_key = config.feature_key
        self.repo_id = config.repo_id
        self.publish_left = bool(config.publish_left)
        self.publish_right = bool(config.publish_right)
        self.loop = bool(config.loop)
        self.left_gripper_controller = config.left_gripper_controller
        self.right_gripper_controller = config.right_gripper_controller
        self.gripper_close_threshold = float(config.gripper_close_threshold)
        self.publish_markers = bool(config.publish_markers)
        self.reset_before_replay = bool(config.reset_before_replay)
        self.require_confirm = bool(config.require_confirm)
        self.initial_left_pose = config.initial_left_pose
        self.initial_right_pose = config.initial_right_pose

        self._left_gripper_pub: Optional[Publisher] = None
        self._right_gripper_pub: Optional[Publisher] = None
        self._last_gripper_command: Dict[str, Optional[int]] = {"left": None, "right": None}
        self._confirm_event = threading.Event()
        self._confirm_thread: Optional[threading.Thread] = None
        self._mode = "reset" if self.reset_before_replay else "replay"

        self.left_topic = config.left_topic
        self.right_topic = config.right_topic
        self.marker_pub = self.create_publisher(Marker, "visualization_marker", 10)
        self._setup_gripper_publishers()

        self._trajectory = load_episode_trajectory(
            self.dataset_root,
            self.episode_index,
            self.feature_key,
            repo_id=self.repo_id,
        )
        self._num_frames = self._trajectory.timestamps.shape[0]
        self._frame_index = 0

        if self.publish_left:
            self.left_pose_pub = self.create_publisher(Pose, self.left_topic, 10)
        else:
            self.left_pose_pub = None

        if self.publish_right:
            self.right_pose_pub = self.create_publisher(Pose, self.right_topic, 10)
        else:
            self.right_pose_pub = None

        if self.publish_rate <= 0.0:
            raise ValueError("publish_rate must be positive")

        self.timer = self.create_timer(1.0 / self.publish_rate, self._on_timer)

        duration = self._trajectory.timestamps[-1] - self._trajectory.timestamps[0]
        self.get_logger().info(
            f"Loaded episode {self.episode_index} from '{self.dataset_root}' "
            f"({self._num_frames} frames, duration {duration:.2f}s)."
        )

        if self._mode == "reset":
            self._enter_reset()

    def _setup_gripper_publishers(self) -> None:
        if self.left_gripper_controller:
            topic = f"/{self.left_gripper_controller}/target_command"
            self._left_gripper_pub = self.create_publisher(Int32, topic, 10)
            self.get_logger().info(f"Left gripper replay enabled on {topic}")
        else:
            self.get_logger().info("Left gripper replay disabled (no controller configured)")

        if self.right_gripper_controller:
            topic = f"/{self.right_gripper_controller}/target_command"
            self._right_gripper_pub = self.create_publisher(Int32, topic, 10)
            self.get_logger().info(f"Right gripper replay enabled on {topic}")
        else:
            self.get_logger().info("Right gripper replay disabled (no controller configured)")

    def _make_pose(self, position: np.ndarray, quaternion: np.ndarray) -> Pose:
        pose = Pose()
        pose.position.x = float(position[0])
        pose.position.y = float(position[1])
        pose.position.z = float(position[2])
        pose.orientation.x = float(quaternion[0])
        pose.orientation.y = float(quaternion[1])
        pose.orientation.z = float(quaternion[2])
        pose.orientation.w = float(quaternion[3])
        return pose

    def _publish(self, pose_pub, position, quaternion) -> None:
        if pose_pub is None:
            return

        pose = self._make_pose(position, quaternion)
        pose_pub.publish(pose)

        if not self.publish_markers:
            return

        marker = Marker()
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.header.frame_id = self.frame_id
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose = pose
        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 0.1
        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        self.marker_pub.publish(marker)

    def _enter_reset(self) -> None:
        self._mode = "reset"
        if self.require_confirm:
            self._start_confirm_wait()
            self.get_logger().info("Resetting to initial pose. Press Enter to start replay...")
        else:
            self.get_logger().info("Resetting to initial pose. Auto-start enabled.")

    def _start_confirm_wait(self) -> None:
        if self._confirm_thread is not None and self._confirm_thread.is_alive():
            return
        self._confirm_event.clear()

        def _wait_for_enter() -> None:
            try:
                input("已回到初始位置，按下回车开始回放...\n")
            except Exception:
                return
            self._confirm_event.set()

        self._confirm_thread = threading.Thread(target=_wait_for_enter, daemon=True)
        self._confirm_thread.start()

    def _publish_initial_poses(self) -> None:
        if self.publish_left and self.left_pose_pub is not None and self.initial_left_pose is not None:
            self._publish(
                self.left_pose_pub,
                self.initial_left_pose.position,
                self.initial_left_pose.quaternion,
            )
        if self.publish_right and self.right_pose_pub is not None and self.initial_right_pose is not None:
            self._publish(
                self.right_pose_pub,
                self.initial_right_pose.position,
                self.initial_right_pose.quaternion,
            )

    def _publish_gripper_command(self, arm: str, pub: Optional[Publisher], raw_value: float) -> None:
        if pub is None:
            return

        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            return

        if np.isnan(value):
            return

        command = int(value >= self.gripper_close_threshold)
        if self._last_gripper_command[arm] == command:
            return

        msg = Int32()
        msg.data = command
        pub.publish(msg)
        self._last_gripper_command[arm] = command

    def _on_timer(self) -> None:
        if self._mode == "reset":
            self._publish_initial_poses()
            if not self.require_confirm or self._confirm_event.is_set():
                self._confirm_event.clear()
                self._mode = "replay"
                self._frame_index = 0
                self.get_logger().info("Starting episode replay.")
            return

        if self._frame_index >= self._num_frames:
            if self.loop:
                self.get_logger().info("Episode replay finished.")
                if self.reset_before_replay:
                    self._enter_reset()
                    return
                else:
                    self.get_logger().info("Looping from start.")
                    self._frame_index = 0
            else:
                self.get_logger().info("Episode replay finished. Stopping publisher timer.")
                self.timer.cancel()
                return

        idx = self._frame_index
        if self.publish_left and self._trajectory.left is not None:
            self._publish(
                self.left_pose_pub,
                self._trajectory.left.positions[idx],
                self._trajectory.left.quaternions_xyzw[idx],
            )
            self._publish_gripper_command(
                "left",
                self._left_gripper_pub,
                self._trajectory.left.gripper[idx],
            )

        if self.publish_right and self._trajectory.right is not None:
            self._publish(
                self.right_pose_pub,
                self._trajectory.right.positions[idx],
                self._trajectory.right.quaternions_xyzw[idx],
            )
            self._publish_gripper_command(
                "right",
                self._right_gripper_pub,
                self._trajectory.right.gripper[idx],
            )

        self._frame_index += 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay LeRobot trajectories via ROS2 topics.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/home/fa/lerobot_ros2/dataset/handover_lerobot_dataset_50trajs"),
        help="Path to LeRobot dataset root",
    )
    parser.add_argument("--episode-index", type=int, default=0, help="Episode index to replay")
    parser.add_argument("--feature-key", type=str, default="action", help="Dataset feature key to replay")
    parser.add_argument("--repo-id", type=str, default="", help="Optional repo_id override")
    parser.add_argument("--publish-rate", type=float, default=30.0, help="Publish rate (Hz)")
    parser.add_argument("--frame-id", type=str, default="arm_base", help="Frame id for PoseStamped")
    parser.add_argument("--publish-left", action="store_true", default=True, help="Publish left arm")
    parser.add_argument("--no-publish-left", action="store_false", dest="publish_left", help="Disable left arm")
    parser.add_argument("--publish-right", action="store_true", default=True, help="Publish right arm")
    parser.add_argument("--no-publish-right", action="store_false", dest="publish_right", help="Disable right arm")
    parser.add_argument("--loop", action="store_true", help="Loop the episode")
    parser.add_argument("--left-topic", type=str, default="left_target", help="Left Pose topic")
    parser.add_argument("--right-topic", type=str, default="right_target", help="Right Pose topic")
    parser.add_argument(
        "--left-gripper-controller",
        type=str,
        default="left_gripper_controller",
        help="Left gripper controller name",
    )
    parser.add_argument(
        "--right-gripper-controller",
        type=str,
        default="right_gripper_controller",
        help="Right gripper controller name",
    )
    parser.add_argument("--gripper-close-threshold", type=float, default=0.5, help="Gripper close threshold")
    parser.add_argument("--no-markers", action="store_true", help="Disable RViz marker publishing")
    parser.add_argument("--no-reset", action="store_true", help="Disable reset to initial pose before replay")
    parser.add_argument("--auto-start", action="store_true", help="Do not wait for Enter before replay")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    left_initial = PoseSpec(
        position=np.array([0.3933272666189703, 0.12494711000828024, 0.26916451531517516], dtype=np.float32),
        quaternion=np.array([0.5657388143990963, -0.4982722862699055, 0.4636322619344259, 0.46552062071538136], dtype=np.float32),
    )
    right_initial = PoseSpec(
        position=np.array([0.39340843062440123, -0.12501497090534242, 0.2691534778752828], dtype=np.float32),
        quaternion=np.array([0.5658124850731148, 0.4982549216702517, 0.46366765288877865, -0.46541440935081585], dtype=np.float32),
    )
    config = ReplayConfig(
        dataset_root=args.dataset_root.expanduser().resolve(),
        episode_index=args.episode_index,
        feature_key=args.feature_key,
        repo_id=args.repo_id or None,
        publish_rate=args.publish_rate,
        frame_id=args.frame_id,
        publish_left=args.publish_left,
        publish_right=args.publish_right,
        loop=args.loop,
        left_topic=args.left_topic,
        right_topic=args.right_topic,
        left_gripper_controller=args.left_gripper_controller,
        right_gripper_controller=args.right_gripper_controller,
        gripper_close_threshold=args.gripper_close_threshold,
        publish_markers=not args.no_markers,
        reset_before_replay=not args.no_reset,
        require_confirm=not args.auto_start,
        initial_left_pose=left_initial,
        initial_right_pose=right_initial,
    )

    rclpy.init()
    node = LeRobotAbsoluteReplayNode(config)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:  # pragma: no cover - CLI convenience
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
