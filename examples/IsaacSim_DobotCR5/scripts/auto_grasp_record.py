#!/usr/bin/env python3
"""
ROS2 Grasp Recording Demo using LeRobot.

python examples/Cr5_ACT/auto_grasp_record.py


This script combines the grasp routine from demo_grasp.py with dataset capture.
It performs a multi-step grasp sequence while recording raw frames (state +
images + pose metadata) to disk. The raw data can later be converted offline
into the LeRobot format by `convert_raw_to_lerobot.py`.
"""

from __future__ import annotations

import json
import pickle
import shutil
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
from sensor_msgs.msg import Image, CameraInfo
from tf2_msgs.msg import TFMessage
from cv_bridge import CvBridge

from lerobot_robot_ros2 import (
    ControlType,
    ROS2Robot,
    ROS2RobotConfig,
    ROS2RobotInterfaceConfig,
)
from lerobot_camera_ros2 import ROS2CameraConfig


# ----------------------- Configuration --------------------------------------
FPS = 30
# 最长等待时间（秒），以防未到达目标位姿也不会无限等待
MAX_STAGE_DURATION = 2.0
# 判定"到达目标"的容差（根据实际硬件精度调整）
POSE_TOL_POS = 0.025     # 米 (20mm，基于实际误差10-40mm设定)
POSE_TOL_ORI = 0.08        # 弧度 (~4.6°)
GRIPPER_TOL = 0.05         # 开合容差（0~1）
RETURN_PAUSE = 1.0         # 回到初始位姿时的短暂停留（秒）

CAMERA_NAME = "global"  # BridgeVLA expects zed_* naming
CAMERA_TOPIC = "/global_camera/rgb"
DEPTH_TOPIC = "/global_camera/depth"
DEPTH_INFO_TOPIC = "/global_camera/camera_info"
WRIST_CAMERA_NAME = "wrist"
WRIST_CAMERA_TOPIC = "/wrist_camera/rgb"
OBJECT_TF_TOPIC = "/isaac/appletf"
TARGET_FRAME_ID = "apple"
USE_OBJECT_ORIENTATION = False
APPROACH_CLEARANCE = 0.12   # Approach 高度：目标上方 120mm
APPROACH_OFFSET_X = 0.12    # Lift 阶段 X 偏移
APPROACH_OFFSET_Y = 0.12    # Lift 阶段 Y 偏移
GRASP_CLEARANCE = -0.03     # Descend 高度：目标上方 5mm（改为正值，避免负Z）
GRIPPER_OPEN = 1.0
GRIPPER_CLOSED = 0.0

# 释放位置配置 (完整的抓取-移动-释放任务)
RELEASE_POSITION_X = -0.5011657941743888
RELEASE_POSITION_Y = 0.36339369774887115
RELEASE_POSITION_Z = 0.37857743925383547
LIFT_HEIGHT = 0.20          # 抬起高度（相对于抓取位置）
TRANSPORT_HEIGHT = 0.4     # 移动时的安全高度
RETRACT_HEIGHT = 0.20       # 释放后撤离高度
POSE_TIMEOUT = 10.0
TASK_NAME = "full_grasp_transport_release"
# Order used for action vectors in dataset/keypoints
ACTION_KEYS = [
    "end_effector.position.x",
    "end_effector.position.y",
    "end_effector.position.z",
    "end_effector.orientation.x",
    "end_effector.orientation.y",
    "end_effector.orientation.z",
    "end_effector.orientation.w",
]
# 是否在关键点采集深度并生成点云，可在开头交互式询问
ENABLE_KEYPOINT_PCD = True
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


class RawDepthListener:
    """Subscribe to raw 32FC1 depth images directly from ROS2 topic."""

    def __init__(self, depth_topic: str) -> None:
        self.depth_topic = depth_topic
        self._latest_depth: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._bridge = CvBridge()

        if not rclpy.ok():
            rclpy.init()

        self._node = rclpy.create_node("grasp_record_depth_listener")
        self._sub = self._node.create_subscription(
            Image, self.depth_topic, self._depth_callback, 10
        )

        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)
        self._thread = threading.Thread(target=self._executor.spin, daemon=True)
        self._thread.start()

        print(f"[RawDepthListener] Subscribed to {depth_topic}")

    def _depth_callback(self, msg: Image) -> None:
        """Callback to receive raw depth images (32FC1)."""
        try:
            # Convert ROS Image message to numpy array, preserving original encoding
            depth_image = self._bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            with self._lock:
                self._latest_depth = depth_image.astype(np.float32)
        except Exception as e:
            print(f"[RawDepthListener] Failed to convert depth image: {e}")

    def get_latest_depth(self) -> Optional[np.ndarray]:
        """Get the most recent raw depth image."""
        with self._lock:
            return self._latest_depth.copy() if self._latest_depth is not None else None

    def shutdown(self) -> None:
        """Clean up resources."""
        self._executor.shutdown()
        self._node.destroy_node()


class DepthCameraInfoListener:
    """Listen to camera info to retrieve intrinsics for depth projection."""

    def __init__(self, info_topic: str) -> None:
        self.info_topic = info_topic
        self._intrinsics: Optional[tuple[float, float, float, float]] = None
        self._lock = threading.Lock()
        self._ready = threading.Event()

        if not rclpy.ok():
            rclpy.init()

        self._node = rclpy.create_node("grasp_record_camera_info_listener")
        self._sub = self._node.create_subscription(
            CameraInfo, self.info_topic, self._info_callback, 1
        )
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)
        self._thread = threading.Thread(target=self._executor.spin, daemon=True)
        self._thread.start()

        print(f"[DepthCameraInfoListener] Subscribed to {info_topic}")

    def _info_callback(self, msg: CameraInfo) -> None:
        fx, fy, cx, cy = msg.k[0], msg.k[4], msg.k[2], msg.k[5]
        with self._lock:
            self._intrinsics = (fx, fy, cx, cy)
            self._ready.set()

    def wait_for_intrinsics(self, timeout: float) -> Optional[tuple[float, float, float, float]]:
        """Wait for a camera info message."""
        if not self._ready.wait(timeout):
            return None
        with self._lock:
            return self._intrinsics

    def shutdown(self) -> None:
        if self._executor is not None:
            self._executor.shutdown()
            self._executor = None
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if hasattr(self, "_sub") and self._sub is not None:
            self._sub.destroy()
            self._sub = None
        if hasattr(self, "_node") and self._node is not None:
            self._node.destroy_node()
            self._node = None


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
        CAMERA_NAME: ROS2CameraConfig(
            topic_name=CAMERA_TOPIC,
            node_name="lerobot_global_camera",
            width=1280,
            height=720,
            fps=FPS,
            encoding="bgr8",
            depth_topic_name=DEPTH_TOPIC,
            depth_encoding="32FC1",
        ),
        WRIST_CAMERA_NAME: ROS2CameraConfig(
            topic_name=WRIST_CAMERA_TOPIC,
            node_name="lerobot_wrist_camera",
            width=1280,
            height=720,
            fps=FPS,
            encoding="bgr8",
        ),
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


def pose_error(
    obs: dict[str, float],
    target: dict[str, float],
    pos_tol: float,
    ori_tol: float,
    grip_tol: float,
) -> tuple[float, float, float]:
    """Compute position/angle/gripper errors between observation and target action dict."""
    dx = obs.get("end_effector.position.x", 0.0) - target.get("end_effector.position.x", 0.0)
    dy = obs.get("end_effector.position.y", 0.0) - target.get("end_effector.position.y", 0.0)
    dz = obs.get("end_effector.position.z", 0.0) - target.get("end_effector.position.z", 0.0)
    pos_err = float(np.sqrt(dx * dx + dy * dy + dz * dz))

    # Orientation error via quaternion dot (clipped to avoid NaNs)
    qw1, qx1, qy1, qz1 = (
        target.get("end_effector.orientation.w", 1.0),
        target.get("end_effector.orientation.x", 0.0),
        target.get("end_effector.orientation.y", 0.0),
        target.get("end_effector.orientation.z", 0.0),
    )
    qw2, qx2, qy2, qz2 = (
        obs.get("end_effector.orientation.w", 1.0),
        obs.get("end_effector.orientation.x", 0.0),
        obs.get("end_effector.orientation.y", 0.0),
        obs.get("end_effector.orientation.z", 0.0),
    )
    dot = qw1 * qw2 + qx1 * qx2 + qy1 * qy2 + qz1 * qz2
    dot = float(np.clip(abs(dot), -1.0, 1.0))
    ori_err = float(2.0 * np.arccos(dot))

    # Gripper observation uses "gripper_joint.pos" key from ROS2
    obs_gripper = obs.get("gripper_joint.pos", obs.get("gripper.position", 0.0))
    target_gripper = target.get("gripper.position", 0.0)
    grip_err = abs(obs_gripper - target_gripper)

    return pos_err, ori_err, grip_err


def depth_to_pointcloud(
    depth: np.ndarray, intrinsics: tuple[float, float, float, float]
) -> np.ndarray:
    """Project depth (H, W) to XYZ (H, W, 3) using pinhole intrinsics."""
    if depth.ndim == 3:
        depth = depth[..., 0]
    fx, fy, cx, cy = intrinsics
    h, w = depth.shape
    u = np.arange(w, dtype=np.float32)
    v = np.arange(h, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)
    z = depth.astype(np.float32)
    # Avoid division by zero
    fx = fx if fx != 0 else 1.0
    fy = fy if fy != 0 else 1.0
    x = (uu - cx) * z / fx
    y = (vv - cy) * z / fy
    pcd = np.stack((x, y, z), axis=-1).astype(np.float32)
    return pcd


def save_pointcloud_frame(
    dataset_dir: Path,
    episode_index: int,
    frame_index: int,
    depth_image: Optional[np.ndarray],
    intrinsics: Optional[tuple[float, float, float, float]],
    rgb_image: Optional[np.ndarray] = None,
    pointcloud_key: str = "observation.points.zed_pcd",
) -> None:
    """Save a single frame point cloud (XYZ + optional RGB) to BridgeVLA-compatible path."""
    # Ensure base dirs exist
    pcd_dir = dataset_dir / "points" / pointcloud_key / "chunk-000" / f"episode_{episode_index:06d}"
    pcd_dir.mkdir(parents=True, exist_ok=True)

    if depth_image is None:
        print(f"[WARN] Depth missing for ep {episode_index} frame {frame_index}, writing zeros pcd")
        depth_image = np.zeros((1, 1), dtype=np.float32)

    if intrinsics is None:
        print("[WARN] Camera intrinsics missing, using identity intrinsics (pcd may be inaccurate)")
        h, w = depth_image.shape[:2]
        intrinsics = (max(w, 1), max(h, 1), w / 2.0, h / 2.0)

    pcd_xyz = depth_to_pointcloud(depth_image, intrinsics)  # (H, W, 3)

    pcd = pcd_xyz
    if rgb_image is not None:
        rgb = np.asarray(rgb_image)
        if rgb.ndim == 2:
            rgb = np.repeat(rgb[:, :, None], 3, axis=2)
        if rgb.shape[:2] != pcd_xyz.shape[:2]:
            print(f"[WARN] RGB shape {rgb.shape} != depth shape {pcd_xyz.shape}, skipping color")
        else:
            rgb = rgb.astype(np.float32)
            if rgb.max() > 1.0:
                rgb = rgb / 255.0
            rgb = np.clip(rgb, 0.0, 1.0)
            pcd = np.concatenate([pcd_xyz, rgb], axis=-1).astype(np.float32)

    out_path = pcd_dir / f"frame_{frame_index:06d}.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(pcd, f)


def build_state_names(robot: ROS2Robot) -> list[str]:
    """Return observation.state names aligned with extract_observation_frame()."""
    names = []
    cfg = robot.config.ros2_interface
    for joint in cfg.joint_names:
        names.append(f"{joint}.pos")
    for joint in cfg.joint_names:
        names.append(f"{joint}.vel")
    if cfg.gripper_enabled:
        names.append(f"{cfg.gripper_joint_name}.pos")
    names.extend(
        [
            "end_effector.position.x",
            "end_effector.position.y",
            "end_effector.position.z",
            "end_effector.orientation.x",
            "end_effector.orientation.y",
            "end_effector.orientation.z",
            "end_effector.orientation.w",
        ]
    )
    return names


def build_raw_output(robot: ROS2Robot, initial_obs: dict[str, float]) -> Path:
    raw_root = Path.cwd() / "raw_dataset" / f"grasp_raw_{int(time.time())}"
    raw_root.mkdir(parents=True, exist_ok=True)
    (raw_root / "episodes").mkdir(parents=True, exist_ok=True)

    camera_shapes = {}
    for cam_name in robot.config.cameras.keys():
        rgb_key = f"{cam_name}.rgb"
        img = None
        if rgb_key in initial_obs:
            img = initial_obs[rgb_key]
        elif cam_name in initial_obs:
            img = initial_obs[cam_name]
        if isinstance(img, np.ndarray):
            camera_shapes[cam_name] = list(img.shape)

    meta = {
        "fps": FPS,
        "robot_type": robot.name,
        "task_name": TASK_NAME,
        "action_keys": ACTION_KEYS,
        "include_gripper": robot.config.ros2_interface.gripper_enabled,
        "state_names": build_state_names(robot),
        "camera_shapes": camera_shapes,
        "camera_names": list(robot.config.cameras.keys()),
    }
    (raw_root / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return raw_root


def cleanup_episode(raw_root: Path, episode_index: int) -> None:
    """Remove episode data (frames/images) and any keypoint pointclouds."""
    episode_dir = raw_root / "episodes" / f"episode_{episode_index:06d}"
    if episode_dir.exists():
        shutil.rmtree(episode_dir, ignore_errors=True)

    points_root = raw_root / "points"
    if points_root.exists():
        target_suffix = f"episode_{episode_index:06d}"
        for path in points_root.rglob(target_suffix):
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)


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

    # 保存相机 RGB（键名规范为 observation.images.<cam> 或 <cam>_rgb）
    for cam_name in robot.config.cameras.keys():
        rgb_key = f"{cam_name}.rgb"
        if rgb_key in obs:
            frame[f"observation.images.{cam_name}_rgb"] = obs[rgb_key]
        elif cam_name in obs:
            frame[f"observation.images.{cam_name}"] = obs[cam_name]

    return frame, ee_pose, gripper_pos




def main() -> None:
    print("ROS2 Grasp Recording Demo (LeRobot)")
    print("=" * 70)

    loops = 1
    try:
        loops = max(1, int(input("How many grasp episodes to record? ")))
    except Exception:
        print("[info] invalid input, defaulting to 1 episode")

    global ENABLE_KEYPOINT_PCD
    use_pcd_input = input("Capture depth+pointcloud at keypoints? [y/N]: ").strip().lower()
    ENABLE_KEYPOINT_PCD = use_pcd_input in {"y", "yes"}

    pause_input = input("Pause between episodes after returning to home? [y/N]: ").strip().lower()
    pause_between_episodes = pause_input in {"y", "yes"}

    robot_config = build_robot_config()
    robot = ROS2Robot(robot_config)
    pose_listener = ObjectPoseListener(OBJECT_TF_TOPIC, TARGET_FRAME_ID)
    depth_listener = RawDepthListener(DEPTH_TOPIC)
    cam_info_listener = DepthCameraInfoListener(DEPTH_INFO_TOPIC)

    def shutdown_handler(sig, frame) -> None:  # type: ignore[override]
        print("\nInterrupt received. Cleaning up...")
        try:
            pose_listener.shutdown()
            depth_listener.shutdown()
            cam_info_listener.shutdown()
            if robot.is_connected:
                robot.disconnect()
        finally:
            sys.exit(0)

    signal.signal(signal.SIGINT, shutdown_handler)

    try:
        print("Connecting robot...")
        robot.connect()
        print("[OK] Robot connected")

        intrinsics = cam_info_listener.wait_for_intrinsics(timeout=POSE_TIMEOUT)
        if intrinsics is None:
            print("[WARN] No camera_info received, will fall back to default intrinsics")
        else:
            fx, fy, cx, cy = intrinsics
            print(f"[OK] Camera intrinsics received: fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")

        initial_obs = robot.get_observation()
        raw_root = build_raw_output(robot, initial_obs)
        print(f"[OK] Raw data will be saved to {raw_root}")

        initial_pose = pose_from_observation(initial_obs)
        initial_action_open = action_from_pose(initial_pose, GRIPPER_OPEN)
        initial_action_closed = action_from_pose(initial_pose, GRIPPER_CLOSED)

        kept_episode = 0
        attempt = 0
        while kept_episode < loops:
            episode_index = kept_episode
            attempt += 1
            print(f"=== Recording episode {kept_episode + 1}/{loops} (attempt {attempt}) ===")
            command_counter = 0
            pcd_save_idx = 0  # 点云文件按保存顺序递增命名
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

            # Lift 阶段：抬起苹果
            lift_pose = Pose()
            lift_pose.position.x = descend_pose.position.x
            lift_pose.position.y = descend_pose.position.y
            lift_pose.position.z = descend_pose.position.z + LIFT_HEIGHT
            lift_pose.orientation = descend_pose.orientation

            # Transport 阶段：移动到释放位置上方
            transport_pose = Pose()
            transport_pose.position.x = RELEASE_POSITION_X
            transport_pose.position.y = RELEASE_POSITION_Y
            transport_pose.position.z = TRANSPORT_HEIGHT
            transport_pose.orientation = descend_pose.orientation

            # Lower 阶段：下降到释放高度
            lower_pose = Pose()
            lower_pose.position.x = RELEASE_POSITION_X
            lower_pose.position.y = RELEASE_POSITION_Y
            lower_pose.position.z = RELEASE_POSITION_Z
            lower_pose.orientation = descend_pose.orientation

            # Retract 阶段：释放后撤离
            retract_pose = Pose()
            retract_pose.position.x = RELEASE_POSITION_X
            retract_pose.position.y = RELEASE_POSITION_Y
            retract_pose.position.z = RELEASE_POSITION_Z + RETRACT_HEIGHT
            retract_pose.orientation = descend_pose.orientation

            # 完整的8阶段序列
            sequence = [
                ("1-Approach", action_from_pose(approach_pose, GRIPPER_OPEN)),
                ("2-Descend", action_from_pose(descend_pose, GRIPPER_OPEN)),
                ("3-Grasp", action_from_pose(descend_pose, GRIPPER_CLOSED)),
                ("4-Lift", action_from_pose(lift_pose, GRIPPER_CLOSED)),
                ("5-Transport", action_from_pose(transport_pose, GRIPPER_CLOSED)),
                ("6-Lower", action_from_pose(lower_pose, GRIPPER_CLOSED)),
                ("7-Release", action_from_pose(lower_pose, GRIPPER_OPEN)),
                ("8-Retract", action_from_pose(retract_pose, GRIPPER_OPEN)),
            ]

            episode_dir = raw_root / "episodes" / f"episode_{episode_index:06d}"
            frames_dir = episode_dir / "frames"
            images_dir = episode_dir / "images"
            frames_dir.mkdir(parents=True, exist_ok=True)
            images_dir.mkdir(parents=True, exist_ok=True)
            frame_meta_path = episode_dir / "frames.jsonl"
            frame_meta_file = frame_meta_path.open("w", encoding="utf-8")

            recorded_count = 0

            pending_keypoint: Optional[dict] = None

            for step_name, action in sequence:
                print(f"[Step] {step_name}")
                robot.send_action(action)
                cmd_gripper = action.get("gripper.position", 0.0)
                pending_keypoint = {
                    "command_index": command_counter,
                }
                command_counter += 1
                start = time.time()
                reached_streak = 0
                while True:
                    obs = robot.get_observation()
                    frame, ee_pose, gripper = extract_observation_frame(robot, obs)
                    frame_idx = recorded_count
                    if ENABLE_KEYPOINT_PCD and pending_keypoint is not None:
                        # Get raw depth directly from ROS2 topic (32FC1, in meters)
                        depth_raw = depth_listener.get_latest_depth()
                        if depth_raw is None:
                            depth_key = f"{CAMERA_NAME}.depth"
                            if depth_key in obs:
                                depth_raw = obs[depth_key].astype(np.float32)

                        # Get RGB image from observation
                        rgb_raw = None
                        for cam_name in robot.config.cameras.keys():
                            rgb_key = f"{cam_name}.rgb"
                            if rgb_key in obs:
                                rgb_raw = obs[rgb_key].astype(np.uint8)
                            elif cam_name in obs:
                                rgb_raw = obs[cam_name].astype(np.uint8)
                            if rgb_raw is not None:
                                break

                        if depth_raw is not None:
                            print(f"[Keypoint] Captured raw depth: shape={depth_raw.shape}, "
                                  f"range=[{depth_raw[depth_raw > 0].min():.3f}, {depth_raw.max():.3f}] m")
                        else:
                            print("[WARNING] Failed to capture raw depth for keypoint")

                        # Save point cloud only at keypoints
                        save_pointcloud_frame(
                            raw_root,
                            episode_index,
                            pcd_save_idx,
                            depth_raw,
                            intrinsics,
                            rgb_image=rgb_raw,
                            pointcloud_key=f"observation.points.{CAMERA_NAME}_pcd",
                        )
                        pcd_save_idx += 1
                        pending_keypoint = None
                    elif pending_keypoint is not None and not ENABLE_KEYPOINT_PCD:
                        # 不采集点云时也要清掉 pending_keypoint，避免后续报错
                        pending_keypoint = None

                    # Save raw frame data (state + pose + gripper + timestamp)
                    ee_pose_arr = np.array(
                        [
                            ee_pose["x"],
                            ee_pose["y"],
                            ee_pose["z"],
                            ee_pose["qx"],
                            ee_pose["qy"],
                            ee_pose["qz"],
                            ee_pose["qw"],
                        ],
                        dtype=np.float32,
                    )
                    frame_file = frames_dir / f"frame_{frame_idx:06d}.npz"
                    np.savez_compressed(
                        frame_file,
                        observation_state=frame["observation.state"],
                        ee_pose=ee_pose_arr,
                        gripper=np.array([gripper], dtype=np.float32),
                        cmd_gripper=np.array([cmd_gripper], dtype=np.float32),
                        timestamp=np.array([time.time()], dtype=np.float64),
                    )

                    # Save RGB images per camera (as .npy to avoid extra dependencies)
                    image_records = {}
                    for cam_name in robot.config.cameras.keys():
                        img = frame.get(f"observation.images.{cam_name}_rgb")
                        if img is None:
                            img = frame.get(f"observation.images.{cam_name}")
                        if isinstance(img, np.ndarray):
                            cam_dir = images_dir / cam_name
                            cam_dir.mkdir(parents=True, exist_ok=True)
                            img_path = cam_dir / f"frame_{frame_idx:06d}.npy"
                            np.save(img_path, img)
                            image_records[cam_name] = str(
                                Path("images") / cam_name / f"frame_{frame_idx:06d}.npy"
                            )

                    frame_meta = {
                        "frame_index": frame_idx,
                        "state_path": str(Path("frames") / f"frame_{frame_idx:06d}.npz"),
                        "images": image_records,
                    }
                    frame_meta_file.write(json.dumps(frame_meta, ensure_ascii=False) + "\n")
                    recorded_count += 1
                    # 判定是否到达目标：连续 3 帧满足容差即进入下一阶段（不考虑夹爪误差）
                    pos_err, ori_err, grip_err = pose_error(
                        obs, action, POSE_TOL_POS, POSE_TOL_ORI, GRIPPER_TOL
                    )
                    if (pos_err <= POSE_TOL_POS) and (ori_err <= POSE_TOL_ORI):
                        reached_streak += 1
                    else:
                        reached_streak = 0

                    if reached_streak >= 3:
                        break

                    if (time.time() - start) > MAX_STAGE_DURATION:
                        print(
                            f"[WARN] Stage '{step_name}' timeout "
                            f"(pos_err={pos_err:.4f}, ori_err={ori_err:.3f}, grip_err={grip_err:.3f})"
                        )
                        break

                    time.sleep(max(0.0, (1.0 / FPS)))

                if pending_keypoint is not None:
                    raise RuntimeError(
                        f"No frame captured after sending action index {pending_keypoint['command_index']}"
                    )

            frame_meta_file.close()

            if recorded_count == 0:
                raise RuntimeError("No frames captured during grasp sequence.")

            print(f"[OK] Episode raw saved with {recorded_count} frames.")
            delete_input = input("Delete this episode data? [y/N]: ").strip().lower()
            keep_episode = delete_input not in {"y", "yes"}
            if not keep_episode:
                cleanup_episode(raw_root, episode_index)
                print(f"[Cleanup] Episode {kept_episode + 1} removed.")

            if robot.config.ros2_interface.gripper_enabled:
                print("Opening gripper before returning to initial pose...")
                current_obs = robot.get_observation()
                open_current = action_from_observation(current_obs, GRIPPER_OPEN)
                robot.send_action(open_current)
                time.sleep(RETURN_PAUSE)

            print("Returning to initial pose...")
            robot.send_action(initial_action_open)
            time.sleep(RETURN_PAUSE)

            if robot.config.ros2_interface.gripper_enabled:
                print("Closing gripper after return...")
                robot.send_action(initial_action_closed)
                time.sleep(RETURN_PAUSE)

            if keep_episode:
                kept_episode += 1

            if pause_between_episodes and kept_episode < loops:
                input("Press Enter to start the next episode...")

        print(f"[OK] Raw dataset available at: {raw_root}")

    except Exception as exc:
        print(f"[ERROR] Recording failed: {exc}")
    finally:
        pose_listener.shutdown()
        depth_listener.shutdown()
        cam_info_listener.shutdown()
        if robot.is_connected:
            robot.disconnect()
            print("Robot disconnected")

 
if __name__ == "__main__":
    main()
