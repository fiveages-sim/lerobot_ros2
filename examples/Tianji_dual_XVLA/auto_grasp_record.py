#!/usr/bin/env python3
"""
ROS2 Grasp Recording Demo using LeRobot.

python examples/Tianji_dual_XVLA/auto_grasp_record.py


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
GRASP_STAGE_DURATION = 3.0   # 抓取阶段固定等待时间（秒），给夹爪足够时间完成闭合动作（考虑位置偏置）
# 判定"到达目标"的容差（根据实际硬件精度调整）
POSE_TOL_POS = 0.025     # 米 (20mm，基于实际误差10-40mm设定)
POSE_TOL_ORI = 0.08        # 弧度 (~4.6°)
GRIPPER_TOL = 0.05         # 开合容差（0~1）
RETURN_PAUSE = 1.0         # 回到初始位姿时的短暂停留（秒）

CAMERA_NAME = "head"  
CAMERA_TOPIC = "/head_camera/rgb"
DEPTH_TOPIC = "/head_camera/depth"
DEPTH_INFO_TOPIC = "/head_camera/camera_info"
# Two wrist cameras for dual-arm setup
LEFT_WRIST_CAMERA_NAME = "left_wrist"
LEFT_WRIST_CAMERA_TOPIC = "/left_wrist_camera/rgb"
RIGHT_WRIST_CAMERA_NAME = "right_wrist"
RIGHT_WRIST_CAMERA_TOPIC = "/right_wrist_camera/rgb"
OBJECT_TF_TOPIC = "/isaac/medicinetf"
TARGET_FRAME_ID = "tablets"
USE_OBJECT_ORIENTATION = True
APPROACH_CLEARANCE = 0.15   # Approach 高度：目标上方 150mm
APPROACH_OFFSET_X = 0.03    # Lift 阶段 X 偏移（负值让抓取位置向x负方向偏移）
APPROACH_OFFSET_Y = 0.06    # Lift 阶段 Y 偏移
GRASP_CLEARANCE = 0.01      
GRIPPER_OPEN = 1.0
GRIPPER_CLOSED = 0.0

# 右手换手位置配置
HANDOVER_POSITION_X = 0.396690552561732
HANDOVER_POSITION_Y = -0.02282794401272727
HANDOVER_POSITION_Z = 0.3910536913261088
HANDOVER_ORIENTATION_QX = 0.5259496034887217
HANDOVER_ORIENTATION_QY = 0.4977646286125966
HANDOVER_ORIENTATION_QZ = 0.49207819248623463
HANDOVER_ORIENTATION_QW = -0.4831836520120486

# 释放位置配置 (完整的抓取-移动-释放任务)
LEFT_RELEASE_POSITION_X = 0.4560101318359375
LEFT_RELEASE_POSITION_Y = 0.3267540919780731
LEFT_RELEASE_POSITION_Z = 0.17857743925383547
LIFT_HEIGHT = 0.06          # 抬起高度（相对于抓取位置）
HANDOVER_OFFSET_X = -0.024
HANDOVER_OFFSET_Y = 0.08    
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


def build_dual_robot_configs() -> tuple[ROS2RobotConfig, ROS2RobotConfig]:
    """Build two ROS2RobotConfig objects: right arm and left arm."""
    camera_cfg = {
        CAMERA_NAME: ROS2CameraConfig(
            topic_name=CAMERA_TOPIC,
            node_name="lerobot_head_camera",
            width=640,
            height=480,
            fps=FPS,
            encoding="bgr8",
            depth_topic_name=DEPTH_TOPIC,
            depth_encoding="32FC1",
        ),
        LEFT_WRIST_CAMERA_NAME: ROS2CameraConfig(
            topic_name=LEFT_WRIST_CAMERA_TOPIC,
            node_name="lerobot_left_wrist_camera",
            width=1280,
            height=720,
            fps=FPS,
            encoding="bgr8",
        ),
        RIGHT_WRIST_CAMERA_NAME: ROS2CameraConfig(
            topic_name=RIGHT_WRIST_CAMERA_TOPIC,
            node_name="lerobot_right_wrist_camera",
            width=640,  
            height=480,
            fps=FPS,
            encoding="bgr8",
        ),
    }

    # Right-arm interface
    right_iface = ROS2RobotInterfaceConfig(
        joint_states_topic="/isaac/joint_states",
        end_effector_pose_topic="/right_current_pose",
        end_effector_target_topic="/right_target",
        control_type=ControlType.CARTESIAN_POSE,
        joint_names=[
            "right_joint1",
            "right_joint2",
            "right_joint3",
            "right_joint4",
            "right_joint5",
            "right_joint6",
            "right_joint7",
        ],
        max_linear_velocity=0.1,
        max_angular_velocity=0.5,
        gripper_enabled=True,
        gripper_joint_name="right_gripper_joint",
        gripper_min_position=0.0,
        gripper_max_position=1.0,
        gripper_command_topic="/right_gripper_joint/position_command",
        right_gripper_controller_name="right_hand_controller",  # For target_command mode
    )

    # Left-arm interface
    left_iface = ROS2RobotInterfaceConfig(
        joint_states_topic="/isaac/joint_states",
        end_effector_pose_topic="/left_current_pose",
        end_effector_target_topic="/left_target",
        control_type=ControlType.CARTESIAN_POSE,
        joint_names=[
            "left_joint1",
            "left_joint2",
            "left_joint3",
            "left_joint4",
            "left_joint5",
            "left_joint6",
            "left_joint7",
        ],
        max_linear_velocity=0.1,
        max_angular_velocity=0.5,
        gripper_enabled=True,
        gripper_joint_name="left_gripper_joint",
        gripper_min_position=0.0,
        gripper_max_position=1.0,
        gripper_command_topic="/left_gripper_joint/position_command",
        left_gripper_controller_name="left_hand_controller",  # For target_command mode
    )

    right_cfg = ROS2RobotConfig(id="ros2_right_arm", cameras=camera_cfg, ros2_interface=right_iface)
    left_cfg = ROS2RobotConfig(id="ros2_left_arm", cameras=camera_cfg, ros2_interface=left_iface)
    return right_cfg, left_cfg


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


def euler_from_quaternion(qx: float, qy: float, qz: float, qw: float) -> tuple[float, float, float]:
    """Return (roll, pitch, yaw) from quaternion (x,y,z,w)."""
    import math

    # roll (x-axis rotation)
    t0 = +2.0 * (qw * qx + qy * qz)
    t1 = +1.0 - 2.0 * (qx * qx + qy * qy)
    roll = math.atan2(t0, t1)

    # pitch (y-axis rotation)
    t2 = +2.0 * (qw * qy - qz * qx)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch = math.asin(t2)

    # yaw (z-axis rotation)
    t3 = +2.0 * (qw * qz + qx * qy)
    t4 = +1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = math.atan2(t3, t4)

    return roll, pitch, yaw


def quaternion_from_euler(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
    """Return quaternion (x,y,z,w) from Euler angles (roll, pitch, yaw)."""
    import math

    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    return qx, qy, qz, qw


def make_grasp_orientation_from_object(obj_pose: Pose) -> tuple[float, float, float, float]:
    """Compute a quaternion for the gripper so it points down (vertical) and rotates
    around the approach axis to align with the object's yaw (so gripper opening aligns with box width).

    Strategy: extract yaw from object's quaternion, set roll=0, pitch=pi (flip so z points down), yaw=obj_yaw.
    """
    import math

    qx = obj_pose.orientation.x
    qy = obj_pose.orientation.y
    qz = obj_pose.orientation.z
    qw = obj_pose.orientation.w
    _, _, obj_yaw = euler_from_quaternion(qx, qy, qz, qw)

    # roll=0, pitch=pi to invert z-axis (gripper pointing down), yaw=obj_yaw
    roll = 0.0
    pitch = math.pi
    yaw = obj_yaw
    return quaternion_from_euler(roll, pitch, yaw)


def action_from_observation(obs: dict[str, float], gripper: float) -> dict[str, float]:
    pose = pose_from_observation(obs)
    return action_from_pose(pose, gripper)


def send_action_with_actor_gripper(robot: ROS2Robot, actor: str, action: dict[str, float]) -> None:
    """Send pose through LeRobot API, and route gripper command by actor side."""
    pose_only_action = {k: v for k, v in action.items() if k != "gripper.position"}
    robot.send_action(pose_only_action)

    if not robot.config.ros2_interface.gripper_enabled:
        return

    if "gripper.position" not in action:
        return

    gripper_position = float(action["gripper.position"])
    iface = robot.ros2_interface

    # Choose handler based on actor
    handler = iface.right_gripper_handler if actor == "right" else iface.left_gripper_handler

    if handler and hasattr(handler, 'target_command_pub') and handler.target_command_pub is not None:
        # Use target command (force feedback mode)
        target_value = 1 if gripper_position > 0.5 else 0
        handler.send_target_command(target_value)
    else:
        # Use position command (fallback)
        if actor == "right":
            # Prefer right gripper publisher when available (dual-arm topic layout).
            if getattr(iface, "right_gripper_handler", None) is not None:
                iface.send_right_gripper_command(gripper_position)
            else:
                # Fallback for setups where right side is configured as the primary gripper.
                iface.send_gripper_command(gripper_position)
        else:
            iface.send_gripper_command(gripper_position)


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


def build_state_names_dual(robot_right: ROS2Robot, robot_left: ROS2Robot) -> list[str]:
    """Build merged state names for a dual-arm robot (prefix with 'right.' and 'left.')."""
    names = []
    # Right arm
    cfg_r = robot_right.config.ros2_interface
    for joint in cfg_r.joint_names:
        names.append(f"right.{joint}.pos")
    for joint in cfg_r.joint_names:
        names.append(f"right.{joint}.vel")
    if cfg_r.gripper_enabled:
        names.append(f"right.{cfg_r.gripper_joint_name}.pos")
    names.extend([
        "right.end_effector.position.x",
        "right.end_effector.position.y",
        "right.end_effector.position.z",
        "right.end_effector.orientation.x",
        "right.end_effector.orientation.y",
        "right.end_effector.orientation.z",
        "right.end_effector.orientation.w",
    ])

    # Left arm
    cfg_l = robot_left.config.ros2_interface
    for joint in cfg_l.joint_names:
        names.append(f"left.{joint}.pos")
    for joint in cfg_l.joint_names:
        names.append(f"left.{joint}.vel")
    if cfg_l.gripper_enabled:
        names.append(f"left.{cfg_l.gripper_joint_name}.pos")
    names.extend([
        "left.end_effector.position.x",
        "left.end_effector.position.y",
        "left.end_effector.position.z",
        "left.end_effector.orientation.x",
        "left.end_effector.orientation.y",
        "left.end_effector.orientation.z",
        "left.end_effector.orientation.w",
    ])

    return names


def build_raw_output_dual(
    robot_right: ROS2Robot,
    robot_left: ROS2Robot,
    initial_obs_right: dict[str, float],
    initial_obs_left: dict[str, float],
) -> Path:
    """Create raw output tree and meta.json for dual-arm recording.

    Merges camera shapes and builds combined state/action keys.
    """
    raw_root = Path.cwd() / "raw_dataset" / f"grasp_raw_{int(time.time())}"
    raw_root.mkdir(parents=True, exist_ok=True)
    (raw_root / "episodes").mkdir(parents=True, exist_ok=True)

    camera_shapes = {}
    # Merge cameras from right robot (they share the same camera_cfg)
    for cam_name in robot_right.config.cameras.keys():
        rgb_key = f"{cam_name}.rgb"
        img = None
        if rgb_key in initial_obs_right:
            img = initial_obs_right[rgb_key]
        elif cam_name in initial_obs_right:
            img = initial_obs_right[cam_name]
        elif rgb_key in initial_obs_left:
            img = initial_obs_left[rgb_key]
        elif cam_name in initial_obs_left:
            img = initial_obs_left[cam_name]
        if isinstance(img, np.ndarray):
            camera_shapes[cam_name] = list(img.shape)

    # Build combined action keys: right then left
    action_keys = [f"right.{k}" for k in ACTION_KEYS] + [f"left.{k}" for k in ACTION_KEYS]

    meta = {
        "fps": FPS,
        "robot_type": "tianji_dual_arm",
        "task_name": TASK_NAME,
        "action_keys": action_keys,
        "include_gripper_right": robot_right.config.ros2_interface.gripper_enabled,
        "include_gripper_left": robot_left.config.ros2_interface.gripper_enabled,
        "state_names": build_state_names_dual(robot_right, robot_left),
        "camera_shapes": camera_shapes,
        "camera_names": list(robot_right.config.cameras.keys()),
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

    right_cfg, left_cfg = build_dual_robot_configs()
    robot_right = ROS2Robot(right_cfg)
    robot_left = ROS2Robot(left_cfg)
    pose_listener = ObjectPoseListener(OBJECT_TF_TOPIC, TARGET_FRAME_ID)
    depth_listener = RawDepthListener(DEPTH_TOPIC)
    cam_info_listener = DepthCameraInfoListener(DEPTH_INFO_TOPIC)

    def shutdown_handler(sig, frame) -> None:  # type: ignore[override]
        print("\nInterrupt received. Cleaning up...")
        try:
            pose_listener.shutdown()
            depth_listener.shutdown()
            cam_info_listener.shutdown()
            try:
                if 'robot_right' in locals() and robot_right.is_connected:
                    robot_right.disconnect()
            except Exception:
                pass
            try:
                if 'robot_left' in locals() and robot_left.is_connected:
                    robot_left.disconnect()
            except Exception:
                pass
        finally:
            sys.exit(0)

    signal.signal(signal.SIGINT, shutdown_handler)

    try:
        print("Connecting both robots...")
        robot_right.connect()
        robot_left.connect()
        print("[OK] Both robots connected")

        intrinsics = cam_info_listener.wait_for_intrinsics(timeout=POSE_TIMEOUT)
        if intrinsics is None:
            print("[WARN] No camera_info received, will fall back to default intrinsics")
        else:
            fx, fy, cx, cy = intrinsics
            print(f"[OK] Camera intrinsics received: fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")

        initial_obs_right = robot_right.get_observation()
        initial_obs_left = robot_left.get_observation()
        # Build raw output metadata for dual-arm robot
        raw_root = build_raw_output_dual(robot_right, robot_left, initial_obs_right, initial_obs_left)
        print(f"[OK] Raw data will be saved to {raw_root}")

        initial_pose_right = pose_from_observation(initial_obs_right)
        initial_action_right_open = action_from_pose(initial_pose_right, GRIPPER_OPEN)
        initial_action_right_closed = action_from_pose(initial_pose_right, GRIPPER_CLOSED)

        initial_pose_left = pose_from_observation(initial_obs_left)
        initial_action_left_open = action_from_pose(initial_pose_left, GRIPPER_OPEN)
        initial_action_left_closed = action_from_pose(initial_pose_left, GRIPPER_CLOSED)

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

            current_obs = robot_right.get_observation()
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
            if USE_OBJECT_ORIENTATION and np.linalg.norm(orientation_vec) >= 1e-3:
                # compute gripper-down orientation aligned with object yaw
                qx, qy, qz, qw = make_grasp_orientation_from_object(target_pose)
                target_pose.orientation.x = qx
                target_pose.orientation.y = qy
                target_pose.orientation.z = qz
                target_pose.orientation.w = qw
            else:
                (
                    target_pose.orientation.x,
                    target_pose.orientation.y,
                    target_pose.orientation.z,
                    target_pose.orientation.w,
                ) = current_orientation

            # Right-arm approach/descend/grasp poses (use object pose from TF)
            approach_pose_r = Pose()
            approach_pose_r.position.x = target_pose.position.x + APPROACH_OFFSET_X
            approach_pose_r.position.y = target_pose.position.y
            approach_pose_r.position.z = target_pose.position.z + APPROACH_CLEARANCE
            approach_pose_r.orientation = target_pose.orientation

            descend_pose_r = Pose()
            descend_pose_r.position.x = target_pose.position.x + APPROACH_OFFSET_X
            descend_pose_r.position.y = target_pose.position.y
            descend_pose_r.position.z = target_pose.position.z + GRASP_CLEARANCE
            descend_pose_r.orientation = target_pose.orientation

            lift_pose_r = Pose()
            lift_pose_r.position.x = descend_pose_r.position.x
            lift_pose_r.position.y = descend_pose_r.position.y
            lift_pose_r.position.z = descend_pose_r.position.z + APPROACH_CLEARANCE
            lift_pose_r.orientation = descend_pose_r.orientation

            # Handover pose (a small offset towards the left arm)
            handover_pose_r = Pose()
            handover_pose_r.position.x = HANDOVER_POSITION_X
            handover_pose_r.position.y = HANDOVER_POSITION_Y
            handover_pose_r.position.z = HANDOVER_POSITION_Z
            handover_pose_r.orientation.x = HANDOVER_ORIENTATION_QX
            handover_pose_r.orientation.y = HANDOVER_ORIENTATION_QY
            handover_pose_r.orientation.z = HANDOVER_ORIENTATION_QZ
            handover_pose_r.orientation.w = HANDOVER_ORIENTATION_QW

            # Left arm pick/handover approach (above the handover position)
            handover_approach_l = Pose()
            handover_approach_l.position.x = HANDOVER_POSITION_X + HANDOVER_OFFSET_X
            handover_approach_l.position.y = -HANDOVER_POSITION_Y + HANDOVER_OFFSET_Y
            handover_approach_l.position.z = HANDOVER_POSITION_Z + LIFT_HEIGHT
            handover_approach_l.orientation.x = HANDOVER_ORIENTATION_QX
            handover_approach_l.orientation.y = -HANDOVER_ORIENTATION_QY
            handover_approach_l.orientation.z = HANDOVER_ORIENTATION_QZ
            handover_approach_l.orientation.w = -HANDOVER_ORIENTATION_QW

            handover_descend_l = Pose()
            handover_descend_l.position.x = HANDOVER_POSITION_X + HANDOVER_OFFSET_X
            handover_descend_l.position.y = -HANDOVER_POSITION_Y - HANDOVER_OFFSET_Y
            handover_descend_l.position.z = HANDOVER_POSITION_Z + LIFT_HEIGHT
            handover_descend_l.orientation.x = HANDOVER_ORIENTATION_QX
            handover_descend_l.orientation.y = -HANDOVER_ORIENTATION_QY
            handover_descend_l.orientation.z = HANDOVER_ORIENTATION_QZ
            handover_descend_l.orientation.w = -HANDOVER_ORIENTATION_QW

            # Left arm final place poses
            place_approach_l = Pose()
            place_approach_l.position.x = LEFT_RELEASE_POSITION_X
            place_approach_l.position.y = LEFT_RELEASE_POSITION_Y
            place_approach_l.position.z = LEFT_RELEASE_POSITION_Z + LIFT_HEIGHT
            place_approach_l.orientation.x = HANDOVER_ORIENTATION_QX
            place_approach_l.orientation.y = -HANDOVER_ORIENTATION_QY
            place_approach_l.orientation.z = HANDOVER_ORIENTATION_QZ
            place_approach_l.orientation.w = -HANDOVER_ORIENTATION_QW

            place_descend_l = Pose()
            place_descend_l.position.x = LEFT_RELEASE_POSITION_X
            place_descend_l.position.y = LEFT_RELEASE_POSITION_Y
            place_descend_l.position.z = LEFT_RELEASE_POSITION_Z
            place_descend_l.orientation.x = HANDOVER_ORIENTATION_QX
            place_descend_l.orientation.y = -HANDOVER_ORIENTATION_QY
            place_descend_l.orientation.z = HANDOVER_ORIENTATION_QZ
            place_descend_l.orientation.w = -HANDOVER_ORIENTATION_QW

            place_retract_l = Pose()
            place_retract_l.position.x = LEFT_RELEASE_POSITION_X
            place_retract_l.position.y = LEFT_RELEASE_POSITION_Y
            place_retract_l.position.z = LEFT_RELEASE_POSITION_Z + RETRACT_HEIGHT
            place_retract_l.orientation.x = HANDOVER_ORIENTATION_QX
            place_retract_l.orientation.y = -HANDOVER_ORIENTATION_QY
            place_retract_l.orientation.z = HANDOVER_ORIENTATION_QZ
            place_retract_l.orientation.w = -HANDOVER_ORIENTATION_QW

            # Build a coordinated sequence: (label, actor, action)
            sequence = [
                ("R-Approach", "right", action_from_pose(approach_pose_r, GRIPPER_OPEN)),
                ("R-Descend", "right", action_from_pose(descend_pose_r, GRIPPER_OPEN)),
                ("R-Grasp", "right", action_from_pose(descend_pose_r, GRIPPER_CLOSED)),
                ("R-Lift", "right", action_from_pose(lift_pose_r, GRIPPER_CLOSED)),
                ("R-MoveToHandover", "right", action_from_pose(handover_pose_r, GRIPPER_CLOSED)),
                ("L-ApproachHandover", "left", action_from_pose(handover_approach_l, GRIPPER_OPEN)),
                ("L-DescendHandover", "left", action_from_pose(handover_descend_l, GRIPPER_OPEN)),
                ("L-GraspFromR", "left", action_from_pose(handover_descend_l, GRIPPER_CLOSED)),
                ("R-OpenToRelease", "right", action_from_pose(handover_pose_r, GRIPPER_OPEN)),
                ("L-LiftWithBox", "left", action_from_pose(handover_approach_l, GRIPPER_CLOSED)),
                ("L-MoveToPlace", "left", action_from_pose(place_approach_l, GRIPPER_CLOSED)),
                ("L-LowerPlace", "left", action_from_pose(place_descend_l, GRIPPER_CLOSED)),
                ("L-Release", "left", action_from_pose(place_descend_l, GRIPPER_OPEN)),
                ("L-Retract", "left", action_from_pose(place_retract_l, GRIPPER_OPEN)),
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

            for step_name, actor, action in sequence:
                print(f"[Step] {step_name} -> {actor}")
                # Choose which robot to act on
                actor_robot = robot_right if actor == "right" else robot_left
                is_grasp_step = step_name in {"R-Grasp", "L-GraspFromR"}
                stage_timeout = GRASP_STAGE_DURATION if is_grasp_step else MAX_STAGE_DURATION
                
                # 对于抓取阶段，直接等待固定时间，不检查位姿（因为设置了位置偏置）
                if is_grasp_step:
                    print(f"[Grasp] Sending gripper close command and waiting {GRASP_STAGE_DURATION:.1f}s...")
                    send_action_with_actor_gripper(actor_robot, actor, action)
                    cmd_gripper = action.get("gripper.position", 0.0)
                    pending_keypoint = {"command_index": command_counter}
                    command_counter += 1
                    start = time.time()
                    
                    # 等待固定时间，同时记录数据
                    while (time.time() - start) < GRASP_STAGE_DURATION:
                        obs_act = actor_robot.get_observation()
                        obs_other = robot_left.get_observation() if actor == "right" else robot_right.get_observation()
                        
                        # Build combined frame by merging both robots' observations
                        frame_r, ee_pose_r, gripper_r = extract_observation_frame(robot_right, robot_right.get_observation())
                        frame_l, ee_pose_l, gripper_l = extract_observation_frame(robot_left, robot_left.get_observation())
                        
                        frame_idx = recorded_count
                        
                        if ENABLE_KEYPOINT_PCD and pending_keypoint is not None:
                            depth_raw = depth_listener.get_latest_depth()
                            if depth_raw is None:
                                depth_key = f"{CAMERA_NAME}.depth"
                                if depth_key in obs_act:
                                    depth_raw = obs_act[depth_key].astype(np.float32)
                            
                            rgb_raw = None
                            for cam_name in actor_robot.config.cameras.keys():
                                rgb_key = f"{cam_name}.rgb"
                                if rgb_key in obs_act:
                                    rgb_raw = obs_act[rgb_key].astype(np.uint8)
                                elif cam_name in obs_act:
                                    rgb_raw = obs_act[cam_name].astype(np.uint8)
                                if rgb_raw is not None:
                                    break
                            
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
                        
                        # Save frame data
                        ee_pose_arr_r = np.array([
                            ee_pose_r["x"], ee_pose_r["y"], ee_pose_r["z"], ee_pose_r["qx"], ee_pose_r["qy"], ee_pose_r["qz"], ee_pose_r["qw"]
                        ], dtype=np.float32)
                        ee_pose_arr_l = np.array([
                            ee_pose_l["x"], ee_pose_l["y"], ee_pose_l["z"], ee_pose_l["qx"], ee_pose_l["qy"], ee_pose_l["qz"], ee_pose_l["qw"]
                        ], dtype=np.float32)
                        
                        frame_file = frames_dir / f"frame_{frame_idx:06d}.npz"
                        np.savez_compressed(
                            frame_file,
                            observation_state_right=frame_r["observation.state"],
                            ee_pose_right=ee_pose_arr_r,
                            gripper_right=np.array([gripper_r], dtype=np.float32),
                            observation_state_left=frame_l["observation.state"],
                            ee_pose_left=ee_pose_arr_l,
                            gripper_left=np.array([gripper_l], dtype=np.float32),
                            cmd_gripper=np.array([cmd_gripper], dtype=np.float32),
                            timestamp=np.array([time.time()], dtype=np.float64),
                        )
                        
                        # Save RGB images
                        image_records = {}
                        for cam_name in actor_robot.config.cameras.keys():
                            img = None
                            if f"{cam_name}.rgb" in frame_r:
                                img = frame_r.get(f"observation.images.{cam_name}_rgb") or frame_r.get(f"observation.images.{cam_name}")
                            if img is None and f"{cam_name}.rgb" in frame_l:
                                img = frame_l.get(f"observation.images.{cam_name}_rgb") or frame_l.get(f"observation.images.{cam_name}")
                            if isinstance(img, np.ndarray):
                                cam_dir = images_dir / cam_name
                                cam_dir.mkdir(parents=True, exist_ok=True)
                                img_path = cam_dir / f"frame_{frame_idx:06d}.npy"
                                np.save(img_path, img)
                                image_records[cam_name] = str(Path("images") / cam_name / f"frame_{frame_idx:06d}.npy")
                        
                        frame_meta = {
                            "frame_index": frame_idx,
                            "state_path": str(Path("frames") / f"frame_{frame_idx:06d}.npz"),
                            "images": image_records,
                        }
                        frame_meta_file.write(json.dumps(frame_meta, ensure_ascii=False) + "\n")
                        recorded_count += 1
                        
                        time.sleep(max(0.0, (1.0 / FPS)))
                    
                    print(f"[Grasp] Completed {GRASP_STAGE_DURATION:.1f}s wait, proceeding to next step.")
                    continue
                
                # 非抓取阶段的正常流程
                # 在右臂 approach 和 descend 阶段，确保夹爪在发送action时就张开
                if actor == "right" and (step_name.startswith("R-Approach") or step_name.startswith("R-Descend")):
                    action = dict(action)
                    action["gripper.position"] = GRIPPER_OPEN
                send_action_with_actor_gripper(actor_robot, actor, action)
                cmd_gripper = action.get("gripper.position", 0.0)
                pending_keypoint = {"command_index": command_counter}
                command_counter += 1
                start = time.time()
                reached_streak = 0
                
                while True:
                    # 在右臂 approach 和 descend 阶段，持续保持夹爪张开
                    if actor == "right" and (step_name.startswith("R-Approach") or step_name.startswith("R-Descend")):
                        hold_action = dict(action)
                        hold_action["gripper.position"] = GRIPPER_OPEN
                        send_action_with_actor_gripper(actor_robot, actor, hold_action)

                    obs_act = actor_robot.get_observation()
                    # Also fetch the other robot's observation for saving combined frame
                    obs_other = robot_left.get_observation() if actor == "right" else robot_right.get_observation()

                    # Build combined frame by merging both robots' observations
                    frame_r, ee_pose_r, gripper_r = extract_observation_frame(robot_right, robot_right.get_observation())
                    frame_l, ee_pose_l, gripper_l = extract_observation_frame(robot_left, robot_left.get_observation())

                    # Compose a single frame record that contains both arms' states
                    frame_idx = recorded_count

                    if ENABLE_KEYPOINT_PCD and pending_keypoint is not None:
                        depth_raw = depth_listener.get_latest_depth()
                        if depth_raw is None:
                            depth_key = f"{CAMERA_NAME}.depth"
                            if depth_key in obs_act:
                                depth_raw = obs_act[depth_key].astype(np.float32)

                        # Try to find an RGB from either wrist or global
                        rgb_raw = None
                        for cam_name in actor_robot.config.cameras.keys():
                            rgb_key = f"{cam_name}.rgb"
                            if rgb_key in obs_act:
                                rgb_raw = obs_act[rgb_key].astype(np.uint8)
                            elif cam_name in obs_act:
                                rgb_raw = obs_act[cam_name].astype(np.uint8)
                            if rgb_raw is not None:
                                break

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

                    # Save combined raw frame data: include both arms' ee_pose and gripper
                    ee_pose_arr_r = np.array([
                        ee_pose_r["x"], ee_pose_r["y"], ee_pose_r["z"], ee_pose_r["qx"], ee_pose_r["qy"], ee_pose_r["qz"], ee_pose_r["qw"]
                    ], dtype=np.float32)
                    ee_pose_arr_l = np.array([
                        ee_pose_l["x"], ee_pose_l["y"], ee_pose_l["z"], ee_pose_l["qx"], ee_pose_l["qy"], ee_pose_l["qz"], ee_pose_l["qw"]
                    ], dtype=np.float32)

                    frame_file = frames_dir / f"frame_{frame_idx:06d}.npz"
                    np.savez_compressed(
                        frame_file,
                        observation_state_right=frame_r["observation.state"],
                        ee_pose_right=ee_pose_arr_r,
                        gripper_right=np.array([gripper_r], dtype=np.float32),
                        observation_state_left=frame_l["observation.state"],
                        ee_pose_left=ee_pose_arr_l,
                        gripper_left=np.array([gripper_l], dtype=np.float32),
                        cmd_gripper=np.array([cmd_gripper], dtype=np.float32),
                        timestamp=np.array([time.time()], dtype=np.float64),
                    )

                    # Save RGB images for both cameras (global + left_wrist + right_wrist)
                    image_records = {}
                    for cam_name in actor_robot.config.cameras.keys():
                        # try right then left then global
                        img = None
                        if f"{cam_name}.rgb" in frame_r:
                            img = frame_r.get(f"observation.images.{cam_name}_rgb") or frame_r.get(f"observation.images.{cam_name}")
                        if img is None and f"{cam_name}.rgb" in frame_l:
                            img = frame_l.get(f"observation.images.{cam_name}_rgb") or frame_l.get(f"observation.images.{cam_name}")
                        if isinstance(img, np.ndarray):
                            cam_dir = images_dir / cam_name
                            cam_dir.mkdir(parents=True, exist_ok=True)
                            img_path = cam_dir / f"frame_{frame_idx:06d}.npy"
                            np.save(img_path, img)
                            image_records[cam_name] = str(Path("images") / cam_name / f"frame_{frame_idx:06d}.npy")

                    frame_meta = {
                        "frame_index": frame_idx,
                        "state_path": str(Path("frames") / f"frame_{frame_idx:06d}.npz"),
                        "images": image_records,
                    }
                    frame_meta_file.write(json.dumps(frame_meta, ensure_ascii=False) + "\n")
                    recorded_count += 1

                    # Use actor's observation to check reach condition
                    pos_err, ori_err, grip_err = pose_error(obs_act, action, POSE_TOL_POS, POSE_TOL_ORI, GRIPPER_TOL)
                    if (pos_err <= POSE_TOL_POS) and (ori_err <= POSE_TOL_ORI):
                        reached_streak += 1
                    else:
                        reached_streak = 0

                    if reached_streak >= 3:
                        break

                    if (time.time() - start) > stage_timeout:
                        print(f"[WARN] Stage '{step_name}' timeout (pos_err={pos_err:.4f}, ori_err={ori_err:.3f}, grip_err={grip_err:.3f})")
                        break

                    time.sleep(max(0.0, (1.0 / FPS)))

            frame_meta_file.close()

            if recorded_count == 0:
                raise RuntimeError("No frames captured during grasp sequence.")

            print(f"[OK] Episode raw saved with {recorded_count} frames.")
            delete_input = input("Delete this episode data? [y/N]: ").strip().lower()
            keep_episode = delete_input not in {"y", "yes"}
            if not keep_episode:
                cleanup_episode(raw_root, episode_index)
                print(f"[Cleanup] Episode {kept_episode + 1} removed.")

            # Open grippers before returning to initial poses
            try:
                if robot_right.config.ros2_interface.gripper_enabled:
                    print("Opening right gripper before returning to initial pose...")
                    curr_r = robot_right.get_observation()
                    open_r = action_from_observation(curr_r, GRIPPER_OPEN)
                    send_action_with_actor_gripper(robot_right, "right", open_r)
                    time.sleep(RETURN_PAUSE)
            except Exception:
                pass
            try:
                if robot_left.config.ros2_interface.gripper_enabled:
                    print("Opening left gripper before returning to initial pose...")
                    curr_l = robot_left.get_observation()
                    open_l = action_from_observation(curr_l, GRIPPER_OPEN)
                    send_action_with_actor_gripper(robot_left, "left", open_l)
                    time.sleep(RETURN_PAUSE)
            except Exception:
                pass

            print("Returning both arms to initial poses...")
            try:
                send_action_with_actor_gripper(robot_right, "right", initial_action_right_open)
            except Exception:
                pass
            try:
                send_action_with_actor_gripper(robot_left, "left", initial_action_left_open)
            except Exception:
                pass
            time.sleep(RETURN_PAUSE)

            # Close grippers after return if desired
            try:
                if robot_right.config.ros2_interface.gripper_enabled:
                    print("Closing right gripper after return...")
                    send_action_with_actor_gripper(robot_right, "right", initial_action_right_closed)
                    time.sleep(RETURN_PAUSE)
            except Exception:
                pass
            try:
                if robot_left.config.ros2_interface.gripper_enabled:
                    print("Closing left gripper after return...")
                    send_action_with_actor_gripper(robot_left, "left", initial_action_left_closed)
                    time.sleep(RETURN_PAUSE)
            except Exception:
                pass

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
        try:
            if 'robot_right' in locals() and robot_right.is_connected:
                robot_right.disconnect()
                print("Right robot disconnected")
        except Exception:
            pass
        try:
            if 'robot_left' in locals() and robot_left.is_connected:
                robot_left.disconnect()
                print("Left robot disconnected")
        except Exception:
            pass

 
if __name__ == "__main__":
    main()
