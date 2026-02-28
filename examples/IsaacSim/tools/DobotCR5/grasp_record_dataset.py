#!/usr/bin/env python3
"""
ROS2 Grasp Recording Demo using LeRobot.

python examples/IsaacSim/record_datasets.py

This script records multi-step grasp episodes and writes a LeRobot dataset
directly. Target poses are obtained from ROS2 entity-state services (world
frame), then converted to base frame for control. Keypoint point clouds are
optionally saved as side files under `points/`.
"""

from __future__ import annotations

import pickle
import shutil
import signal
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Optional

import numpy as np
import rclpy
from geometry_msgs.msg import Pose
from rclpy.executors import SingleThreadedExecutor
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
from ros2_robot_interface import FSM_HOLD, FSM_OCS2
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # pyright: ignore[reportMissingImports]
from lerobot.datasets.utils import build_dataset_frame, hw_to_dataset_features  # pyright: ignore[reportMissingImports]

from lerobot_robot_ros2 import (
    ROS2Robot,
    ROS2RobotConfig,
    build_single_arm_pick_sequence,
    build_single_arm_place_sequence,
    build_single_arm_return_home_sequence,
)
from lerobot_camera_ros2 import ROS2CameraConfig

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))
from record_config import RecordConfig
from record_config import RECORD_CFG

COMMON_ISAAC_DIR = Path(__file__).resolve().parents[2] / "common"
if str(COMMON_ISAAC_DIR) not in sys.path:
    sys.path.append(str(COMMON_ISAAC_DIR))
ROBOT_CFG_DIR = Path(__file__).resolve().parents[2] / "robots" / "DobotCR5"
if str(ROBOT_CFG_DIR) not in sys.path:
    sys.path.append(str(ROBOT_CFG_DIR))

from isaac_ros2_sim_common import (  # pyright: ignore[reportMissingImports]
    SimTimeHelper,
    action_from_pose,
    get_entity_pose_world_service,
    get_object_pose_from_service,
    reset_simulation_and_randomize_object,
)
from pick_place_flow_demo_common import PickPlaceFlowTaskConfig  # pyright: ignore[reportMissingImports]
from flow_configs.pick_place_flow import FLOW_CONFIG as PICK_PLACE_FLOW_FLOW_CONFIG  # pyright: ignore[reportMissingImports]
from robot_config import ROBOT_CFG  # pyright: ignore[reportMissingImports]


# ----------------------- Recording options -----------------------------------
ROBOT_OPTIONS: dict[str, dict[str, Any]] = {
    "dobot_cr5": {
        "label": "Dobot CR5",
        "robot_cfg": ROBOT_CFG,
    }
}
TASK_OPTIONS: dict[str, dict[str, Any]] = {
    "pick_place_flow": {
        "label": "Pick Place Flow",
        "task_cfg": PickPlaceFlowTaskConfig(**PICK_PLACE_FLOW_FLOW_CONFIG["base_task_overrides"]),
    }
}
RECORD_PROFILES: dict[str, RecordConfig] = {
    "default": RECORD_CFG,
    "fast": replace(RECORD_CFG, image_writer_threads=12, video_encoding_batch_size=10),
}
# ----------------------------------------------------------------------------


def _select_option(
    *,
    title: str,
    options: dict[str, dict[str, Any]] | dict[str, RecordConfig],
    default_key: str,
) -> str:
    keys = list(options.keys())
    print(f"\n{title}")
    for idx, key in enumerate(keys, start=1):
        value = options[key]
        label = value["label"] if isinstance(value, dict) and "label" in value else key
        suffix = " (default)" if key == default_key else ""
        print(f"  {idx}. {label} [{key}]{suffix}")
    raw = input("Select option (press Enter for default): ").strip()
    if raw == "":
        return default_key
    if raw.isdigit():
        idx = int(raw) - 1
        if 0 <= idx < len(keys):
            return keys[idx]
    if raw in options:
        return raw
    print(f"[info] Invalid option '{raw}', using default '{default_key}'.")
    return default_key


def _collect_runtime_options(default_enable_keypoint_pcd: bool) -> tuple[int, bool, bool]:
    loops = 1
    try:
        loops = max(1, int(input("How many episodes to record? ")))
    except Exception:
        print("[info] invalid input, defaulting to 1 episode")

    default_pcd = "Y" if default_enable_keypoint_pcd else "N"
    use_pcd_input = input(f"Capture depth+pointcloud at keypoints? [y/{default_pcd}]: ").strip().lower()
    if use_pcd_input == "":
        enable_keypoint_pcd = default_enable_keypoint_pcd
    else:
        enable_keypoint_pcd = use_pcd_input in {"y", "yes"}

    manual_check_input = input("Manually review each episode after return-to-home? [y/N]: ").strip().lower()
    enable_manual_episode_check = manual_check_input in {"y", "yes"}
    return loops, enable_keypoint_pcd, enable_manual_episode_check


def _resolve_gripper_values(gripper_control_mode: str) -> tuple[float, float]:
    if gripper_control_mode == "target_command":
        return 1.0, 0.0
    raise ValueError("grasp_record_dataset only supports gripper_control_mode='target_command'")


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


def build_robot_config(*, robot_cfg: Any, record_cfg: RecordConfig) -> ROS2RobotConfig:
    if not getattr(robot_cfg, "cameras", None):
        raise ValueError("robot_cfg.cameras must contain at least one camera definition")
    camera_cfg: dict[str, ROS2CameraConfig] = {
        name: replace(cam, fps=record_cfg.fps)
        for name, cam in robot_cfg.cameras.items()
    }
    ros2_interface = robot_cfg.ros2_interface
    return ROS2RobotConfig(
        id=robot_cfg.robot_id,
        cameras=camera_cfg,
        ros2_interface=ros2_interface,
        gripper_control_mode=robot_cfg.gripper_control_mode,
    )


def pose_from_observation(obs: dict[str, float]) -> Pose:
    pose = Pose()
    pose.position.x = obs.get("left_ee.pos.x", 0.0)
    pose.position.y = obs.get("left_ee.pos.y", 0.0)
    pose.position.z = obs.get("left_ee.pos.z", 0.0)
    pose.orientation.x = obs.get("left_ee.quat.x", 0.0)
    pose.orientation.y = obs.get("left_ee.quat.y", 0.0)
    pose.orientation.z = obs.get("left_ee.quat.z", 0.0)
    pose.orientation.w = obs.get("left_ee.quat.w", 1.0)
    return pose


def pose_error(
    obs: dict[str, float],
    target: dict[str, float],
    gripper_obs_key: str = "gripper_joint.pos",
) -> tuple[float, float, float]:
    """Compute position/angle/gripper errors between observation and target action dict."""
    dx = obs.get("left_ee.pos.x", 0.0) - target.get("left_ee.pos.x", 0.0)
    dy = obs.get("left_ee.pos.y", 0.0) - target.get("left_ee.pos.y", 0.0)
    dz = obs.get("left_ee.pos.z", 0.0) - target.get("left_ee.pos.z", 0.0)
    pos_err = float(np.sqrt(dx * dx + dy * dy + dz * dz))

    # Orientation error via quaternion dot (clipped to avoid NaNs)
    qw1, qx1, qy1, qz1 = (
        target.get("left_ee.quat.w", 1.0),
        target.get("left_ee.quat.x", 0.0),
        target.get("left_ee.quat.y", 0.0),
        target.get("left_ee.quat.z", 0.0),
    )
    qw2, qx2, qy2, qz2 = (
        obs.get("left_ee.quat.w", 1.0),
        obs.get("left_ee.quat.x", 0.0),
        obs.get("left_ee.quat.y", 0.0),
        obs.get("left_ee.quat.z", 0.0),
    )
    dot = qw1 * qw2 + qx1 * qx2 + qy1 * qy2 + qz1 * qz2
    dot = float(np.clip(abs(dot), -1.0, 1.0))
    ori_err = float(2.0 * np.arccos(dot))

    obs_gripper = obs.get(gripper_obs_key, 0.0)
    target_gripper = target.get("left_gripper.pos", 0.0)
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


def build_dataset_features(
    robot: ROS2Robot,
    *,
    include_depth_feature: bool,
) -> tuple[dict[str, dict], list[str], bool]:
    # Keep points as side-files; depth can be toggled via config.
    obs_hw_features = dict(robot.observation_features)
    if not include_depth_feature:
        obs_hw_features = {
            k: v
            for k, v in obs_hw_features.items()
            if not (isinstance(v, tuple) and str(k).endswith(".depth"))
        }
    features = {
        **hw_to_dataset_features(robot.action_features, "action", use_video=True),
        **hw_to_dataset_features(obs_hw_features, "observation", use_video=True),
    }
    action_names = list(features["action"]["names"])
    include_gripper = bool(robot.config.ros2_interface.gripper_enabled)
    return features, action_names, include_gripper


def cleanup_episode_points(dataset_root: Path, episode_index: int) -> None:
    """Remove side pointcloud files for a rejected episode."""
    points_root = dataset_root / "points"
    if points_root.exists():
        target_suffix = f"episode_{episode_index:06d}"
        for path in points_root.rglob(target_suffix):
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)


def extract_observation_frame(
    dataset_features: dict[str, dict],
    obs: dict[str, float],
) -> tuple[dict[str, object], dict[str, float]]:
    """Build observation frame from official feature mapping + keep ee pose for action labels."""
    frame = build_dataset_frame(dataset_features, obs, prefix="observation")
    ee_pose = {
        "x": obs.get("left_ee.pos.x", 0.0),
        "y": obs.get("left_ee.pos.y", 0.0),
        "z": obs.get("left_ee.pos.z", 0.0),
        "qx": obs.get("left_ee.quat.x", 0.0),
        "qy": obs.get("left_ee.quat.y", 0.0),
        "qz": obs.get("left_ee.quat.z", 0.0),
        "qw": obs.get("left_ee.quat.w", 1.0),
    }
    return frame, ee_pose


def action_from_next(
    next_ee_pose: np.ndarray,
    next_cmd_gripper: float,
    action_keys: list[str],
    include_gripper: bool,
) -> np.ndarray:
    action_dict = {
        "left_ee.pos.x": float(next_ee_pose[0]),
        "left_ee.pos.y": float(next_ee_pose[1]),
        "left_ee.pos.z": float(next_ee_pose[2]),
        "left_ee.quat.x": float(next_ee_pose[3]),
        "left_ee.quat.y": float(next_ee_pose[4]),
        "left_ee.quat.z": float(next_ee_pose[5]),
        "left_ee.quat.w": float(next_ee_pose[6]),
    }
    if include_gripper:
        action_dict["left_gripper.pos"] = float(np.clip(next_cmd_gripper, 0.0, 1.0))
    values = [action_dict.get(k, 0.0) for k in action_keys]
    if include_gripper:
        values.append(action_dict.get("left_gripper.pos", 0.0))
    return np.asarray(values, dtype=np.float32)


def build_pick_place_flow_sequence(
    *,
    target_pose: Pose,
    task_cfg: Any,
    initial_action_open: dict[str, float],
    initial_action_closed: dict[str, float],
    gripper_enabled: bool,
    gripper_open: float,
    gripper_closed: float,
) -> list[tuple[str, dict[str, float]]]:
    """Build one episode flow sequence from target pose."""
    if task_cfg.place_position is None or task_cfg.place_orientation is None:
        raise ValueError("CR5 pick-place flow requires place_position/place_orientation in task config")
    sequence: list[tuple[str, dict[str, float]]] = []
    sequence.extend(
        build_single_arm_pick_sequence(
            target_pose=target_pose,
            approach_clearance=task_cfg.approach_clearance,
            grasp_clearance=task_cfg.grasp_clearance,
            grasp_orientation=task_cfg.grasp_orientation,
            grasp_direction=task_cfg.grasp_direction,
            grasp_direction_vector=task_cfg.grasp_direction_vector,
            retreat_direction_extra=task_cfg.retreat_direction_extra,
            retreat_raise_z=task_cfg.retreat_raise_z,
            gripper_open=gripper_open,
            gripper_closed=gripper_closed,
            ee_prefix="left_ee",
            gripper_key="left_gripper.pos",
            stage_prefix="PickPlaceFlow",
        )
    )
    sequence.extend(
        build_single_arm_place_sequence(
            place_position=task_cfg.place_position,
            place_orientation=task_cfg.place_orientation,
            post_release_retract_offset=task_cfg.post_release_retract_offset,
            gripper_open=gripper_open,
            gripper_closed=gripper_closed,
            ee_prefix="left_ee",
            gripper_key="left_gripper.pos",
            stage_prefix="PickPlaceFlow",
            start_index=5,
        )
    )
    sequence.extend(
        build_single_arm_return_home_sequence(
            home_action=initial_action_closed if gripper_enabled else initial_action_open,
            stage_name="PickPlaceFlow-8-ReturnHomeHold",
        )
    )
    return sequence


def record_episode_sequence(
    *,
    robot: ROS2Robot,
    sim_time: SimTimeHelper,
    sequence: list[tuple[str, dict[str, float]]],
    task_cfg: PickPlaceFlowTaskConfig,
    dataset_features: dict[str, dict],
    dataset_root: Path,
    episode_index: int,
    intrinsics: tuple[float, float, float, float] | None,
    depth_listener: RawDepthListener,
    record_cfg: RecordConfig,
    enable_keypoint_pcd: bool,
    camera_name: str,
) -> list[dict[str, object]]:
    """Execute one sequence while collecting per-frame records for LeRobot writing."""
    records: list[dict[str, object]] = []
    command_counter = 0
    pcd_save_idx = 0
    pending_keypoint: Optional[dict] = None

    for step_name, action in sequence:
        print(f"[Step] {step_name}")
        robot.send_action(action)
        cmd_gripper = float(action.get("left_gripper.pos", 0.0))
        pending_keypoint = {
            "command_index": command_counter,
        }
        command_counter += 1
        start_sim = sim_time.now_seconds()
        wall_start = time.monotonic()
        reached_streak = 0
        while True:
            obs = robot.get_observation()
            observation_frame, ee_pose = extract_observation_frame(dataset_features, obs)
            frame_idx = len(records)
            if enable_keypoint_pcd and pending_keypoint is not None:
                # Get raw depth directly from ROS2 topic (32FC1, in meters)
                depth_raw = depth_listener.get_latest_depth()
                if depth_raw is None:
                    depth_key = f"{camera_name}.depth"
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
                    print(
                        f"[Keypoint] Captured raw depth: shape={depth_raw.shape}, "
                        f"range=[{depth_raw[depth_raw > 0].min():.3f}, {depth_raw.max():.3f}] m"
                    )
                else:
                    print("[WARNING] Failed to capture raw depth for keypoint")

                # Save point cloud only at keypoints
                save_pointcloud_frame(
                    dataset_root,
                    episode_index,
                    pcd_save_idx,
                    depth_raw,
                    intrinsics,
                    rgb_image=rgb_raw,
                    pointcloud_key=f"observation.points.{camera_name}_pcd",
                )
                pcd_save_idx += 1
                pending_keypoint = None
            elif pending_keypoint is not None and not enable_keypoint_pcd:
                # 不采集点云时也要清掉 pending_keypoint，避免后续报错
                pending_keypoint = None

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
            records.append(
                {
                    "observation_frame": observation_frame,
                    "ee_pose": ee_pose_arr,
                    "cmd_gripper": cmd_gripper,
                }
            )
            # 判定是否到达目标：连续 3 帧满足容差即进入下一阶段（不考虑夹爪误差）
            pos_err, ori_err, grip_err = pose_error(
                obs,
                action,
                gripper_obs_key=f"{robot.config.ros2_interface.gripper_joint_name}.pos",
            )
            ori_ok = (ori_err <= task_cfg.pose_tol_ori) if task_cfg.require_orientation_reach else True
            if (pos_err <= task_cfg.pose_tol_pos) and ori_ok:
                reached_streak += 1
            else:
                reached_streak = 0

            if reached_streak >= 3:
                break

            elapsed_sim = sim_time.now_seconds() - start_sim
            if elapsed_sim > task_cfg.max_stage_duration:
                print(
                    f"[WARN] Stage '{step_name}' timeout "
                    f"(pos_err={pos_err:.4f}, ori_err={ori_err:.3f}, grip_err={grip_err:.3f})"
                )
                break

            # Wall-time guard avoids hard hang if /clock stops advancing unexpectedly.
            if (time.monotonic() - wall_start) > max(task_cfg.max_stage_duration + 5.0, 10.0):
                print(
                    f"[WARN] Stage '{step_name}' wall-time guard triggered "
                    f"(sim_elapsed={elapsed_sim:.3f}s)."
                )
                break

            sim_time.sleep(max(0.0, (1.0 / record_cfg.fps)))

        if pending_keypoint is not None:
            raise RuntimeError(
                f"No frame captured after sending action index {pending_keypoint['command_index']}"
            )
    return records


def append_episode_to_dataset(
    *,
    dataset: LeRobotDataset,
    records: list[dict[str, object]],
    action_names: list[str],
    include_gripper: bool,
    task_name: str,
) -> None:
    action_keys = [name for name in action_names if name != "left_gripper.pos"]
    for idx, rec in enumerate(records):
        frame: dict[str, object] = {"task": task_name}
        frame.update(rec["observation_frame"])  # type: ignore[arg-type]

        next_idx = idx + 1 if idx + 1 < len(records) else idx
        next_rec = records[next_idx]
        frame["action"] = action_from_next(
            next_ee_pose=next_rec["ee_pose"],  # type: ignore[arg-type]
            next_cmd_gripper=float(next_rec["cmd_gripper"]),  # type: ignore[arg-type]
            action_keys=action_keys,
            include_gripper=include_gripper,
        )
        dataset.add_frame(frame)

    dataset.save_episode()




def run_pick_place_recording(
    *,
    robot_cfg: Any,
    task_cfg: Any,
    record_cfg: RecordConfig,
    loops: int,
    enable_keypoint_pcd: bool,
    enable_manual_episode_check: bool,
) -> None:
    gripper_open, gripper_closed = _resolve_gripper_values(robot_cfg.gripper_control_mode)
    robot_config = build_robot_config(robot_cfg=robot_cfg, record_cfg=record_cfg)
    robot = ROS2Robot(robot_config)
    depth_cam_name = robot_cfg.depth_camera_name
    depth_cam = robot_cfg.cameras.get(depth_cam_name)
    if depth_cam is None or depth_cam.depth_topic_name is None:
        raise ValueError("depth_camera_name must reference a configured camera with depth_topic_name")
    depth_listener = RawDepthListener(depth_cam.depth_topic_name)
    cam_info_listener = DepthCameraInfoListener(robot_cfg.depth_info_topic)
    sim_time = SimTimeHelper()

    def shutdown_handler(sig, frame) -> None:  # type: ignore[override]
        print("\nInterrupt received. Cleaning up...")
        try:
            depth_listener.shutdown()
            cam_info_listener.shutdown()
            if robot.is_connected:
                robot.disconnect()
        finally:
            sim_time.shutdown()
            sys.exit(0)

    signal.signal(signal.SIGINT, shutdown_handler)

    try:
        print("Connecting robot...")
        robot.connect()
        print("[OK] Robot connected")

        print("Switching FSM: HOLD -> OCS2 ...")
        robot.ros2_interface.send_fsm_command(FSM_HOLD)
        time.sleep(robot_cfg.fsm_switch_delay)
        robot.ros2_interface.send_fsm_command(FSM_OCS2)
        print("[OK] FSM switched to OCS2")

        intrinsics = cam_info_listener.wait_for_intrinsics(timeout=record_cfg.camera_info_timeout)
        if intrinsics is None:
            print("[WARN] No camera_info received, will fall back to default intrinsics")
        else:
            fx, fy, cx, cy = intrinsics
            print(f"[OK] Camera intrinsics received: fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")

        initial_obs = robot.get_observation()
        features, action_names, include_gripper = build_dataset_features(
            robot,
            include_depth_feature=record_cfg.include_depth_feature,
        )
        out_root = (Path.cwd() / "lerobot_dataset" / f"grasp_record_{int(time.time())}").resolve()
        out_root.parent.mkdir(parents=True, exist_ok=True)
        dataset = LeRobotDataset.create(
            repo_id=str(out_root),
            fps=record_cfg.fps,
            features=features,
            robot_type=robot.name,
            use_videos=True,
            image_writer_processes=record_cfg.image_writer_processes,
            image_writer_threads=record_cfg.image_writer_threads,
            batch_encoding_size=record_cfg.video_encoding_batch_size,
        )
        print(f"[OK] LeRobot dataset will be saved to {out_root}")

        initial_pose = pose_from_observation(initial_obs)
        initial_action_open = action_from_pose(initial_pose, gripper_open)
        initial_action_closed = action_from_pose(initial_pose, gripper_closed)

        kept_episode = 0
        attempt = 0
        while kept_episode < loops:
            episode_index = kept_episode
            attempt += 1
            print(f"=== Recording episode {kept_episode + 1}/{loops} (attempt {attempt}) ===")
            print("Resetting simulation state...")
            reset_simulation_and_randomize_object(
                task_cfg.source_object_entity_path,
                xyz_offset=task_cfg.object_xyz_random_offset,
                post_reset_wait=robot_cfg.post_reset_wait,
                sleep_fn=sim_time.sleep,
            )
            base_world_pos, base_world_quat = get_entity_pose_world_service(robot_cfg.base_link_entity_path)
            print("Querying object pose from simulation service...")
            target_pose = get_object_pose_from_service(
                base_world_pos,
                base_world_quat,
                task_cfg.source_object_entity_path,
                include_orientation=task_cfg.use_object_orientation,
            )

            current_obs = robot.get_observation()
            current_orientation = (
                current_obs["left_ee.quat.x"],
                current_obs["left_ee.quat.y"],
                current_obs["left_ee.quat.z"],
                current_obs["left_ee.quat.w"],
            )
            orientation_vec = np.array(
                [
                    target_pose.orientation.x,
                    target_pose.orientation.y,
                    target_pose.orientation.z,
                    target_pose.orientation.w,
                ]
            )
            if not task_cfg.use_object_orientation or np.linalg.norm(orientation_vec) < 1e-3:
                (
                    target_pose.orientation.x,
                    target_pose.orientation.y,
                    target_pose.orientation.z,
                    target_pose.orientation.w,
                ) = current_orientation

            sequence = build_pick_place_flow_sequence(
                target_pose=target_pose,
                task_cfg=task_cfg,
                initial_action_open=initial_action_open,
                initial_action_closed=initial_action_closed,
                gripper_enabled=robot.config.ros2_interface.gripper_enabled,
                gripper_open=gripper_open,
                gripper_closed=gripper_closed,
            )
            episode_records = record_episode_sequence(
                robot=robot,
                sim_time=sim_time,
                sequence=sequence,
                task_cfg=task_cfg,
                dataset_features=features,
                dataset_root=out_root,
                episode_index=episode_index,
                intrinsics=intrinsics,
                depth_listener=depth_listener,
                record_cfg=record_cfg,
                enable_keypoint_pcd=enable_keypoint_pcd,
                camera_name=depth_cam_name,
            )
            if len(episode_records) == 0:
                raise RuntimeError("No frames captured during grasp sequence.")

            print(f"[OK] Episode captured with {len(episode_records)} frames.")
            keep_episode = True
            if enable_manual_episode_check:
                delete_input = input("Delete this episode data? [y/N]: ").strip().lower()
                keep_episode = delete_input not in {"y", "yes"}
                if not keep_episode:
                    cleanup_episode_points(out_root, episode_index)
                    print(f"[Cleanup] Episode {kept_episode + 1} removed.")

            if keep_episode:
                append_episode_to_dataset(
                    dataset=dataset,
                    records=episode_records,
                    action_names=action_names,
                    include_gripper=include_gripper,
                    task_name=record_cfg.task_name,
                )
                kept_episode += 1
                print(f"[OK] Episode {kept_episode} saved.")

        print(f"[OK] LeRobot dataset available at: {out_root}")
    except Exception as exc:
        print(f"[ERROR] Recording failed: {exc}")
    finally:
        depth_listener.shutdown()
        cam_info_listener.shutdown()
        sim_time.shutdown()
        if robot.is_connected:
            robot.disconnect()
            print("Robot disconnected")


def main() -> None:
    print("ROS2 Generic Record Entry (LeRobot)")
    print("=" * 70)
    robot_key = _select_option(title="Select robot", options=ROBOT_OPTIONS, default_key="dobot_cr5")
    task_key = _select_option(title="Select task", options=TASK_OPTIONS, default_key="pick_place_flow")
    profile_key = _select_option(title="Select record profile", options=RECORD_PROFILES, default_key="default")
    selected_robot_cfg = ROBOT_OPTIONS[robot_key]["robot_cfg"]
    selected_task_cfg = TASK_OPTIONS[task_key]["task_cfg"]
    selected_record_cfg = RECORD_PROFILES[profile_key]
    loops, enable_keypoint_pcd, enable_manual_episode_check = _collect_runtime_options(
        selected_record_cfg.enable_keypoint_pcd
    )
    print(
        f"[Selection] robot={robot_key}, task={task_key}, profile={profile_key}, "
        f"episodes={loops}, pointcloud={enable_keypoint_pcd}, manual_review={enable_manual_episode_check}"
    )
    run_pick_place_recording(
        robot_cfg=selected_robot_cfg,
        task_cfg=selected_task_cfg,
        record_cfg=selected_record_cfg,
        loops=loops,
        enable_keypoint_pcd=enable_keypoint_pcd,
        enable_manual_episode_check=enable_manual_episode_check,
    )

 
if __name__ == "__main__":
    main()
