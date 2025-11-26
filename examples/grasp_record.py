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

import pickle
import signal
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from types import MethodType
from typing import Optional

import numpy as np
import rclpy
from geometry_msgs.msg import Pose
from rclpy.executors import SingleThreadedExecutor
from sensor_msgs.msg import Image, CameraInfo
from tf2_msgs.msg import TFMessage
from cv_bridge import CvBridge

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

CAMERA_NAME = "zed"  # BridgeVLA expects zed_* naming
CAMERA_TOPIC = "/global_camera/rgb"
DEPTH_TOPIC = "/global_camera/depth"
DEPTH_INFO_TOPIC = "/global_camera/camera_info"
OBJECT_TF_TOPIC = "/isaac/tf"
TARGET_FRAME_ID = "apple"
USE_OBJECT_ORIENTATION = False
APPROACH_CLEARANCE = 0.12
GRASP_CLEARANCE = -0.03
GRIPPER_OPEN = 1.0
GRIPPER_CLOSED = 0.0
POSE_TIMEOUT = 10.0
TASK_NAME = "grasp_apples"
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


def action_dict_to_array(action: dict[str, float], include_gripper: bool) -> np.ndarray:
    values = [action.get(key, 0.0) for key in ACTION_KEYS]
    if include_gripper:
        values.append(action.get("gripper.position", 0.0))
    return np.array(values, dtype=np.float32)


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


def build_dataset(robot: ROS2Robot) -> tuple[LeRobotDataset, Path]:
    features = {}
    features.update(hw_to_dataset_features(robot.action_features, "action"))
    features.update(hw_to_dataset_features(robot.observation_features, "observation"))
    # Normalize camera feature keys to BridgeVLA style (observation.images.<cam>_rgb)
    # Drop depth images from the dataset (we keep raw depth only for point clouds)
    remapped = {}
    for key, info in features.items():
        new_key = key.replace(".rgb", "_rgb").replace(".depth", "_depth")
        if new_key.endswith("_depth"):
            continue
        remapped[new_key] = info
    features = remapped
    dataset_root = Path.cwd() / "dataset"
    dataset_root.mkdir(parents=True, exist_ok=True)
    dataset_dir = dataset_root / f"grasp_dataset_{int(time.time())}"
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

        dataset, dataset_path = build_dataset(robot)
        include_gripper = robot.config.ros2_interface.gripper_enabled
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

        def _clear_episode_buffer_keep_images(self, delete_images: bool = True) -> None:
            return LeRobotDataset.clear_episode_buffer(self, delete_images=False)

        dataset.clear_episode_buffer = MethodType(_clear_episode_buffer_keep_images, dataset)
        print(f"[OK] Dataset initialized at {dataset_path}")

        initial_obs = robot.get_observation()
        initial_pose = pose_from_observation(initial_obs)
        initial_action_open = action_from_pose(initial_pose, GRIPPER_OPEN)
        initial_action_closed = action_from_pose(initial_pose, GRIPPER_CLOSED)

        for episode in range(loops):
            print(f"=== Recording episode {episode + 1}/{loops} ===")
            command_counter = 0
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

            pending_keypoint: Optional[dict] = None

            for step_name, action in sequence:
                print(f"[Step] {step_name}")
                robot.send_action(action)
                pending_keypoint = {
                    "command_index": command_counter,
                    "action_vec": action_dict_to_array(action, include_gripper),
                }
                command_counter += 1
                start = time.time()
                while (time.time() - start) < STAGE_DURATION:
                    obs = robot.get_observation()
                    frame, ee_pose, gripper = extract_observation_frame(robot, obs)
                    frame["_ee_pose"] = ee_pose
                    frame["_gripper"] = gripper
                    frame_idx = len(recorded_frames)
                    if pending_keypoint is not None:
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
                            dataset_path,
                            episode,
                            frame_idx,
                            depth_raw,
                            intrinsics,
                            rgb_image=rgb_raw,
                            pointcloud_key=f"observation.points.{CAMERA_NAME}_pcd",
                        )
                        pending_keypoint = None

                    recorded_frames.append(frame)
                    time.sleep(max(0.0, (1.0 / FPS)))

                if pending_keypoint is not None:
                    raise RuntimeError(
                        f"No frame captured after sending action index {pending_keypoint['command_index']}"
                    )

            if not recorded_frames:
                raise RuntimeError("No frames captured during grasp sequence.")

            for idx, frame in enumerate(recorded_frames):
                if idx < len(recorded_frames) - 1:
                    ref = recorded_frames[idx + 1]
                else:
                    ref = recorded_frames[idx]

                pose = ref["_ee_pose"]
                action_dict = {
                    "end_effector.position.x": pose["x"],
                    "end_effector.position.y": pose["y"],
                    "end_effector.position.z": pose["z"],
                    "end_effector.orientation.x": pose["qx"],
                    "end_effector.orientation.y": pose["qy"],
                    "end_effector.orientation.z": pose["qz"],
                    "end_effector.orientation.w": pose["qw"],
                }
                if include_gripper:
                    action_dict["gripper.position"] = ref["_gripper"]

                frame["action"] = action_dict_to_array(action_dict, include_gripper)
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

            if pause_between_episodes and episode < loops - 1:
                input("Press Enter to start the next episode...")

        print(f"[OK] Dataset available at: {dataset_path}")

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
