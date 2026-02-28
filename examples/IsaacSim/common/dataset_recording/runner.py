#!/usr/bin/env python3
"""Generic dataset record runner for IsaacSim tasks."""

from __future__ import annotations

import pickle
import signal
import sys
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Optional

import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Pose
from rclpy.executors import SingleThreadedExecutor
from sensor_msgs.msg import CameraInfo, Image

from ros2_robot_interface import FSM_HOLD, FSM_OCS2  # pyright: ignore[reportMissingImports]

from lerobot.datasets.lerobot_dataset import LeRobotDataset  # pyright: ignore[reportMissingImports]
from lerobot_robot_ros2 import (  # pyright: ignore[reportMissingImports]
    ROS2Robot,
    ROS2RobotConfig,
    execute_stage_sequence,
)
from dataset_recording.recorder import DatasetRecorder  # pyright: ignore[reportMissingImports]
from motion_generation.handover import build_handover_record_sequence  # pyright: ignore[reportMissingImports]
from motion_generation.pick_place import build_single_arm_pick_place_sequence  # pyright: ignore[reportMissingImports]
from isaac_ros2_sim_common import (  # pyright: ignore[reportMissingImports]
    SimTimeHelper,
    get_entity_pose_world_service,
    get_object_pose_from_service,
    reset_simulation_and_randomize_object,
)

SUPPORTED_RECORD_KINDS: tuple[str, ...] = ("pick_place", "handover")


@dataclass(frozen=True)
class RecordConfig:
    fps: int = 30
    camera_info_timeout: float = 10.0
    task_name: str = "pick_place"
    enable_keypoint_pcd: bool = False
    include_depth_feature: bool = False
    # LeRobot dataset write/encode tuning (does not change frame alignment semantics)
    image_writer_processes: int = 0
    image_writer_threads: int = 8
    video_encoding_batch_size: int = 5


DEFAULT_RECORD_CFG = RecordConfig()


class RawDepthListener:
    def __init__(self, depth_topic: str) -> None:
        self._latest_depth: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._bridge = CvBridge()
        if not rclpy.ok():
            rclpy.init()
        self._node = rclpy.create_node("record_depth_listener")
        self._sub = self._node.create_subscription(Image, depth_topic, self._depth_callback, 10)
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)
        self._thread = threading.Thread(target=self._executor.spin, daemon=True)
        self._thread.start()

    def _depth_callback(self, msg: Image) -> None:
        try:
            depth_image = self._bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            with self._lock:
                self._latest_depth = depth_image.astype(np.float32)
        except Exception:
            return

    def get_latest_depth(self) -> Optional[np.ndarray]:
        with self._lock:
            return self._latest_depth.copy() if self._latest_depth is not None else None

    def shutdown(self) -> None:
        self._executor.shutdown()
        self._node.destroy_node()


class DepthCameraInfoListener:
    def __init__(self, info_topic: str) -> None:
        self._intrinsics: Optional[tuple[float, float, float, float]] = None
        self._lock = threading.Lock()
        self._ready = threading.Event()
        if not rclpy.ok():
            rclpy.init()
        self._node = rclpy.create_node("record_camera_info_listener")
        self._sub = self._node.create_subscription(CameraInfo, info_topic, self._info_callback, 1)
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)
        self._thread = threading.Thread(target=self._executor.spin, daemon=True)
        self._thread.start()

    def _info_callback(self, msg: CameraInfo) -> None:
        with self._lock:
            self._intrinsics = (msg.k[0], msg.k[4], msg.k[2], msg.k[5])
            self._ready.set()

    def wait_for_intrinsics(self, timeout: float) -> Optional[tuple[float, float, float, float]]:
        if not self._ready.wait(timeout):
            return None
        with self._lock:
            return self._intrinsics

    def shutdown(self) -> None:
        self._executor.shutdown()
        self._node.destroy_node()


def _resolve_gripper_values(gripper_control_mode: str) -> tuple[float, float]:
    if gripper_control_mode == "target_command":
        return 1.0, 0.0
    raise ValueError("record runner only supports gripper_control_mode='target_command'")


def _build_robot_config(*, robot_cfg: Any, record_cfg: Any) -> ROS2RobotConfig:
    if not getattr(robot_cfg, "cameras", None):
        raise ValueError("robot_cfg.cameras must contain at least one camera definition")
    camera_cfg = {name: replace(cam, fps=record_cfg.fps) for name, cam in robot_cfg.cameras.items()}
    robot_id = getattr(robot_cfg, "robot_id", None)
    if not robot_id:
        robot_id = getattr(robot_cfg, "ROBOT_KEY", None) or robot_cfg.__class__.__name__.lower()
    return ROS2RobotConfig(
        id=str(robot_id),
        cameras=camera_cfg,
        ros2_interface=robot_cfg.ros2_interface,
        gripper_control_mode=robot_cfg.gripper_control_mode,
    )


def _pose_from_observation(obs: dict[str, float], *, ee_prefix: str = "left_ee") -> Pose:
    pose = Pose()
    pose.position.x = obs.get(f"{ee_prefix}.pos.x", 0.0)
    pose.position.y = obs.get(f"{ee_prefix}.pos.y", 0.0)
    pose.position.z = obs.get(f"{ee_prefix}.pos.z", 0.0)
    pose.orientation.x = obs.get(f"{ee_prefix}.quat.x", 0.0)
    pose.orientation.y = obs.get(f"{ee_prefix}.quat.y", 0.0)
    pose.orientation.z = obs.get(f"{ee_prefix}.quat.z", 0.0)
    pose.orientation.w = obs.get(f"{ee_prefix}.quat.w", 1.0)
    return pose


def _save_pointcloud_frame(
    dataset_dir: Path,
    episode_index: int,
    frame_index: int,
    depth_image: Optional[np.ndarray],
    intrinsics: Optional[tuple[float, float, float, float]],
    rgb_image: Optional[np.ndarray] = None,
    pointcloud_key: str = "observation.points.zed_pcd",
) -> None:
    pcd_dir = dataset_dir / "points" / pointcloud_key / "chunk-000" / f"episode_{episode_index:06d}"
    pcd_dir.mkdir(parents=True, exist_ok=True)
    if depth_image is None:
        depth_image = np.zeros((1, 1), dtype=np.float32)
    if intrinsics is None:
        h, w = depth_image.shape[:2]
        intrinsics = (max(w, 1), max(h, 1), w / 2.0, h / 2.0)
    fx, fy, cx, cy = intrinsics
    if depth_image.ndim == 3:
        depth_image = depth_image[..., 0]
    h, w = depth_image.shape
    uu, vv = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    z = depth_image.astype(np.float32)
    x = (uu - cx) * z / (fx if fx != 0 else 1.0)
    y = (vv - cy) * z / (fy if fy != 0 else 1.0)
    pcd_xyz = np.stack((x, y, z), axis=-1).astype(np.float32)
    pcd = pcd_xyz
    if rgb_image is not None and rgb_image.shape[:2] == pcd_xyz.shape[:2]:
        rgb = rgb_image.astype(np.float32)
        if rgb.max() > 1.0:
            rgb = rgb / 255.0
        pcd = np.concatenate([pcd_xyz, np.clip(rgb, 0.0, 1.0)], axis=-1).astype(np.float32)
    with open(pcd_dir / f"frame_{frame_index:06d}.pkl", "wb") as f:
        pickle.dump(pcd, f)


def run_recording(
    *,
    robot_cfg: Any,
    task_cfg: Any,
    record_cfg: Any,
    loops: int,
    enable_keypoint_pcd: bool,
    enable_manual_episode_check: bool,
    task_kind: str = "pick_place",
) -> None:
    if task_kind not in SUPPORTED_RECORD_KINDS:
        raise NotImplementedError(f"Record runner does not support task kind: {task_kind}")

    gripper_open, gripper_closed = _resolve_gripper_values(robot_cfg.gripper_control_mode)
    robot = ROS2Robot(_build_robot_config(robot_cfg=robot_cfg, record_cfg=record_cfg))
    depth_required = bool(enable_keypoint_pcd)
    default_cam_name = next(iter(robot_cfg.cameras.keys()), "")
    depth_cam_name = getattr(robot_cfg, "depth_camera_name", default_cam_name)
    depth_cam = robot_cfg.cameras.get(depth_cam_name) if getattr(robot_cfg, "cameras", None) else None
    depth_topic = getattr(depth_cam, "depth_topic_name", None) if depth_cam is not None else None
    depth_info_topic = getattr(robot_cfg, "depth_info_topic", None)
    depth_listener: RawDepthListener | None = None
    cam_info_listener: DepthCameraInfoListener | None = None
    if depth_required:
        if depth_topic is None:
            raise ValueError("Depth recording requested, but no camera depth topic is configured")
        if depth_info_topic is None:
            raise ValueError("Depth recording requested, but depth_info_topic is not configured")
        depth_listener = RawDepthListener(depth_topic)
        cam_info_listener = DepthCameraInfoListener(depth_info_topic)
    sim_time = SimTimeHelper()

    def shutdown_handler(sig, frame) -> None:  # type: ignore[override]
        try:
            if depth_listener is not None:
                depth_listener.shutdown()
            if cam_info_listener is not None:
                cam_info_listener.shutdown()
            if robot.is_connected:
                robot.disconnect()
        finally:
            sim_time.shutdown()
            sys.exit(0)

    signal.signal(signal.SIGINT, shutdown_handler)

    try:
        robot.connect()
        robot.ros2_interface.send_fsm_command(FSM_HOLD)
        time.sleep(robot_cfg.fsm_switch_delay)
        robot.ros2_interface.send_fsm_command(FSM_OCS2)

        intrinsics = (
            cam_info_listener.wait_for_intrinsics(timeout=record_cfg.camera_info_timeout)
            if cam_info_listener is not None
            else None
        )
        features, action_names, include_gripper = DatasetRecorder.build_dataset_features(
            robot, include_depth_feature=record_cfg.include_depth_feature
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

        initial_obs = robot.get_observation()
        source_is_right = False
        source_home_pose: Pose | None = None
        receiver_home_pose: Pose | None = None
        if task_kind == "handover":
            initial_arm = task_cfg.initial_grasp_arm.lower()
            if initial_arm not in {"left", "right"}:
                raise ValueError("initial_grasp_arm must be 'left' or 'right'")
            source_is_right = initial_arm == "right"
            source_ee_prefix = "right_ee" if source_is_right else "left_ee"
            receiver_ee_prefix = "left_ee" if source_is_right else "right_ee"
            source_home_pose = _pose_from_observation(initial_obs, ee_prefix=source_ee_prefix)
            receiver_home_pose = _pose_from_observation(initial_obs, ee_prefix=receiver_ee_prefix)

        kept_episode = 0
        while kept_episode < loops:
            episode_index = kept_episode
            reset_simulation_and_randomize_object(
                task_cfg.source_object_entity_path,
                xyz_offset=task_cfg.object_xyz_random_offset,
                post_reset_wait=robot_cfg.post_reset_wait,
                sleep_fn=sim_time.sleep,
            )
            base_world_pos, base_world_quat = get_entity_pose_world_service(robot_cfg.base_link_entity_path)
            target_pose = get_object_pose_from_service(
                base_world_pos,
                base_world_quat,
                task_cfg.source_object_entity_path,
                include_orientation=getattr(task_cfg, "use_object_orientation", False),
            )

            if task_kind == "pick_place":
                current_obs = robot.get_observation()
                orientation_vec = np.array(
                    [target_pose.orientation.x, target_pose.orientation.y, target_pose.orientation.z, target_pose.orientation.w]
                )
                if not getattr(task_cfg, "use_object_orientation", False) or np.linalg.norm(orientation_vec) < 1e-3:
                    target_pose.orientation.x = current_obs["left_ee.quat.x"]
                    target_pose.orientation.y = current_obs["left_ee.quat.y"]
                    target_pose.orientation.z = current_obs["left_ee.quat.z"]
                    target_pose.orientation.w = current_obs["left_ee.quat.w"]
                sequence = build_single_arm_pick_place_sequence(
                    target_pose=target_pose,
                    task_cfg=task_cfg,
                    home_pose=_pose_from_observation(initial_obs),
                    ee_prefix="left_ee",
                    gripper_key="left_gripper.pos",
                    gripper_open=gripper_open,
                    gripper_closed=gripper_closed,
                    grasp_orientation=task_cfg.grasp_orientation,
                    grasp_direction=task_cfg.grasp_direction,
                    grasp_direction_vector=task_cfg.grasp_direction_vector,
                )
            elif task_kind == "handover":
                if source_home_pose is None or receiver_home_pose is None:
                    raise RuntimeError("Failed to initialize handover home poses")
                sequence = build_handover_record_sequence(
                    handover_task_cfg=task_cfg,
                    source_target_pose=target_pose,
                    source_home_pose=source_home_pose,
                    receiver_home_pose=receiver_home_pose,
                    source_is_right=source_is_right,
                    gripper_open=gripper_open,
                    gripper_closed=gripper_closed,
                )
            else:
                raise NotImplementedError(f"Unsupported task kind: {task_kind}")
            recorder = DatasetRecorder(
                robot=robot,
                sim_time=sim_time,
                task_cfg=task_cfg,
                dataset_features=features,
                dataset_root=out_root,
                episode_index=episode_index,
                intrinsics=intrinsics,
                depth_listener=depth_listener,
                record_cfg=record_cfg,
                enable_keypoint_pcd=enable_keypoint_pcd,
                camera_name=depth_cam_name,
                save_pointcloud_frame_fn=_save_pointcloud_frame,
            )
            recorder.start_sequence()
            if task_kind == "pick_place":
                execute_stage_sequence(
                    robot=robot,
                    sequence=sequence,
                    wait_both_arms=False,
                    single_arm_part="left_arm",
                    arrival_timeout=robot_cfg.arrival_timeout,
                    arrival_poll=robot_cfg.arrival_poll,
                    time_now_fn=sim_time.now_seconds,
                    sleep_fn=sim_time.sleep,
                    gripper_action_wait=robot_cfg.gripper_action_wait,
                    warn_prefix="PickPlace stage timeout",
                    on_stage_start=recorder.on_stage_start,
                    on_stage_poll=recorder.on_stage_poll,
                )
            elif task_kind == "handover":
                pickup_sequence = [item for item in sequence if item[0].startswith("Pickup-")]
                handover_sequence = [item for item in sequence if item[0].startswith("Handover-")]
                place_sequence = [item for item in sequence if item[0].startswith("Place-")]
                if pickup_sequence:
                    execute_stage_sequence(
                        robot=robot,
                        sequence=pickup_sequence,
                        wait_both_arms=False,
                        single_arm_part="right_arm" if source_is_right else "left_arm",
                        arrival_timeout=robot_cfg.arrival_timeout,
                        arrival_poll=robot_cfg.arrival_poll,
                        time_now_fn=sim_time.now_seconds,
                        sleep_fn=sim_time.sleep,
                        gripper_action_wait=robot_cfg.gripper_action_wait,
                        warn_prefix="Pickup stage timeout",
                        on_stage_start=recorder.on_stage_start,
                        on_stage_poll=recorder.on_stage_poll,
                    )
                if handover_sequence:
                    execute_stage_sequence(
                        robot=robot,
                        sequence=handover_sequence,
                        wait_both_arms=True,
                        arrival_timeout=robot_cfg.arrival_timeout,
                        arrival_poll=robot_cfg.arrival_poll,
                        time_now_fn=sim_time.now_seconds,
                        sleep_fn=sim_time.sleep,
                        gripper_action_wait=robot_cfg.gripper_action_wait,
                        left_arrival_guard_stage="Handover-1-SyncMove" if source_is_right else None,
                        warn_prefix="Handover stage timeout",
                        on_stage_start=recorder.on_stage_start,
                        on_stage_poll=recorder.on_stage_poll,
                    )
                if place_sequence:
                    execute_stage_sequence(
                        robot=robot,
                        sequence=place_sequence,
                        wait_both_arms=True,
                        arrival_timeout=robot_cfg.arrival_timeout,
                        arrival_poll=robot_cfg.arrival_poll,
                        time_now_fn=sim_time.now_seconds,
                        sleep_fn=sim_time.sleep,
                        gripper_action_wait=robot_cfg.gripper_action_wait,
                        warn_prefix="Place stage timeout",
                        on_stage_start=recorder.on_stage_start,
                        on_stage_poll=recorder.on_stage_poll,
                    )
            episode_records = recorder.finish_sequence()
            if not episode_records:
                raise RuntimeError("No frames captured during sequence.")

            keep_episode = True
            if enable_manual_episode_check:
                delete_input = input("Delete this episode data? [y/N]: ").strip().lower()
                keep_episode = delete_input not in {"y", "yes"}
                if not keep_episode:
                    points_root = out_root / "points"
                    if points_root.exists():
                        target_suffix = f"episode_{episode_index:06d}"
                        for path in points_root.rglob(target_suffix):
                            if path.is_dir():
                                import shutil
                                shutil.rmtree(path, ignore_errors=True)
            if keep_episode:
                DatasetRecorder.append_episode_to_dataset(
                    dataset=dataset,
                    records=episode_records,
                    action_names=action_names,
                    include_gripper=include_gripper,
                    task_name=record_cfg.task_name,
                )
                kept_episode += 1
    finally:
        if depth_listener is not None:
            depth_listener.shutdown()
        if cam_info_listener is not None:
            cam_info_listener.shutdown()
        sim_time.shutdown()
        if robot.is_connected:
            robot.disconnect()
