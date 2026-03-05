#!/usr/bin/env python3
"""Shared dataset recording runtime for IsaacSim motion sequences."""

from __future__ import annotations

import threading
import time
from typing import Any, Callable, Optional

import numpy as np
from lerobot.datasets.utils import build_dataset_frame, hw_to_dataset_features  # pyright: ignore[reportMissingImports]


class DatasetRecorder:
    """Execute staged actions and write aligned LeRobot frames."""

    def __init__(
        self,
        *,
        robot: Any,
        sim_time: Any,
        task_cfg: Any,
        dataset_features: dict[str, dict],
        dataset_root: Any,
        episode_index: int,
        intrinsics: tuple[float, float, float, float] | None,
        depth_listener: Any,
        record_cfg: Any,
        enable_keypoint_pcd: bool,
        camera_name: str,
        save_pointcloud_frame_fn: Callable[..., None],
    ) -> None:
        self.robot = robot
        self.sim_time = sim_time
        self.task_cfg = task_cfg
        self.dataset_features = dataset_features
        self.dataset_root = dataset_root
        self.episode_index = episode_index
        self.intrinsics = intrinsics
        self.depth_listener = depth_listener
        self.record_cfg = record_cfg
        self.enable_keypoint_pcd = enable_keypoint_pcd
        self.camera_name = camera_name
        self.save_pointcloud_frame_fn = save_pointcloud_frame_fn
        self._records: list[dict[str, object]] = []
        self._command_counter = 0
        self._pcd_save_idx = 0
        self._pending_keypoint: Optional[dict[str, int]] = None
        self._current_action: dict[str, float] | None = None
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._sampler_thread: threading.Thread | None = None

    @staticmethod
    def build_dataset_features(
        robot: Any,
        *,
        include_depth_feature: bool,
    ) -> tuple[dict[str, dict], list[str], bool]:
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

    @staticmethod
    def append_episode_to_dataset(
        *,
        dataset: Any,
        records: list[dict[str, object]],
        action_names: list[str],
        include_gripper: bool,
        task_name: str,
    ) -> None:
        action_keys = list(action_names)
        for idx, rec in enumerate(records):
            frame: dict[str, object] = {"task": task_name}
            frame.update(rec["observation_frame"])  # type: ignore[arg-type]

            next_idx = idx + 1 if idx + 1 < len(records) else idx
            next_rec = records[next_idx]
            frame["action"] = DatasetRecorder._action_from_next(
                next_action=next_rec["command_action"],  # type: ignore[arg-type]
                action_keys=action_keys,
            )
            dataset.add_frame(frame)

        dataset.save_episode()

    def start_sequence(self) -> None:
        with self._lock:
            self._records = []
            self._command_counter = 0
            self._pcd_save_idx = 0
            self._pending_keypoint = None
            self._current_action = None
        self._stop_event.clear()
        self._sampler_thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._sampler_thread.start()

    def on_stage_start(self, stage_name: str, action: dict[str, float]) -> None:
        del stage_name
        with self._lock:
            self._current_action = dict(action)
            self._pending_keypoint = {"command_index": self._command_counter}
            self._command_counter += 1

    def on_stage_poll(self, stage_name: str, part: str, result: dict[str, object] | None, elapsed: float) -> None:
        del stage_name, part, result, elapsed
        return

    def finish_sequence(self) -> list[dict[str, object]]:
        self._stop_event.set()
        if self._sampler_thread is not None:
            self._sampler_thread.join(timeout=2.0)
            self._sampler_thread = None
        with self._lock:
            pending = self._pending_keypoint
            records = list(self._records)
        if pending is not None:
            raise RuntimeError(
                f"No frame captured after sending action index {pending['command_index']}"
            )
        return records

    def _sample_loop(self) -> None:
        period = max(1.0 / float(getattr(self.record_cfg, "fps", 30)), 1e-3)
        while not self._stop_event.is_set():
            self._capture_once()
            time.sleep(period)

    def _capture_once(self) -> None:
        with self._lock:
            current_action = None if self._current_action is None else dict(self._current_action)
            pending_keypoint = None if self._pending_keypoint is None else dict(self._pending_keypoint)
        if current_action is None:
            return
        obs = self.robot.get_observation()
        observation_frame, ee_pose = self._extract_observation_frame(obs)
        if self.enable_keypoint_pcd and pending_keypoint is not None:
            depth_raw = self.depth_listener.get_latest_depth() if self.depth_listener is not None else None
            if depth_raw is None:
                depth_key = f"{self.camera_name}.depth"
                if depth_key in obs:
                    depth_raw = obs[depth_key].astype(np.float32)

            rgb_raw = None
            for cam_name in self.robot.config.cameras.keys():
                rgb_key = f"{cam_name}.rgb"
                if rgb_key in obs:
                    rgb_raw = obs[rgb_key].astype(np.uint8)
                elif cam_name in obs:
                    rgb_raw = obs[cam_name].astype(np.uint8)
                if rgb_raw is not None:
                    break

            self.save_pointcloud_frame_fn(
                self.dataset_root,
                self.episode_index,
                self._pcd_save_idx,
                depth_raw,
                self.intrinsics,
                rgb_image=rgb_raw,
                pointcloud_key=f"observation.points.{self.camera_name}_pcd",
            )
            with self._lock:
                self._pcd_save_idx += 1
                self._pending_keypoint = None
        elif pending_keypoint is not None and not self.enable_keypoint_pcd:
            with self._lock:
                self._pending_keypoint = None

        ee_pose_arr = np.array(
            [ee_pose["x"], ee_pose["y"], ee_pose["z"], ee_pose["qx"], ee_pose["qy"], ee_pose["qz"], ee_pose["qw"]],
            dtype=np.float32,
        )
        with self._lock:
            self._records.append(
                {
                    "observation_frame": observation_frame,
                    "ee_pose": ee_pose_arr,
                    "command_action": current_action,
                }
            )

    def _extract_observation_frame(self, obs: dict[str, float]) -> tuple[dict[str, object], dict[str, float]]:
        frame = build_dataset_frame(self.dataset_features, obs, prefix="observation")
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

    @staticmethod
    def _action_from_next(
        *,
        next_action: dict[str, float],
        action_keys: list[str],
    ) -> np.ndarray:
        values = [float(next_action.get(k, 0.0)) for k in action_keys]
        return np.asarray(values, dtype=np.float32)

