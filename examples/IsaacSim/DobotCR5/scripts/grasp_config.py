#!/usr/bin/env python3
"""Unified grasp configuration object for demo and dataset scripts."""

from __future__ import annotations

from dataclasses import dataclass
from ros2_robot_interface import FSM_HOLD, FSM_OCS2


@dataclass(frozen=True)
class MotionConfig:
    approach_clearance: float = 0.12
    grasp_clearance: float = -0.03
    gripper_open: float = 1.0
    gripper_closed: float = 0.0
    grasp_orientation: tuple[float, float, float, float] = (0.7, 0.7, 0.0, 0.0)
    release_position: tuple[float, float, float] = (-0.5011657941743888, 0.36339369774887115, 0.34)
    transport_height: float = 0.4
    retract_offset_y: float = -0.2
    place_orientation: tuple[float, float, float, float] = (0.64, 0.64, -0.28, -0.28)


@dataclass(frozen=True)
class RuntimeConfig:
    fsm_hold: int = FSM_HOLD
    fsm_ocs2: int = FSM_OCS2
    fsm_switch_delay: float = 0.1
    post_reset_wait: float = 1.0
    object_xyz_random_offset: tuple[float, float, float] = (0.5, 0.5, 0.0)
    base_link_entity_path: str = "/World/DobotCR5_ROS2/DobotCR5/base_link"


@dataclass(frozen=True)
class SharedBehaviorConfig:
    object_entity_path: str = "/World/apple/apple/apple"
    use_object_orientation: bool = False


@dataclass(frozen=True)
class SingleDemoConfig:
    gripper_action_wait: float = 0.3
    stage_arrival_timeout: float = 3.0
    stage_check_period: float = 0.05
    stage_fallback_wait: float = 0.2
    return_pose_timeout: float = 8.0
    return_pos_tol: float = 0.02
    return_check_period: float = 0.05


@dataclass(frozen=True)
class RecordConfig:
    fps: int = 30
    max_stage_duration: float = 2.0
    pose_tol_pos: float = 0.025
    pose_tol_ori: float = 0.08
    gripper_tol: float = 0.05
    require_orientation_reach: bool = False
    return_pause: float = 1.0
    pose_timeout: float = 10.0
    task_name: str = "full_grasp_transport_release"
    enable_keypoint_pcd: bool = True


@dataclass(frozen=True)
class GraspConfig:
    motion: MotionConfig = MotionConfig()
    runtime: RuntimeConfig = RuntimeConfig()
    shared: SharedBehaviorConfig = SharedBehaviorConfig()
    single: SingleDemoConfig = SingleDemoConfig()
    record: RecordConfig = RecordConfig()


GRASP_CFG = GraspConfig()
