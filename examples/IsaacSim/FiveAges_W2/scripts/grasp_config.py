#!/usr/bin/env python3
"""Unified bimanual grasp configuration for FiveAges W2 IsaacSim demos."""

from __future__ import annotations

from dataclasses import dataclass
from ros2_robot_interface import FSM_HOLD, FSM_OCS2  # pyright: ignore[reportMissingImports]


@dataclass(frozen=True)
class MotionConfig:
    approach_clearance: float = 0.2
    grasp_clearance: float = 0.05
    grasp_orientation: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    transport_height: float = 0.35
    left_retract_offset_y: float = 0.15
    right_retract_offset_y: float = -0.15
    gripper_open: float = 1.0
    gripper_closed: float = 0.0
    left_release_position: tuple[float, float, float] = (-0.40, 0.25, 0.30)
    right_release_position: tuple[float, float, float] = (-0.40, -0.25, 0.30)
    # Handover pose used after right-arm grasp
    handover_position: tuple[float, float, float] = (
        0.55,
        0.0,
        1.35,
    )
    handover_orientation: tuple[float, float, float, float] = (
        0.7,
        0.3,
        0.7,
        -0.3,
    )
    left_handover_orientation: tuple[float, float, float, float] = (
        0.6,
        -0.3,
        0.6,
        0.3,
    )
    left_place_position: tuple[float, float, float] = (
        0.7,
        0.3,
        1.1,
    )
    left_place_orientation: tuple[float, float, float, float] = (
        0.6532920183548978,
        -0.12549174013187747,
        0.7390193620374562,
        0.10635668500949247,
    )


@dataclass(frozen=True)
class RuntimeConfig:
    joint_states_topic: str = "/joint_states"
    left_current_pose_topic: str = "/left_current_pose"
    right_current_pose_topic: str = "/right_current_pose"
    left_target_topic: str = "/left_target"
    right_target_topic: str = "/right_target"
    left_current_target_topic: str = "/left_current_target"
    right_current_target_topic: str = "/right_current_target"
    left_gripper_command_topic: str = "/left_gripper_joint/position_command"
    right_gripper_command_topic: str = "/right_gripper_joint/position_command"
    gripper_control_mode: str = "target_command"
    left_gripper_joint_name: str = "left_gripper_joint"
    joint_names: tuple[str, ...] = (
        "body_joint1",
        "body_joint2",
        "body_joint3",
        "body_joint4",
        "head_joint1",
        "head_joint2",
        "left_gripper_joint",
        "left_joint1",
        "left_joint2",
        "left_joint3",
        "left_joint4",
        "left_joint5",
        "left_joint6",
        "left_joint7",
        "right_gripper_joint",
        "right_joint1",
        "right_joint2",
        "right_joint3",
        "right_joint4",
        "right_joint5",
        "right_joint6",
        "right_joint7",
    )
    fsm_hold: int = FSM_HOLD
    fsm_ocs2: int = FSM_OCS2
    fsm_switch_delay: float = 0.1
    post_reset_wait: float = 1.0
    object_xyz_random_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class SceneConfig:
    # Adjust this base path to your FiveAges W2 prim path in IsaacSim.
    base_link_entity_path: str = "/World/FiveAges_W2/LinkHou_S2/base_footprint"
    left_object_entity_path: str = "/World/left_object"
    right_object_entity_path: str = "/World/medicine_handover/FinasterideTablets/tablets/tablets"


@dataclass(frozen=True)
class SingleDemoConfig:
    right_grasp_only: bool = True
    stop_after_grasp: bool = False
    stop_after_lift: bool = True
    arrival_timeout: float = 3.0
    arrival_poll: float = 0.05
    gripper_action_wait: float = 0.3


@dataclass(frozen=True)
class GraspConfig:
    motion: MotionConfig = MotionConfig()
    runtime: RuntimeConfig = RuntimeConfig()
    scene: SceneConfig = SceneConfig()
    single: SingleDemoConfig = SingleDemoConfig()


GRASP_CFG = GraspConfig()
