#!/usr/bin/env python3
"""Robot profile config for Agibot G1 IsaacSim demos."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RobotConfig:
    # Robot profile
    gripper_control_mode: str = "target_command"
    joint_names: tuple[str, ...] = (
        "body_joint1",
        "body_joint2",
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
    base_link_entity_path: str = "/World/Agibot_G1_ROS2/Agibot_G1/base_footprint"
    fsm_switch_delay: float = 0.1
    post_reset_wait: float = 1.0
    # Default flow
    arrival_timeout: float = 3.0
    arrival_poll: float = 0.05
    gripper_action_wait: float = 0.3


ROBOT_CFG = RobotConfig()
