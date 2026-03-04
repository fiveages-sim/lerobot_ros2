#!/usr/bin/env python3
"""Robot profile config for Marvin M6CCS IsaacSim demos."""

from __future__ import annotations

from dataclasses import dataclass, field

from ros2_robot_interface import ROS2RobotInterfaceConfig  # pyright: ignore[reportMissingImports]


ROBOT_KEY = "marvin_m6ccs"
ROBOT_LABEL = "Marvin M6CCS"


@dataclass(frozen=True)
class RobotConfig:
    gripper_control_mode: str = "target_command"
    ros2_interface: ROS2RobotInterfaceConfig = field(
        default_factory=lambda: ROS2RobotInterfaceConfig.default_bimanual(
            joint_names=(
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
            ),
        )
    )
    base_link_entity_path: str = "/World/M6_CCS_Dual/Dual_Stand1/root_joint"
    fsm_switch_delay: float = 0.1
    post_reset_wait: float = 1.0
    arrival_timeout: float = 3.0
    arrival_poll: float = 0.05
    gripper_action_wait: float = 0.3


ROBOT_CFG = RobotConfig()
