#!/usr/bin/env python3
"""Robot profile config for Dobot CR5 IsaacSim demos."""

from __future__ import annotations

from dataclasses import dataclass, field

from lerobot_camera_ros2 import ROS2CameraConfig  # pyright: ignore[reportMissingImports]
from ros2_robot_interface import ROS2RobotInterfaceConfig  # pyright: ignore[reportMissingImports]

ROBOT_KEY = "dobot_cr5"
ROBOT_LABEL = "Dobot CR5"


@dataclass(frozen=True)
class RobotConfig:
    robot_id: str = "ros2_grasp_robot"
    gripper_control_mode: str = "target_command"
    ros2_interface: ROS2RobotInterfaceConfig = field(
        default_factory=lambda: ROS2RobotInterfaceConfig.default_single_arm(
            joint_names=("joint1", "joint2", "joint3", "joint4", "joint5", "joint6"),
            gripper_joint_name="gripper_joint",
            gripper_command_topic="gripper_joint/position_command",
        )
    )
    base_link_entity_path: str = "/World/DobotCR5_ROS2/DobotCR5/base_link"
    fsm_switch_delay: float = 0.1
    post_reset_wait: float = 1.0
    arrival_timeout: float = 3.0
    arrival_poll: float = 0.05
    gripper_action_wait: float = 0.3
    cameras: dict[str, ROS2CameraConfig] = field(
        default_factory=lambda: {
            "global": ROS2CameraConfig(
                topic_name="/global_camera/rgb",
                node_name="lerobot_global_camera",
                depth_topic_name="/global_camera/depth",
            ),
            "wrist": ROS2CameraConfig(
                topic_name="/wrist_camera/rgb",
                node_name="lerobot_wrist_camera",
            ),
        }
    )
    depth_camera_name: str = "global"
    depth_info_topic: str = "/global_camera/camera_info"


ROBOT_CFG = RobotConfig()
