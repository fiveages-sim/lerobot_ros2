"""
LeRobot ROS2 Robot Plugin

This package provides ROS 2 robot integration for the LeRobot framework,
following LeRobot's plugin naming conventions.
"""

from .config import ROS2RobotConfig
from .motion_generation import (
    build_handover_sequence_for_arms,
    build_single_arm_pick_sequence,
    build_single_arm_place_sequence,
    build_single_arm_return_home_sequence,
    compose_bimanual_synchronized_sequence,
    execute_stage_sequence,
)
from .robot import ROS2Robot

# Re-export from ros2_robot_interface for convenience
from ros2_robot_interface import ROS2RobotInterfaceConfig, ControlType

__version__ = "0.1.0"

__all__ = [
    "ROS2RobotConfig",
    "ROS2RobotInterfaceConfig", 
    "ControlType",
    "ROS2Robot",
    "compose_bimanual_synchronized_sequence",
    "build_single_arm_pick_sequence",
    "build_single_arm_place_sequence",
    "build_single_arm_return_home_sequence",
    "build_handover_sequence_for_arms",
    "execute_stage_sequence",
]
