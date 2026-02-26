"""
LeRobot ROS2 Robot Plugin

This package provides ROS 2 robot integration for the LeRobot framework,
following LeRobot's plugin naming conventions.
"""

from .config import ROS2RobotConfig
from .motion_generation import (
    build_single_arm_grasp_transport_release_sequence,
    compose_bimanual_synchronized_sequence,
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
    "build_single_arm_grasp_transport_release_sequence",
    "compose_bimanual_synchronized_sequence",
]
