"""
LeRobot ROS2 Robot Plugin

This package provides ROS 2 robot integration for the LeRobot framework,
following LeRobot's plugin naming conventions.
"""

from .config import ROS2RobotConfig, ROS2RobotInterfaceConfig, ControlType
from .robot import ROS2Robot

__version__ = "0.1.0"

__all__ = [
    "ROS2RobotConfig",
    "ROS2RobotInterfaceConfig", 
    "ControlType",
    "ROS2Robot",
]
