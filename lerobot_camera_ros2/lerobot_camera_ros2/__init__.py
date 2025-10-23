"""
LeRobot ROS2 Camera Plugin

This package provides ROS 2 camera integration for the LeRobot framework,
following LeRobot's plugin naming conventions.
"""

from .config import ROS2CameraConfig
from .camera import ROS2Camera

__version__ = "0.1.0"

__all__ = [
    "ROS2CameraConfig",
    "ROS2Camera",
]
