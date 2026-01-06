"""
ROS 2 Robot Configuration

Configuration classes for ROS 2 robots in LeRobot.
"""

from dataclasses import dataclass, field
from typing import Any

from lerobot.cameras import CameraConfig
from lerobot.robots import RobotConfig

# Import from the standalone ros2_robot_interface package
from ros2_robot_interface import ControlType, ROS2RobotInterfaceConfig


@RobotConfig.register_subclass("lerobot_robot_ros2")
@dataclass
class ROS2RobotConfig(RobotConfig):
    """Configuration for ROS 2 robots in LeRobot.
    
    This configuration allows LeRobot to interface with robots through ROS 2,
    supporting joint state monitoring and end-effector control.
    
    Example:
        ```python
        config = ROS2RobotConfig(
            id="my_robot",
            cameras={
                "wrist_camera": ROS2CameraConfig(
                    topic_name="/camera/image_raw",
                    width=640,
                    height=480
                )
            },
            ros2_interface=ROS2RobotInterfaceConfig(
                joint_states_topic="/joint_states",
                end_effector_pose_topic="/left_current_pose",
                end_effector_target_topic="/left_target",
                control_type=ControlType.CARTESIAN_POSE
            )
        )
        ```
    """
    
    # ROS 2 interface configuration
    ros2_interface: ROS2RobotInterfaceConfig = field(default_factory=ROS2RobotInterfaceConfig)
    
    # Camera configurations
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
    
    # Safety settings
    max_relative_target: float | None = None  # Maximum relative movement for safety
    
    # Robot-specific settings
    robot_name: str = "ros2_robot"
    namespace: str = ""
