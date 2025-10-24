"""
ROS 2 Robot Configuration

Configuration classes for ROS 2 robots in LeRobot.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from lerobot.cameras import CameraConfig
from lerobot.robots import RobotConfig


class ControlType(Enum):
    """Control type for the robot."""
    CARTESIAN_POSE = "cartesian_pose"


@dataclass
class ROS2RobotInterfaceConfig:
    """Configuration for ROS 2 robot interface."""
    
    # ROS 2 topics
    joint_states_topic: str = "/joint_states"
    end_effector_pose_topic: str = "/left_current_pose"
    end_effector_target_topic: str = "/left_target"
    
    # Joint names
    joint_names: list[str] = field(default_factory=lambda: [
        "joint1", "joint2", "joint3", "joint4", "joint5", "joint6"
    ])
    
    # Gripper configuration
    gripper_enabled: bool = True
    gripper_joint_name: str = "gripper_joint"
    gripper_state_topic: str = "/gripper_state"  # Optional: separate gripper state topic
    gripper_command_topic: str = "/gripper_command"  # Required for gripper control
    gripper_min_position: float = 0.0  # Closed position
    gripper_max_position: float = 1.0  # Open position
    
    # Control parameters
    control_type: ControlType = ControlType.CARTESIAN_POSE
    
    # Safety limits
    max_linear_velocity: float = 0.1  # m/s
    max_angular_velocity: float = 0.5  # rad/s
    
    # Joint limits (optional)
    min_joint_positions: list[float] | None = None
    max_joint_positions: list[float] | None = None
    
    # Timeout settings
    joint_state_timeout: float = 1.0  # seconds
    end_effector_pose_timeout: float = 1.0  # seconds


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
