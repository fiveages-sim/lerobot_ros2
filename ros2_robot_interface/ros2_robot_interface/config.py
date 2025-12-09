"""
ROS 2 Robot Interface Configuration

Configuration classes for ROS 2 robot interface.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ControlType(Enum):
    """Control type for the robot."""
    CARTESIAN_POSE = "cartesian_pose"


@dataclass
class ROS2RobotInterfaceConfig:
    """Configuration for ROS 2 robot interface.
    
    This configuration class is independent of LeRobot and can be used
    in any ROS 2 environment.
    
    Example:
        ```python
        from ros2_robot_interface import ROS2RobotInterfaceConfig, ControlType
        
        config = ROS2RobotInterfaceConfig(
            joint_states_topic="/joint_states",
            end_effector_pose_topic="/left_current_pose",
            end_effector_target_topic="/left_target",
            control_type=ControlType.CARTESIAN_POSE,
            joint_names=["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
        )
        ```
    """
    
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
    gripper_command_topic: str = "gripper_joint/position_command"  # Required for gripper control
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
    # Set to 0 to disable timeout checking (useful when ROS2 nodes may restart)
    joint_state_timeout: float = 0.0  # seconds
    end_effector_pose_timeout: float = 0.0  # seconds
    
    # ROS 2 node namespace (optional)
    namespace: str = ""

