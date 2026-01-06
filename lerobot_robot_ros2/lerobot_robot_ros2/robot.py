"""
ROS 2 Robot Implementation for LeRobot

This module provides a ROS 2 robot implementation that integrates with LeRobot,
supporting joint state monitoring and end-effector control through ROS 2 topics.
"""

import logging
import time
from functools import cached_property
from typing import Any

import numpy as np
from geometry_msgs.msg import Pose
# Import will be done dynamically to use the patched version
from lerobot.robots import Robot
from lerobot.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from .config import ROS2RobotConfig
# Import from the standalone ros2_robot_interface package
from ros2_robot_interface import ROS2RobotInterface

logger = logging.getLogger(__name__)


class ROS2Robot(Robot):
    """ROS 2 robot implementation for LeRobot.
    
    This robot class interfaces with ROS 2 robots through topics:
    - Subscribes to /joint_states for joint state information
    - Subscribes to /left_current_pose for current end-effector pose
    - Publishes to /left_target for end-effector target commands
    
    Example:
        ```python
        from lerobot_ros2.robots import ROS2Robot, ROS2RobotConfig
        
        # Create configuration
        config = ROS2RobotConfig(
            id="my_robot",
            ros2_interface=ROS2RobotInterfaceConfig(
                joint_states_topic="/joint_states",
                end_effector_pose_topic="/left_current_pose",
                end_effector_target_topic="/left_target",
                control_type=ControlType.CARTESIAN_POSE
            )
        )
        
        # Create and connect robot
        robot = ROS2Robot(config)
        robot.connect()
        
        # Get observation
        obs = robot.get_observation()
        
        # Send action
        action = {"end_effector_pose": {...}}
        robot.send_action(action)
        
        # Disconnect when done
        robot.disconnect()
        ```
    """
    
    config_class = ROS2RobotConfig
    name = "ros2_robot"
    
    def __init__(self, config: ROS2RobotConfig):
        """Initialize the ROS 2 robot.
        
        Args:
            config: ROS 2 robot configuration
        """
        super().__init__(config)
        self.config = config
        self.ros2_interface = ROS2RobotInterface(config.ros2_interface)
        # Create cameras with ROS2 support
        self.cameras = self._create_cameras_with_ros2_support(config.cameras)
    
    def _create_cameras_with_ros2_support(self, camera_configs: dict[str, Any]) -> dict[str, Any]:
        """Create cameras with support for ROS2 cameras.
        
        Args:
            camera_configs: Dictionary of camera configurations
            
        Returns:
            Dictionary of camera instances
        """
        cameras = {}
        
        for key, cfg in camera_configs.items():
            if cfg.type == "lerobot_camera_ros2":
                # Special handling for ROS2 cameras
                try:
                    from lerobot_camera_ros2 import ROS2Camera
                    cameras[key] = ROS2Camera(cfg)
                    logger.info(f"Created ROS2 camera: {key}")
                except ImportError as e:
                    logger.error(f"Failed to import ROS2Camera: {e}")
                    raise ImportError(
                        "ROS2Camera not available. Please install lerobot_camera_ros2 package."
                    ) from e
            else:
                # Use standard LeRobot camera creation for other types
                from lerobot.cameras.utils import make_cameras_from_configs
                other_cameras = make_cameras_from_configs({key: cfg})
                cameras.update(other_cameras)
        
        return cameras
    
    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        """Get the observation features for this robot.
        
        Returns:
            Dictionary mapping observation keys to their types.
        """
        features = {}
        
        # Joint positions
        for joint_name in self.config.ros2_interface.joint_names:
            features[f"{joint_name}.pos"] = float
        
        # Joint velocities
        for joint_name in self.config.ros2_interface.joint_names:
            features[f"{joint_name}.vel"] = float
        
        # Joint efforts (torques)
        for joint_name in self.config.ros2_interface.joint_names:
            features[f"{joint_name}.effort"] = float
        
        # Gripper position (if enabled)
        if self.config.ros2_interface.gripper_enabled:
            features[f"{self.config.ros2_interface.gripper_joint_name}.pos"] = float
        
        # End-effector pose
        features["end_effector.position.x"] = float
        features["end_effector.position.y"] = float
        features["end_effector.position.z"] = float
        features["end_effector.orientation.x"] = float
        features["end_effector.orientation.y"] = float
        features["end_effector.orientation.z"] = float
        features["end_effector.orientation.w"] = float
        
        # Camera features
        for cam_name, cam_config in self.config.cameras.items():
            height = cam_config.height or 720
            width = cam_config.width or 1280
            if getattr(cam_config, "depth_topic_name", None):
                features[f"{cam_name}.rgb"] = (height, width, 3)
                features[f"{cam_name}.depth"] = (height, width, 1)
            else:
                features[cam_name] = (height, width, 3)
        
        return features
    
    @cached_property
    def action_features(self) -> dict[str, type]:
        """Get the action features for this robot.
        
        Returns:
            Dictionary mapping action keys to their types.
        """
        features = {
            "end_effector.position.x": float,
            "end_effector.position.y": float,
            "end_effector.position.z": float,
            "end_effector.orientation.x": float,
            "end_effector.orientation.y": float,
            "end_effector.orientation.z": float,
            "end_effector.orientation.w": float,
        }
        
        # Add gripper control feature if enabled
        if self.config.ros2_interface.gripper_enabled:
            features["gripper.position"] = float
        
        return features
    
    @property
    def is_connected(self) -> bool:
        """Check if the robot is connected.
        
        Returns:
            True if robot and cameras are connected, False otherwise.
        """
        return self.ros2_interface.is_connected and all(cam.is_connected for cam in self.cameras.values())
    
    def connect(self, calibrate: bool = True) -> None:
        """Connect to the robot.
        
        Args:
            calibrate: Whether to perform calibration (ignored for ROS 2 robots)
        """
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")
        
        # Connect cameras first
        for cam in self.cameras.values():
            cam.connect()
        
        # Connect ROS 2 interface
        self.ros2_interface.connect()
        
        logger.info(f"Connected {self}")
    
    @property
    def is_calibrated(self) -> bool:
        """Check if the robot is calibrated.
        
        Returns:
            True (ROS 2 robots are considered pre-calibrated)
        """
        return True
    
    def calibrate(self) -> None:
        """Calibrate the robot.
        
        Note: ROS 2 robots are considered pre-calibrated.
        """
        logger.info("ROS 2 robots are considered pre-calibrated")
    
    def configure(self) -> None:
        """Configure the robot.
        
        Note: ROS 2 robots are configured through their ROS 2 system.
        """
        logger.info("ROS 2 robots are configured through their ROS 2 system")
    
    def get_observation(self) -> dict[str, Any]:
        """Get the current observation from the robot.
        
        Returns:
            Dictionary containing joint states, end-effector pose, and camera images.
            
        Raises:
            DeviceNotConnectedError: If robot is not connected
            RuntimeError: If required data is not available
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected")
        
        obs_dict: dict[str, Any] = {}
        
        # Get joint states
        joint_state = self.ros2_interface.get_joint_state()
        
        # When timeout is 0, use default values if joint state not available
        # This allows the system to continue working when ROS2 nodes restart
        if joint_state is None:
            if self.config.ros2_interface.joint_state_timeout > 0:
                raise RuntimeError("Joint state not available")
            else:
                # Timeout is 0, use default values for all joints
                logger.debug("Joint state not available, using default values (timeout=0)")
                for joint_name in self.config.ros2_interface.joint_names:
                    obs_dict[f"{joint_name}.pos"] = 0.0
                    obs_dict[f"{joint_name}.vel"] = 0.0
                    obs_dict[f"{joint_name}.effort"] = 0.0
                if self.config.ros2_interface.gripper_enabled:
                    gripper_joint_name = self.config.ros2_interface.gripper_joint_name
                    obs_dict[f"{gripper_joint_name}.pos"] = 0.0
        else:
            # Extract joint positions, velocities, and efforts
            joint_names = joint_state["names"]
            joint_positions = joint_state["positions"]
            joint_velocities = joint_state["velocities"]
            joint_efforts = joint_state["efforts"]
            
            for joint_name in self.config.ros2_interface.joint_names:
                try:
                    idx = joint_names.index(joint_name)
                    obs_dict[f"{joint_name}.pos"] = joint_positions[idx]
                    obs_dict[f"{joint_name}.vel"] = joint_velocities[idx]
                    obs_dict[f"{joint_name}.effort"] = joint_efforts[idx]
                except ValueError:
                    logger.warning(f"Joint '{joint_name}' not found in joint state")
                    obs_dict[f"{joint_name}.pos"] = 0.0
                    obs_dict[f"{joint_name}.vel"] = 0.0
                    obs_dict[f"{joint_name}.effort"] = 0.0
        
        # Extract gripper position (if enabled and not already set)
        if self.config.ros2_interface.gripper_enabled and joint_state is not None:
            gripper_joint_name = self.config.ros2_interface.gripper_joint_name
            if f"{gripper_joint_name}.pos" not in obs_dict:
                joint_names = joint_state["names"]
                joint_positions = joint_state["positions"]
                try:
                    idx = joint_names.index(gripper_joint_name)
                    obs_dict[f"{gripper_joint_name}.pos"] = joint_positions[idx]
                except ValueError:
                    logger.warning(f"Gripper joint '{gripper_joint_name}' not found in joint state")
                    obs_dict[f"{gripper_joint_name}.pos"] = 0.0
        
        # Get end-effector pose
        end_effector_pose = self.ros2_interface.get_end_effector_pose()
        if end_effector_pose is not None:
            obs_dict["end_effector.position.x"] = end_effector_pose.position.x
            obs_dict["end_effector.position.y"] = end_effector_pose.position.y
            obs_dict["end_effector.position.z"] = end_effector_pose.position.z
            obs_dict["end_effector.orientation.x"] = end_effector_pose.orientation.x
            obs_dict["end_effector.orientation.y"] = end_effector_pose.orientation.y
            obs_dict["end_effector.orientation.z"] = end_effector_pose.orientation.z
            obs_dict["end_effector.orientation.w"] = end_effector_pose.orientation.w
        else:
            # Set default values if pose not available
            obs_dict["end_effector.position.x"] = 0.0
            obs_dict["end_effector.position.y"] = 0.0
            obs_dict["end_effector.position.z"] = 0.0
            obs_dict["end_effector.orientation.x"] = 0.0
            obs_dict["end_effector.orientation.y"] = 0.0
            obs_dict["end_effector.orientation.z"] = 0.0
            obs_dict["end_effector.orientation.w"] = 1.0
        
        # Capture images from cameras
        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            try:
                cam_data = cam.async_read(timeout_ms=300)
                if isinstance(cam_data, dict):
                    for modality, value in cam_data.items():
                        if value is None:
                            continue
                        if modality == "depth":
                            obs_dict[f"{cam_key}.{modality}"] = value[..., None] if value.ndim == 2 else value
                        else:
                            obs_dict[f"{cam_key}.{modality}"] = value
                else:
                    obs_dict[cam_key] = cam_data
            except Exception as e:
                logger.error(f"Failed to read camera {cam_key}: {e}")
                if getattr(cam, "depth_topic_name", None):
                    obs_dict[f"{cam_key}.rgb"] = None
                    obs_dict[f"{cam_key}.depth"] = None
                else:
                    obs_dict[cam_key] = None
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")
        
        return obs_dict
    
    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """Send an action to the robot.
        
        Args:
            action: Dictionary containing the action to send
            
        Returns:
            The action that was actually sent to the robot
            
        Raises:
            DeviceNotConnectedError: If robot is not connected
            ValueError: If action format is invalid
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected")
        
        # Create pose from action
        pose = Pose()
        pose.position.x = action.get("end_effector.position.x", 0.0)
        pose.position.y = action.get("end_effector.position.y", 0.0)
        pose.position.z = action.get("end_effector.position.z", 0.0)
        pose.orientation.x = action.get("end_effector.orientation.x", 0.0)
        pose.orientation.y = action.get("end_effector.orientation.y", 0.0)
        pose.orientation.z = action.get("end_effector.orientation.z", 0.0)
        pose.orientation.w = action.get("end_effector.orientation.w", 1.0)
        
        # Send the target pose
        self.ros2_interface.send_end_effector_target(pose)
        
        # Send gripper command if provided and gripper is enabled
        if self.config.ros2_interface.gripper_enabled and "gripper.position" in action:
            gripper_position = action["gripper.position"]
            self.ros2_interface.send_gripper_command(gripper_position)
            logger.debug(f"Sent gripper command: {gripper_position}")
        
        return action
    
    def disconnect(self) -> None:
        """Disconnect from the robot and cleanup resources."""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected")
        
        # Disconnect cameras
        for cam in self.cameras.values():
            cam.disconnect()
        
        # Disconnect ROS 2 interface
        self.ros2_interface.disconnect()
        
        logger.info(f"Disconnected {self}")
