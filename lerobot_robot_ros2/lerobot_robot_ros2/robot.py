"""
ROS 2 Robot Implementation for LeRobot

This module provides a ROS 2 robot implementation that integrates with LeRobot,
supporting joint state monitoring and end-effector control through ROS 2 topics.
"""

import logging
import time
from collections.abc import Sequence
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

    LEFT_EE_PREFIX = "left_ee"
    RIGHT_EE_PREFIX = "right_ee"
    LEFT_GRIPPER_KEY = "left_gripper.pos"
    RIGHT_GRIPPER_KEY = "right_gripper.pos"
    TARGET_COMMAND_OPEN = 1
    TARGET_COMMAND_CLOSE = 0
    
    def __init__(self, config: ROS2RobotConfig):
        """Initialize the ROS 2 robot.
        
        Args:
            config: ROS 2 robot configuration
        """
        super().__init__(config)
        self.config = config
        self.ros2_interface = ROS2RobotInterface(config.ros2_interface)
        self._use_target_command = False
        self._gripper_command_threshold = 0.5
        self._joint_name_to_index: dict[str, int] = {}
        self._joint_names_signature: tuple[str, ...] | None = None
        self._refresh_gripper_command_config()
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

    def _is_dual_arm_enabled(self) -> bool:
        """Whether right arm topics are configured (manually or auto-detected)."""
        cfg = self.ros2_interface.config
        return bool(cfg.right_end_effector_pose_topic and cfg.right_end_effector_target_topic)

    def _refresh_gripper_command_config(self) -> None:
        """Cache gripper mode-related config to avoid per-action repeated lookups."""
        self._use_target_command = self.config.gripper_control_mode == "target_command"
        self._gripper_command_threshold = float(self.config.gripper_command_threshold)

    def _to_target_command(self, raw_value: Any) -> int:
        """Map input value to gripper target command (open/close)."""
        value = float(raw_value)
        as_int = int(value)
        if as_int in (self.TARGET_COMMAND_OPEN, self.TARGET_COMMAND_CLOSE):
            return as_int
        return (
            self.TARGET_COMMAND_OPEN
            if value >= self._gripper_command_threshold
            else self.TARGET_COMMAND_CLOSE
        )

    def _send_gripper_value(self, raw_value: Any, *, is_right: bool) -> None:
        """Send one gripper command according to cached mode config."""
        if self._use_target_command:
            handler_name = "right_gripper_handler" if is_right else "left_gripper_handler"
            side_label = "Right" if is_right else "Left"
            handler = getattr(self.ros2_interface, handler_name, None)
            if handler is None:
                logger.warning(f"{side_label} gripper handler not initialized")
                return
            handler.send_target_command(self._to_target_command(raw_value))
            return

        if is_right:
            self.ros2_interface.send_right_gripper_command(raw_value)
        else:
            self.ros2_interface.send_gripper_command(raw_value)

    def _refresh_joint_index_cache(self, joint_names: Sequence[str]) -> None:
        """Build joint-name indices once and refresh only when names change."""
        signature = tuple(joint_names)
        if signature == self._joint_names_signature:
            return

        self._joint_names_signature = signature
        self._joint_name_to_index = {name: idx for idx, name in enumerate(joint_names)}

    @staticmethod
    def _add_ee_features(features: dict[str, type | tuple], prefix: str) -> None:
        features[f"{prefix}.pos.x"] = float
        features[f"{prefix}.pos.y"] = float
        features[f"{prefix}.pos.z"] = float
        features[f"{prefix}.quat.x"] = float
        features[f"{prefix}.quat.y"] = float
        features[f"{prefix}.quat.z"] = float
        features[f"{prefix}.quat.w"] = float
    
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
        
        # End-effector pose
        self._add_ee_features(features, self.LEFT_EE_PREFIX)
        if self._is_dual_arm_enabled():
            self._add_ee_features(features, self.RIGHT_EE_PREFIX)
        
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
            f"{self.LEFT_EE_PREFIX}.pos.x": float,
            f"{self.LEFT_EE_PREFIX}.pos.y": float,
            f"{self.LEFT_EE_PREFIX}.pos.z": float,
            f"{self.LEFT_EE_PREFIX}.quat.x": float,
            f"{self.LEFT_EE_PREFIX}.quat.y": float,
            f"{self.LEFT_EE_PREFIX}.quat.z": float,
            f"{self.LEFT_EE_PREFIX}.quat.w": float,
        }
        if self._is_dual_arm_enabled():
            features.update({
                f"{self.RIGHT_EE_PREFIX}.pos.x": float,
                f"{self.RIGHT_EE_PREFIX}.pos.y": float,
                f"{self.RIGHT_EE_PREFIX}.pos.z": float,
                f"{self.RIGHT_EE_PREFIX}.quat.x": float,
                f"{self.RIGHT_EE_PREFIX}.quat.y": float,
                f"{self.RIGHT_EE_PREFIX}.quat.z": float,
                f"{self.RIGHT_EE_PREFIX}.quat.w": float,
            })
        
        # Add gripper control feature if enabled
        if self.config.ros2_interface.gripper_enabled:
            features[self.LEFT_GRIPPER_KEY] = float
            if self._is_dual_arm_enabled():
                features[self.RIGHT_GRIPPER_KEY] = float
        
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
        # Refresh feature schemas after topic auto-discovery updates config.
        self.__dict__.pop("observation_features", None)
        self.__dict__.pop("action_features", None)
        self._refresh_gripper_command_config()
        
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
        else:
            # Extract joint positions, velocities, and efforts
            joint_names = joint_state["names"]
            joint_positions = joint_state["positions"]
            joint_velocities = joint_state["velocities"]
            joint_efforts = joint_state["efforts"]
            self._refresh_joint_index_cache(joint_names)
            
            for joint_name in self.config.ros2_interface.joint_names:
                idx = self._joint_name_to_index.get(joint_name)
                if idx is not None:
                    obs_dict[f"{joint_name}.pos"] = joint_positions[idx]
                    obs_dict[f"{joint_name}.vel"] = joint_velocities[idx]
                    obs_dict[f"{joint_name}.effort"] = joint_efforts[idx]
                else:
                    logger.warning(f"Joint '{joint_name}' not found in joint state")
                    obs_dict[f"{joint_name}.pos"] = 0.0
                    obs_dict[f"{joint_name}.vel"] = 0.0
                    obs_dict[f"{joint_name}.effort"] = 0.0
        
        # Get end-effector pose (left)
        end_effector_pose = self.ros2_interface.get_end_effector_pose()
        if end_effector_pose is not None:
            obs_dict[f"{self.LEFT_EE_PREFIX}.pos.x"] = end_effector_pose.position.x
            obs_dict[f"{self.LEFT_EE_PREFIX}.pos.y"] = end_effector_pose.position.y
            obs_dict[f"{self.LEFT_EE_PREFIX}.pos.z"] = end_effector_pose.position.z
            obs_dict[f"{self.LEFT_EE_PREFIX}.quat.x"] = end_effector_pose.orientation.x
            obs_dict[f"{self.LEFT_EE_PREFIX}.quat.y"] = end_effector_pose.orientation.y
            obs_dict[f"{self.LEFT_EE_PREFIX}.quat.z"] = end_effector_pose.orientation.z
            obs_dict[f"{self.LEFT_EE_PREFIX}.quat.w"] = end_effector_pose.orientation.w
        else:
            # Set default values if pose not available
            obs_dict[f"{self.LEFT_EE_PREFIX}.pos.x"] = 0.0
            obs_dict[f"{self.LEFT_EE_PREFIX}.pos.y"] = 0.0
            obs_dict[f"{self.LEFT_EE_PREFIX}.pos.z"] = 0.0
            obs_dict[f"{self.LEFT_EE_PREFIX}.quat.x"] = 0.0
            obs_dict[f"{self.LEFT_EE_PREFIX}.quat.y"] = 0.0
            obs_dict[f"{self.LEFT_EE_PREFIX}.quat.z"] = 0.0
            obs_dict[f"{self.LEFT_EE_PREFIX}.quat.w"] = 1.0

        # Right end-effector pose (dual-arm mode)
        if self._is_dual_arm_enabled():
            right_pose = self.ros2_interface.get_right_end_effector_pose()
            if right_pose is not None:
                obs_dict[f"{self.RIGHT_EE_PREFIX}.pos.x"] = right_pose.position.x
                obs_dict[f"{self.RIGHT_EE_PREFIX}.pos.y"] = right_pose.position.y
                obs_dict[f"{self.RIGHT_EE_PREFIX}.pos.z"] = right_pose.position.z
                obs_dict[f"{self.RIGHT_EE_PREFIX}.quat.x"] = right_pose.orientation.x
                obs_dict[f"{self.RIGHT_EE_PREFIX}.quat.y"] = right_pose.orientation.y
                obs_dict[f"{self.RIGHT_EE_PREFIX}.quat.z"] = right_pose.orientation.z
                obs_dict[f"{self.RIGHT_EE_PREFIX}.quat.w"] = right_pose.orientation.w
            else:
                obs_dict[f"{self.RIGHT_EE_PREFIX}.pos.x"] = 0.0
                obs_dict[f"{self.RIGHT_EE_PREFIX}.pos.y"] = 0.0
                obs_dict[f"{self.RIGHT_EE_PREFIX}.pos.z"] = 0.0
                obs_dict[f"{self.RIGHT_EE_PREFIX}.quat.x"] = 0.0
                obs_dict[f"{self.RIGHT_EE_PREFIX}.quat.y"] = 0.0
                obs_dict[f"{self.RIGHT_EE_PREFIX}.quat.z"] = 0.0
                obs_dict[f"{self.RIGHT_EE_PREFIX}.quat.w"] = 1.0
        
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
        
        has_left_ee = any(k.startswith(f"{self.LEFT_EE_PREFIX}.") for k in action)
        has_right_ee = any(k.startswith(f"{self.RIGHT_EE_PREFIX}.") for k in action)

        # Send left arm pose only when left ee keys are provided.
        if has_left_ee:
            left_pose = Pose()
            left_pose.position.x = action.get(f"{self.LEFT_EE_PREFIX}.pos.x", 0.0)
            left_pose.position.y = action.get(f"{self.LEFT_EE_PREFIX}.pos.y", 0.0)
            left_pose.position.z = action.get(f"{self.LEFT_EE_PREFIX}.pos.z", 0.0)
            left_pose.orientation.x = action.get(f"{self.LEFT_EE_PREFIX}.quat.x", 0.0)
            left_pose.orientation.y = action.get(f"{self.LEFT_EE_PREFIX}.quat.y", 0.0)
            left_pose.orientation.z = action.get(f"{self.LEFT_EE_PREFIX}.quat.z", 0.0)
            left_pose.orientation.w = action.get(f"{self.LEFT_EE_PREFIX}.quat.w", 1.0)
            self.ros2_interface.send_end_effector_target(left_pose)

        # Create and send right arm pose (if dual-arm keys provided)
        if self._is_dual_arm_enabled() and has_right_ee:
            right_pose = Pose()
            right_pose.position.x = action.get(f"{self.RIGHT_EE_PREFIX}.pos.x", 0.0)
            right_pose.position.y = action.get(f"{self.RIGHT_EE_PREFIX}.pos.y", 0.0)
            right_pose.position.z = action.get(f"{self.RIGHT_EE_PREFIX}.pos.z", 0.0)
            right_pose.orientation.x = action.get(f"{self.RIGHT_EE_PREFIX}.quat.x", 0.0)
            right_pose.orientation.y = action.get(f"{self.RIGHT_EE_PREFIX}.quat.y", 0.0)
            right_pose.orientation.z = action.get(f"{self.RIGHT_EE_PREFIX}.quat.z", 0.0)
            right_pose.orientation.w = action.get(f"{self.RIGHT_EE_PREFIX}.quat.w", 1.0)
            self.ros2_interface.send_right_end_effector_target(right_pose)
        
        # Send gripper command(s) if provided and gripper is enabled
        if self.config.ros2_interface.gripper_enabled:
            if self.LEFT_GRIPPER_KEY in action:
                left_gripper_position = action[self.LEFT_GRIPPER_KEY]
                self._send_gripper_value(left_gripper_position, is_right=False)
                logger.debug(f"Sent left gripper command: {left_gripper_position}")
            if self._is_dual_arm_enabled() and self.RIGHT_GRIPPER_KEY in action:
                right_gripper_position = action[self.RIGHT_GRIPPER_KEY]
                self._send_gripper_value(right_gripper_position, is_right=True)
                logger.debug(f"Sent right gripper command: {right_gripper_position}")
        
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
