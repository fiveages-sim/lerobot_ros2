"""
ROS 2 Robot Interface

Interface class for communicating with ROS 2 robots through topics.
"""

import logging
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import rclpy
from geometry_msgs.msg import Pose, PoseStamped, Twist
from lerobot.errors import DeviceNotConnectedError
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.publisher import Publisher
from rclpy.subscription import Subscription
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64

from .config import ControlType, ROS2RobotInterfaceConfig

logger = logging.getLogger(__name__)


class ROS2RobotInterface:
    """Interface for communicating with ROS 2 robots.
    
    This class handles:
    - Subscribing to joint states from /joint_states topic
    - Subscribing to current end-effector pose from /left_current_pose topic
    - Publishing target end-effector pose to /left_target topic
    """
    
    def __init__(self, config: ROS2RobotInterfaceConfig):
        """Initialize the ROS 2 robot interface.
        
        Args:
            config: ROS 2 robot interface configuration
        """
        self.config = config
        self.robot_node: Node | None = None
        self.executor: SingleThreadedExecutor | None = None
        self.executor_thread: threading.Thread | None = None
        
        # Subscriptions
        self.joint_state_sub: Subscription | None = None
        self.end_effector_pose_sub: Subscription | None = None
        
        # Publishers
        self.end_effector_target_pub: Publisher | None = None
        self.gripper_command_pub: Publisher | None = None
        
        # Data storage
        self.latest_joint_state: Dict[str, Any] | None = None
        self.latest_end_effector_pose: Pose | None = None
        
        # Thread safety
        self.data_lock = threading.Lock()
        
        # Connection state
        self._connected = False
        
        # Timing
        self.last_joint_state_time = 0.0
        self.last_end_effector_pose_time = 0.0
        
        # Track if we previously had no data (for recovery logging)
        self._had_joint_state = False
        self._had_end_effector_pose = False
    
    @property
    def is_connected(self) -> bool:
        """Check if the interface is connected."""
        return self._connected and self.robot_node is not None
    
    def connect(self) -> None:
        """Connect to ROS 2 and create subscriptions/publishers."""
        if self.is_connected:
            raise DeviceAlreadyConnectedError("ROS2RobotInterface already connected")
        
        try:
            # Initialize ROS 2 if not already done
            if not rclpy.ok():
                rclpy.init()
            
            # Create ROS 2 node
            self.robot_node = Node(
                "lerobot_ros2_robot_interface",
                namespace=self.config.namespace if hasattr(self.config, 'namespace') else ""
            )
            
            # Create joint state subscription
            self.joint_state_sub = self.robot_node.create_subscription(
                JointState,
                self.config.joint_states_topic,
                self._joint_state_callback,
                10
            )
            
            # Create end-effector pose subscription
            self.end_effector_pose_sub = self.robot_node.create_subscription(
                PoseStamped,
                self.config.end_effector_pose_topic,
                self._end_effector_pose_callback,
                10
            )
            
            # Create end-effector target publisher
            self.end_effector_target_pub = self.robot_node.create_publisher(
                Pose,
                self.config.end_effector_target_topic,
                10
            )
            
            # Create gripper command publisher (if gripper is enabled)
            if self.config.gripper_enabled and self.config.gripper_command_topic:
                self.gripper_command_pub = self.robot_node.create_publisher(
                    Float64,
                    self.config.gripper_command_topic,
                    10
                )
                logger.info(f"Created gripper command publisher on topic: {self.config.gripper_command_topic}")
            
            # Start executor in separate thread
            self.executor = SingleThreadedExecutor()
            self.executor.add_node(self.robot_node)
            self.executor_thread = threading.Thread(
                target=self.executor.spin,
                daemon=True
            )
            self.executor_thread.start()
            
            # Wait for connection to establish
            time.sleep(1.0)
            
            self._connected = True
            logger.info("Connected to ROS 2 robot interface")
            
        except Exception as e:
            logger.error(f"Failed to connect to ROS 2 robot interface: {e}")
            self.disconnect()
            raise
    
    def _joint_state_callback(self, msg: JointState) -> None:
        """Callback for joint state messages."""
        with self.data_lock:
            # Check if we're recovering from no data state or stale data
            # Recovery happens if:
            # 1. We never had data before (first time receiving), OR
            # 2. Data was missing/None, OR  
            # 3. If timeout > 0, data was stale (older than timeout)
            current_time = time.time()
            was_recovering = False
            
            if not self._had_joint_state:
                # First time receiving data
                was_recovering = True
            elif self.latest_joint_state is None:
                # Data was lost/missing, now recovering
                was_recovering = True
            elif self.config.joint_state_timeout > 0:
                # Check if previous data was stale (only if timeout is enabled)
                time_since_last = current_time - self.last_joint_state_time
                if time_since_last > self.config.joint_state_timeout:
                    was_recovering = True
            
            self.latest_joint_state = {
                "names": list(msg.name),
                "positions": list(msg.position),
                "velocities": list(msg.velocity),
                "efforts": list(msg.effort),
                "timestamp": current_time
            }
            self.last_joint_state_time = current_time
            self._had_joint_state = True

            
            # Log recovery if we just started receiving data again
            if was_recovering:
                logger.info("Joint state data recovery: Started receiving joint state messages again")
    
    def _end_effector_pose_callback(self, msg: PoseStamped) -> None:
        """Callback for end-effector pose messages."""
        with self.data_lock:
            # Check if we're recovering from no data state or stale data
            # Recovery happens if:
            # 1. We never had data before (first time receiving), OR
            # 2. Data was missing/None, OR
            # 3. If timeout > 0, data was stale (older than timeout)
            current_time = time.time()
            was_recovering = False
            
            if not self._had_end_effector_pose:
                # First time receiving data
                was_recovering = True
            elif self.latest_end_effector_pose is None:
                # Data was lost/missing, now recovering
                was_recovering = True
            elif self.config.end_effector_pose_timeout > 0:
                # Check if previous data was stale (only if timeout is enabled)
                time_since_last = current_time - self.last_end_effector_pose_time
                if time_since_last > self.config.end_effector_pose_timeout:
                    was_recovering = True
            
            self.latest_end_effector_pose = msg.pose
            self.last_end_effector_pose_time = current_time
            self._had_end_effector_pose = True
            
            # Log recovery if we just started receiving data again
            if was_recovering:
                logger.info("End-effector pose data recovery: Started receiving end-effector pose messages again")
    
    def get_joint_state(self) -> Dict[str, Any] | None:
        """Get the latest joint state.
        
        Returns:
            Dict containing joint names, positions, velocities, efforts, and timestamp,
            or None if no joint state has been received yet.
            When timeout is 0, returns the last received state even if stale.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError("ROS2RobotInterface is not connected")
        
        with self.data_lock:
            # Check if joint state is recent enough (skip check if timeout is 0)
            if self.config.joint_state_timeout > 0:
                if (time.time() - self.last_joint_state_time) > self.config.joint_state_timeout:
                    logger.warning("Joint state data is stale")
                    return None
            
            # When timeout is 0, always return last received state if available
            # This allows the system to continue working when ROS2 nodes restart
            return self.latest_joint_state.copy() if self.latest_joint_state else None
    
    def get_end_effector_pose(self) -> Pose | None:
        """Get the latest end-effector pose.
        
        Returns:
            Pose message containing position and orientation,
            or None if no end-effector pose has been received yet.
            When timeout is 0, returns the last received pose even if stale.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError("ROS2RobotInterface is not connected")
        
        with self.data_lock:
            # Check if end-effector pose is recent enough (skip check if timeout is 0)
            if self.config.end_effector_pose_timeout > 0:
                if (time.time() - self.last_end_effector_pose_time) > self.config.end_effector_pose_timeout:
                    logger.warning("End-effector pose data is stale")
                    return None
            
            # When timeout is 0, always return last received pose if available
            # This allows the system to continue working when ROS2 nodes restart
            return self.latest_end_effector_pose
    
    def send_end_effector_target(self, pose: Pose) -> None:
        """Send target end-effector pose.
        
        Args:
            pose: Target pose for the end-effector
        """
        if not self.is_connected:
            raise DeviceNotConnectedError("ROS2RobotInterface is not connected")
        
        if self.end_effector_target_pub is None:
            raise DeviceNotConnectedError("End-effector target publisher not initialized")
        
        # Publish the target pose directly
        self.end_effector_target_pub.publish(pose)
        logger.debug(f"Published end-effector target: {pose}")
    
    def send_gripper_command(self, position: float) -> None:
        """Send gripper position command.
        
        Args:
            position: Target gripper position (typically 0.0 to 1.0)
        """
        if not self.is_connected:
            raise DeviceNotConnectedError("ROS2RobotInterface is not connected")
        
        if not self.config.gripper_enabled:
            logger.warning("Gripper is not enabled in configuration")
            return
        
        if self.gripper_command_pub is None:
            logger.warning("Gripper command publisher not initialized")
            return
        
        # Clamp position to configured limits
        clamped_position = max(
            self.config.gripper_min_position,
            min(position, self.config.gripper_max_position)
        )
        
        # Create and publish Float64 message
        gripper_msg = Float64()
        gripper_msg.data = clamped_position
        
        self.gripper_command_pub.publish(gripper_msg)
        logger.debug(f"Published gripper command: {clamped_position}")
    
    def send_cartesian_velocity(self, linear: Tuple[float, float, float], angular: Tuple[float, float, float]) -> None:
        """Send cartesian velocity commands.
        
        Args:
            linear: Linear velocity (x, y, z) in m/s
            angular: Angular velocity (rx, ry, rz) in rad/s
        """
        if not self.is_connected:
            raise DeviceNotConnectedError("ROS2RobotInterface is not connected")
        
        # For cartesian velocity control, we would need a different topic
        # This is a placeholder implementation
        logger.warning("Cartesian velocity control not implemented yet")
    
    def disconnect(self) -> None:
        """Disconnect from ROS 2 and cleanup resources."""
        self._connected = False
        
        # Stop executor
        if self.executor:
            self.executor.shutdown()
            self.executor = None
        
        # Wait for thread to finish
        if self.executor_thread:
            self.executor_thread.join(timeout=2.0)
            self.executor_thread = None
        
        # Destroy subscriptions and publishers
        if self.joint_state_sub:
            self.joint_state_sub.destroy()
            self.joint_state_sub = None
        
        if self.end_effector_pose_sub:
            self.end_effector_pose_sub.destroy()
            self.end_effector_pose_sub = None
        
        if self.end_effector_target_pub:
            self.end_effector_target_pub.destroy()
            self.end_effector_target_pub = None
        
        if self.gripper_command_pub:
            self.gripper_command_pub.destroy()
            self.gripper_command_pub = None
        
        # Destroy node
        if self.robot_node:
            self.robot_node.destroy_node()
            self.robot_node = None
        
        # Clear data
        with self.data_lock:
            self.latest_joint_state = None
            self.latest_end_effector_pose = None
        
        # Shutdown rclpy
        try:
            rclpy.shutdown()
        except Exception as e:
            logger.warning(f"Error during rclpy shutdown: {e}")
        
        logger.info("Disconnected from ROS 2 robot interface")
