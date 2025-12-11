"""
ROS 2 Robot Interface

Interface class for communicating with ROS 2 robots through topics.
This is a standalone implementation independent of LeRobot.
"""

import logging
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import rclpy
from geometry_msgs.msg import Pose, PoseStamped, Twist
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.publisher import Publisher
from rclpy.subscription import Subscription
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64, Float64MultiArray, Int32
from .config import ControlType, ROS2RobotInterfaceConfig
from .exceptions import ROS2AlreadyConnectedError, ROS2NotConnectedError

logger = logging.getLogger(__name__)


class ROS2RobotInterface:
    """Interface for communicating with ROS 2 robots.
    
    This class handles:
    - Subscribing to joint states from /joint_states topic
    - Subscribing to current end-effector pose from /left_current_pose topic
    - Publishing target end-effector pose to /left_target topic
    
    This is a standalone implementation that does not depend on LeRobot.
    It can be used in any ROS 2 environment.
    
    Example:
        ```python
        from ros2_robot_interface import ROS2RobotInterface, ROS2RobotInterfaceConfig
        
        config = ROS2RobotInterfaceConfig(
            joint_states_topic="/joint_states",
            end_effector_pose_topic="/left_current_pose",
            end_effector_target_topic="/left_target"
        )
        
        interface = ROS2RobotInterface(config)
        interface.connect()
        
        # Get joint state
        joint_state = interface.get_joint_state()
        
        # Send target pose
        from geometry_msgs.msg import Pose
        pose = Pose()
        # ... set pose values ...
        interface.send_end_effector_target(pose)
        
        interface.disconnect()
        ```
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
        self.right_end_effector_pose_sub: Subscription | None = None  # For dual-arm
        
        # Publishers
        self.end_effector_target_pub: Publisher | None = None
        self.right_end_effector_target_pub: Publisher | None = None  # For dual-arm
        self.gripper_command_pub: Publisher | None = None
        self.right_gripper_command_pub: Publisher | None = None  # For dual-arm
        self.fsm_command_pub: Publisher | None = None  # For FSM state switching
        self.head_joint_controller_pub: Publisher | None = None  # For head joint control
        self.body_joint_controller_pub: Publisher | None = None  # For body joint control
        
        # Data storage
        self.latest_joint_state: Dict[str, Any] | None = None
        self.latest_end_effector_pose: Pose | None = None
        self.latest_right_end_effector_pose: Pose | None = None  # For dual-arm
        
        # Thread safety
        self.data_lock = threading.Lock()
        
        # Connection state
        self._connected = False
        
        # Timing
        self.last_joint_state_time = 0.0
        self.last_end_effector_pose_time = 0.0
        self.last_right_end_effector_pose_time = 0.0  # For dual-arm
        
        # Track if we previously had no data (for recovery logging)
        self._had_joint_state = False
        self._had_end_effector_pose = False
        self._had_right_end_effector_pose = False  # For dual-arm
        
        # Target positions for head and body joints (automatically set when sending commands)
        self.head_target_positions: Optional[List[float]] = None
        self.body_target_positions: Optional[List[float]] = None
        self.position_threshold: float = 0.05  # Default threshold in radians (≈2.87 degrees)
    
    @property
    def is_connected(self) -> bool:
        """Check if the interface is connected."""
        return self._connected and self.robot_node is not None
    
    def connect(self) -> None:
        """Connect to ROS 2 and create subscriptions/publishers."""
        if self.is_connected:
            raise ROS2AlreadyConnectedError("ROS2RobotInterface already connected")
        
        try:
            # Initialize ROS 2 if not already done
            if not rclpy.ok():
                rclpy.init()
            
            # Auto-detect dual-arm mode by checking if right arm topics exist in ROS 2
            # Check for right_target or right_current_pose topics to determine dual-arm mode
            # We need to create a temporary node first to check topics, then create the actual node with correct name
            is_dual_arm_detected = False
            
            # Auto-detect by checking ROS 2 topics
            try:
                # Create temporary node for topic detection
                temp_node = Node(
                    "ros2_robot_interface_temp",
                    namespace=self.config.namespace if self.config.namespace else ""
                )
                
                # Create executor to allow discovery
                temp_executor = SingleThreadedExecutor()
                temp_executor.add_node(temp_node)
                
                # Wait for topics to be discovered (ROS 2 discovery takes time)
                # Use a more robust discovery mechanism: wait until topic count stabilizes
                max_attempts = 30
                topic_names = []
                stable_count = 0
                last_count = 0
                
                print("[DEBUG] Starting topic discovery...")
                for attempt in range(max_attempts):
                    # Spin multiple times to process discovery messages more aggressively
                    for _ in range(5):
                        temp_executor.spin_once(timeout_sec=0.05)
                    time.sleep(0.3)  # Wait 300ms between attempts
                    
                    # Get topics
                    topic_names_and_types = temp_node.get_topic_names_and_types()
                    topic_names = [name for name, _ in topic_names_and_types]
                    current_count = len(topic_names)
                    
                    # Check if topic count is stable (same count for 3 consecutive attempts)
                    if current_count == last_count and current_count > 2:
                        stable_count += 1
                        if stable_count >= 3:
                            print(f"[DEBUG] Topic discovery stabilized after {attempt + 1} attempts, found {current_count} topics")
                            break
                    else:
                        stable_count = 0
                    
                    last_count = current_count
                    
                    # Progress indicator
                    if attempt % 5 == 0:
                        print(f"[DEBUG] Discovery attempt {attempt + 1}/{max_attempts}, found {current_count} topics...")
                
                if stable_count < 3:
                    print(f"[DEBUG] Topic discovery completed (may be incomplete), found {len(topic_names)} topics after {max_attempts} attempts")
                
                # Cleanup
                temp_executor.shutdown()
                temp_node.destroy_node()
                
                # Debug: Print all discovered topics
                print(f"\n[DEBUG] Discovered {len(topic_names)} ROS 2 topics:")
                print("-" * 70)
                # Print topics in columns for better readability
                for i, topic_name in enumerate(sorted(topic_names), 1):
                    print(f"  {i:3d}. {topic_name}")
                print("-" * 70)
                
                # Filter topics containing "right" for easier inspection
                right_related = [t for t in topic_names if "right" in t.lower()]
                if right_related:
                    print(f"\n[DEBUG] Topics containing 'right' ({len(right_related)}):")
                    for topic in sorted(right_related):
                        print(f"  - {topic}")
                else:
                    print("\n[DEBUG] No topics containing 'right' found!")
                print()
                
                # Check for right arm topics (look for topics containing "right" and "target" or "current_pose")
                right_target_topic = "/right_target"
                right_pose_topic = "/right_current_pose"
                
                # Check if right arm topics exist
                if right_target_topic in topic_names or right_pose_topic in topic_names:
                    is_dual_arm_detected = True
                    # Auto-configure right arm topics
                    if right_pose_topic in topic_names:
                        self.config.right_end_effector_pose_topic = right_pose_topic
                    if right_target_topic in topic_names:
                        self.config.right_end_effector_target_topic = right_target_topic
                    
                    # Auto-configure right gripper topic (if exists)
                    # In dual-arm mode, right gripper uses standard naming with leading slash
                    right_gripper_topic = "/right_gripper_joint/position_command"
                    if right_gripper_topic in topic_names:
                        self.config.right_gripper_command_topic = right_gripper_topic
                        self.config.gripper_command_topic = "/left_gripper_joint/position_command"
                    
                    print(f"Auto-detected dual-arm mode from ROS 2 topics")
                    print(f"  - Right pose topic: {self.config.right_end_effector_pose_topic}")
                    print(f"  - Right target topic: {self.config.right_end_effector_target_topic}")
                    if self.config.gripper_command_topic == "/left_gripper_joint/position_command":
                        print(f"  - Left gripper topic: {self.config.gripper_command_topic}")
                    else:
                        print(f"  - Left gripper topic not found")
                    
                    if self.config.right_gripper_command_topic:
                        print(f"  - Right gripper topic: {self.config.right_gripper_command_topic}")
                    else:
                        print(f"  - Right gripper topic not found")
                else:
                    print("Single-arm mode detected (no right arm topics found)")
                
                # Auto-detect head and body joint controller topics
                head_joint_controller_topic = "/head_joint_controller/target_joint_position"
                body_joint_controller_topic = "/body_joint_controller/target_joint_position"
                
                if head_joint_controller_topic in topic_names:
                    self.config.head_joint_controller_topic = head_joint_controller_topic
                    print(f"Auto-detected head joint controller topic: {head_joint_controller_topic}")
                else:
                    print(f"Head joint controller topic not found: {head_joint_controller_topic}")
                
                if body_joint_controller_topic in topic_names:
                    self.config.body_joint_controller_topic = body_joint_controller_topic
                    print(f"Auto-detected body joint controller topic: {body_joint_controller_topic}")
                else:
                    print(f"Body joint controller topic not found: {body_joint_controller_topic}")
            except Exception as e:
                logger.warning(f"Failed to auto-detect dual-arm mode: {e}")
                is_dual_arm_detected = False
            
            # Output detection result
            print("\n" + "=" * 60, flush=True)
            if is_dual_arm_detected:
                print("✓ DUAL-ARM MODE DETECTED", flush=True)
            else:
                print("✓ SINGLE-ARM MODE DETECTED", flush=True)
            print("=" * 60 + "\n", flush=True)
            
            # Determine node name based on detection result or configuration
            if self.config.node_name:
                node_name = self.config.node_name
            elif is_dual_arm_detected:
                node_name = "ros2_robot_interface_dual_arm"
            else:
                node_name = "ros2_robot_interface"
            
            # Create ROS 2 node with correct name
            self.robot_node = Node(
                node_name,
                namespace=self.config.namespace if self.config.namespace else ""
            )
            
            # Create joint state subscription
            self.joint_state_sub = self.robot_node.create_subscription(
                JointState,
                self.config.joint_states_topic,
                self._joint_state_callback,
                10
            )
            
            # Create end-effector pose subscription (left arm)
            self.end_effector_pose_sub = self.robot_node.create_subscription(
                PoseStamped,
                self.config.end_effector_pose_topic,
                self._end_effector_pose_callback,
                10
            )
            
            # Create right arm end-effector pose subscription (if dual-arm mode)
            if self.config.right_end_effector_pose_topic:
                self.right_end_effector_pose_sub = self.robot_node.create_subscription(
                    PoseStamped,
                    self.config.right_end_effector_pose_topic,
                    self._right_end_effector_pose_callback,
                    10
                )
                logger.info(f"Created right arm pose subscription on topic: {self.config.right_end_effector_pose_topic}")
            
            # Create end-effector target publisher (left arm)
            self.end_effector_target_pub = self.robot_node.create_publisher(
                Pose,
                self.config.end_effector_target_topic,
                10
            )
            
            # Create right arm end-effector target publisher (if dual-arm mode)
            if self.config.right_end_effector_target_topic:
                self.right_end_effector_target_pub = self.robot_node.create_publisher(
                    Pose,
                    self.config.right_end_effector_target_topic,
                    10
                )
                logger.info(f"Created right arm target publisher on topic: {self.config.right_end_effector_target_topic}")
            
            # Create gripper command publisher (left arm, if gripper is enabled)
            if self.config.gripper_enabled and self.config.gripper_command_topic:
                self.gripper_command_pub = self.robot_node.create_publisher(
                    Float64,
                    self.config.gripper_command_topic,
                    10
                )
                logger.info(f"Created gripper command publisher on topic: {self.config.gripper_command_topic}")
            
            # Create right gripper command publisher (if dual-arm mode)
            if self.config.right_gripper_command_topic:
                self.right_gripper_command_pub = self.robot_node.create_publisher(
                    Float64,
                    self.config.right_gripper_command_topic,
                    10
                )
                logger.info(f"Created right gripper command publisher on topic: {self.config.right_gripper_command_topic}")
            
            # Create FSM command publisher (for state switching)
            self.fsm_command_pub = self.robot_node.create_publisher(
                Int32,
                "/fsm_command",
                10
            )
            logger.info("Created FSM command publisher on topic: /fsm_command")
            
            # Create head joint controller publisher (if configured)
            if self.config.head_joint_controller_topic:
                self.head_joint_controller_pub = self.robot_node.create_publisher(
                    Float64MultiArray,
                    self.config.head_joint_controller_topic,
                    10
                )
                logger.info(f"Created head joint controller publisher on topic: {self.config.head_joint_controller_topic}")
            
            # Create body joint controller publisher (if configured)
            if self.config.body_joint_controller_topic:
                self.body_joint_controller_pub = self.robot_node.create_publisher(
                    Float64MultiArray,
                    self.config.body_joint_controller_topic,
                    10
                )
                logger.info(f"Created body joint controller publisher on topic: {self.config.body_joint_controller_topic}")
            
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
    
    def _right_end_effector_pose_callback(self, msg: PoseStamped) -> None:
        """Callback for right end-effector pose messages (dual-arm mode)."""
        with self.data_lock:
            current_time = time.time()
            was_recovering = False
            
            if not self._had_right_end_effector_pose:
                was_recovering = True
            elif self.latest_right_end_effector_pose is None:
                was_recovering = True
            elif self.config.end_effector_pose_timeout > 0:
                time_since_last = current_time - self.last_right_end_effector_pose_time
                if time_since_last > self.config.end_effector_pose_timeout:
                    was_recovering = True
            
            self.latest_right_end_effector_pose = msg.pose
            self.last_right_end_effector_pose_time = current_time
            self._had_right_end_effector_pose = True
            
            if was_recovering:
                logger.info("Right end-effector pose data recovery: Started receiving right end-effector pose messages again")
    
    def _categorize_joints(self, joint_names: List[str], positions: List[float], 
                          velocities: List[float], efforts: List[float]) -> Dict[str, Dict[str, Any]]:
        """Categorize joints by body part (left arm, right arm, head, body).
        
        Uses the dual-arm detection result from connect() to determine categorization:
        - Dual-arm: left_ → left_arm, right_ → right_arm
        - Single-arm: joint1-6 → arm
        - head → head
        - body → body
        
        Args:
            joint_names: List of joint names
            positions: List of joint positions
            velocities: List of joint velocities
            efforts: List of joint efforts
            
        Returns:
            Dict with keys: 'arm'/'left_arm', 'right_arm', 'head', 'body', 'other',
            each containing: {'names': [...], 'positions': [...], 'velocities': [...], 'efforts': [...]}
        """
        # Determine if dual-arm mode based on config (right_end_effector_pose_topic is set if dual-arm)
        is_dual_arm = self.config.right_end_effector_pose_topic is not None
        
        # Initialize categories based on dual-arm mode
        if is_dual_arm:
            categories = {
                'left_arm': {'names': [], 'positions': [], 'velocities': [], 'efforts': []},
                'right_arm': {'names': [], 'positions': [], 'velocities': [], 'efforts': []},
                'head': {'names': [], 'positions': [], 'velocities': [], 'efforts': []},
                'body': {'names': [], 'positions': [], 'velocities': [], 'efforts': []},
                'other': {'names': [], 'positions': [], 'velocities': [], 'efforts': []}
            }
        else:
            categories = {
                'arm': {'names': [], 'positions': [], 'velocities': [], 'efforts': []},
                'head': {'names': [], 'positions': [], 'velocities': [], 'efforts': []},
                'body': {'names': [], 'positions': [], 'velocities': [], 'efforts': []},
                'other': {'names': [], 'positions': [], 'velocities': [], 'efforts': []}
            }
        
        for i, name in enumerate(joint_names):
            name_lower = name.lower()
            category = None
            
            # Categorize based on joint name prefix and dual-arm mode
            if is_dual_arm:
                # Dual-arm mode: use left_arm and right_arm
                if name_lower.startswith('left_'):
                    category = 'left_arm'
                elif name_lower.startswith('right_'):
                    category = 'right_arm'
                elif 'head' in name_lower:
                    category = 'head'
                elif 'body' in name_lower:
                    category = 'body'
                else:
                    category = 'other'
            else:
                # Single-arm mode: use arm
                if 'head' in name_lower:
                    category = 'head'
                elif 'body' in name_lower:
                    category = 'body'
                elif 'joint' in name_lower or 'gripper' in name_lower:
                    # All arm joints (including gripper) go to 'arm' in single-arm mode
                    category = 'arm'
                else:
                    category = 'other'
            
            categories[category]['names'].append(name)
            if i < len(positions):
                categories[category]['positions'].append(positions[i])
            if i < len(velocities):
                categories[category]['velocities'].append(velocities[i])
            if i < len(efforts):
                categories[category]['efforts'].append(efforts[i])
        
        return categories
    
    def get_joint_state(self, categorized: bool = False) -> Dict[str, Any] | None:
        """Get the latest joint state.
        
        Args:
            categorized: If True, returns joints categorized by body part.
                        If False, returns all joints together (default).
        
        Returns:
            If categorized=False:
                Dict containing joint names, positions, velocities, efforts, and timestamp,
                or None if no joint state has been received yet.
            If categorized=True:
                For single-arm robots:
                    Dict with keys: 'arm', 'head', 'body', 'other', 'timestamp'
                For dual-arm robots:
                    Dict with keys: 'left_arm', 'right_arm', 'head', 'body', 'other', 'timestamp'
                Each category contains: {'names': [...], 'positions': [...], 'velocities': [...], 'efforts': [...]}
            When timeout is 0, returns the last received state even if stale.
        """
        if not self.is_connected:
            raise ROS2NotConnectedError("ROS2RobotInterface is not connected")
        
        with self.data_lock:
            # Check if joint state is recent enough (skip check if timeout is 0)
            if self.config.joint_state_timeout > 0:
                if (time.time() - self.last_joint_state_time) > self.config.joint_state_timeout:
                    logger.warning("Joint state data is stale")
                    return None
            
            if self.latest_joint_state is None:
                return None
            
            # If not categorized, return original format
            if not categorized:
                return self.latest_joint_state.copy()
            
            # Categorize joints
            categories = self._categorize_joints(
                self.latest_joint_state['names'],
                self.latest_joint_state['positions'],
                self.latest_joint_state['velocities'],
                self.latest_joint_state.get('efforts', [])
            )
            
            # Add timestamp
            categories['timestamp'] = self.latest_joint_state.get('timestamp', 0.0)
            
            return categories
    
    def get_end_effector_pose(self) -> Pose | None:
        """Get the latest end-effector pose (left arm).
        
        Returns:
            Pose message containing position and orientation,
            or None if no end-effector pose has been received yet.
            When timeout is 0, returns the last received pose even if stale.
        """
        if not self.is_connected:
            raise ROS2NotConnectedError("ROS2RobotInterface is not connected")
        
        with self.data_lock:
            # Check if end-effector pose is recent enough (skip check if timeout is 0)
            if self.config.end_effector_pose_timeout > 0:
                if (time.time() - self.last_end_effector_pose_time) > self.config.end_effector_pose_timeout:
                    logger.warning("End-effector pose data is stale")
                    return None
            
            # When timeout is 0, always return last received pose if available
            # This allows the system to continue working when ROS2 nodes restart
            return self.latest_end_effector_pose
    
    def get_right_end_effector_pose(self) -> Pose | None:
        """Get the latest right end-effector pose (dual-arm mode).
        
        Returns:
            Pose message containing position and orientation,
            or None if no right end-effector pose has been received yet or dual-arm mode is not enabled.
            When timeout is 0, returns the last received pose even if stale.
        """
        if not self.is_connected:
            raise ROS2NotConnectedError("ROS2RobotInterface is not connected")
        
        if not self.config.right_end_effector_pose_topic:
            logger.warning("Right end-effector pose topic not configured. Dual-arm mode not enabled.")
            return None
        
        with self.data_lock:
            # Check if end-effector pose is recent enough (skip check if timeout is 0)
            if self.config.end_effector_pose_timeout > 0:
                if (time.time() - self.last_right_end_effector_pose_time) > self.config.end_effector_pose_timeout:
                    logger.warning("Right end-effector pose data is stale")
                    return None
            
            return self.latest_right_end_effector_pose
    
    def send_end_effector_target(self, pose: Pose) -> None:
        """Send target end-effector pose (left arm).
        
        Args:
            pose: Target pose for the end-effector
        """
        if not self.is_connected:
            raise ROS2NotConnectedError("ROS2RobotInterface is not connected")
        
        if self.end_effector_target_pub is None:
            raise ROS2NotConnectedError("End-effector target publisher not initialized")
        
        # Publish the target pose directly
        self.end_effector_target_pub.publish(pose)
        logger.debug(f"Published end-effector target: {pose}")
    
    def send_right_end_effector_target(self, pose: Pose) -> None:
        """Send target right end-effector pose (dual-arm mode).
        
        Args:
            pose: Target pose for the right end-effector
        """
        if not self.is_connected:
            raise ROS2NotConnectedError("ROS2RobotInterface is not connected")
        
        if not self.config.right_end_effector_target_topic:
            raise ROS2NotConnectedError("Right end-effector target topic not configured. Dual-arm mode not enabled.")
        
        if self.right_end_effector_target_pub is None:
            raise ROS2NotConnectedError("Right end-effector target publisher not initialized")
        
        # Publish the target pose directly
        self.right_end_effector_target_pub.publish(pose)
        logger.debug(f"Published right end-effector target: {pose}")
    
    def send_gripper_command(self, position: float) -> None:
        """Send gripper position command.
        
        Args:
            position: Target gripper position (typically 0.0 to 1.0)
        """
        if not self.is_connected:
            raise ROS2NotConnectedError("ROS2RobotInterface is not connected")
        
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
    
    def send_right_gripper_command(self, position: float) -> None:
        """Send right gripper position command (dual-arm mode).
        
        Args:
            position: Target gripper position (typically 0.0 to 1.0)
        """
        if not self.is_connected:
            raise ROS2NotConnectedError("ROS2RobotInterface is not connected")
        
        if not self.config.right_gripper_command_topic:
            logger.warning("Right gripper command topic not configured. Dual-arm mode not enabled.")
            return
        
        if self.right_gripper_command_pub is None:
            logger.warning("Right gripper command publisher not initialized")
            return
        
        # Clamp position to configured limits
        clamped_position = max(
            self.config.gripper_min_position,
            min(position, self.config.gripper_max_position)
        )
        
        # Create and publish Float64 message
        gripper_msg = Float64()
        gripper_msg.data = clamped_position
        
        self.right_gripper_command_pub.publish(gripper_msg)
        logger.debug(f"Published right gripper command: {clamped_position}")
    
    def send_fsm_command(self, command: int) -> None:
        """Send FSM command for state switching.
        
        FSM command values:
        - 1: Switch to HOME state
        - 2: Switch to HOLD state
        - 3: Switch to OCS2/MOVE state
        - 4: Switch to MOVEJ state (from HOLD) or switch pose (in HOME)
        - 100: Switch pose (in HOME state)
        - 0: Reset
        
        Args:
            command: FSM command value (0, 1, 2, 3, 4, or 100)
        """
        if not self.is_connected:
            raise ROS2NotConnectedError("ROS2RobotInterface is not connected")
        
        if self.fsm_command_pub is None:
            logger.warning("FSM command publisher not initialized")
            return
        
        # Create and publish Int32 message
        fsm_msg = Int32()
        fsm_msg.data = command
        self.fsm_command_pub.publish(fsm_msg)
        logger.info(f"Published FSM command: {command}")
    
    def send_head_joint_positions(self, positions: List[float]) -> None:
        """Send target joint positions for head joints.
        
        Args:
            positions: List of target joint positions (radians) in the same order as configured joints
        """
        if not self.is_connected:
            raise ROS2NotConnectedError("ROS2RobotInterface is not connected")
        
        if self.head_joint_controller_pub is None:
            logger.warning("Head joint controller publisher not initialized. Set head_joint_controller_topic in config.")
            return
        
        # Create and publish Float64MultiArray message
        msg = Float64MultiArray()
        msg.data = positions
        self.head_joint_controller_pub.publish(msg)
        logger.debug(f"Published head joint positions: {positions}")
        
        # Automatically record target positions for arrival checking
        self.head_target_positions = positions.copy() if positions else None
    
    def send_body_joint_positions(self, positions: List[float]) -> None:
        """Send target joint positions for body joints.
        
        Args:
            positions: List of target joint positions (radians) in the same order as configured joints
        """
        if not self.is_connected:
            raise ROS2NotConnectedError("ROS2RobotInterface is not connected")
        
        if self.body_joint_controller_pub is None:
            logger.warning("Body joint controller publisher not initialized. Set body_joint_controller_topic in config.")
            return
        
        # Create and publish Float64MultiArray message
        msg = Float64MultiArray()
        msg.data = positions
        self.body_joint_controller_pub.publish(msg)
        logger.debug(f"Published body joint positions: {positions}")
        
        # Automatically record target positions for arrival checking
        self.body_target_positions = positions.copy() if positions else None
    
    def check_arrive(self, part: Optional[str] = None, position_threshold: Optional[float] = None) -> Dict[str, Any]:
        """Check if head or body joints have arrived at target positions.
        
        Args:
            part: 'head' or 'body'. If None, returns results for both.
            position_threshold: Distance threshold in radians. If None, uses self.position_threshold.
        
        Returns:
            If part is specified:
                {'arrived': bool, 'distance': float}
            If part is None:
                {'head': {'arrived': bool, 'distance': float}, 
                 'body': {'arrived': bool, 'distance': float}}
        """
        if not self.is_connected:
            raise ROS2NotConnectedError("ROS2RobotInterface is not connected")
        
        threshold = position_threshold if position_threshold is not None else self.position_threshold
        result = {}
        
        # Get current joint state (categorized)
        categorized_state = self.get_joint_state(categorized=True)
        if not categorized_state:
            if part:
                return {'arrived': False, 'distance': float('inf')}
            else:
                return {'head': {'arrived': False, 'distance': float('inf')}, 
                       'body': {'arrived': False, 'distance': float('inf')}}
        
        # Calculate Euclidean distance
        def calculate_distance(current: List[float], target: List[float]) -> float:
            """Calculate Euclidean distance between two position vectors."""
            if not current or not target:
                return float('inf')
            if len(current) != len(target):
                return float('inf')
            return sum((c - t) ** 2 for c, t in zip(current, target)) ** 0.5
        
        # Check head arrival
        if part is None or part == 'head':
            head_arrived = False
            head_distance = float('inf')
            if self.head_target_positions is not None and 'head' in categorized_state:
                current_head = categorized_state['head']['positions']
                if len(current_head) == len(self.head_target_positions):
                    head_distance = calculate_distance(current_head, self.head_target_positions)
                    head_arrived = head_distance < threshold
                    
                    # Output head position information
                    print(f"  [位置检查-HEAD] 当前位置: {[f'{p:.4f}' for p in current_head]}")
                    print(f"  [位置检查-HEAD] 目标位置: {[f'{p:.4f}' for p in self.head_target_positions]}")
                    print(f"  [位置检查-HEAD] 距离: {head_distance:.4f} 弧度 (阈值: {threshold:.4f})")
                    if head_arrived:
                        print(f"  [位置检查-HEAD] ✓ 已到达目标位置")
                    else:
                        print(f"  [位置检查-HEAD] ✗ 未到达目标位置")
            
            if part == 'head':
                return {'arrived': head_arrived, 'distance': head_distance}
            result['head'] = {'arrived': head_arrived, 'distance': head_distance}
        
        # Check body arrival
        if part is None or part == 'body':
            body_arrived = False
            body_distance = float('inf')
            if self.body_target_positions is not None and 'body' in categorized_state:
                current_body = categorized_state['body']['positions']
                if len(current_body) == len(self.body_target_positions):
                    body_distance = calculate_distance(current_body, self.body_target_positions)
                    body_arrived = body_distance < threshold
                    
                    # Output body position information
                    print(f"  [位置检查-BODY] 当前位置: {[f'{p:.4f}' for p in current_body]}")
                    print(f"  [位置检查-BODY] 目标位置: {[f'{p:.4f}' for p in self.body_target_positions]}")
                    print(f"  [位置检查-BODY] 距离: {body_distance:.4f} 弧度 (阈值: {threshold:.4f})")
                    if body_arrived:
                        print(f"  [位置检查-BODY] ✓ 已到达目标位置")
                    else:
                        print(f"  [位置检查-BODY] ✗ 未到达目标位置")
            
            if part == 'body':
                return {'arrived': body_arrived, 'distance': body_distance}
            result['body'] = {'arrived': body_arrived, 'distance': body_distance}
        
        return result
    
    def send_cartesian_velocity(self, linear: Tuple[float, float, float], angular: Tuple[float, float, float]) -> None:
        """Send cartesian velocity commands.
        
        Args:
            linear: Linear velocity (x, y, z) in m/s
            angular: Angular velocity (rx, ry, rz) in rad/s
        """
        if not self.is_connected:
            raise ROS2NotConnectedError("ROS2RobotInterface is not connected")
        
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
        
        if self.right_end_effector_pose_sub:
            self.right_end_effector_pose_sub.destroy()
            self.right_end_effector_pose_sub = None
        
        if self.end_effector_target_pub:
            self.end_effector_target_pub.destroy()
            self.end_effector_target_pub = None
        
        if self.right_end_effector_target_pub:
            self.right_end_effector_target_pub.destroy()
            self.right_end_effector_target_pub = None
        
        if self.gripper_command_pub:
            self.gripper_command_pub.destroy()
            self.gripper_command_pub = None
        
        if self.right_gripper_command_pub:
            self.right_gripper_command_pub.destroy()
            self.right_gripper_command_pub = None
        
        if self.fsm_command_pub:
            self.fsm_command_pub.destroy()
            self.fsm_command_pub = None
        
        if self.head_joint_controller_pub:
            self.head_joint_controller_pub.destroy()
            self.head_joint_controller_pub = None
        
        if self.body_joint_controller_pub:
            self.body_joint_controller_pub.destroy()
            self.body_joint_controller_pub = None
        
        # Destroy node
        if self.robot_node:
            self.robot_node.destroy_node()
            self.robot_node = None
        
        # Clear data
        with self.data_lock:
            self.latest_joint_state = None
            self.latest_end_effector_pose = None
            self.latest_right_end_effector_pose = None
        
        # Shutdown rclpy
        try:
            rclpy.shutdown()
        except Exception as e:
            logger.warning(f"Error during rclpy shutdown: {e}")
        
        logger.info("Disconnected from ROS 2 robot interface")

