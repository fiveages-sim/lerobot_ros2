"""
ROS 2 Camera Implementation

This module provides a ROS 2 camera implementation for LeRobot,
allowing cameras to receive image data through ROS 2 topics.
"""

import logging
import threading
import time
from typing import Any

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from lerobot.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import Image

from lerobot.cameras.camera import Camera
from .config import ROS2CameraConfig

logger = logging.getLogger(__name__)


class ROS2Camera(Camera):
    """ROS 2 camera implementation for LeRobot.
    
    This class provides a LeRobot camera interface that receives image data
    from ROS 2 topics. It supports both synchronous and asynchronous frame reading.
    
    Example:
        ```python
        from lerobot_ros2_devices.cameras import ROS2Camera, ROS2CameraConfig
        
        # Create configuration
        config = ROS2CameraConfig(
            topic_name="/camera/image_raw",
            fps=30,
            width=640,
            height=480
        )
        
        # Create and connect camera
        camera = ROS2Camera(config)
        camera.connect()
        
        # Read frames
        image = camera.read()
        async_image = camera.async_read()
        
        # Disconnect when done
        camera.disconnect()
        ```
    """
    
    def __init__(self, config: ROS2CameraConfig):
        """Initialize the ROS 2 camera.
        
        Args:
            config: ROS 2 camera configuration
        """
        super().__init__(config)
        self.config = config
        self.topic_name = config.topic_name
        self.node_name = config.node_name
        self.namespace = config.namespace
        self.timeout_ms = config.timeout_ms
        self.queue_size = config.queue_size
        self.encoding = config.encoding
        
        # ROS 2 components
        self.ros_node: Node | None = None
        self.executor: SingleThreadedExecutor | None = None
        self.executor_thread: threading.Thread | None = None
        self.image_subscription = None
        
        # Image processing
        self.bridge = CvBridge()
        self.latest_image: np.ndarray | None = None
        self.image_lock = threading.Lock()
        self.image_received_event = threading.Event()
        self._dimensions_initialized = False
        
        # Connection state
        self._connected = False
    
    def __str__(self) -> str:
        return f"ROS2Camera(topic={self.topic_name}, node={self.node_name})"
    
    @property
    def is_connected(self) -> bool:
        """Check if the camera is currently connected.
        
        Returns:
            bool: True if connected and ready to capture frames, False otherwise.
        """
        return self._connected and self.ros_node is not None
    
    def get_actual_image_dimensions(self) -> tuple[int, int] | None:
        """Get the actual image dimensions from the received images.
        
        Returns:
            tuple[int, int] | None: (width, height) if images have been received, None otherwise.
        """
        with self.image_lock:
            if self.latest_image is not None:
                return self.latest_image.shape[1], self.latest_image.shape[0]  # (width, height)
            return None
    
    @staticmethod
    def find_cameras() -> list[dict[str, Any]]:
        """Discover available ROS 2 camera topics.
        
        This method scans the ROS 2 system for available image topics.
        
        Returns:
            List[Dict[str, Any]]: List of dictionaries containing camera information.
        """
        cameras = []
        
        try:
            # Initialize ROS 2 if not already done
            if not rclpy.ok():
                rclpy.init()
            
            # Create temporary node for discovery
            temp_node = Node("ros2_camera_discovery")
            
            # Get all topic names and types
            topic_names_and_types = temp_node.get_topic_names_and_types()
            
            # Filter for image topics
            for topic_name, topic_types in topic_names_and_types:
                if 'sensor_msgs/msg/Image' in topic_types:
                    cameras.append({
                        "topic_name": topic_name,
                        "type": "ros2_camera",
                        "description": f"ROS 2 Image topic: {topic_name}",
                        "namespace": topic_name.split('/')[1] if len(topic_name.split('/')) > 2 else ""
                    })
            
            # Clean up
            temp_node.destroy_node()
            
        except Exception as e:
            logger.warning(f"Failed to discover ROS 2 cameras: {e}")
        
        return cameras
    
    def connect(self, warmup: bool = True) -> None:
        """Connect to the ROS 2 camera topic.
        
        Args:
            warmup: If True, wait for the first image before returning.
        """
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")
        
        try:
            # Initialize ROS 2 if not already done
            if not rclpy.ok():
                rclpy.init()
            
            # Create ROS 2 node with unique name
            self.ros_node = Node(
                self.node_name, 
                namespace=self.namespace
            )
            
            # Create image subscription
            self.image_subscription = self.ros_node.create_subscription(
                Image,
                self.topic_name,
                self._image_callback,
                self.queue_size
            )
            
            # Start executor in separate thread
            self.executor = SingleThreadedExecutor()
            self.executor.add_node(self.ros_node)
            self.executor_thread = threading.Thread(
                target=self.executor.spin, 
                daemon=True
            )
            self.executor_thread.start()
            
            # Wait for connection to establish
            time.sleep(1.0)
            
            # Warmup: wait for first image
            if warmup:
                logger.info(f"Warming up {self}...")
                if not self.image_received_event.wait(timeout=5.0):
                    logger.warning(f"No image received from {self.topic_name} during warmup")
            
            self._connected = True
            logger.info(f"Connected to ROS 2 camera: {self.topic_name}")
            
        except Exception as e:
            logger.error(f"Failed to connect to ROS 2 camera: {e}")
            self.disconnect()
            raise
    
    def _image_callback(self, msg: Image) -> None:
        """ROS 2 image callback function.
        
        Args:
            msg: ROS 2 Image message
        """
        try:
            # Extract actual image dimensions from the message (only on first image)
            if not self._dimensions_initialized:
                actual_width = msg.width
                actual_height = msg.height
                
                # Set camera dimensions from first received image
                if self.width is None or self.height is None:
                    self.width = actual_width
                    self.height = actual_height
                    logger.info(f"Auto-detected image dimensions: {actual_width}x{actual_height}")
                elif self.width != actual_width or self.height != actual_height:
                    logger.warning(
                        f"Image dimensions mismatch: configured {self.width}x{self.height}, "
                        f"actual {actual_width}x{actual_height}. Updating to actual dimensions."
                    )
                    self.width = actual_width
                    self.height = actual_height
                
                self._dimensions_initialized = True
            
            # Convert ROS image message to OpenCV format
            try:
                cv_image = self.bridge.imgmsg_to_cv2(msg, self.encoding)
            except Exception as e:
                logger.error(f"Failed to convert ROS image message: {e}")
                return
            
            # Convert to RGB format for LeRobot
            try:
                if self.encoding in ["bgr8", "bgra8"]:
                    rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
                elif self.encoding in ["rgb8", "rgba8"]:
                    rgb_image = cv_image
                elif self.encoding == "mono8":
                    rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_GRAY2RGB)
                else:
                    # Default to BGR to RGB conversion
                    rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            except Exception as e:
                logger.error(f"Failed to convert image color format: {e}")
                # Fallback: use the image as-is
                rgb_image = cv_image
            
            # Update latest image
            with self.image_lock:
                self.latest_image = rgb_image.copy()
                self.image_received_event.set()
                
        except Exception as e:
            logger.error(f"Error processing image: {e}")
    
    def read(self, color_mode=None) -> np.ndarray:
        """Synchronously read a frame from the camera.
        
        Args:
            color_mode: Color mode (ignored for ROS 2 cameras, always returns RGB)
            
        Returns:
            np.ndarray: Captured frame as numpy array in RGB format
            
        Raises:
            DeviceNotConnectedError: If camera is not connected
            RuntimeError: If no image is available
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected")
        
        with self.image_lock:
            if self.latest_image is None:
                raise RuntimeError("No image available")
            return self.latest_image.copy()
    
    def async_read(self, timeout_ms: float = None) -> np.ndarray:
        """Asynchronously read a frame from the camera.
        
        Args:
            timeout_ms: Maximum time to wait for a frame in milliseconds
            
        Returns:
            np.ndarray: Captured frame as numpy array in RGB format
            
        Raises:
            DeviceNotConnectedError: If camera is not connected
            TimeoutError: If no image is received within timeout
            RuntimeError: If no image is available
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected")
        
        timeout_ms = timeout_ms or self.timeout_ms
        
        # Wait for new image
        if not self.image_received_event.wait(timeout=timeout_ms / 1000.0):
            raise TimeoutError(f"No image received within {timeout_ms}ms")
        
        with self.image_lock:
            if self.latest_image is None:
                raise RuntimeError("No image available")
            return self.latest_image.copy()
    
    def disconnect(self) -> None:
        """Disconnect from the camera and release resources."""
        self._connected = False
        
        # Stop executor
        if self.executor:
            self.executor.shutdown()
            self.executor = None
        
        # Wait for thread to finish
        if self.executor_thread:
            self.executor_thread.join(timeout=2.0)
            self.executor_thread = None
        
        # Destroy node
        if self.ros_node:
            self.ros_node.destroy_node()
            self.ros_node = None
        
        # Clean up resources
        with self.image_lock:
            self.latest_image = None
            self.image_received_event.clear()
            self._dimensions_initialized = False
        
        logger.info(f"Disconnected from ROS 2 camera: {self.topic_name}")
