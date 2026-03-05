"""
ROS 2 Camera Configuration

Configuration classes for ROS 2 cameras in LeRobot.
"""

from dataclasses import dataclass
from lerobot.cameras.configs import CameraConfig


@CameraConfig.register_subclass("lerobot_camera_ros2")
@dataclass
class ROS2CameraConfig(CameraConfig):
    """Configuration class for ROS 2 cameras.
    
    This configuration allows LeRobot to receive image data from ROS 2 topics,
    enabling integration with ROS 2-based camera systems.
    
    Example:
        ```python
        config = ROS2CameraConfig(
            topic_name="/camera/image_raw",
            node_name="wrist_camera_node",
            namespace="robot",
            fps=30,
            width=640,
            height=480
        )
        ```
    
    Args:
        topic_name: ROS 2 image topic name (e.g., "/camera/image_raw")
        node_name: ROS 2 node name for this camera (e.g., "wrist_camera_node")
        namespace: ROS 2 namespace (optional, defaults to empty string)
        timeout_ms: Timeout for receiving images in milliseconds
        queue_size: ROS 2 subscription queue size
        encoding: Expected image encoding (e.g., "bgr8", "rgb8", "mono8")
    """
    
    topic_name: str = "/camera/image_raw"
    node_name: str = "lerobot_ros2_camera"
    namespace: str = ""
    timeout_ms: float = 1000.0
    queue_size: int = 10
    encoding: str = "bgr8"
    depth_topic_name: str | None = None
    depth_encoding: str = "32FC1"
    # LeRobot required parameters with default values
    width: int = 1280
    height: int = 720
    fps: int = 30
