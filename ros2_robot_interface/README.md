# ROS2 Robot Interface

A standalone Python package for communicating with ROS 2 robots through topics. This package is independent of LeRobot and can be used in any ROS 2 environment.

## Features

- Subscribe to joint states from ROS 2 topics
- Subscribe to end-effector pose information
- Publish target end-effector poses
- Control gripper position
- Thread-safe data access
- Configurable timeouts and recovery mechanisms

## Installation

### From Source (Development)

```bash
cd ros2_robot_interface
pip install -e .
```

## Usage

### Basic Example

```python
from ros2_robot_interface import ROS2RobotInterface, ROS2RobotInterfaceConfig
from geometry_msgs.msg import Pose

# Create configuration
config = ROS2RobotInterfaceConfig(
    joint_states_topic="/joint_states",
    end_effector_pose_topic="/left_current_pose",
    end_effector_target_topic="/left_target",
    joint_names=["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
)

# Create and connect interface
interface = ROS2RobotInterface(config)
interface.connect()

# Get joint state
joint_state = interface.get_joint_state()
if joint_state:
    print(f"Joint positions: {joint_state['positions']}")

# Get end-effector pose
pose = interface.get_end_effector_pose()
if pose:
    print(f"End-effector position: ({pose.position.x}, {pose.position.y}, {pose.position.z})")

# Send target pose
target_pose = Pose()
target_pose.position.x = 0.5
target_pose.position.y = 0.0
target_pose.position.z = 0.3
target_pose.orientation.w = 1.0
interface.send_end_effector_target(target_pose)

# Control gripper
interface.send_gripper_command(0.5)  # 50% open

# Disconnect
interface.disconnect()
```

### Configuration Options

```python
from ros2_robot_interface import ROS2RobotInterfaceConfig, ControlType

config = ROS2RobotInterfaceConfig(
    # ROS 2 topics
    joint_states_topic="/joint_states",
    end_effector_pose_topic="/left_current_pose",
    end_effector_target_topic="/left_target",
    
    # Joint names
    joint_names=["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"],
    
    # Gripper configuration
    gripper_enabled=True,
    gripper_joint_name="gripper_joint",
    gripper_command_topic="gripper_joint/position_command",
    gripper_min_position=0.0,
    gripper_max_position=1.0,
    
    # Control parameters
    control_type=ControlType.CARTESIAN_POSE,
    
    # Safety limits
    max_linear_velocity=0.1,  # m/s
    max_angular_velocity=0.5,  # rad/s
    
    # Timeout settings (0 = disabled)
    joint_state_timeout=0.0,  # seconds
    end_effector_pose_timeout=0.0,  # seconds
    
    # ROS 2 namespace
    namespace=""
)
```

## API Reference

### ROS2RobotInterface

Main interface class for ROS 2 robot communication.

#### Methods

- `connect()` - Connect to ROS 2 and create subscriptions/publishers
- `disconnect()` - Disconnect and cleanup resources
- `get_joint_state()` - Get the latest joint state
- `get_end_effector_pose()` - Get the latest end-effector pose
- `send_end_effector_target(pose)` - Send target end-effector pose
- `send_gripper_command(position)` - Send gripper position command
- `send_cartesian_velocity(linear, angular)` - Send cartesian velocity commands (placeholder)

#### Properties

- `is_connected` - Check if the interface is connected

### ROS2RobotInterfaceConfig

Configuration dataclass for the interface.

### Exceptions

- `ROS2InterfaceError` - Base exception for ROS 2 interface errors
- `ROS2NotConnectedError` - Raised when trying to use the interface while not connected
- `ROS2AlreadyConnectedError` - Raised when trying to connect while already connected

## Requirements

- Python >= 3.10
- ROS 2 (tested with Humble and later)
- rclpy
- sensor-msgs
- geometry-msgs
- std-msgs
- numpy

## Development

For local development, install in editable mode:

```bash
cd ros2_robot_interface
pip install -e .
```

## License

Apache-2.0
