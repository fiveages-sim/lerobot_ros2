# LeRobot ROS2

LeRobot ROS2 集成，支持通过 ROS2 话题与机器人通信。

## 项目结构

本项目包含两个包：

1. **ros2_robot_interface** - 独立的 ROS2 机器人接口包（不依赖 LeRobot）
2. **lerobot_robot_ros2** - LeRobot 的 ROS2 机器人集成插件

## 安装

### 前置要求

- ROS2 (测试版本: Jazzy)
- Python >= 3.10
- Conda 环境（推荐）

### 安装步骤

```bash
# 1. Clone and install LeRobot
git clone https://github.com/huggingface/lerobot
cd lerobot
git checkout 55e752f0c2e7fab0d989c5ff999fbe3b6d8872ab
pip install -e .

# 2. Clone LeRobot ROS2
cd ..
git clone https://github.com/fiveages-sim/lerobot_ros2
cd lerobot_ros2

# 3. 激活 conda 环境（确保已配置 ROS2 和系统 OpenCV）
conda activate lerobot-ros2

# 4. 安装 ros2_robot_interface（独立包）
cd ros2_robot_interface
pip install -e .

# 5. 安装 lerobot_robot_ros2（LeRobot 插件）
# 注意：由于需要使用 ROS2 的系统级 OpenCV，安装时需要使用 --ignore-installed
cd ../lerobot_robot_ros2
pip install -e . --ignore-installed
```

### 重要说明

**ROS2 OpenCV 依赖：** 此项目需要使用 ROS2 的系统级 OpenCV（通过 `PYTHONPATH` 配置），而不是 conda 环境中的 OpenCV。这会导致某些系统包出现在 Python 路径中，安装时需要使用 `--ignore-installed` 选项来避免权限错误。

详细安装说明和常见问题请参考 [DEVELOPMENT.md](DEVELOPMENT.md)。

## 使用

### 使用 LeRobot 集成

```python
from lerobot_robot_ros2 import ROS2Robot, ROS2RobotConfig, ROS2RobotInterfaceConfig

config = ROS2RobotConfig(
    id="my_robot",
    ros2_interface=ROS2RobotInterfaceConfig(
        joint_states_topic="/joint_states",
        end_effector_pose_topic="/left_current_pose",
        end_effector_target_topic="/left_target"
    )
)

robot = ROS2Robot(config)
robot.connect()
# ... 使用机器人 ...
robot.disconnect()
```

### 独立使用（不依赖 LeRobot）

```python
from ros2_robot_interface import ROS2RobotInterface, ROS2RobotInterfaceConfig

config = ROS2RobotInterfaceConfig(
    joint_states_topic="/joint_states",
    end_effector_pose_topic="/left_current_pose",
    end_effector_target_topic="/left_target"
)

interface = ROS2RobotInterface(config)
interface.connect()
# ... 使用接口 ...
interface.disconnect()
```

更多使用示例请参考 [examples/](examples/) 目录。