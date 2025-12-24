# LeRobot ROS2

LeRobot ROS2 集成，支持通过 ROS2 话题与机器人通信。

## 项目结构

本项目包含两个包：

1. **ros2_robot_interface** - 独立的 ROS2 机器人接口包（不依赖 LeRobot）
2. **lerobot_robot_ros2** - LeRobot 的 ROS2 机器人集成插件

## 子模块

本项目使用 Git 子模块来管理 `ros2_robot_interface` 包。该子模块是一个独立的仓库，位于 `git@github.com:fiveages-sim/ros2_robot_interface.git`。

### 克隆包含子模块的项目

首次克隆项目时，需要同时初始化子模块：

```bash
git clone --recursive git@github.com:fiveages-sim/lerobot_ros2.git
cd lerobot_ros2
```

如果已经克隆了项目但没有包含子模块，可以运行：

```bash
git submodule update --init --recursive
```

### 更新子模块

要更新子模块到最新版本：

```bash
git submodule update --remote ros2_robot_interface
```

### 子模块开发

如果需要在子模块中进行开发：

```bash
cd ros2_robot_interface
# 进行修改后，在子模块目录中提交
git add .
git commit -m "Your changes"
git push

# 返回父仓库，更新子模块引用
cd ..
git add ros2_robot_interface
git commit -m "Update ros2_robot_interface submodule"
git push
```

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

# 2. Clone LeRobot ROS2（包含子模块）
cd ..
git clone --recursive git@github.com:fiveages-sim/lerobot_ros2.git
# 或者如果使用 HTTPS:
# git clone --recursive https://github.com/fiveages-sim/lerobot_ros2.git
cd lerobot_ros2

# 3. 激活 conda 环境（确保已配置 ROS2 和系统 OpenCV）
conda activate lerobot-ros2

# 4. 安装 ros2_robot_interface（子模块，独立包）
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