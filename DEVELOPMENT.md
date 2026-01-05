# 开发说明

## 包结构

本项目现在包含两个独立的包：

1. **ros2_robot_interface** - 独立的 ROS2 机器人接口包（不依赖 LeRobot）
2. **lerobot_robot_ros2** - LeRobot 的 ROS2 机器人集成插件（依赖 ros2_robot_interface）

## 环境要求

**重要：** 请确保在 conda 虚拟环境中安装，避免权限错误。

### 使用 Conda 环境

```bash
# 激活 conda 环境（如果使用 conda）
conda activate lerobot-ros2

# 验证环境是否正确激活
which python  # 应该显示 conda 环境的路径，而不是 /usr/bin/python
which pip     # 应该显示 conda 环境的 pip
```

如果 `which python` 显示 `/usr/bin/python`，说明 conda 环境未正确激活，这会导致安装时出现权限错误（如 `Permission denied: '/usr/lib/python3/dist-packages/mpmath'`）。

## 本地开发安装

**重要说明：** 此项目需要使用 ROS2 的系统级 OpenCV（通过 `PYTHONPATH` 配置），而不是 conda 环境中的 OpenCV。这会导致某些系统包（如 `mpmath`）出现在 Python 路径中，安装时需要使用 `--ignore-installed` 选项。

由于 `ros2-robot-interface` 尚未发布到 PyPI，在本地开发时需要先安装它：

```bash
# 确保 conda 环境已激活
conda activate lerobot-ros2

# 1. 首先安装 ros2_robot_interface（独立包）
cd ros2_robot_interface
pip install -e .

# 2. 然后安装 lerobot_robot_ros2（LeRobot 插件）
# 注意：由于 ROS2 环境配置，需要使用 --ignore-installed 来避免系统包冲突
cd ../lerobot_robot_ros2
pip install -e . --ignore-installed
```

**为什么需要 `--ignore-installed`？**

- ROS2 环境配置了 `PYTHONPATH=/usr/lib/python3/dist-packages:/opt/ros/jazzy/lib/python3.12/site-packages` 以使用系统级 OpenCV
- 这导致系统包（如 `mpmath`）优先于 conda 环境的包
- pip 检测到系统包时会尝试卸载它们来安装新版本，但没有权限
- `--ignore-installed` 选项会忽略已安装的系统包，直接在 conda 环境中安装新版本

### 常见问题

**问题：** 安装时出现 `Permission denied: '/usr/lib/python3/dist-packages/...'` 错误

**原因：** 
1. 为了使用 ROS2 的系统级 OpenCV，环境配置了 `PYTHONPATH=/usr/lib/python3/dist-packages:/opt/ros/jazzy/lib/python3.12/site-packages`
2. 这导致系统级别的包（如 `mpmath`）优先于 conda 环境的包
3. pip 检测到系统级别的旧版本包，尝试卸载它来安装新版本，但没有权限修改系统包

**解决方案：**

**推荐方法：在安装包时使用 `--ignore-installed`**
```bash
conda activate lerobot-ros2
cd lerobot_robot_ros2
pip install -e . --ignore-installed
```

**如果特定包有问题，可以单独处理：**
```bash
conda activate lerobot-ros2
# 对于有问题的包，使用 --ignore-installed
pip install --ignore-installed <package-name>
# 然后正常安装其他包
pip install -e .
```

**注意：** 
- 使用 `--ignore-installed` 不会删除系统包，只是在 conda 环境中安装新版本
- conda 环境的包会优先使用（因为 conda 环境的 site-packages 在 sys.path 中排在系统包之后，但 Python 会优先使用 conda 环境中的包）
- 系统包仍然保留，ROS2 的 OpenCV 可以继续正常工作

## 使用独立包

`ros2_robot_interface` 可以在不使用 LeRobot 的环境中使用：

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

## 使用 LeRobot 集成

如果使用 LeRobot，可以通过 `lerobot_robot_ros2` 包：

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

## 发布到 PyPI

当 `ros2-robot-interface` 发布到 PyPI 后，`lerobot_robot_ros2` 的 `pyproject.toml` 中的依赖会自动解析。

