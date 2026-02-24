# LeRobot ROS2

LeRobot 与 ROS2 的集成工程，支持通过 ROS2 话题与机器人通信。

## 项目结构

本项目主要包含：

1. `submodules/ros2_robot_interface`：独立 ROS2 机器人接口包
2. `lerobot_robot_ros2`：LeRobot 的 ROS2 机器人插件
3. `lerobot_camera_ros2`：LeRobot 的 ROS2 相机插件
4. `submodules/lerobot`：固定提交版本的 LeRobot 子模块

## 前置要求

- ROS2（测试版本：Jazzy）
- Python >= 3.12
- Conda（推荐）

## 安装

<details>
<summary>方式一：脚本安装（推荐）</summary>

```bash
# 1) 克隆项目
git clone --recursive git@github.com:fiveages-sim/lerobot_ros2.git
cd lerobot_ros2

# 2) 按需执行初始化脚本
./init.sh
```

脚本菜单中建议顺序：

1. 初始化子模块
2. 初始化 lerobot（固定提交）
3. 创建 conda 环境
4. 安装 interface
5. 安装 lerobot 插件（会安装 CUDA/PyTorch/ffmpeg/evdev，并安装 lerobot 与插件）

</details>

<details>
<summary>方式二：手动安装（默认参考）</summary>

```bash
# 1) 克隆项目与子模块
git clone --recursive git@github.com:fiveages-sim/lerobot_ros2.git
cd lerobot_ros2

# 2) 初始化/更新子模块
git submodule update --init --recursive

# 3) 固定 lerobot 到指定提交
cd submodules/lerobot
git fetch --all --tags
git checkout 55e752f0c2e7fab0d989c5ff999fbe3b6d8872ab
cd ../..

# 4) 创建并激活 conda 环境
conda create -n lerobot-ros2 python=3.12 -y
conda activate lerobot-ros2

# 5) 安装运行依赖
conda install -y -c nvidia cuda-toolkit=12.8
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
conda install -y ffmpeg -c conda-forge
conda install -y evdev -c conda-forge
pip install "numpy<2"

# 6) 安装本地包
pip install -e submodules/ros2_robot_interface
pip install -e submodules/lerobot
pip install -e lerobot_robot_ros2 --no-deps
pip install -e lerobot_camera_ros2

# 7) 再次固定 numpy，避免被依赖升级到 2.x
pip install "numpy<2"
```

</details>

## 说明

- 本项目依赖 ROS2 的系统 OpenCV，当前建议固定 `numpy<2`，避免 `cv_bridge` 与 NumPy 2.x 的 ABI 冲突。
- `lerobot_robot_ros2` 依赖本地 `ros2-robot-interface`，请先安装 `submodules/ros2_robot_interface`。

## 使用

```python
from lerobot_robot_ros2 import ROS2Robot, ROS2RobotConfig, ROS2RobotInterfaceConfig

config = ROS2RobotConfig(
    id="my_robot",
    ros2_interface=ROS2RobotInterfaceConfig(
        joint_states_topic="/joint_states",
        end_effector_pose_topic="/left_current_pose",
        end_effector_target_topic="/left_target",
    ),
)

robot = ROS2Robot(config)
robot.connect()
# ...
robot.disconnect()
```

更多示例见 `examples/`。