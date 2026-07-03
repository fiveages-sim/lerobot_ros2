# LeRobot ROS2

LeRobot 与 ROS2 的集成工程，支持通过 ROS2 话题与机器人通信。

## 项目结构

本项目主要包含：

1. `submodules/ros2_robot_interface`：独立 ROS2 机器人接口包
2. `submodules/robot_action_composer`：机器人动作编排
3. `lerobot_robot_ros2`：LeRobot 的 ROS2 机器人插件
4. `lerobot_camera_ros2`：LeRobot 的 ROS2 相机插件

LeRobot 核心库通过 PyPI 安装（默认 `lerobot==0.5.1`，版本在 `.fa-env.toml` 中配置；首次由 [`.fa-env.toml.example`](.fa-env.toml.example) 生成）。

## 前置要求

- ROS2（测试版本：Jazzy）
- Python >= 3.12
- [uv](https://docs.astral.sh/uv/)（推荐）或 Conda

## 安装

<details>
<summary>方式一：脚本安装（推荐）</summary>

```bash
# 1) 克隆项目
git clone --recursive git@github.com:fiveages-sim/lerobot_ros2.git
cd lerobot_ros2

# 2) 安装 uv（若尚未安装）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 3) 全量初始化（子模块 + 环境 + 任务编排 + lerobot）
./init.sh all
# 仅任务编排（不含 lerobot）：
# ./init.sh all-motion
source .venv/bin/activate   # 默认 backend=uv
```

交互菜单（`./init.sh`）：

1. 初始化子模块
2. 按当前 backend 创建环境
3. 安装任务编排（`ros2_robot_interface` + `robot_action_composer`）
4. 安装 lerobot 相关（PyTorch + lerobot + 插件包）
5. 全部执行（任务编排：1 + 2 + 3）
6. 全部执行（任务编排 + lerobot：1 + 2 + 3 + 4）

仅跑 `motion-generation` 时，执行 **3** 或 **`./init.sh all-motion`** 即可；录制 / 推理需 **4** 或 **`./init.sh install-lerobot`**。

切换 backend：

```bash
./init.sh set-backend uv      # 或 conda
./init.sh env 3.12
```

</details>

<details>
<summary>方式二：手动安装（uv）</summary>

```bash
git clone --recursive git@github.com:fiveages-sim/lerobot_ros2.git
cd lerobot_ros2
git submodule update --init --recursive

# 创建 uv 环境（--system-site-packages 以使用系统 ROS2 包）
uv venv --python python3.12 --system-site-packages .venv
source .venv/bin/activate

# 安装 PyTorch 与 lerobot
uv pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
  --index-url https://download.pytorch.org/whl/cu128
uv pip install "lerobot==0.5.1"

# 安装本地包（ROS 依赖由系统提供）
uv pip install -e submodules/ros2_robot_interface --no-deps
uv pip install numpy pyyaml
uv pip install -e submodules/robot_action_composer --no-deps
uv pip install -e lerobot_robot_ros2 --no-deps
uv pip install "lerobot==0.5.1"
uv pip install -e lerobot_camera_ros2 --no-deps
uv pip install "lerobot==0.5.1" numpy
```

</details>

## 环境配置

运行时配置见 [`.fa-env.toml.example`](.fa-env.toml.example)（复制为本地 `.fa-env.toml`，不纳入 git）：

| 配置项 | 说明 |
|--------|------|
| `backend` | `uv` 或 `conda` |
| `[conda].name` | conda 环境名（默认 `lerobot-ros2`） |
| `[uv].venv` | uv 虚拟环境路径（默认 `.venv`） |
| `[lerobot].version` | PyPI lerobot 版本（默认 `0.5.1`） |
| `[ros2].workspace` | ROS2 工作空间，激活环境时自动 source |

个人覆盖：`.fa-env.local.toml`（已 gitignore）。本地 `.fa-env.toml` 亦已 gitignore。

## 说明

- `lerobot_robot_ros2` 依赖本地 `ros2-robot-interface`，请先安装 `submodules/ros2_robot_interface`。
- 相机插件默认使用 **manual 图像转换**（不依赖 `cv_bridge`），以兼容 `lerobot==0.5.1` 所需的 numpy 2.x。若环境支持 cv_bridge，会自动尝试使用；也可设置 `LEROBOT_ROS2_DISABLE_CV_BRIDGE=1` 强制禁用。
- uv 路径需确保系统已安装 `ffmpeg`（如 `sudo apt install ffmpeg`）。
- `install-plugins` 会额外安装 `scipy>=1.14`，避免 `--system-site-packages` 下系统 SciPy 与 numpy 2.x 不兼容。

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


https://github.com/user-attachments/assets/a824835a-7614-4833-9d39-4c0005474dbe

