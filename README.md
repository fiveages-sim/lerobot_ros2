# LeRobot ROS2

新机器从头部署、以及部署中踩到的报错：根目录 **[部署使用说明.md](部署使用说明.md)**（操作笔记，和 `docs/` 里的设计说明分开）。

在 Isaac Sim 里按配置编排机器人任务，并通过 ROS 2 话题执行。任务编排（运控启动 + 技能序列）可以单独使用。灵巧手抓取是可选插件：人手姿态生成、手型 retarget、收拢执行器，既能开调试界面，也能作为一个技能接到任务队列里。

从零配置见 **[docs/SETUP.md](docs/SETUP.md)**。文档按用途分组见 **[docs/README.md](docs/README.md)**。

## 项目结构

| 路径 | 作用 | 是否必须 |
|------|------|----------|
| `submodules/ros2_robot_interface` | ROS 2 机器人接口 | 任务编排需要 |
| `submodules/robot_action_composer` | 任务编排、`ros2-stack`、`motion-generation`；抓取库也在这里 | 任务编排需要 |
| `submodules/hug` | 可选。RGB-D → MANO 人手。独立 Python 3.10 环境 | 不要灵巧手推理就不装 |
| `submodules/wuji-retargeting` | 可选。开源 retarget 参考仓。运行时用 PyPI `wuji-sdk` | 参考手型需要 sdk，不需要嵌套仿真 |
| `lerobot_robot_ros2` / `lerobot_camera_ros2` | LeRobot 录制与推理插件 | 只做编排或抓取可以不装 |

LeRobot 核心库从 PyPI 安装（默认 `lerobot==0.5.1`，写在 `.fa-env.toml`）。`submodules/lerobot` 不是子模块，不要提交。

参考实现（相机、手型、物体 prim 都走配置）：`examples/IsaacSim/robots/FiveAges_W2/`（Wuji Hand2 + W2）。仿真场景在资产仓，当前用的是 FaSim 的 `wuji_hand2_new.usda`。换环境、换手、以及 B4 为什么要手资产上的接触力，见 [docs/ENV_AND_HAND_CONTRACT.md](docs/ENV_AND_HAND_CONTRACT.md)。

## 前置要求

- ROS 2（测试版本：Jazzy）
- Python >= 3.12（父仓环境）
- [uv](https://docs.astral.sh/uv/)（推荐）或 Conda
- 灵巧手推理另要 Python 3.10（只在 `submodules/hug/.venv`）

## 安装

<details>
<summary>方式一：脚本安装（推荐）</summary>

```bash
git clone git@github.com:fiveages-sim/lerobot_ros2.git
cd lerobot_ros2

# 任务编排（不含灵巧手）
./init.sh all-motion
./init.sh ros2-workspace
source .venv/bin/activate

# 可选：抓取 extras + HUG 环境（MANO 仍需自行注册下载）
# ./init.sh all-grasp
# ./init.sh doctor
```

交互菜单：`./init.sh`。

- **子模块**：1 全部检出 / 2 更新到 main。只要编排用 `./init.sh submodules-core`
- **安装**：4 任务编排 / `i` 抓取 extras / 5 lerobot / `h` HUG 环境 / `d` 检查缺什么
- **一键**：6 只编排 / 7 编排 + lerobot / `g` 编排 + 抓取

分步和 MANO、权重、不要递归拉取 retarget 嵌套仓库，见 [docs/SETUP.md](docs/SETUP.md)。

</details>

<details>
<summary>方式二：手动安装（uv，仅任务编排）</summary>

```bash
git clone git@github.com:fiveages-sim/lerobot_ros2.git
cd lerobot_ros2
git submodule update --init -- submodules/ros2_robot_interface submodules/robot_action_composer

uv venv --python python3.12 --system-site-packages .venv
source .venv/bin/activate

# 本地包必须 --no-deps，否则会去 PyPI 找 rclpy
uv pip install -e submodules/ros2_robot_interface --no-deps
uv pip install numpy pyyaml
uv pip install -e submodules/robot_action_composer --no-deps
```

可选抓取界面与参考手型 sdk（仍在这个 `.venv` 里，不要装进 `submodules/hug/.venv`）：

```bash
uv pip install "viser>=0.2" "wuji-sdk>=0.10.0"
./init.sh hug-env
```

录制 / 策略推理再装 PyTorch 与 `lerobot==0.5.1`，以及 `lerobot_robot_ros2`、`lerobot_camera_ros2`（同样 `--no-deps`）。脚本入口是 `./init.sh install-lerobot`。

</details>

## 环境配置

运行时配置见 [`.fa-env.toml.example`](.fa-env.toml.example)（复制为本地 `.fa-env.toml`，不纳入 git）：

| 配置项 | 说明 |
|--------|------|
| `backend` | `uv` 或 `conda` |
| `[conda].name` | conda 环境名（默认 `lerobot-ros2`） |
| `[uv].venv` | uv 虚拟环境路径（默认 `.venv`） |
| `[lerobot].version` | PyPI lerobot 版本（默认 `0.5.1`） |
| `[ros2].workspace` | ROS 2 工作空间，激活环境时自动 source |

个人覆盖：`.fa-env.local.toml`（已 gitignore）。

### 常见坑

Ubuntu 上裸跑 `pip install` 会打到系统 pip，报 `externally-managed-environment`。

1. 父仓 CLI 只用本仓库的 `.venv`（或配置的 conda 环境）。不要用 `submodules/hug/.venv` 装 `ros2-stack` / `grasp-generation`。
2. 本地包用 `uv pip install -e ... --no-deps`。
3. HUG 权重 `checkpoints/hug_full.safetensors` 和 `assets/mano_models/` 不进 git。MANO 要到官网注册后自己放。详见 [docs/SETUP.md](docs/SETUP.md)。
4. `./init.sh all` 会安装 NumPy 2.x。纯编排或抓取不要顺手跑这条。

更完整的环境说明：[`docs/ENV_THIS_CHECKOUT.md`](docs/ENV_THIS_CHECKOUT.md)。

## 使用

编排与运控：

```bash
cd examples/IsaacSim
ros2-stack launch --robot <robot_key>
motion-generation
```

可选抓取（参考机器人 `fiveages_w2`）：

```bash
grasp-generation serve --robot fiveages_w2 --workspace .
```

LeRobot 插件示例：

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

更多示例见 `examples/`。Isaac 仿真步骤见 [`examples/IsaacSim/README.md`](examples/IsaacSim/README.md)。


https://github.com/user-attachments/assets/a824835a-7614-4833-9d39-4c0005474dbe
