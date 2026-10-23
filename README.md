# LeRobot ROS2

LeRobot 与 ROS 2 的集成仓库。可以用 ROS 2 话题连接机器人，做数据采集和策略推理；也可以在 Isaac Sim 里按 YAML 编排任务。

任务编排可以单独使用：`ros2-stack` 启动运控和导航，`motion-generation` 按任务文件依次执行技能。灵巧手抓取是可选项，参考实现是 FiveAges W2 加 Wuji Hand2。它既能开一个网页调试界面，也能作为同一次任务里「导航靠近后再抓」的一步。没有这份抓取配置的机器人，仍然只跑编排。

从零把仿真、运控和这条任务跑起来的逐步说明（含部署时会遇到的报错）在飞书：[仿真任务编排与灵巧手抓取：从零部署](https://lcnnwa8es0bc.feishu.cn/wiki/QjCpwrpWMiCsUbkJ8iCcN7n4ncx)。本文是仓库入口。

## 仓库里有什么

| 路径 | 作用 |
|------|------|
| `submodules/ros2_robot_interface` | ROS 2 机器人接口。关节、末端、导航目标都从这里进出 |
| `submodules/robot_action_composer` | 任务编排。提供 `ros2-stack`、`motion-generation`，以及可选的 `grasp-generation` |
| `lerobot_robot_ros2` | LeRobot 的机器人插件，调用上面的接口包 |
| `lerobot_camera_ros2` | LeRobot 的相机插件 |
| `submodules/hug` | 可选。由头相机 RGB-D 生成人手姿态。独立 Python 3.10 环境 |
| `submodules/wuji-retargeting` | 可选。开源重定向参考仓库。运行时安装的是 PyPI 上的 `wuji-sdk`，不要递归克隆它里面的仿真子模块 |
| `submodules/internnav_w2` | W2 DualVLN 部署脚本，见 [docs/INTERNAV_W2_SUBMODULE.md](docs/INTERNAV_W2_SUBMODULE.md) |
| `submodules/internnav_arx_lift2s` | Lift2S DualVLN 部署脚本，见 [docs/INTERNAV_ARX_LIFT2S_SUBMODULE.md](docs/INTERNAV_ARX_LIFT2S_SUBMODULE.md) |
| `submodules/LightNav-0_W2` | W2 LightNav-0 部署脚本，见 [docs/LIGHTNAV_W2_SUBMODULE.md](docs/LIGHTNAV_W2_SUBMODULE.md) |

LeRobot 库从 PyPI 安装，版本写在 [`.fa-env.toml.example`](.fa-env.toml.example)（默认 `0.5.1`）。本仓库不再把 `submodules/lerobot` 当作子模块。

导航那三项是公司自己的部署脚本，不包含上游 InternNav / LightNav-0 的模型仓库。已经克隆本仓库的话：

```bash
git submodule update --init \
  submodules/internnav_w2 \
  submodules/internnav_arx_lift2s \
  submodules/LightNav-0_W2
```

然后打开对应子模块里的 Quick Start。

仿真场景和 OCS2 运控不在本仓库里。参考场景在 FaSim-Isaac：`robots/humanoid/FiveAges/Gen2/W2/env/wuji_hand2_new.usda`。机器人配置在 `examples/IsaacSim/robots/`，W2 加 Wuji 的例子是 `examples/IsaacSim/robots/FiveAges_W2/`。

## 依赖

| 项 | 要求 |
|----|------|
| 系统 | Ubuntu 24.04 |
| ROS 2 | Jazzy。编排和抓取使用系统里的 `rclpy` |
| Python | 父环境 3.12。HUG 单独使用 3.10，由脚本创建 |
| 环境工具 | [uv](https://docs.astral.sh/uv/)（默认）或 Conda |
| 仿真 | Isaac Sim 6.0.1 到 6.2.0，以及 FaSim-Isaac。只做实机录制可以不装 |
| 运控 | 已编译的 ROS 2 工作空间。W2 参考目录是 `~/fa_w2_ws`，以你机器上的路径为准 |
| 导航 | `ros-jazzy-navigation2`、`ros-jazzy-nav2-bringup`，并与 `ros-jazzy-behaviortree-cpp` 版本对齐 |
| 灵巧手推理 | NVIDIA 驱动和 CUDA。只跑编排可以不装 |
| MANO | 许可模型，需在 [mano.is.tue.mpg.de](https://mano.is.tue.mpg.de/) 注册后自行放入，脚本不会下载 |

编排和灵巧手共用父环境，NumPy 保持 1.x，这样才和 ROS 2 的系统 OpenCV 兼容。录制和策略推理会安装 NumPy 2，给 lerobot 0.5 使用，请另建环境，不要和抓取混在一起。

## 安装

<details>
<summary>方式一：脚本安装（推荐）</summary>

```bash
git clone git@github.com:fiveages-sim/lerobot_ros2.git
cd lerobot_ros2

# 只做任务编排
./init.sh all-motion
./init.sh ros2-workspace
source .venv/bin/activate
./init.sh doctor
```

`ros2-workspace` 会询问运控工作空间路径。填写本机真实目录，例如 `~/fa_w2_ws`。直接回车会写成默认的 `~/ros2_ws`。

还要灵巧手时，在仓库根目录继续：

```bash
./init.sh all-grasp
./init.sh doctor
```

`all-grasp` 已经包含编排那一段。它会安装 `viser` 和 `wuji-sdk`，并创建 `submodules/hug/.venv`。权重可以由脚本下载。MANO 需要自己放进：

```text
submodules/hug/assets/mano_models/models/MANO_RIGHT.pkl
```

录制或策略推理再执行 `./init.sh install-lerobot`（菜单里的 5 或 7）。这条会把当前环境的 NumPy 升到 2，之后不要再用这个环境采头相机。

交互菜单是 `./init.sh`：

| 菜单 | 作用 |
|------|------|
| 1 | 检出编排和灵巧手子模块。不递归克隆 retarget 里的仿真仓库 |
| 4 | 只安装任务编排 |
| 6 | 一键：编排子模块、环境、安装 |
| i / h / g | 抓取界面、HUG 环境、一键编排加抓取 |
| 5 / 7 | LeRobot 插件。会安装 NumPy 2 |
| 10 | 把运控工作空间写进环境激活脚本 |
| d | 检查缺什么，不安装 |

分步说明见 [docs/SETUP.md](docs/SETUP.md)。

</details>

<details>
<summary>方式二：手动安装（只装任务编排）</summary>

```bash
git clone git@github.com:fiveages-sim/lerobot_ros2.git
cd lerobot_ros2
git submodule update --init -- \
  submodules/ros2_robot_interface \
  submodules/robot_action_composer

uv venv --python 3.12 --system-site-packages .venv
source .venv/bin/activate

# 本地包必须 --no-deps，否则会到 PyPI 上找 rclpy
uv pip install -e submodules/ros2_robot_interface --no-deps
uv pip install 'numpy>=1.26,<2' pyyaml
uv pip install -e submodules/robot_action_composer --no-deps
```

`lerobot_robot_ros2` 依赖已经装好的 `ros2_robot_interface`。录制插件用 `./init.sh install-lerobot`，不要和上面这个环境混用。

</details>

### 环境文件

复制 [`.fa-env.toml.example`](.fa-env.toml.example) 为 `.fa-env.toml`（不提交）。个人覆盖用 `.fa-env.local.toml`。

| 配置 | 含义 |
|------|------|
| `backend` | `uv` 或 `conda` |
| `[conda].name` | Conda 环境名，默认 `lerobot-ros2` |
| `[uv].venv` | 虚拟环境路径，默认 `.venv` |
| `[lerobot].version` | PyPI 上的 lerobot 版本，默认 `0.5.1` |
| `[ros2].workspace` | 激活环境时自动 source 的运控工作空间 |

激活后确认解释器在本仓库的 `.venv` 里，而不是系统 Python，也不是 `submodules/hug/.venv`：

```bash
which python
which ros2-stack
which motion-generation
```

## 使用

每次新开终端，在仓库根目录 `source .venv/bin/activate`。仿真用 FaSim-Isaac 的 `./run.sh` 打开，并让 Isaac 和终端使用同一套 `ROS_DOMAIN_ID` 与 `RMW_IMPLEMENTATION`。

### 任务编排

```bash
cd examples/IsaacSim
ros2-stack launch --robot <robot_key>
```

另一个已经激活同一环境的终端：

```bash
cd examples/IsaacSim
motion-generation
```

菜单里选择机器人、任务和场景。这一步不需要 HUG、MANO 或 `wuji-sdk`。`<robot_key>` 与 `examples/IsaacSim/robots/<Robot>/robot.yaml` 里的 `key` 一致，例如 `fiveages_w2`。

### 参考例子：W2 导航后用 Wuji 抓取

1. Isaac 打开 `wuji_hand2_new.usda` 并 Play。启用 `isaacsim.ros2.sim_control` 和 `isaacsim.ros2.bridge`。
2. 把 `examples/IsaacSim/robots/FiveAges_W2/robot.yaml` 里的 `robot_profile` 改成这台电脑上的绝对路径，指向运控工作空间中的 `config/robot.profiles/wuji_hand2.yaml`。脚本不会展开 `~`。这行是机器自己的路径，不要提交。
3. 终端 A：

```bash
cd examples/IsaacSim
ros2-stack launch --robot fiveages_w2 --group dex_hand
```

确认启动参数里是 `nav2_profile:=map_only`，地图路径含 `east_rural_open/map.yaml`。不选 `dex_hand` 时，导航使用仓库自带的货架地图。

4. 终端 B：

```bash
cd examples/IsaacSim
motion-generation --robot fiveages_w2 --task-key navigate_and_dex_grasp --scene apple
```

也可以不带参数，在菜单里选同一任务。场景名以任务配置里实际有的为准。

只在图像上点物体、不走导航：

```bash
cd examples/IsaacSim
grasp-generation serve --robot fiveages_w2 --workspace .
```

浏览器打开 Viser，默认端口 8080。顺序是采头相机、点物体、看规划、再执行手臂和合手。

### LeRobot 插件

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

更多脚本在 `examples/`。Isaac 的公共步骤在 [`examples/IsaacSim/README.md`](examples/IsaacSim/README.md)。文档索引在 [`docs/README.md`](docs/README.md)。

https://github.com/user-attachments/assets/a824835a-7614-4833-9d39-4c0005474dbe
