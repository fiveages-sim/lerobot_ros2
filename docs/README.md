# 文档索引

仓库入口是根目录 [README.md](../README.md)。克隆之后先看 [SETUP.md](SETUP.md)。仿真、运控和 W2 抓取的逐步操作在飞书：[从零部署](https://lcnnwa8es0bc.feishu.cn/wiki/QjCpwrpWMiCsUbkJ8iCcN7n4ncx)。

## 上手

| 文档 | 何时读 |
|------|--------|
| [SETUP.md](SETUP.md) | 安装任务编排，以及可选的灵巧手 |
| [ENV_THIS_CHECKOUT.md](ENV_THIS_CHECKOUT.md) | 父环境、HUG 环境、录制环境为什么要分开 |
| [examples/IsaacSim/README.md](../examples/IsaacSim/README.md) | Isaac Sim、仿真控制服务、机器人列表 |
| [submodules/robot_action_composer/README.md](../submodules/robot_action_composer/README.md) | 编排分层和命令行 |
| [submodules/robot_action_composer/docs/ROS2_STACK.md](../submodules/robot_action_composer/docs/ROS2_STACK.md) | `ros2-stack` 如何拉起运控和导航 |
| [submodules/robot_action_composer/docs/TASK_CONFIG_YAML.md](../submodules/robot_action_composer/docs/TASK_CONFIG_YAML.md) | 任务 YAML 如何合并 |
| [submodules/robot_action_composer/docs/SKILLS_REFERENCE.md](../submodules/robot_action_composer/docs/SKILLS_REFERENCE.md) | 每个技能的参数 |
| [submodules/robot_action_composer/docs/ROBOT_CONFIG.md](../submodules/robot_action_composer/docs/ROBOT_CONFIG.md) | `robot.yaml` 与机器人目录 |
| [submodules/robot_action_composer/docs/GRASP_GENERATION.md](../submodules/robot_action_composer/docs/GRASP_GENERATION.md) | 抓取命令行和 Viser |
| [submodules/robot_action_composer/docs/CLOSE_EXECUTORS.md](../submodules/robot_action_composer/docs/CLOSE_EXECUTORS.md) | 合手的几种执行方式 |
| [ENV_AND_HAND_CONTRACT.md](ENV_AND_HAND_CONTRACT.md) | 新场景至少要有哪些物体和手；接触力做在手的资产上 |

## 子模块

| 文档 | 何时读 |
|------|--------|
| [INTERNAV_W2_SUBMODULE.md](INTERNAV_W2_SUBMODULE.md) | W2 DualVLN 胶水仓 |
| [INTERNAV_ARX_LIFT2S_SUBMODULE.md](INTERNAV_ARX_LIFT2S_SUBMODULE.md) | Lift2S DualVLN 胶水仓 |
| [LIGHTNAV_W2_SUBMODULE.md](LIGHTNAV_W2_SUBMODULE.md) | W2 LightNav-0 胶水仓 |
| [HUG_SUBMODULE.md](HUG_SUBMODULE.md) | HUG 公司仓，以及哪些文件不能提交 |
| [WUJI_RETARGETING_SUBMODULE.md](WUJI_RETARGETING_SUBMODULE.md) | 重定向公司仓；初始化时不要递归进嵌套仿真 |
| [WUJI_RETARGETING.md](WUJI_RETARGETING.md) | `wuji-sdk` 和开源仓库各管什么 |
| [submodules/hug/docs/SETUP_LOCAL.md](../submodules/hug/docs/SETUP_LOCAL.md) | MANO 目录和模型下载 |

LeRobot 从 PyPI 安装，版本在 `.fa-env.toml`。`submodules/lerobot` 不是本仓库的子模块。

## 参考实现笔记

下面几篇是 FiveAges W2 加 Wuji Hand2 联调时的测量记录。换机器人时改配置，不要把这里的坐标系和话题当成唯一写法。可运行的配置在 `examples/IsaacSim/robots/FiveAges_W2/`。

| 文档 | 内容 |
|------|------|
| [SIM_GRASP_EE_ALIGN.md](SIM_GRASP_EE_ALIGN.md) | 手腕目标对应装手后的工具坐标，下发的是末端位姿 |
| [SIM_GRASP_HUG_TEST_GUIDE.md](SIM_GRASP_HUG_TEST_GUIDE.md) | 头相机图像进入 HUG 时，各条命令在检查什么 |
| [SIM_GRASP_WUJI_HAND2.md](SIM_GRASP_WUJI_HAND2.md) | 参考手的关节顺序和指尖偏移 |
| [SIM_GRASP_ASSET_HANDOFF.md](SIM_GRASP_ASSET_HANDOFF.md) | 参考场景的相机、话题和运控类型 |
| [SIM_GRASP_MILESTONE_AND_ROADMAP.md](SIM_GRASP_MILESTONE_AND_ROADMAP.md) | 实现顺序：先手腕，再重定向，再合手 |
