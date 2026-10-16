# 文档索引

先看 [`SETUP.md`](SETUP.md)：从零把任务编排跑起来，以及可选的灵巧手抓取。下面按用途分组。阶段笔记里的「规划中」是当时的进度，坑和轴约定仍然有效；换机器人时不要把参考实现里的 prim、话题抄成唯一写法。

## 上手

| 文档 | 何时读 |
|------|--------|
| [`SETUP.md`](SETUP.md) | 克隆之后配环境、决定要不要灵巧手 |
| [`ENV_THIS_CHECKOUT.md`](ENV_THIS_CHECKOUT.md) | 环境混用、系统 pip、HUG venv 和父仓 venv |
| [`examples/IsaacSim/README.md`](../examples/IsaacSim/README.md) | Isaac Sim、ROS 2 仿真服务、机器人列表 |
| [`submodules/robot_action_composer/README.md`](../submodules/robot_action_composer/README.md) | 编排分层、技能表、CLI |
| [`submodules/robot_action_composer/docs/GRASP_GENERATION.md`](../submodules/robot_action_composer/docs/GRASP_GENERATION.md) | 抓取 CLI、Viser、三段接口（MANO / retarget / close） |
| [`submodules/robot_action_composer/docs/SKILLS_REFERENCE.md`](../submodules/robot_action_composer/docs/SKILLS_REFERENCE.md) | 任务 YAML 里每个 skill 的参数 |
| [`submodules/robot_action_composer/docs/TASK_CONFIG_YAML.md`](../submodules/robot_action_composer/docs/TASK_CONFIG_YAML.md) | 任务文件怎么合并 |
| [`submodules/robot_action_composer/docs/ROBOT_CONFIG.md`](../submodules/robot_action_composer/docs/ROBOT_CONFIG.md) | `robot.yaml` 与 `robots/` 目录 |
| [`submodules/robot_action_composer/docs/ROS2_STACK.md`](../submodules/robot_action_composer/docs/ROS2_STACK.md) | `ros2-stack` 与场景级运控 |
| [`submodules/robot_action_composer/docs/CLOSE_EXECUTORS.md`](../submodules/robot_action_composer/docs/CLOSE_EXECUTORS.md) | 抓取收拢 B0–B5；B2 开环和 B4 门控不是一回事 |
| [`ENV_AND_HAND_CONTRACT.md`](ENV_AND_HAND_CONTRACT.md) | 新环境最少要什么；接触力在手资产上；按手型选执行器 |

## 子模块从哪来

| 文档 | 何时读 |
|------|--------|
| [`HUG_SUBMODULE.md`](HUG_SUBMODULE.md) | 公司仓与上游的关系；哪些路径不能提交 |
| [`WUJI_RETARGETING_SUBMODULE.md`](WUJI_RETARGETING_SUBMODULE.md) | retarget 公司仓；不要 `--recursive` 拉嵌套仿真 |
| [`WUJI_RETARGETING.md`](WUJI_RETARGETING.md) | 21 点进、关节角出；`wuji-sdk` 与开源仓的分工 |
| [`submodules/hug/docs/SETUP_LOCAL.md`](../submodules/hug/docs/SETUP_LOCAL.md) | MANO 目录、HF 镜像、离线 DINOv2 |

`submodules/lerobot` 不是子模块，父仓不跟踪它。LeRobot 从 PyPI 安装，版本在 `.fa-env.toml`。

## 参考实现笔记（FiveAges W2）

这些是联调时留下的测量和弯路，不是「只支持这一台机器人」的说明。当前可跑的配置在 `examples/IsaacSim/robots/FiveAges_W2/`。

| 文档 | 保留的内容 |
|------|------------|
| [`SIM_GRASP_EE_ALIGN.md`](SIM_GRASP_EE_ALIGN.md) | 腕目标对的是装手的 tcp，下发的是 eef；live TF，不要把腕 XYZ 直接当掌心 |
| [`SIM_GRASP_HUG_TEST_GUIDE.md`](SIM_GRASP_HUG_TEST_GUIDE.md) | 头相机 RGB-D 进 HUG 时，命令分别在验证什么 |
| [`SIM_GRASP_WUJI_HAND2.md`](SIM_GRASP_WUJI_HAND2.md) | 参考手型的关节序、tip 约 7 cm、`mano_to_tip_rpy` |
| [`SIM_GRASP_ASSET_HANDOFF.md`](SIM_GRASP_ASSET_HANDOFF.md) | 参考场景的相机 prim、话题、运控 `type` |
| [`SIM_GRASP_MILESTONE_AND_ROADMAP.md`](SIM_GRASP_MILESTONE_AND_ROADMAP.md) | 阶段划分：先腕，再 retarget，再 close；HUG 保持黑盒 |
