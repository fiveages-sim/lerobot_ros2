# Isaac Sim + LeRobot ROS2（通用说明）

本文档说明 `examples/IsaacSim` 下不同机器人示例共享的环境与 Isaac ROS2 配置步骤。

## 1. 环境准备

### 1.1 版本与平台

- 操作系统：`Ubuntu 24.04`
- ROS2 发行版：`Jazzy`
- Isaac Sim：`6.0`
- 已测试平台：`RTX 5060 Laptop`、`RTX 4090`

### 1.2 一键初始化（推荐）

```bash
./init.sh install-plugins
```

### 1.3 推理运行时库配置（可选）

用于避免推理阶段的 `libtiff/libjpeg` 运行时冲突：

```bash
./init.sh conda-runtime
```

## 2. 配置 Isaac Sim 的 ROS2 能力

### 2.1 启停仿真控制（Simulation State Service）

1) 按官方文档安装并配置 ROS2 工作空间：  
[Setup ROS2 Workspaces](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/install_ros.html#setup-ros-2-workspaces)

2) 启用支持仿真控制的扩展：  
[Enabling Simulation Control Extension](https://docs.isaacsim.omniverse.nvidia.com/latest/ros2_tutorials/tutorial_ros2_simulation_control.html#enabling-the-extension)

![Isaac ROS2 Setup](.images/isaac_ros2.png)
![Isaac ROS2 Sim Control](.images/isaac_ros2_sim_control.png)

3) 验证仿真状态服务：

```bash
# 启动/继续仿真（1 = playing）
ros2 service call /set_simulation_state simulation_interfaces/srv/SetSimulationState "{state: {state: 1}}"
```

```bash
# 重置仿真（0 = reset）
ros2 service call /set_simulation_state simulation_interfaces/srv/SetSimulationState "{state: {state: 0}}"
```

### 2.2 物体属性服务（Prim Service）

用于读取/修改场景内物体属性（例如位置随机化）：  
[ROS2 Prim Service](https://docs.isaacsim.omniverse.nvidia.com/latest/ros2_tutorials/tutorial_ros2_prim_service.html)

## 3. 启动仿真与 ROS2 控制（通用）

### 3.1 Isaac Sim 侧

- 按 [robot_usds](https://github.com/fiveages-sim/robot_usds) 配置机器人与场景资产。
- 打开目标机器人对应的 USD 场景并运行仿真。

### 3.2 ROS2 控制侧

- 按 [open-deploy-ws](https://github.com/fiveages-sim/open-deploy-ws) 配置 ROS2 工作空间。
- 编译对应机器人的 `description`、控制器与 topic-based ros2 control。
- 确认可通过 OCS2 arm controller 控制仿真机器人。

## 4. 机器人专有文档

- [Dobot CR5](robots/DobotCR5/README.md) — 单臂协作机器人
- [Agibot G1](robots/Agibot_G1/README.md) — 双臂人形机器人
- [FiveAges W2](robots/FiveAges_W2/README.md) — 双臂人形机器人
- [Marvin M6CCS](robots/Marvin_M6CCS/README.md) — 双臂机器人

## 5. robot-action-composer 包

动作编排与录制逻辑在子模块 [`submodules/robot_action_composer`](../../submodules/robot_action_composer/)（可 `pip install -e submodules/robot_action_composer`）。  
本目录下的 `motion_generation.py`、`record_datasets.py` 与 `robots/registry_loader.py` 仅为薄封装；开发与调试可直接 `import robot_action_composer`。

**任务 YAML、`robot_config.py` 与 `robots/` 目录约定**以 composer 内文档为准（不再在 `examples/IsaacSim/docs` 重复维护正文）：

- [任务 YAML 说明](../../submodules/robot_action_composer/docs/TASK_CONFIG_YAML.md)
- [机器人配置说明](../../submodules/robot_action_composer/docs/ROBOT_CONFIG.md)
- [包架构与 README](../../submodules/robot_action_composer/README.md)

- **`train.py`** → **`policy_training/train.py`**：只读 **LeRobot 数据集**与训练配置，**不会**加载 `robots/*/robot_config.py`。要与仿真/实物一致，请用与录制时相同的相机与状态键（数据集 `meta` 即真源）。
- **`inference.py`**：在线推理时通过 **`robot_action_composer.discovery.registry_loader.load_motion_entries`** 与 **`robot_action_composer.task_config_io.flatten_queue_task_overrides`** 读取与 motion/录制 **同一套** `robots/` 资产；策略与 ROS2 循环在 **`online_infer/core.py`**；仿真服务仍用 `common/isaac_ros2_sim_common`。

任务队列运行时（runner、registry、内置 skills）与 **数据集录制**（`robot_action_composer.dataset_recording`、`cli.record_main`）均在 **`robot_action_composer`**；`common/` 保留 Isaac 仿真辅助（如 `isaac_ros2_sim_common`）。
