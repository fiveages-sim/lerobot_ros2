# Isaac Sim + LeRobot ROS2（Dobot CR5）

本文档用于说明如何在 Isaac Sim 中完成 Dobot CR5 的抓取任务、数据采集、训练与推理。

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

说明：`Grasp_Apple`场景中已预先配置好物体属性服务（Prim Service），通常无需额外手动配置。

## 3. 启动仿真与 ROS2 控制

### 3.1 Isaac Sim 侧

- 按 [robot_usds](https://github.com/fiveages-sim/robot_usds) 配置机器人与场景资产。
- 打开并运行 `Grasp_Apple.usd`。

### 3.2 ROS2 控制侧

- 按 [open-deploy-ws](https://github.com/fiveages-sim/open-deploy-ws) 配置 ROS2 工作空间。
- 编译 Dobot CR5 的 `description`、控制器与 topic-based ros2 control。
- 确认可通过 OCS2 arm controller 控制仿真机器人。

演示视频：  

https://github.com/user-attachments/assets/78cad128-95e7-475b-828f-d12a9ff4b84e

## 4. 功能脚本

### 4.1 单次抓取放置

```bash
python scripts/grasp_single_demo.py
```

演示视频：  

https://github.com/user-attachments/assets/d75b44b9-e6a4-4131-bd86-3ea269ea96aa

### 4.2 录制数据集

```bash
python scripts/grasp_record_dataset.py
```

### 4.3 转换为 LeRobot 数据集

```bash
python scripts/data_convert_to_lerobot.py
```

说明：默认视频编码为 AV1。建议使用 Cursor 内置播放器，或升级 VLC 到 4.0 及以上版本。

### 4.4 训练模型

```bash
python scripts/train.py
```

`train.py` 已内置常用 ACT 默认参数（`chunk-size=16`、`n-action-steps=8`、`steps=20000`、`batch-size=8`、`device=cuda`）。

### 4.5 推理模型

```bash
python scripts/inference.py
```

演示视频：  

https://github.com/user-attachments/assets/87a48b3b-db63-41e1-9e7b-be89aa801782


