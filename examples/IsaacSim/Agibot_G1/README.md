# Isaac Sim + LeRobot ROS2（Agibot G1）

本文档说明 Agibot G1 在 Isaac Sim 下的双臂抓取与交接示例。

通用环境与 Isaac ROS2 配置请先阅读：[`../README.md`](../README.md)

## 1. 启动仿真与 ROS2 控制

### 1.1 Isaac Sim 侧

- 按 [robot_usds](https://github.com/fiveages-sim/robot_usds) 配置机器人与场景资产。
- 打开并运行 `medicine_handover.usd`。
- `Grasp_Apple` 场景中已预先配置好 Prim Service，通常无需额外手动配置。

### 1.2 ROS2 控制侧

- 按 [open-deploy-ws](https://github.com/fiveages-sim/open-deploy-ws) 配置 ROS2 工作空间。
- 编译 Dobot CR5 的 `description`、控制器与 topic-based ros2 control。
- 确认可通过 OCS2 arm controller 控制仿真机器人。

演示视频：  

https://github.com/user-attachments/assets/78cad128-95e7-475b-828f-d12a9ff4b84e


## 1. 当前示例能力

- 右臂先完成抓取
- 右臂移动到 handover 位，左臂同步接近
- 左臂到位后夹取，随后右臂松开
- 左臂将物体移动到放置位并释放
- 右臂回初始位，最后左臂回初始位

核心脚本：

- 机器人配置：`scripts/robot_config.py`
- 默认流程配置：`scripts/robot_config.py`（与机器人配置已合并）
- 任务配置（handover）：`scripts/handover_demo.py`（文件内 `HANDOVER_TASK_CFG`）
- 执行：`scripts/handover_demo.py`

## 2. 运行方式

在当前目录下执行：

```bash
python scripts/handover_demo.py
```

## 3. 关键配置项

在以下文件中可调：

- `scripts/robot_config.py`：
  - 抓取参数：`grasp_orientation`、`grasp_clearance`
  - 运行参数：`arrival_timeout`、`arrival_poll`、`gripper_action_wait`
  - `initial_grasp_arm`（`left` 或 `right`）
- `scripts/handover_demo.py`（`HANDOVER_TASK_CFG`）：
  - 交接位姿：`handover_position`、`source_handover_orientation`、`receiver_handover_orientation`
  - 放置位姿：`receiver_place_position` / `receiver_place_orientation`
  - 场景位姿服务对象：`source_object_entity_path`


## 4. 说明

- 夹爪控制当前由 robot 层参数 `gripper_control_mode` 驱动，并直接调用 interface 的 gripper handler 开关接口。
- 如需进一步扩展到完整双臂 handover 策略（避障、时序优化、回退策略），建议在当前 handover 阶段序列基础上继续配置化。
