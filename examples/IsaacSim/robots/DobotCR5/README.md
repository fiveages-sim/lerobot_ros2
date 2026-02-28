# Isaac Sim + LeRobot ROS2（Dobot CR5）

本文档用于说明 Dobot CR5 在 Isaac Sim 中的抓取任务、数据采集、训练与推理。

通用环境与 Isaac ROS2 配置请先阅读：[`../../README.md`](../../README.md)

## 1. 启动仿真与 ROS2 控制

### 1.1 Isaac Sim 侧

- 按 [robot_usds](https://github.com/fiveages-sim/robot_usds) 配置机器人与场景资产。
- 打开并运行 `Grasp_Apple.usd`。
- `Grasp_Apple` 场景中已预先配置好 Prim Service，通常无需额外手动配置。

### 1.2 ROS2 控制侧

- 按 [open-deploy-ws](https://github.com/fiveages-sim/open-deploy-ws) 配置 ROS2 工作空间。
- 编译 Dobot CR5 的 `description`、控制器与 topic-based ros2 control。
- 确认可通过 OCS2 arm controller 控制仿真机器人。

演示视频：  

https://github.com/user-attachments/assets/78cad128-95e7-475b-828f-d12a9ff4b84e

## 2. 功能脚本

### 2.1 单次抓取放置

```bash
python ../../run_motion_generation.py
```

演示视频：  

https://github.com/user-attachments/assets/d75b44b9-e6a4-4131-bd86-3ea269ea96aa

### 2.2 录制数据集

```bash
python ../../record_datasets.py
```

脚本启动后会进入交互式入口，依次选择：
- 机器人（当前首版支持 `dobot_cr5`）
- 任务（当前首版支持 `pick_place_flow`）
- 录制配置（`default` / `fast`）
- 运行时选项（录制条数、是否录制点云、是否手动审核每条 episode）

说明：默认视频编码为 AV1。建议使用 Cursor 内置播放器，或升级 VLC 到 4.0 及以上版本。

### 2.3 训练模型

```bash
python ../../tools/DobotCR5/train.py
```

`train.py` 已内置常用 ACT 默认参数（`chunk-size=16`、`n-action-steps=8`、`steps=20000`、`batch-size=8`、`device=cuda`）。

### 2.4 推理模型

```bash
python ../../tools/DobotCR5/inference.py
```

演示视频：  

https://github.com/user-attachments/assets/87a48b3b-db63-41e1-9e7b-be89aa801782


