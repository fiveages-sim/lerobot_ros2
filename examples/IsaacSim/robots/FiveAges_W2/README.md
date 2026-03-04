# FiveAges W2

双臂人形机器人，支持以下 Isaac Sim 任务：

| 任务 | 类型 | 场景预设 | 数据采集 |
|------|------|----------|----------|
| Pick Place | 单臂抓取放置 | `grab_medicine`、`grab_bottle` | - |
| Handover | 双臂交接 | `grab_medicine`、`grab_bottle` | - |

配置文件：

- `robot_config.py` — 机器人参数（关节、阈值等）
- `task_configs/pick_place.py` — 抓取放置任务配置
- `task_configs/handover.py` — 双臂交接任务配置

ROS2夹爪控制器参数：
```yaml
right_gripper_controller:
  ros__parameters:
    joint: right_gripper_joint
    force_feedback_ratio: 0.01
    force_threshold: 1.5
```