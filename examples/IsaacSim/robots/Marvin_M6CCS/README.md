# Marvin M6CCS

双臂机器人，支持以下 Isaac Sim 任务：

| 任务 | 类型 | 场景预设 | 数据采集 |
|------|------|----------|----------|
| Handover | 双臂交接 | `grab_medicine`、`grab_bottle` | 支持 |

配置文件：

- `robot_config.py` — 机器人参数（关节、阈值等）
- `task_configs/handover.py` — 双臂交接任务配置

ROS2夹爪控制器参数：
```yaml
right_gripper_controller:
  ros__parameters:
    joint: right_gripper_joint
    force_feedback_ratio: 0.05
    force_threshold: 0.05
```



https://github.com/user-attachments/assets/c11167b6-c935-4386-9a65-252192247d9d

