# FiveAges W2

双臂人形机器人，支持以下 Isaac Sim 任务：

| 任务 | 类型 | 场景预设 | 数据采集 |
|------|------|----------|----------|
| Pick Place | 单臂抓取放置 | `grab_medicine`、`grab_bottle` | - |
| Handover | 双臂交接 | `grab_medicine`、`grab_bottle` | - |
| Bimanual Carry | 双臂同步搬运 | `box01`、`warehouse_box01` | - |

配置文件：

- `robot_config.py` — 机器人参数（关节、阈值等）
- `task_configs/*.yaml` — 任务配置均为 **YAML**（需 PyYAML）。`pick_place`、`handover`、`bimanual_carry`、`navigate_and_pick`、`navigate_and_carry` 等见各文件与 `docs/TASK_CONFIG_YAML.md`。

ROS2夹爪控制器参数：
```yaml
right_gripper_controller:
  ros__parameters:
    joint: right_gripper_joint
    force_feedback_ratio: 0.01
    force_threshold: 1.5
```



https://github.com/user-attachments/assets/87fb7c00-027f-4e23-ad9b-9f2610029a94



https://github.com/user-attachments/assets/012bd4f9-c59b-4c3c-ac9a-66959adbb9ac


### PTC Demo
```bash
source ~/ros2_ws/install/setup.bash
ros2 launch ocs2_arm_controller full_body.launch.py type:=rg75 robot:=fiveages_w2 hardware:=isaac
```