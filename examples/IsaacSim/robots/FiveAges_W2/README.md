# FiveAges W2

双臂人形机器人，支持以下 Isaac Sim 任务：

| 任务 | 类型 | 场景预设 | 数据采集 |
|------|------|----------|----------|
| Pick Place | 单臂抓取放置 | `grab_medicine`、`grab_bottle` | - |
| Handover | 双臂交接 | `grab_medicine`、`grab_bottle` | - |
| Bimanual Carry | 双臂同步搬运 | `box01`、`warehouse_box01` | - |

配置文件：

- `robot.yaml` — 机器人参数与可选 `ros2_stack`（运控/导航默认启动）
- `task_configs/*.yaml` — 任务编排 YAML（需 PyYAML）。见 `docs/TASK_CONFIG_YAML.md`
- `task_configs/<场景>/.meta/ros2_stack.yaml` — 场景级导航/运控覆盖（如 Wind turbo blade 的 `map` / `profile`）；见 `docs/ROS2_STACK.md`

启动运控与导航：

```bash
cd examples/IsaacSim
ros2-stack launch --robot fiveages_w2
```

交互菜单里可以选择任务组。例如叶片场景：

```bash
ros2-stack launch --robot fiveages_w2 --group "projets/siemens/Wind turbo blade" --force-nav
motion-generation --robot fiveages_w2 --task-key transfer_blade --ensure-ros2-stack
```

导航靠近后再用 Wuji Hand2 抓取。仿真场景是 FaSim-Isaac 的 `wuji_hand2_new.usda`。`robot.yaml` 里的 `robot_profile` 要改成这台电脑上运控工作空间的绝对路径（`config/robot.profiles/wuji_hand2.yaml`），不要提交这行改动。

```bash
ros2-stack launch --robot fiveages_w2 --group dex_hand
motion-generation --robot fiveages_w2 --task-key navigate_and_dex_grasp --scene apple
```

只在图像上点选、不导航时，用 `grasp-generation serve --robot fiveages_w2 --workspace .`。安装和依赖见仓库根目录 README。

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


### PTC Demo（手动 launch，等价于 `ros2_stack.motion.preset: ocs2-fullbody`）
```bash
source ~/ros2_ws/install/setup.bash
ros2 launch ocs2_arm_controller full_body.launch.py type:=rg75 robot:=fiveages_w2 hardware:=isaac
```