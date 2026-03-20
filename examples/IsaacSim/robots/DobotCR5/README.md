# Dobot CR5

单臂协作机器人，支持以下 Isaac Sim 任务：

| 任务 | 类型 | 场景预设 | 数据采集 |
|------|------|----------|----------|
| Pick Place | 单臂抓取放置 | `apple`, `nailong` | 支持 |
| Drawer + Pick Place | 开抽屉 → 取放 → 关抽屉 | `drawer_001`, `drawer_002` | 支持 |

配置文件：

- `robot_config.py` — 机器人参数（关节、阈值等）
- `task_configs/pick_place.yaml` — 抓取放置任务（**YAML**；与同名 `.py` 二选一）。`base_task_overrides` / `scene_presets` 仍可用嵌套 `pick` / `place`；需要 **PyYAML**。说明见 `docs/TASK_CONFIG_YAML.md`。若含 **`task_queue`**，`motion_generation.py` 会走 `run_single_arm_task_queue`（技能含末尾 **`single_arm.movej_return_initial`** 做关节回零；可从队列中删掉该块以禁用 MoveJ）；去掉 `task_queue` 则由 `run_pick_place_demo` 在末尾统一 MoveJ。
- `task_configs/drawer_pick_place.yaml` — 抽屉复合任务（`drawer_pick_place` / **Drawer Pick Place**）；默认 ``task_queue`` 末尾含 **`single_arm.movej_return_initial`**（与 pick_place 队列相同机制）。去掉 ``task_queue`` 时仍走旧版 `run_drawer_demo`（末尾无 MoveJ）。

## 演示视频

ROS2 控制仿真机器人：

https://github.com/user-attachments/assets/78cad128-95e7-475b-828f-d12a9ff4b84e

单次抓取放置：

https://github.com/user-attachments/assets/d75b44b9-e6a4-4131-bd86-3ea269ea96aa

模型推理：

https://github.com/user-attachments/assets/87a48b3b-db63-41e1-9e7b-be89aa801782
