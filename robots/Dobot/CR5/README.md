# Dobot CR5

单臂协作机器人，支持以下 Isaac Sim 任务：

| 任务 | 类型 | 场景预设 | 数据采集 |
|------|------|----------|----------|
| Pick Place | 单臂抓取放置 | `apple`, `nailong` | 支持 |
| Drawer + Pick Place | 开抽屉 → 取放 → 关抽屉 | `drawer_001`, `drawer_002` | 支持 |

配置文件：

- `robot_config.py` — 机器人参数（关节、阈值等）
- `task_configs/pick_place.yaml` — 抓取放置任务（嵌套 `pick` / `place` 等；须含非空 **`task_queue`**）。说明见 `docs/TASK_CONFIG_YAML.md`。
- `task_configs/drawer_pick_place.yaml` — 抽屉复合任务（须含非空 **`task_queue`**；末尾为 **`single_arm.goto_cache_pose`** 回到 **`robot.cache_ee_pose`** 写入的起势位姿）。

## 演示视频

ROS2 控制仿真机器人：

https://github.com/user-attachments/assets/78cad128-95e7-475b-828f-d12a9ff4b84e

单次抓取放置：

https://github.com/user-attachments/assets/d75b44b9-e6a4-4131-bd86-3ea269ea96aa

模型推理：

https://github.com/user-attachments/assets/87a48b3b-db63-41e1-9e7b-be89aa801782
