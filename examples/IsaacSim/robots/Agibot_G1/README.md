# Agibot G1

双臂人形机器人，支持以下 Isaac Sim 任务：

| 任务 | 类型 | 场景预设 | 数据采集 |
|------|------|----------|----------|
| Pick Place | 单臂抓取放置 | `grab_medicine`、`grab_bottle` | - |
| Handover | 双臂交接 | `grab_medicine` | 支持 |

配置文件：

- `robot_config.py` — 机器人参数（关节、阈值、相机等）
- `task_configs/pick_place.yaml` — 抓取放置任务配置（需 PyYAML；见 `docs/TASK_CONFIG_YAML.md`）
- `task_configs/handover.yaml` — 双臂交接任务配置



https://github.com/user-attachments/assets/77a536df-8430-497f-8710-7d60406aac06

