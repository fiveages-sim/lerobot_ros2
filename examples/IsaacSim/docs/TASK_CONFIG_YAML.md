# 任务配置：YAML 说明

## 是否可以用 YAML？

可以。`examples/IsaacSim/common/task_config_io.py` 中的 `discover_task_configs()` 会为每个**基名**（如 `pick_place`）加载**一种**来源：

- **`pick_place.yaml`** 或 **`pick_place.yml`**，或  
- **`pick_place.py`**（导出 `TASK_CONFIG` / `FLOW_CONFIG`）

**同一基名不能同时存在 `.py` 与 `.yaml`/`.yml`**，否则会 `ValueError`，避免两套真源冲突。

`motion_generation.py` 与 `robots/registry_loader.py`（录制 / 推理注册表）共用该发现逻辑。

## 依赖

解析 YAML 需要 **PyYAML**：

```bash
pip install pyyaml
```

（子模块 `ros2_robot_interface` 的 `pyproject.toml` 已声明 `pyyaml`，随该环境安装即可。）

## 类型注意

- YAML 里没有 Python `tuple`，向量请写成列表，例如 `grasp_orientation: [0.7, 0.7, 0.0, 0.0]`。加载时会将**仅含数字的非空列表**规范化为 `tuple`，供 `PickPlaceFlowTaskConfig` 等 dataclass 使用。
- `null` → Python `None`。
- 布尔值为 `true` / `false`（小写，YAML 1.2 常见写法）。

## 示例

- `robots/DobotCR5/task_configs/pick_place.yaml`
- `robots/DobotCR5/task_configs/drawer_pick_place.yaml`
