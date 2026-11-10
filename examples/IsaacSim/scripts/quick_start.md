# G1D 中石化试管任务 — 半自动数据采集 Quick Start

> 分支：`feature/g1d-sinopec-auto-collect`  
> **请勿在未评审前合并到 `main`。**  
> English: [quick_start.EN.md](./quick_start.EN.md)

本流程用于采集 **右臂取管 → 双手交接 → 左臂放管** 的 ACT 训练数据（pose 模式 + 夹爪 `target_percent`）。

---

## 1. 需要哪些仓库 / 分支

| 仓库 | 检出分支 | 作用 |
|------|----------|------|
| `lerobot_ros2` | `feature/g1d-sinopec-auto-collect` | `auto_collect_g1d_tube.sh`、任务 YAML、孔位 meta |
| `data-collector-platform` | `feature/g1d-sinopec-auto-collect` | 订阅夹爪 `target_percent`、G1D 关节/pose 配置 |
| `FaSim-Isaac` | `feature/g1d-sinopec-auto-collect`（可选） | 冷启动 Play 时更长的 ros2_control activate 等待 |

```bash
git -C ~/lerobot_ros2 fetch origin && git -C ~/lerobot_ros2 checkout feature/g1d-sinopec-auto-collect
git -C ~/data-collector-platform fetch origin && git -C ~/data-collector-platform checkout feature/g1d-sinopec-auto-collect
# 可选
git -C ~/FaSim-Isaac fetch origin && git -C ~/FaSim-Isaac checkout feature/g1d-sinopec-auto-collect
```

---

## 2. 环境约定

与现有 FaSim / ros2_control 习惯一致，例如：

```bash
export RMW_IMPLEMENTATION=rmw_zenoh_cpp
export ROS_DOMAIN_ID=72          # 按团队实际修改
export ROS_USE_SIM_TIME=true
```

原始数据与转换输出默认路径（脚本可覆盖）：

- 原始：`/data/agent_client_installer/Agent/raw_data`
- LeRobot / 转换结果：`/data/agent_client_installer/Agent/data_collected`

---

## 3. 启动顺序（每次采集会话）

### 3.1 Isaac Sim

1. 打开 Sinopec 场景中的 G1D `room.usda`（FaSim 工程路径以团队为准）。
2. **第一次 Play** 常出现 PhysX tensor “Simulation view object is invalidated” 刷屏：等刷完后 **Stop → 等约 5s → 再 Play**。  
   第二次 Play 应看到控制器激活成功（如 `Successfully switched controllers` / HOLD）。  
   **不要在日志写 `ignoring Stop during activate` 时点 Stop。**
3. 确认话题存在：`/robot_description`、`/ocs2_wbc_controller/current_state`、相机与夹爪相关 topic。

### 3.2 数据采集器

```bash
cd ~/data-collector-platform/unified_data_platform
# 按团队习惯 activate conda / source ROS
python unified_collector_converter.py
```

看到 `Node initialized` 即可；脚本会通过 `/teleop/controller_state` 发 `50` 开始录、`51` 停止录。

### 3.3 半自动采集脚本

```bash
cd ~/lerobot_ros2/examples/IsaacSim
bash scripts/auto_collect_g1d_tube.sh
# 或指定默认集数提示：bash scripts/auto_collect_g1d_tube.sh 20
```

---

## 4. 交互流程

1. **How many episodes?** — 输入计划采集条数（可中途 `q` 提前结束）。
2. 每一集：
   - **start hole 1–6**：右侧架 `test_tube_rack_01` 取管孔（默认 6）
   - **place hole 1–6**：左侧架 `test_tube_rack` 放管孔（默认 3）→ 对应 scene `place_hole_N`
3. 脚本自动：
   - 双臂重新使能 → FSM **HOME** → 张开夹爪  
   - 把试管 teleport 到起始孔  
   - 发 `50` 开始录 → `motion-generation … --no-reset` → 发 `51` 停录  
4. **Keep this episode? [y/N]** — 目视 Isaac：取管/交接/放入目标孔是否成功。  
   Soft CSV 会检查左右夹爪命令是否含 `0/1`（失败也可强制 keep，但建议丢弃）。
5. **Continue collecting? [y / q]** — 继续或退出。

默认 **不会** 对 Isaac 做 Stop/Play。仅在你明确需要整场景重置时加 `--with-sim-reset`（慢，且需自己等控制器再次就绪）。

---

## 5. 数据落盘与转换

- 每条原始 episode：`raw_data/YYYYMMDD_HHMMSS_xxxxxx/`
- Collector 后台会排队转成 LeRobot 格式到 `data_collected/`
- Keep/Discard 只影响是否保留该 raw（及对应输出目录）；**不必等转换完成再决定 keep**

丢弃时脚本会删除对应 `raw_data` 与 `data_collected` 下同名目录。

---

## 6. 常见问题

| 现象 | 处理 |
|------|------|
| 等待 stack：`ocs2_wbc=False` 但 `ros2 topic list` 有 topic | 多为 QoS：`current_state` 是 TRANSIENT_LOCAL；本脚本已按 latched 订阅。确认第二次 Play 已激活控制器。 |
| 冷启动 PhysX 刷屏 | Stop → 等几秒 → Play；勿在 activate 中途 Stop。 |
| 上集结束后手臂停在空中 | 本脚本每集前会 HOME；若 HOME 失败请检查 `/fsm_command` 与双臂 mode。 |
| 试管没放进孔 / 插歪 | Discard 该集；属运动/对齐问题，可稍后调 `pick_tube_full.yaml` 的 insert 参数。 |
| motion-generation 报 `robot_description` / BASE_LOCK 超时 | 栈未就绪或上次 Stop 打断了 activate；按 §3.1 重新 Play。 |
| 夹爪 CSV 全是 1 或缺失 | 确认 collector 在本 feature 分支，且订阅了 `/*/target_percent`。 |

---

## 7. 场景与物体约定

- 右架（取）：`/World/room/scene/test_tube_rack_01`，孔 `hole_1`…`hole_6`
- 左架（放）：`/World/room/scene/test_tube_rack`
- 试管：`/World/room/scene/test_tube`
- 任务：`pick_tube_full`，scene：`place_hole_N`
- 全程 `BASE_LOCK`（底盘不挪），便于 ACT 复现末端轨迹

---

## 8. 给队友的最短命令清单

```bash
export RMW_IMPLEMENTATION=rmw_zenoh_cpp ROS_DOMAIN_ID=72 ROS_USE_SIM_TIME=true

# 终端 A：Isaac 已 Play 且控制器 OK
# 终端 B：
cd ~/data-collector-platform/unified_data_platform && python unified_collector_converter.py

# 终端 C：
cd ~/lerobot_ros2/examples/IsaacSim && bash scripts/auto_collect_g1d_tube.sh
```

问题反馈时请附上：`/tmp/g1d_auto_collect/run_*.log` 与 Isaac 中 activate / HOLD 相关日志片段。
