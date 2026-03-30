# IsaacSim 任务队列架构（设计草案）

目标：**从「每种任务一个大模板 + 一个 `run_*_demo`」**，演进为 **「可自由组合的技能块队列 + 统一执行器」**，便于后续用配置或脚本拼装长流程（多段 pick / place / 抽屉 / 递物 / 搬运等）。

---

## 1. 现状与痛点

| 层面 | 现状 | 问题 |
|------|------|------|
| 入口 | `motion_generation.py` 按 `kind` 分发到不同 `run_*_demo` | 新增组合 = 新 kind 或复制粘贴整段流程 |
| 配置 | `PickPlaceFlowTaskConfig` / `HandoverTaskConfig` 等并列 | 语义是「整任务」，不是「可复用步骤」 |
| 执行 | 底层已是 `list[StageTarget]` + `execute_stage_sequence` | 但拼装逻辑散在各自 `run_*` 里，**SendMode 切换**（单臂 / 双臂同步）写死在分支中 |
| 段间状态 | 如 drawer：前一段的几何影响后一段 `place_pose` | 缺少「流水线上下文」，难以在配置里拆成独立块 |

底层能力（`ros2_robot_interface` 的 `StageTarget`、`compose_bimanual_synchronized_sequence` 等）**已经足够组合**；缺的是 **统一的「技能 → 阶段」契约和运行时上下文**。

---

## 2. 目标形态（概念）

### 2.1 三层模型

1. **技能（Skill / Block）**  
   - 输入：只读的 **机器人与场景配置** + **可变运行时上下文（Context）**  
   - 输出：**一段** `list[StageTarget]`，以及本段所需的 **`ExecutionMeta`**（如 `SendMode`、可选的 `left_arrival_guard_stage`、frame_id 规则等）。
   - 技能本身**不负责** `reset_simulation`、连接机器人等；由外层 Runner 统一做。

2. **任务队列（TaskQueue）**  
   - 有序列表：`[BlockSpec, BlockSpec, ...]`  
   - 每格 `BlockSpec`：`block_id`（字符串，用于日志）、`skill`（注册名）、`params`（该技能专用参数，可为嵌套 dict / 小型 dataclass）。
   - 可选：**条件 / 循环** 可第二期再做；第一期先做线性队列。

3. **执行器（QueueRunner）**  
   - 依次：更新 Context（如前一步输出坐标）→ 调注册表解析 `skill` → `build_stages(...)` → `execute_stage_sequence(...)`。  
   - 统一处理：`use_stamped`、异常与 `FSM_HOLD`。关节回零由可选技能 ``single_arm.movej_return_initial`` 编排（Runner 仅缓存连接时关节角到 ``ctx``）。

### 2.2 运行时上下文（Context）——组合任务的关键

**命名空间**：通用单臂字段放在 `SingleArmMotionContext`；**家具机构**（抽屉、**柜门/开关柜门** 等）应使用独立子对象，例如当前的 ``ctx.drawer: DrawerPhaseState``、未来的 ``ctx.door``，避免上下文变成「上帝结构体」。

建议 Context 中显式保存（只列方向，具体字段实现时再定）：

- 双臂 home / 初始关节、`ee_frame_id`、仿真时间工具  
- **可写「上一步结果」**：例如 `last_object_pose`、`drawer_place_reference`、`active_arm`  
- **命名空间参数**：与 `params` 合并策略（scene 预设覆盖 block 默认）

这样「抽屉 → 取放」可以拆成两个技能块，第二块从 Context 读抽屉阶段写入的参考位姿，而**不必**再写第三个「巨型模板」。

---

## 3. 与当前 `flatten_*` / 分层配置的关系

- **短期**：配置文件可从「整任务 overrides」变为 **「队列 + 每块的参数字典」**；若仍存在「单块内 pick/place 分层」，仍可在 **加载层** 合并成小 dataclass，等价于今天的 `flatten`，只是作用域缩小到**块**而不是整任务。
- **长期**：每种 **skill** 自带自己的小配置类型（如 `PickSkillParams`），**不再需要**一个包罗万象的 `PickPlaceFlowTaskConfig` 驱动所有流程；大 dataclass 可拆碎或仅保留为「兼容旧入口的 facade」。

---

## 4. 需要提前约定的契约

1. **SendMode 策略**  
   - 每个技能返回的 meta 必须声明本段是 `STAMPED` / `UNSTAMPED` / `DUAL_ARM_STAMPED`。  
   - 当前 handover「先单臂 pickup 再双臂」= 队列里两段，或单技能内返回「已切段的子列表」；**推荐**拆成两个逻辑块，队列执行两次 `execute_stage_sequence`，与现有行为一致。

2. **命名与阶段前缀**  
   - 保留可读 `stage.name`（便于超时日志与录制对齐）；技能内命名规范统一（如 `Pick-{robot_id}-...`）。

3. **注册表**  
   - `register_skill("pick.single_arm", build_fn)`；配置里只写字符串 key，避免 import 地狱。

4. **仿真重置策略**  
   - 块级可选字段 `reset_env: bool` / `randomize_entity: str | None`，由 Runner 在块边界执行，而不是每个技能内部随意 reset。

---

## 5. 分阶段落地（建议）

| 阶段 | 内容 | 产出 |
|------|------|------|
| **0** | 本设计评审 + 锁定术语（Skill / Block / Context） | 共识 |
| **1** | 新建 `common/task_runtime/`：`SingleArmMotionContext`、`ExecutionMeta`、`run_single_arm_task_queue`；通用技能 ID `single_arm.{pregrasp,pick,place,return_home}`（与具体机器人品牌无关） | **已实现**：`robots/DobotCR5/task_configs/pick_place.py` 中示例 ``task_queue``；`motion_generation` 在存在 ``task_queue`` 时走队列路径；其他单臂机器人可复用同一技能 ID |
| **2** | 把 `run_pick_place_demo` 主路径改为「内置默认队列」调用 Runner；旧 API 保留薄封装 | 行为对齐、回归容易 |
| **3** | 迁移 handover / bimanual_carry / drawer：各拆成 2～N 个 skill + 块间写 Context | `motion_generation.py` 逐步去掉大分支 |
| **4** | `TASK_CONFIG` 支持 `"task_queue": [...]`；`kind` 可deprecated 为 `queue_v1` 或并存 | 配置驱动组合 |
| **5**（可选） | 可视化 / 录制episode与 block_id 对齐；策略里插入条件分支 | 产品化 |

---

## 6. 风险与缓解

- **状态泄漏**：多 skill 共享 Context 要有清晰「由谁写入、谁读取」文档；复杂流可上 Session/Trace 对象。  
- **类型散**：每 skill 一个小 `Params` dataclass，用 Union 或按 skill 名分 typing。  
- **一次性大改**：严格按阶段 1→2 **先跑通再迁移**，避免停产式重构。

---

## 7. 结论

- **「彻底改进」方向正确**：任务队列 + 技能注册 + Context 与当前 `StageTarget` 流水线对齐，是从「模板复制」到「自由组合」的自然演进。  
- **`flatten_pick_place_task_overrides` 之类**：在阶段 1–2 仍可保留为**块级参数加载**工具；阶段 4 起应由 **per-skill 配置** 逐步取代「超大扁平 dataclass」。  
- **建议下一步**：实现阶段 1 的 **Runner + 两个技能 + 一条 YAML/Python 队列**，不碰 handover/drawer，验证 SendMode 与 stamp 规则在队列边界上正确。

---

任务配置文件除 `.py` 外，可为 **YAML**（与同名 `.py` 互斥）：见 `TASK_CONFIG_YAML.md`。

**抽屉任务**：`kind: drawer` 且配置 ``task_queue`` 时，走 ``run_drawer_pick_place_task_queue``；在通用 ``single_arm.pregrasp`` / ``pick`` / ``place`` 之间插入 ``single_arm.drawer.pull_open``、``close_push``、``retreat_to_home``（``close_push`` 内为与旧版一致的中间步，在技能内执行部分 ``execute_stage_sequence``）。

*文档版本：草案 v0.1 — 随实现迭代可修改「Context 字段」与「阶段划分」。*
