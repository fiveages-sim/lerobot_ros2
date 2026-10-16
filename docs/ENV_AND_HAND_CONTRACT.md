# 仿真环境、手型与执行器的边界

编排不打开 USD。它只认配置里的机器人、话题和 prim 路径。仿真资产放在 FaSim-Isaac（或以后别的资产仓），本仓不把某一张桌子、某一只手写进技能调度。

当前仍在用的参考环境：

`FaSim-Isaac/robots/humanoid/FiveAges/Gen2/W2/env/wuji_hand2_new.usda`

参考手型是 Wuji Hand2，参考收拢是 `close_executor: B4`。下面说的是换环境、换手、换 retarget 或换执行器时各自该动哪一层。现在这条参考链路可以继续用，不必为了「通用」先改 B4。

## 进编排最少要什么

普通任务（导航、笛卡尔、夹爪）只需要：

1. **一台运控能驱动的机器人。** `robots/<Robot>/robot.yaml` 里的 `ros2_stack`、关节和末端坐标系，和仿真里实际在动的那台一致。
2. **要交互的物体 prim。** 路径写在该任务的 YAML（`object_prim_path` 等）里。换一张桌子或换一组道具，改这段配置，不改 `motion-generation`。

灵巧手抓取在这两条之上再加一份 `grasp_generation.yaml`：

- 头相机话题（HUG 要 RGB-D）
- `hand_profile`（哪只手）
- `grasp_objects` 里每个物体的 `pose_prim`，需要稳定点击点时再加 `click_prim`
- `close_executor`（这只手允许的收拢方式）

没有接触力话题，仍然可以进编排，也可以跟腕、做开环合手（B0–B2）。**接触力不是进编排的条件。**

## 接触力属于手资产，不属于场景

B3、B4、B5 会等手指碰到物体再停。力从哪来由**手的 USD 变体**决定，不由桌子或机身决定。

参考手型的资产说明在 FaSim：

`robots/dexhands/Wuji/Hand2/payloads/Sensor/CONTACT_FORCE_README.md`

那里已经把传感器挂在手 prim 上：`Side=right|left` 且 `Sensor=contact_force` 才发布 `/wuji_rh/*` 或 `/wuji_lh/*`（`in_contact`、`contact_wrench`、`contact_partners`）。`Sensor=none` 是默认。换臂、换 stage，只要挂的是这只手并打开这个变体，话题还在。Newton 下没有这套 PhysX ContactSensor，要先切回 PhysX。

因此：

| 想做的事 | 资产侧 | 本仓配置 |
|----------|--------|----------|
| 只导航、只动臂、夹爪任务 | 机器人 + 物体 prim | `robot.yaml` + 任务 YAML |
| 灵巧手开环合上（B0–B2） | 手能收关节即可，传感器可关 | `hand_profile` + `close_executor: B0` 等 |
| 接触门控（B3–B5，当前默认 B4） | 该手打开接触力变体，PhysX | 同一 `hand_profile`，执行器选 B3/B4/B5 |

新环境若忘了开传感器，B4 会一直等接触或乱冻。这是资产变体没配对，不是编排坏了。开环执行器不读这些话题。

## 按手型选执行方式，而不是全球共用一套 B

`close_executor` 写在**这台机器人的** `grasp_generation.yaml` 里，和 `hand_profile` 放在一起。换手时一起换，不要假设 B4 的「冻 MCP、再卷指尖」对每只手都成立。

一只手在本仓里是一小包，三块一起注册：

| 块 | 现在的位置 | 换的时候保持不变的契约 |
|----|------------|------------------------|
| Retarget | `grasp_generation/retarget/` | 输入 MANO 21 点（米，腕为原点），输出 `q`，长度等于该手自由度 |
| 执行 profile | `grasp_generation/execution/profiles/` | 关节分组、指尖关节、接触 link 名、左右话题前缀 |
| 收拢执行器 | `grasp_generation/execution/executors/` | 只通过 profile 解释 `q` 和接触；不写死 USD 路径 |

允许的搭配由这只手决定，例如：

- 参考手型 `wuji_hand2`：retarget 用 `wuji-sdk`，收拢可以用 B0–B5，其中 B3–B5 要求资产发布接触力。
- 以后的手：可以只实现开环执行器，或另写一套门控。没实现的 id 不要出现在那份 yaml 里。
- 不需要接触的手：`close_executor` 选开环，资产不必做 ContactSensor。

HUG 仍只出相机系人手。换 retarget 策略时改上表第一块，并让 `hand_profile` 指到它。`proposal.json` 里的 `T_camera_wrist` 和 `landmarks_3d` 保持不变，Viser 和 `dex_hand.auto_grasp` 不用跟着改。

## 新环境怎么接（不改执行器代码）

1. 场景里有已挂手的机器人，运控 profile 与 `robot.yaml` 一致。
2. 物体留在作者摆好的位置。编排不负责清空桌子。
3. 在 `grasp_generation.yaml` 的 `grasp_objects` 写 `pose_prim`（刚体，用来读位姿）。抓取点不在几何中心时，在物体上建 Xform，把路径写入 `click_prim`。
4. 任务 YAML 的 `scene_presets` 只覆盖「导航去哪个 prim」和「抓取用哪个 `object_key`」。物体几何细节留在 `grasp_objects`，避免两处各写一套互相漂移的路径时没人知道以谁为准——导航 prim 和抓取 prim 可以是同一个根，但要在任务文件里点明。
5. 要用 B4：确认该手的 `Sensor=contact_force`（或这只手自己的等价变体）在这个 stage 里是打开的。

参考场景里苹果、瓶子的 prim 和「不要把 `default_environment` 放进 `ignore_contact_partners`」是这条桌子的坑，写在 `examples/IsaacSim/robots/FiveAges_W2/grasp_generation.yaml`。新场景复制键，改路径，不要把新桌子的名字写进 Python。

## 现在还绑在参考手型上的地方

接口评论写了「执行器与具体手无关」，调用处还没有完全做到。下一只手不要只加一个 yaml 就认为 B4 能用。

- **Retarget 入口只构造了 Wuji Hand2。** `retarget/pipeline.py` 里 `hand_profile` 不是 `wuji_hand2` 会直接拒绝。新策略要在这里分发，而不是改 HUG。
- **接触订阅类是 Wuji 的话题形状。** `execution/contact/wuji_ros.py` 认 `/wuji_rh|wuji_lh/in_contact/<link>`。另一只手的话题不同，就另写一个 `ContactSensor`，由 `hand_profile` 选用。没有传感器时用已有的 `NullContactSensor`（开环）。
- **B3–B5 的「冻近端、卷指尖」按 Hand2 的每指 4 关节来分组。** profile 里的 `tip_joint_indices` 是正确挂钩，但自由度、link 名不同的手（例如更早对照过的 O7）不能原样套 B4。
- **手下发节流按 Hand2 控制器的 MOVEJ 时长（约 0.55 s）。** 写在 `retarget/pipeline.py` 的 `send_hand_q`。别的控制器时长不同，会把还在走的轨迹打断、手指弹回张开。这是控制器 profile 的参数，不该永远是常数。
- **接触过滤里有参考场景和参考手的名字。** 忽略列表含 `wuji_`、各指 link；保留列表含 `sm_`、`apple`、`bottle` 等。新物体若名字对不上，真接触会被当成桌面滤掉，B4 就不停。新场景应把忽略词放在 yaml 的 `ignore_contact_partners`，物体命名规则不要继续堆进这份默认元组。
- **配置默认值是 W2 + Hand2。** `GraspGenerationConfig` 没写的字段会落到参考相机 prim 和 `hand_profile: wuji_hand2`。新机器人的 yaml 要把相机、`hand_profile`、`close_executor` 写全。

这些是以后维护时会绊人的点，不是要求现在拆掉 B4。参考实现继续放在 `examples/IsaacSim/robots/FiveAges_W2/`。
