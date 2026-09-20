# Phase 2B 计划：HUG MANO → Wuji Hand2 指关节 Retarget

> 状态：规划中（2026-09）。**2A 腕跟已在 O7 验收**；当前先在 **Wuji Hand2** 上复现腕跟，再接官方 retarget。  
> 原则：HUG 仍为黑盒；手指差异走 `hand_profile` + retarget 适配层，不改 HUG 权重。

## 0. 仿真 / 运控切换（先做，复现腕跟）

| 项 | O7（已通） | Wuji Hand2（当前） |
|----|------------|-------------------|
| Isaac env | `.../env/dexhand_o7_new.usda` | `.../env/wuji_hand2_new.usda` |
| `robot.yaml` motion `type` | `linkerhand_o7` | **`wuji_hand2`** |
| 手装点 | `right_tcp` | 同（`tcp→hand_mount` + Rx(π)） |
| OCS2 tip | `right_eef` | 同（alias of `right_hand_eef`） |
| `hand_base→eef` | O7 ≈ `(0.03,0,0.1)` | Wuji ≈ **`(0,0,0.07)`** |
| 掌心 / 指尖轴 | tip +Z ≈ 伸出 | **约定对齐**（`hand_base` +Z = 指伸，与 O7 思路一致） |

腕跟代码路径不变：`hug_tip=right_tcp`，`control=right_eef`，**live TF** 吃掉 eef 偏置差；`mano_to_tip_rpy_deg: [0,-90,0]` 先沿用（掌心朝向一致时）。

冒烟：

```bash
# Isaac: 打开 wuji_hand2_new.usda 并 Play
cd examples/IsaacSim
ros2-stack stop --robot fiveages_w2
ros2-stack launch --robot fiveages_w2   # 确认 type:=wuji_hand2
# 另终端
grasp-generation diagnose-frames --robot fiveages_w2 --workspace .
grasp-generation diagnose-orientation --robot fiveages_w2 --workspace .
grasp-generation serve --robot fiveages_w2 --workspace .
```

关注：`right_eef − right_tcp` 的 `|Δ|` 应约 **7 cm 量级**（不是 O7 的 ~12 cm）；Capture→点选→Plan/Execute 掌跟仍贴 dry。

---

## 1. 资料结论（Wuji Retargeting）

正式入口（软件仓，非独立论文）：

- 开源算法仓：[wuji-technology/wuji-retargeting](https://github.com/wuji-technology/wuji-retargeting)（DexRetargeting 系）
- 算法附录：[docs …/wuji-retargeting/…/appendix](https://docs.wuji.tech/docs/en/wuji-retargeting/latest/appendix/)
- SDK 接口：[docs …/wuji-sdk/…/retargeting](https://docs.wuji.tech/docs/en/wuji-sdk/latest/retargeting/)
- **本仓集成说明：** [`WUJI_RETARGETING.md`](WUJI_RETARGETING.md)（I/O、双路径、与 HUG 同风格的干净依赖）

本地已就绪（2026-09）：`submodules/wuji-retargeting` clone + 父仓 `.venv` 中 `wuji-sdk`；`RetargetSession` 冒烟通过。

**产品化路径（推荐写入我们栈）：** `pip install wuji-sdk` → `RetargetSession`

### 右手冒烟（已实现）→ 现以 **serve** 为主界面

Viser（推荐）：

```bash
cd examples/IsaacSim
grasp-generation serve --robot fiveages_w2 --workspace .
# UI: arm left|right → Capture → click → Plan → 4) Retarget → 5) Close/squeeze
```

CLI 仍可用：

```bash
grasp-generation retarget-hand --robot fiveages_w2 --workspace . --arm left --squeeze 0.35
grasp-generation retarget-hand --robot fiveages_w2 --workspace . --execute   # 真下发
```

HUG 为右手 MANO：`hug_mano_is_right: true` 时左右臂都用 Right `RetargetSession` 出 `q`，再发到所选 `{arm}_hand_controller`。轴差调 `keypoint_rpy_deg`；贴面靠 `squeeze_amount`（不改腕位）。

```python
from wuji_sdk import Handedness, HandModel, RetargetSession
session = RetargetSession.for_hand(HandModel.WujiHand2, side=Handedness.Right)
q = session.step(keypoints_21x3)   # → (20,) float32，可直接下发
```

输入约定：

- `(21, 3)` **米**，MediaPipe 点序（腕=0 为原点）
- MANO 21 点序与 MediaPipe 同结构（腕 + 拇/食/中/无/小各 4 点）；HUG `landmarks_3d` 可经腕系化后喂入
- 输出 20 维与 `wuji_description` / `{side}_hand_controller` 关节顺序一致（thumb→pinky，含 flex/abd）

开源仓额外能力（调试用）：AdaptiveOptimizer（指尖方向 / 全手向量切换）、`segment_scaling`、pkl 回放、`tuning_tool`。量产联调优先 **SDK `RetargetSession`**，减少自维护优化器。

HUG 论文侧：把「人手关键点 → 灵巧手」交给下游；Wuji 方案即该下游的官方实现，适合 Hand2，不必自研 O7 式解析映射。

---

## 2. 目标架构（composer 内）

```text
                    ┌─────────────────────┐
  head RGB-D        │  hug_worker (3.10)  │
  + click UV   ───► │  → proposal.json    │
                    │  T_cam_wrist        │
                    │  landmarks_3d (21)  │
                    └──────────┬──────────┘
                               │
         ┌─────────────────────▼─────────────────────┐
         │         grasp_generation (ROS 3.12)        │
         │  bridge: T_base + tip map (tcp→eef)  2A ✓ │
         │  retarget/: hand_profile 插件         2B   │
         │    · wuji_hand2 → wuji_sdk.RetargetSession │
         │    · linkerhand_o7 → (可选，后做)          │
         │  service Viser: Propose / Plan腕 / 指      │
         └─────────────────────┬─────────────────────┘
                               │
              send_target_stamped (腕) + send_*_hand_joint_positions (20)
                               │
                         ros2-stack / OCS2 + hand_controller
```

### 包边界

| 位置 | 职责 |
|------|------|
| `submodules/hug` | 只出 MANO；**不**依赖 wuji-sdk |
| `grasp_generation/retarget/` | 新包：关键点预处理 + `HandRetarget` 协议 |
| `grasp_generation/retarget/wuji_hand2.py` | 调 `RetargetSession`；关节名/维数校验 |
| `grasp_generation.yaml` | `hand_profile: wuji_hand2` |
| `ros2_robot_interface` | 已有 `send_right_hand_joint_positions`；尽量不改 |

### 关键点预处理（必做）

```text
landmarks_cam (21,3)
  → p_wrist = landmarks[0]
  → kp_wrist = landmarks - p_wrist          # 腕原点（SDK 要求）
  → 可选：用 T_camera_wrist 的 R 把点旋到「掌系」再对齐 MediaPipe 轴
  → float32 (21,3) → session.step
```

若 SDK 内部已假定某一掌系，轴不对时只加 **固定 `R_mano_mediapipe`**（yaml），不改 HUG。

### 配置草案

```yaml
hand_profile: wuji_hand2
retarget:
  backend: wuji_sdk          # wuji_sdk | none
  side: right
  model: WujiHand2
  # 可选轴校正（度）
  keypoint_rpy_deg: [0, 0, 0]
  send_hand_joints: true
```

环境：retarget 跑在 **composer/.venv（3.12）**；`wuji-sdk` 仅 Linux x86_64/aarch64 wheel。HUG 仍 3.10 子进程。

---

## 3. 开发里程碑（建议顺序）

### 2A-W — Wuji 腕跟复现（本迭代）

1. ~~`robot.yaml` → `type: wuji_hand2`~~  
2. Isaac `wuji_hand2_new` + diagnose + serve 跟腕验收  
3. 若姿态轴偏：只调 `mano_to_tip_rpy_deg`（掌心已对齐则多半不用）

### 2B.1 — Retarget MVP

1. `grasp_generation/retarget/` + `WujiHand2Retarget`  
2. CLI：`grasp-generation retarget-hand --proposal ...` → 打印 / 可选下发 20 维  
3. 单元：固定 `proposal.json` 出 `q` 可重复；维数=20；限幅内  

### 2B.2 — Viser 串联

1. Propose 后按钮 **「4) Retarget fingers」**（默认不下发，先 dry）  
2. 可选在左侧画简化指尖目标 vs FK（后做）  
3. Execute 腕 + 指可分步勾选  

### 2B.3 — 质量闸门

1. 指尖到物体深度阈值、关节速度正则（SDK 已有 warm-start / LPF）  
2. `session.reset()` 在换物体 / 久停后调用  
3. 失败样本落盘供对照开源 `tuning_tool`

### 2C+ — 技能 / 采数

`dex.auto_grasp`：click → HUG → 腕到位 → retarget 合拢 → 抬起；成功标签入库。

---

## 4. 改动清单（文件级）

| 文件 / 目录 | 动作 |
|-------------|------|
| `examples/.../robot.yaml` | `type: wuji_hand2`（已改） |
| `examples/.../grasp_generation.yaml` | Wuji 注释、`hand_profile`、独立 `capture_dir`（已改） |
| `grasp_generation/config.py` | `hand_profile` 字段（已加） |
| `grasp_generation/retarget/__init__.py` | **新建** 协议 |
| `grasp_generation/retarget/wuji_hand2.py` | **新建** SDK 封装 |
| `grasp_generation/retarget/keypoints.py` | **新建** MANO→腕系 MediaPipe |
| `cli/grasp_main.py` | 子命令 `retarget-hand` |
| `service.py` | UI 步进 4） |
| `docs/SIM_GRASP_*` | 里程碑与本文件 |
| `pyproject` / setup 脚本 | 可选依赖 `wuji-sdk`（Linux） |
| FaSim USD / fa_w2_ws | **不改**（只读用现成手） |

---

## 5. 风险与不做

| 风险 | 缓解 |
|------|------|
| MANO 点序 / 掌系 ≠ MediaPipe | 腕原点 + 可配 `keypoint_rpy`；用 SDK 例程对照 |
| `wuji-sdk` 仅 Linux wheel | 文档写明；CI 用 Linux |
| 腕 TF 用 plan 时刻而非 capture | 2A 已知；头不动时 OK，后可冻 `T_base_cam` |
| 自碰 HOLD | 跟姿 / 合拢分步；先 dry |

**不做：** 改 HUG 权重；为 O7 重写一套优化器（O7 另开 `hand_profile`）；未同意改 USD。

---

## 7. 抓取贴面 / 结实度（2B 之后 · 当前痛点）

**现象：** 右手 retarget 与 MANO 形状很接近，但指尖离物体表面仍有缝；Isaac 里要很大力才捏得住。

**原因分层（映射已通 ≠ 能抓）：**

| 层 | 说明 |
|----|------|
| HUG | 绿骨架本身常是「靠近/预抓」而非贴面；landmark 在手体积内，不是接触点 |
| Retarget | `session.step` **只复现** 关键点形状，不会自动合拢压向物体 |
| 仿真 | 有缝则无接触力；PhysX 摩擦/驱动刚度不够时更要「挤」才稳 |

**推荐改法（由易到难，不改 HUG 权重）：**

1. **Squeeze 偏置（优先做）**  
   retarget 得到 `q0` 后，对 flex 类关节加一小步闭合：`q = q0 + α·(q_closed − q0)` 或按指尖方向微增 flex；Viser 滑条调 `α`。不依赖物体几何，立刻改善「看起来抓但不实」。

2. **腕部 approach 内收**  
   在现有腕跟目标上，沿掌法向 / 指向点击深度点再推进 `δ`（1–2 cm 级）。解决「整只手悬在物体外」；与 squeeze 互补。

3. **物体感知贴面（后做）**  
   用头相机点云 + 点击邻域估表面；把指尖关键点沿法向收一段再 `step`，或二次优化。才是真正的 contact-aware。

4. **仿真侧（并行）**  
   手/物摩擦、finger drive stiffness、适当允许轻微穿透或提高 grip effort；与运动学闭环分开调。

**不做：** 为贴面去微调 HUG；把缝隙怪罪于已验收的腕/指映射。
