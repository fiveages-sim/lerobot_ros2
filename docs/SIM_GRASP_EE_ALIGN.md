# Phase 2A.1 — HUG 腕 ↔ 臂 tip 对齐（经验沉淀）

> 状态：**平移对齐已通**（2026-09，用户确认：O7 掌跟 / tcp 与 dry 目标基本重合）。  
> 朝向：当前默认 **`position_only: true`** → 掌心朝向不变（故意的，不是 HUG 没发姿态）。

## 结论（先看这个）

| 问题 | 答案 |
|------|------|
| 为何原先腕偏一截？ | 把 HUG **腕** XYZ 直接当 OCS2 **`right_eef`（掌心）** 目标；O7 掌心相对 tcp 固定偏约 12 cm |
| 怎么修的？ | `hug_tip_frame=right_tcp`，`control_tip_frame=right_eef`，用 **live TF** 做 `cmd = hug_xyz + (eef−tcp)` |
| 为何掌心不翻转？ | **`position_only: true`**：只改平移，姿态沿用当前 EE。HUG 的 `T_camera_wrist` **有完整旋转**，只是没下发 |
| HUG 掌心相对腕会变、机器人不会？ | 对。MANO 抓姿几何在变；机器人 **tcp→eef 是 URDF 固定刚体**。臂阶段只把 HUG **腕系**对齐 **tcp**；掌心由固定 `T_tcp_eef` 推出。手指开合是 2B |

```text
HUG (变)          机器人 (刚体链，env 可改 tcp 朝向)
腕系 / landmarks  →  应对齐 right_tcp（手装在 tcp 上）
「掌心」随抓型变  ≠  right_eef = tcp ⊕ 固定偏置（live TF）
```

---

## 已验证做法（W2 + O7 + dexhand_o7_new）

### 改了哪些文件

| 文件 | 改什么 |
|------|--------|
| `examples/IsaacSim/robots/FiveAges_W2/robot.yaml` | `type: linkerhand_o7`；联调时 `headless: false`（要 RViz） |
| `examples/IsaacSim/robots/FiveAges_W2/grasp_generation.yaml` | 相机话题/TF、`hug_tip_frame` / `control_tip_frame` |
| `…/grasp_generation/bridge.py` | tip 重映射（live TF） |
| `…/grasp_generation/config.py` + `default_config.yaml` | 上述字段默认值 |

### 映射公式（`position_only`）

```text
T_base_wrist = T_base_cam @ T_camera_wrist     # HUG 腕 → base
offset       = xyz(base←eef) − xyz(base←tcp)   # live TF，含当前手姿与 env tcp 朝向
cmd_xyz      = xyz(T_base_wrist) + offset       # 发给 OCS2 的 right_eef
cmd_quat     = 当前 get_pose() 姿态              # 不翻转
```

`position_only: false` 时（下一小步可选）：

```text
T_base_cmd = T_base_wrist @ T_tcp_eef     # T_tcp_eef = lookup(tcp, eef)
```

把 HUG 腕的 **旋转也当作期望 tcp 姿态**，再右乘固定掌心偏置。可能触发自碰 → HOLD，需小心测。

### 诊断命令

```bash
grasp-generation diagnose-frames --robot fiveages_w2 --workspace .
```

注意：`plan − tip` = 当前 tip 离目标还差多远，**不是**选 tip 名的判据。看 `eef−tcp≈0.12m`、`get_pose≡eef`、相机 TF 是否 OK。

---

## 朝向轴不对（掌心该翻、指尖却乱甩）

**原因：** `position_only: false` 后直接把 MANO 腕的 `R` 当成 `right_tcp` 的 `R`。  
两边轴向约定不同：

| 系 | 指尖 / 伸出 | 掌法向（约） |
|----|-------------|--------------|
| HUG MANO 腕 | **−X** | **+Y** |
| O7（tcp→eef） | 常沿 tip 系 **+Z**（掌心偏置） | 另两轴 |

恒等映射会把「绕指尖轴翻掌」拧成「大幅改指尖朝向」。

### 排查（照做）

```bash
# 已有 proposal.json（Capture + 点物体）
cd examples/IsaacSim
grasp-generation diagnose-orientation --robot fiveages_w2 --workspace .
```

看：`wrist→mid_tip ≈ −X`，以及 `eef in tcp` 主轴是否 `+Z`。

### 校正

`grasp_generation.yaml`：

```yaml
position_only: false
mano_to_tip_rpy_deg: [0, -90, 0]   # W2+O7 起始猜测：tip+Z ← MANO −X
```

公式：`T_cmd = T_mano @ T_mano_tip(rpy) @ T_tcp_eef`

重启 serve → **只 Plan / dry-run**，在 Viser 看绿手与期望；不对就改 rpy（±90° 步进：`[0,90,0]`、`[90,0,0]`、`[0,0,90]`…）直到「指尖方向稳、掌心翻转」。

---

## 朝向 /「掌跟 vs 掌心」怎么理解

1. **HUG 有局限吗？**  
   - 对臂：HUG 输出相机系 **MANO 腕** `T_camera_wrist`（含 R）+ 21 关节，**不**输出机器人 `right_eef`。  
   - 对指：不感知 O7 自由度（2B retarget），与本次「掌心不转」无关。

2. **你现在看到掌心朝向不变**  
   - 原因是 composer **`position_only`**，不是 HUG 只给了水平目标。  
   - Viser 绿手会按 HUG 旋转画；真机 EE 姿态被我们锁住了，所以「手翻转了、臂没翻」。

3. **可变的 HUG「掌」 vs 固定的机器人偏置**  
   - MANO：腕到指尖/掌心相对关系随抓取变化。  
   - 机器人：`tcp → hand_base → eef` 是装配刚体；env 若改 **tcp 相对 link7 的朝向**，live TF 会自动带上，无需改代码常数。  
   - 臂对齐只保证：**tcp ≈ HUG 腕**；掌心在 tcp 前方固定处。真要「掌心贴物体某朝向」，要么关 `position_only` 跟 HUG 腕旋，要么以后单独定义物体接近轴（仍不是改 HUG 权重）。

---

## 新 env / 新手怎么适配

### 1. 运控侧（必查）

`examples/IsaacSim/robots/<Robot>/robot.yaml`：

```yaml
ros2_stack:
  motion:
    headless: false          # 要 RViz
    args:
      type: linkerhand_o7    # 或该 env 真实 EEF key（勿留 rg75 若仿真正是 O7）
```

### 2. grasp 侧（每机器人一份）

`…/robots/<Robot>/grasp_generation.yaml`（可复制 W2 改）：

| 键 | 填什么 |
|----|--------|
| `rgb_topic` / `depth_topic` / `info_topic` | 头相机 ROS 话题 |
| `camera_frames` | TF 名，优先光学/相机 link（W2：`head_camera`） |
| `camera_prim` / `base_prim` | 仅 entity_state 回退；TF 通时可后补 |
| `base_frame` / `motion_frame_id` | 通常 `base_link` |
| `hug_tip_frame` | **手装在哪个 link**（O7：`right_tcp`；若手挂在别的法兰则改名） |
| `control_tip_frame` | OCS2 `eeFrame`（O7：`right_eef`） |
| `position_only` | 先 `true` 对平移；要对齐翻转再 `false` |
| `arm` | `right` / `left`（`{arm}_tcp` 写法也行） |

### 3. 适配检查清单

1. `ros2-stack launch` + `type` 与仿真手一致。  
2. `diagnose-frames`：`base←camera` OK；`eef−tcp` 合理；`get_pose` 与 `control_tip` 重合。  
3. Viser dry 手腕 ≈ 物体；execute 后 **tcp/掌跟** ≈ dry（`position_only`）。  
4. 再考虑 `position_only: false` 跟姿态；自碰则 HOLD，先拉开再试。  
5. 手指 → 2B，不改 HUG 权重。

### 4. 不要改的

- 未同意前：FaSim USD / fa_w2_ws 原始文件  
- HUG 权重（黑盒）  
- 写死 `eef−tcp` 常数（应用 live TF，方便不同 env 拧 tcp）

---

## 建议下一小步

1. 保持 `position_only: true` 做更多近桌点击，确认水平+高度稳定性。  
2. 需要翻转时：yaml 设 `position_only: false`，重启 serve，先 dry 看 `wrist_quat`，再小范围 execute。  
3. 姿态稳了再进 **2B** MANO→O7 指关节。
