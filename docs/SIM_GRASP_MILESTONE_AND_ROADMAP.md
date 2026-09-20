# 里程碑：W2 头相机 × HUG 联调（更新 2026-09）

## 结论

**Phase 1 + 2A（O7）：已通。** 头相机 → HUG → Viser → `right_tcp`→`right_eef` 跟腕（含姿态轴校正）。

**当前主线：Wuji Hand2。** 仿真 `wuji_hand2_new.usda`；运控 `type: wuji_hand2`；先复现腕跟，再接 **wuji-sdk RetargetSession** 做 MANO→20 关节。详见 [`SIM_GRASP_WUJI_HAND2.md`](SIM_GRASP_WUJI_HAND2.md)。

### 实测（O7，用户确认）

- 近桌 HUG 贴合好；Viser 点云 + 绿手可用  
- tcp/掌跟与 dry 基本重合；`mano_to_tip_rpy` 后可跟翻转  

### 已知局限

| 现象 | 说明 |
|------|------|
| 手指未跟 | **2B**：Wuji retarget（非改 HUG） |
| MANO 可变 vs 机器人刚体 tip | 腕对齐 tcp；eef 用 live TF |
| Plan 用当前 TF 外参 | 拍照后头勿大动；可后冻 capture 外参 |

### 已具备 / 未具备

| 已具备 | 未具备 |
|--------|--------|
| O7 腕跟 + 诊断 CLI | **Wuji 腕跟验收**（配置已切） |
| tip / 轴映射经验 | MANO→Hand2 `RetargetSession` |
| 文档与双 Python 环境 | Viser 指关节步进、采数闭环 |

---

## 路线图

```text
RGB-D + click
  → HUG (MANO)                         ← 冻结黑盒
  → T_cam_wrist → tip map (tcp→eef)    ← 2A 已通（O7）；2A-W 复现于 Wuji
  → landmarks_21 → wuji RetargetSession ← 2B（当前规划）
  → send_target_stamped + hand q[20]
  → 可行性 → dex.auto_grasp → 采数
```

### Phase 2A-W — Wuji 腕跟（进行中）

1. Isaac：`wuji_hand2_new.usda`  
2. `robot.yaml`：`type: wuji_hand2`（已改）  
3. `diagnose-frames` / serve：确认 `eef−tcp≈7cm`，掌跟贴 dry  

### Phase 2B — MANO → Wuji Hand2

见 [`SIM_GRASP_WUJI_HAND2.md`](SIM_GRASP_WUJI_HAND2.md)：

1. `retarget/` + `wuji_sdk.RetargetSession`  
2. CLI / Viser「指关节」步进  
3. 闸门与 `session.reset()`  

（`linkerhand_o7` 手指另作 `hand_profile`，不阻塞 Hand2。）

### Phase 2C / 2D / 3

可行性 → `dex.auto_grasp` → 自动采数。

---

## 明确不做

- 微调 HUG；未同意改 FaSim / fa_w2_ws  
- 本阶段不 push 各开发分支（除非另行通知）  

## 建议下一动作

1. 开 `wuji_hand2_new` + `ros2-stack`（确认 `wuji_hand2`）→ 跟腕验收  
2. 通过后装 `wuji-sdk`，实现 `retarget-hand` MVP  
