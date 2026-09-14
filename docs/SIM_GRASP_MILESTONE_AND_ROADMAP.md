# 里程碑：W2 头相机 × HUG 联调（2026-09）

## 结论

**Phase 1 + 2A + 2A.1 平移对齐：已通。**  
头相机 → HUG → Viser → `right_tcp` 映射 → `right_eef` 命令；用户确认 **O7 掌跟与 dry 目标基本重合**。

**当前默认不跟姿态：** `position_only: true`（掌心朝向不变是配置行为）。详见 [`SIM_GRASP_EE_ALIGN.md`](SIM_GRASP_EE_ALIGN.md)。

### 实测观感（用户确认）

- 近桌点击：HUG 人手贴合好、推理快；Viser 点云 + 绿手可用
- **tcp / 掌跟与 dry 目标基本重合**（tip live-TF 映射后）
- 掌心朝向暂不随 HUG 翻转（`position_only`）

### 已知局限

| 现象 | 说明 |
|------|------|
| 姿态未跟随 | **配置**：`position_only`；关后可跟 HUG 腕旋（注意自碰） |
| MANO「掌」几何可变 vs 机器人 tcp→eef 固定 | 臂阶段只对齐 **腕↔tcp**；掌心用刚体偏置 |
| 人手大小 / 右手 / O7 做不到的手型 | 2B retarget；不改 HUG 权重 |

### 已具备 / 未具备

| 已具备 | 未具备 |
|--------|--------|
| RGB-D + HUG + Viser | `position_only: false` 姿态跟腕验收 |
| `hug_tip=tcp` → `control=eef` live TF | MANO→O7 指关节（2B） |
| `diagnose-frames` + 新 env 适配说明 | 可行性过滤、`dex.auto_grasp`、采数 |

测试入口：[`SIM_GRASP_HUG_TEST_GUIDE.md`](SIM_GRASP_HUG_TEST_GUIDE.md)、[`SIM_GRASP_EE_ALIGN.md`](SIM_GRASP_EE_ALIGN.md)。

---

## 后续计划（HUG 不动权重，下游适配）

原则：**HUG = 相机系人类抓取先验；composer = 机器人化。** 臂对齐先于手指。

```text
RGB-D + click
    → HUG (MANO, camera)                 ← 已通，冻结黑盒
    → T_cam_wrist → T_base_wrist         ← 已通（TF head_camera）
    → **腕定义 → OCS2 tip 映射**         ← 当前 2A.1
    → 臂 IK / send_target_stamped        ← 已通，待对齐后验收 cm 级
    → MANO → O7 retarget（指）           ← 2B
    → 可行性过滤 → dex.auto_grasp → 采数
```

### Phase 2A.1 — HUG 腕 ↔ 臂 tip（平移已通）

文档：[`SIM_GRASP_EE_ALIGN.md`](SIM_GRASP_EE_ALIGN.md)（含新 env 改哪些 yaml）。

1. ~~diagnose-frames / tip 映射~~ → **`right_tcp`→`right_eef` live TF 已落地**
2. 可选：`position_only: false` 跟 HUG 腕姿态
3. 再进 2B 指关节

### Phase 2A — 坐标系与臂

| 项 | 状态 |
|----|------|
| Capture → click → HUG → Plan/move | 已通 |
| Viser 点云 + dry MANO | 已通 |
| tip 映射（tcp→eef） | **平移已通** |
| 姿态跟随 | 默认关（`position_only`） |
### Phase 2B — MANO→O7 retarget（手指，对齐后）

1. MANO 指尖 → O7 关节映射 + 限幅  
2. 尺度归一；右手优先  
3. **不**微调 HUG

### Phase 2C / 2D / 3

可行性闸门 → `dex.auto_grasp` → 自动采数（同前，不变）。

---

## 明确不做（本阶段）

- 微调 / 重训 HUG 去拟合 O7  
- Newton 第一优先  
- 未同意前改 FaSim / fa_w2_ws / USD  
- 推送 `submodules/hug` 远程  

---

## 建议下一迭代

1. 保持 `position_only` 多测近桌稳定性；需要翻转时再开 `position_only: false`
2. 新 env：按 [`SIM_GRASP_EE_ALIGN.md`](SIM_GRASP_EE_ALIGN.md) 改 `robot.yaml` + `grasp_generation.yaml`
3. 姿态可接受后再开 **2B** 指关节 retarget
