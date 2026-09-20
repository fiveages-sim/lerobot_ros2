# 资产 / 运控侧对接（W2 + Wuji Hand2 当前 / O7 参考）

> 未经同意，自动化流程不改 FaSim-Isaac / fa_w2_ws / USD 原始文件。

## A. 头相机（已对齐 · 2026-09）

| 项 | 值 |
|----|-----|
| usda（O7） | `humanoid/FiveAges/Gen2/W2/env/dexhand_o7_new.usda` |
| usda（**当前 Wuji**） | `humanoid/FiveAges/Gen2/W2/env/wuji_hand2_new.usda` |
| camera_prim | `/World/FiveAges_W2/Head_V1/head_link2/head_camera/head_camera` |
| rgb / depth / info | `/head_camera/{rgb,depth,camera_info}` |
| resolution | 640×480，fx≈fy |
| physics | PhysX |

## B. fa_w2_ws / 运控

### 当前：Wuji Hand2

- Launch：`type:=wuji_hand2`（父仓 `robot.yaml` 已切）  
- 关节 20 维（thumb→pinky，含 flex/abd）；控制器 `{side}_hand_controller`  
- Tip：`right_tcp` → `right_hand_mount` → `hand_base`（+Z 指伸）→ `eef` @ `(0,0,0.07)` → `right_eef`  
- 下发：`send_right_hand_joint_positions`（20 floats）  
- 手指 retarget：官方 `wuji-sdk` `RetargetSession`（见 [`SIM_GRASP_WUJI_HAND2.md`](SIM_GRASP_WUJI_HAND2.md)）

### 参考：LinkerHand O7（已验收腕跟）

O7 关节顺序（`o7.side.yaml`）：拇指 3 + 四指各 1。  
Launch：`type:=linkerhand_o7` + env `dexhand_o7_new`。

## C. 父仓 / composer 状态

| 项 | 状态 |
|----|------|
| 分支 | 父仓 / composer / hug 本地开发；**暂不 push 各模块分支** |
| `submodules/hug` | 公司远端已同步过；本地冒烟可用 |
| O7 腕跟（2A） | **已验收** — [`SIM_GRASP_EE_ALIGN.md`](SIM_GRASP_EE_ALIGN.md) |
| Wuji 腕跟（2A-W） | 配置已切 `wuji_hand2`；待 `wuji_hand2_new` 冒烟 |
| MANO→Hand2 指关节（2B） | 规划见 [`SIM_GRASP_WUJI_HAND2.md`](SIM_GRASP_WUJI_HAND2.md) |

## D. 怎么测

- 腕跟 / Viser：[`SIM_GRASP_HUG_TEST_GUIDE.md`](SIM_GRASP_HUG_TEST_GUIDE.md)（Isaac 改开 `wuji_hand2_new`）  
- 路线图：[`SIM_GRASP_MILESTONE_AND_ROADMAP.md`](SIM_GRASP_MILESTONE_AND_ROADMAP.md)  
- Hand2 + retarget：[`SIM_GRASP_WUJI_HAND2.md`](SIM_GRASP_WUJI_HAND2.md)