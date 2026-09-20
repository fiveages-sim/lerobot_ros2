# Wuji Retargeting：说明与仓库集成

> 与 HUG 分工：HUG 只出 MANO / 21 点；本层把关键点变成 Wuji Hand2 的 20 维关节角。  
> 正式引用入口（无独立学术论文）：[wuji-technology/wuji-retargeting](https://github.com/wuji-technology/wuji-retargeting)

## 1. 它是什么 & 在 HUG 链路里干什么

**Wuji Retargeting** 是 DexRetargeting 系的人手→灵巧手映射：用优化（Adaptive TipDir / FullHand 向量 + 速度正则）把 **人手骨架关键点** 解成 **机器人手关节角**。

在 HUG 抓取管线中的位置：

```text
头相机 RGB-D + 点击
  → HUG（黑盒）→ MANO 腕位姿 + landmarks_3d (21,)
  → [已通] 腕 / tip map → OCS2 跟腕
  → [2B]   Wuji Retarget → q[20] → hand_controller
```

HUG 论文把「人手表征 → 机器人手」留给下游；对 Wuji Hand2，官方下游就是这套 retarget（不必自研 O7 式解析映射）。

两条官方交付物（用途不同）：

| 交付 | 用途 | 依赖重量 |
|------|------|----------|
| **`wuji-sdk`（`RetargetSession`）** | **产品联调 / 我们栈默认路径** | 轻：`pip install wuji-sdk`（算法已打进 wheel） |
| **开源仓 `wuji-retargeting`** | 读算法、调参、`tuning_tool`、MuJoCo 回放 | 重：pin / nlopt / mujoco / … |

## 2. 输入 / 输出（对接时只盯这个）

### 输入

| 项 | 约定 |
|----|------|
| 形状 | `(21, 3)` `float32`（或 63 维 row-major） |
| 单位 | **米** |
| 点序 | **MediaPipe Hands**（腕=0 为原点） |
| 坐标系 | 腕点为原点；左右手各自一帧 |

MediaPipe 索引：0 腕；拇 1–4；食 5–8；中 9–12；无 13–16；小 17–20。

HUG `landmarks_3d` 同为 21 点腕+四指结构，对接前一般要：

1. `kp = landmarks - landmarks[0]`（腕原点，SDK/开源都要求）  
2. 若掌系与 MediaPipe 轴不一致：加固定 `keypoint_rpy`（yaml），**不改 HUG**

### 输出

| 项 | 约定 |
|----|------|
| 形状 | `(20,)` `float32` 关节角（rad） |
| 语义 | 与 `wuji-description` / `{side}_hand_controller` 一致：5 指 × 4 关节（含 flex/abd） |
| 用法 | 可直接 `send_right_hand_joint_positions(q)`（维数校验后） |

产品 API（推荐）：

```python
from wuji_sdk import Handedness, HandModel, RetargetSession
session = RetargetSession.for_hand(HandModel.WujiHand2, side=Handedness.Right)
q = session.step(keypoints_21x3)   # → (20,)
# 换物体 / 久停后：session.reset()
```

开源 API（调试）：`Retargeter.from_yaml(...).retarget(raw_keypoints)` → 同样 `(20,)`。

## 3. 仓库怎么保持干净（对齐 HUG 习惯）

原则：**运行时依赖用 pip；算法参考仓用 submodule；我们只写薄适配层。**

```text
lerobot_ros2/
  .venv/                          # pip install wuji-sdk   ← 运行时（勿提交）
  submodules/hug/                 # HUG 子模块（已有）
  submodules/wuji-retargeting/    # 官方开源仓（参考/调参，见下）
  submodules/robot_action_composer/
    grasp_generation/retarget/    # 我们拥有的适配代码（待建）
      keypoints.py                # MANO → 腕系 MediaPipe
      wuji_hand2.py               # 调 RetargetSession
```

| 放什么 | 放哪 | 他人怎么拿 |
|--------|------|------------|
| `wuji-sdk` | 父仓 `.venv`（或 composer 环境） | `pip install wuji-sdk`（Linux x86_64/aarch64） |
| 开源算法 + URDF | `submodules/wuji-retargeting` | submodule init / 本脚本 clone |
| 业务适配 | `grasp_generation/retarget/` | 跟 composer 一起走 |
| HUG | `submodules/hug` | 已有流程，**不**依赖 wuji |

**不要：** 把 pin/mujoco/整份 retarget 源码 vendoring 进 composer；不要改 HUG 权重；不要把 `.venv` / 大 mesh 提交进父仓。

当前本机状态（2026-09）：

- 已 clone（含 `wuji-description`、`mujoco-sim` 子模块）→ `submodules/wuji-retargeting` @ `531f6ed`  
- 尚未登记进父仓 `.gitmodules`（避免把整树当普通文件误提交）  
- 父仓 `.venv` 已装通：`wuji-sdk==2026.8.31`，`RetargetSession` 冒烟 OK  

登记 submodule（需要时，**请明确让我做**）：

```bash
# 若目录已存在，先移走再 add，或按团队习惯改 remote
git submodule add https://github.com/wuji-technology/wuji-retargeting.git submodules/wuji-retargeting
# recurse：该仓还有 wuji-description / mujoco-sim
git submodule update --init --recursive submodules/wuji-retargeting
```

一般**不需要**公司 fork，除非我们要提交补丁；日常只读上游即可。

## 4. 本机准备（脚本）

```bash
bash scripts/setup_wuji_retargeting.sh
# 默认：确保 clone 存在 + 在父仓 .venv 装 wuji-sdk
# 加 --editable-oss：额外 pip install -e submodules/wuji-retargeting（重依赖，仅调参需要）
```

## 5. 开发顺序建议

1. ~~腕跟 Wuji Hand2~~（已通）  
2. ~~本机 SDK + 开源仓~~（已通）  
3. ~~右手 `retarget-hand` dry 冒烟~~（已通：`grasp_generation/retarget/`）  
4. `--execute` 看真机手指；轴差调 `keypoint_rpy_deg`  
5. Viser「指关节」步进；失败样本对照开源 `tuning_tool`

右手冒烟：

```bash
cd examples/IsaacSim
grasp-generation retarget-hand --robot fiveages_w2 --workspace .
# 真下发：加 --execute（ros2-stack + wuji_hand2）
```
