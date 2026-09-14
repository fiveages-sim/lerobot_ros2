# 第一期测试攻略：W2 头相机 → HUG / MANO

面向：仿真已能发 `/head_camera/{rgb,depth,camera_info}`，想搞清楚 **怎么测、命令什么意思、应该看到什么**。

---

## 与 ros2-stack 联用（跟腕 · 推荐）

运控仍用 `ros2-stack`；抓取辅助用新 CLI（**不改编排**）。环境只使用本仓，见 [`ENV_THIS_CHECKOUT.md`](ENV_THIS_CHECKOUT.md)。

详见 [`submodules/robot_action_composer/docs/GRASP_GENERATION.md`](../submodules/robot_action_composer/docs/GRASP_GENERATION.md)。

```bash
cd /home/fiveages/lerobot_ros2
source .venv/bin/activate
cd examples/IsaacSim
ros2-stack launch --robot fiveages_w2          # 终端 A
# 终端 B（同样 source .venv）
grasp-generation serve --robot fiveages_w2 --workspace .
# 浏览器 Capture → UV → Propose → Plan/move（先 dry-run）
```

## 0. 先建立直觉（30 秒）

| 名词 | 是什么 | 在本项目里干什么 |
|------|--------|------------------|
| **RGB-D** | 彩色图 + 每像素距离 | 头相机看到的「画面 + 远近」 |
| **HUG** | 神经网络：给你一张图 + 你点一下物体 → 猜「人手怎么抓」 | 输出相机坐标系下的抓取姿态 |
| **MANO** | 标准「人手」骨骼/网格模型（不是 LinkerHand O7） | HUG 用它表达手指弯曲；**还不能直接下发到 O7** |
| **checkpoint** | 训练好的权重 `hug_full.safetensors` | 没有它就无法推理 |
| **click / 条件点** | 你在图上点的抓取目标（物体表面一点） | HUG 必须知道「抓哪儿」 |

**当前阶段成功标准（很重要）：**

- 机器人 **不会** 自动伸手、也不会动 O7。
- 成功 = 程序打出 `OK: HUG propose succeeded`，并给出 `T_camera_wrist`（手腕在 **相机坐标系** 里的 4×4 位姿）。
- 「看懂抓姿」优先用官方 **Viser 网页**（路径 A）；仿真联调优先用 **冒烟脚本**（路径 B）。

```text
Isaac 头相机 ──RGB+Depth+K──► HUG(+MANO) ──► T_camera_wrist / 21个关节点
                                      │
                                      ▼
                         （下一批才做）→ 变到机体坐标系 → 臂 IK → O7 手指
```

---

## 1. 开测前检查清单

在 **任意终端**（Isaac 已开、话题在发）：

```bash
# 三个话题都应有数据
ros2 topic hz /head_camera/rgb
ros2 topic hz /head_camera/depth
ros2 topic hz /head_camera/camera_info
```

建议再看一眼：

```bash
ros2 topic echo /head_camera/camera_info --once   # 应有 k[] 内参
# depth 编码应为 32FC1（米）；编排侧会转成 HUG 要的毫米 uint16
```

场景建议：

- usda：`.../W2/env/dexhand_o7_new.usda`（PhysX）
- 桌上放 **一对能被头相机清楚看到** 的物体（不要被手挡住）
- 头/腰姿态固定，物体在画面中部偏下通常更好点

环境（**必须用 HUG 自己的 uv**，不要混父仓 conda/composer 主环境）：

```bash
cd /home/fiveages/lerobot_ros2/submodules/hug
source .venv/bin/activate

# 应看到 Python 3.10.x
python -V

# 权重与 MANO 应已在本地（之前装过可跳过）
ls checkpoints/hug_full.safetensors
ls assets/mano_models/models/MANO_RIGHT.pkl
```

Composer 脚本需要能被 import（一次性）：

```bash
export PYTHONPATH=/home/fiveages/lerobot_ros2/submodules/robot_action_composer:${PYTHONPATH}
```

可把上一行写进该终端会话；关终端就没了。

---

## 2. 路径 A：不连仿真，先「看见」HUG（推荐先做）

目的：弄清 HUG 网页怎么点、手长什么样。用官方演示数据，**不依赖 Isaac**。

> **若终端刷 `Retrying ... huggingface.co`、网页打不开：**  
> 先 **Ctrl+C**。这是 DINOv2 在直连官网失败。本仓已改成优先读本地缓存；请用：
>
> ```bash
> cd /home/fiveages/lerobot_ros2/submodules/hug
> source .venv/bin/activate
> bash scripts/run_app.sh
> ```
>
> 详见 `submodules/hug/docs/SETUP_LOCAL.md`「网络问题」一节。

### A1. 一键冒烟（只验证环境）

```bash
cd /home/fiveages/lerobot_ros2/submodules/hug
source .venv/bin/activate
bash scripts/smoke_check.sh
```

**期望：** 末尾无报错；能 import torch / hug / manotorch。

### A2. 交互网页（最直观）

```bash
cd /home/fiveages/lerobot_ros2/submodules/hug
source .venv/bin/activate
bash scripts/run_app.sh
# 或手动：
# source scripts/hf_env.sh
# python -m hug.app \
#   --checkpoint-path checkpoints/hug_full.safetensors \
#   --dataset-path data/hug_bench \
#   --port 8080 --sampling-steps 1 --save-pred
```

| 参数 | 意思 |
|------|------|
| `--checkpoint-path` | 模型权重 |
| `--dataset-path` | 里面一堆 `.pkl` 样本（已含图+深度+内参） |
| `--port 8080` | 浏览器打开 `http://127.0.0.1:8080` |
| `--sampling-steps 1` | 采样步数；1=快（冒烟），正式可试 10～50（更慢更稳） |
| `--save-pred` | 每次点击把结果写到 `data/hug_bench/grasp_pred/` |

**你怎么操作：**

1. 浏览器打开 `http://127.0.0.1:8080`（若远程机器，用 SSH 端口转发）。
2. 左侧有 RGB 缩略图；**在物体上点一下**。
3. 右侧/中间 3D 场景应出现一只 **MANO 人手网格**（半透明），手腕附近有坐标系。
4. 可开 `Animate Grasp` 看「预抓 → 合拢」插值动画。

**期望效果：**

- 点在物体表面 → 手大致贴着物体、手指弯曲合理。
- 点在背景/空洞 → 深度可能为 0，手会飞掉或离谱（正常，换点重试）。

**这和机器人无关：** 你看到的是「相机坐标系里的人手」，不是 O7。

### A3. 无界面批处理推理（可选）

```bash
python -m hug.inference \
  --checkpoint-path checkpoints/hug_full.safetensors \
  --dataset-path data/hug_bench \
  --num-samples 1 \
  --batch-size 1 \
  --sampling-steps 1
```

**期望：** 打印 Inference timing 表；生成 `data/hug_bench/grasp_pred/*.pkl`。

---

## 3. 路径 B：仿真开着 → 真头相机测（你要的主流程）

### B0. 仿真侧你要做什么

1. 启动 Isaac，加载 **`dexhand_o7_new`**（PhysX），桌上有物体。
2. 确认头相机 graph 在跑，三话题有 hz。
3. **保持仿真 Play**；另开一个终端跑下面的 Python（不要关 Isaac）。
4. 暂时 **不用** 开完整 motion / O7 控制也能测 HUG（只读相机）。

### B1. 在线抓一帧并跑 HUG（主命令）

> **Python 版本坑：** HUG 用 uv **3.10**；ROS Jazzy 的 `rclpy` 只给 **3.12**。  
> 在 hug `.venv` 里直接 `import rclpy` 会报 `_rclpy_pybind11` —— 已改成自动用 `/usr/bin/python3.12` 采帧。

**推荐（一键两段）：**

```bash
cd /home/fiveages/lerobot_ros2/submodules/hug
source .venv/bin/activate
export PYTHONPATH=/home/fiveages/lerobot_ros2/submodules/robot_action_composer:${PYTHONPATH}
# 终端需已 source /opt/ros/jazzy/setup.bash；Isaac Play
bash scripts/smoke_w2_head_hug.sh --u-norm 0.5 --v-norm 0.55
```

**或：**

```bash
python -m robot_action_composer.grasp_generation.smoke_head_rgbd --from-ros \
  --save-dir data/w2_head_capture \
  --u-norm 0.52 \
  --v-norm 0.55 \
  --sampling-steps 1 \
  --timeout 30
```

#### 这条命令在干什么（逐步）

1. 订阅 `/head_camera/rgb`、`/depth`、`/camera_info`，等到三样都齐 → 组成一帧。
2. 把帧存进 `--save-dir`（方便你事后用图检查点得对不对）。
3. 把深度从 **米(float)** 转成 HUG 要的 **毫米(uint16)**，中心裁成方形再缩到 224×224。
4. 把你的 `--u-norm/--v-norm`（全图 0～1）映射到 224 空间的点击。
5. 加载 MANO + 权重，跑网络 → 打印手腕平移等。

#### 参数怎么选

| 参数 | 含义 | 怎么调 |
|------|------|--------|
| `--from-ros` | 从 ROS 实时取一帧 | Isaac 必须在发话题 |
| `--save-dir` | 落盘目录 | 建议保留，出问题好查 |
| `--u-norm` | 点击横坐标，0=左，1=右 | **对准物体中心**；先 0.5 再微调 |
| `--v-norm` | 点击纵坐标，0=上，1=下 | 桌面物体常在 0.45～0.65 |
| `--u-px` / `--v-px` | 用像素点（与 norm 二选一） | 640×480 时物体中心常约 `(320, 280)` |
| `--sampling-steps` | 流匹配采样步数 | 冒烟用 1；观感差再试 10 |
| `--timeout` | 等多久话题 | 没图就超时报错 |

**怎么知道 u/v 点对了？**

看保存的图：

```bash
# 一般会有类似：
ls data/w2_head_capture/
# w2_head_rgb.png  w2_head_depth.png  w2_head_intrinsics.txt  proposal.json
```

用看图软件打开 `w2_head_rgb.png`：

- 图像宽 W、高 H；点击像素约 `(u_norm*(W-1), v_norm*(H-1))`。
- 若点在桌子空隙 / 背景，把 `--u-norm/--v-norm` 改到物体上再跑一遍。

### B2. 成功时长什么样

终端大致类似：

```text
Waiting for /head_camera RGB-D ...
frame rgb=(480, 640, 3) depth=[0.3, 2.5]m  fx=... fy=...
click_uv_224 (...) depth_m 0.8xx
T_camera_wrist translation (m): [x, y, z]
landmarks_3d wrist: [x, y, z]
wrote .../proposal.json
OK: HUG propose succeeded
```

解读：

| 输出 | 含义 |
|------|------|
| `frame rgb=(H,W,3)` | 收到的分辨率（预期约 480×640） |
| `depth=[min,max]m` | 有效深度范围；若 min 很大或几乎没有效值，深度有问题 |
| `depth_m` at click | **点击处**深度；若 ≈0，点到了无效深度，抓姿不可信 |
| `T_camera_wrist translation` | 手腕在相机系下的位置（米）；z 通常是「往前」距离量级 |
| `proposal.json` | 完整 4×4 矩阵 + 21×3 关节点，给后续程序用 |

**合理经验：**

- 点击处 `depth_m` 多在 **0.3～2.0 m**（看你头到桌面距离）。
- `translation` 的 z 与点击深度同量级，不会是 NaN，也不会全接近 0。

### B3. 失败时先查什么

| 现象 | 可能原因 | 处理 |
|------|----------|------|
| `ModuleNotFoundError: rclpy._rclpy_pybind11` / `cpython-310` | 在 HUG 3.10 venv 里直接 import 了 Jazzy rclpy | 用 `bash scripts/smoke_w2_head_hug.sh` 或更新后的 `--from-ros`（自动 3.12 采帧） |
| `timeout waiting for ...` | Isaac 没发话题 / 不在同一 ROS_DOMAIN | `ros2 topic list \| grep head_camera` |
| `HUG not available` | 没进 hug `.venv` 或缺权重 | `source .venv/bin/activate`；检查 checkpoint |
| `depth_m 0` / 手飞掉 | 点到天空/空洞，或深度坏 | 改 u/v；检查 depth 话题 |
| `AssertionError: rgb_pcl` | 旧 inference 补丁丢失 | 确认用的是本仓改过的 `src/inference.py` |
| CUDA OOM | 显存被 Isaac 占满 | `--sampling-steps 1`；或暂时关其它 GPU 进程 |
| 手姿态很怪但程序 OK | 点击偏、遮挡、物体不像训练集 | 换物体/换点击；steps 加大 |

### B4. 离线复跑同一帧（不用再开 ROS）

Isaac 关掉后也能复现：

```bash
python -m robot_action_composer.grasp_generation.smoke_head_rgbd \
  --from-dir data/w2_head_capture \
  --u-norm 0.52 --v-norm 0.55 \
  --sampling-steps 1
```

`--from-dir` 会读目录里文件名含 `rgb` / `depth` / `intrinsics` 的文件。

### B5.（可选）把仿真帧丢进官方 Viser 里点着玩

```bash
# 1) 先用 --from-ros --save-dir 存一帧（上面 B1）
# 2) 目录里已有 png + intrinsics 时，做成 HUG pkl：
python -m hug.prepare_inputs --dataset-path data/w2_head_capture

# 3) 用 app 打开该目录（会递归找 .pkl）
python -m hug.app \
  --checkpoint-path checkpoints/hug_full.safetensors \
  --dataset-path data/w2_head_capture \
  --port 8080 \
  --sampling-steps 1 \
  --save-pred
```

这样你在 **自己的头相机画面** 上点物体，3D 里看 MANO 手 —— 这是「仿真视觉联调」最直观的方式。

---

## 4. MANO 相关：你需要知道的

- **文件：** `assets/mano_models/models/MANO_{LEFT,RIGHT}.pkl`（已从你下的 zip 整理好）。
- **库：** `manotorch` 在 hug `.venv` 里；推理时自动加载。
- **输出里的 MANO 量：**
  - `T_camera_wrist`：手腕刚体位姿（相机系）
  - `landmarks_3d`：21 个关键节点（手腕+手指）
  - `pose` / `shape`：手指轴角与手型参数
- **不是机器人：** MANO ≠ LinkerHand O7。O7 只有更少的关节；**映射（retarget）还没做**，所以现在测通了也不会动真手。

自检 MANO：

```bash
source .venv/bin/activate
python -c "from manotorch.manolayer import ManoLayer; print('manotorch OK')"
ls assets/mano_models/models/MANO_RIGHT.pkl
```

---

## 5. 建议的半天测试顺序

1. **路径 A2**（`hug.app` + `hug_bench`）—— 确认你会点、能看见手。  
2. Isaac Play + `ros2 topic hz` —— 确认三话题。  
3. **路径 B1**（`--from-ros`）—— 终端出现 `OK` + `proposal.json`。  
4. 打开 `w2_head_rgb.png`，确认点击落在物体上；不对就改 `--u-norm/--v-norm` 再跑。  
5. （加分）**路径 B5** 用 Viser 在自己的仿真帧上点。  

全部过关 = 「头相机数据能进 HUG，MANO 抓姿能算出来」。  
**下一步才是：** 相机系 → 机体/底座系 → 臂运动到腕目标 → MANO 手指映到 O7。

---

## 6. 和「整机抓取」的差距（避免预期错位）

| 已具备 | 尚未具备 |
|--------|----------|
| 头相机 RGB-D ROS | 自动选点击点（现在要你手动给 u/v） |
| HUG 推理 + MANO | `T_camera_wrist` → base / EE |
| 落盘 `proposal.json` | 臂 IK 执行 |
| | MANO → O7 关节 |
| | 自动采数闭环 |

若你跑完 B1 得到合理的 `depth_m` 和 `T_camera_wrist`，第一期「测 HUG/MANO」就算过关，把 `proposal.json` 路径或终端摘要发回来即可继续下一批。
