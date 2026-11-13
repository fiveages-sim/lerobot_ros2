# LeRobot 训练与推理（已跑通）

本文记录**实机验证过**的训练与推理方式，环境为 lerobot 0.5.1 / numpy 2.2.6 /
torch 2.7.1+cu128 / RTX 4090。

---

## 1. 环境：为什么有两个 venv

| 环境 | numpy | lerobot | 用途 |
|---|---|---|---|
| `.venv` | **1.26.4** | 0.3.4（旧 submodule editable） | 编排 / 采集 / 导出 / 数据处理 |
| `.venv-train` | **2.0~2.2** | **0.5.1（PyPI）** | 只跑训练与推理 |

`lerobot 0.5.x` 强制 `numpy>=2.0.0,<2.3.0`；而系统 `cv_bridge` 按 numpy 1.x ABI
编译，升到 2.x 会破坏头相机采集（`sdg.sh capture` 的 head 相机直接没图）。
两者不能共存，所以拆开。

```bash
# 创建训练环境（首次）
cd ~/lerobot_ros2
./init.sh setup-train-env
```

> **在 Cursor 内置终端执行时**要覆盖 `UV_CACHE_DIR`：
> ```bash
> env UV_CACHE_DIR="$HOME/.cache/uv" ./init.sh setup-train-env
> ```
> 否则 uv 把包写到 `/tmp/cursor-sandbox-cache/...`，`.venv-train` 永远是 100K 空壳。

验证：

```bash
.venv-train/bin/python -c "
import numpy, torch, lerobot
print('numpy  ', numpy.__version__)     # 2.0~2.2
print('lerobot', lerobot.__version__)   # 0.5.1
print('torch  ', torch.__version__, torch.cuda.is_available())
"
```

> ⚠️ `.venv-train` **不要** `source /opt/ros/jazzy/setup.bash` ——
> 那会把系统 numpy 1.26 塞进 PYTHONPATH，破坏训练依赖。

---

## 2. 数据：LeRobot v3.0 数据集

训练吃的是 LeRobot 格式。用 sdg 导出：

```bash
cd ~/FaSim-Isaac/exts/fasim.replicator.sdg

# 会话 → LeRobot（--action-space absolute 是最稳的起点）
./scripts/sdg.sh export ~/fasim_data/unitree_g1d/sinopec_test_tube/run_04 \
    --out ~/lerobot_datasets/tube_smoke \
    --action-space absolute

# 必做校验：能否被官方 loader 加载
./scripts/sdg.sh export-check ~/lerobot_datasets/tube_smoke
# 必须看到 ✅ 数据集可被 LeRobotDataset 正常加载
```

`--action-space` 三选：

| 选项 | action 内容 | 什么时候用 |
|---|---|---|
| `absolute`（默认） | `joint_pos_cmd` 绝对关节指令 | 推荐起点；部署直接输出目标角，无累积误差 |
| `delta` | `joint_pos[t+1] - joint_pos[t]` | 对齐用 delta 的外部数据集，便于混训 |
| `ee_delta` | 末端相邻增量 14 维 | 想学任务空间（需 IK 执行） |

---

## 3. 训练

### 3.1 已验证的命令

```bash
cd ~/lerobot_ros2
source .venv-train/bin/activate

python -m lerobot.scripts.lerobot_train \
  --dataset.repo_id=tube_smoke \
  --dataset.root=$HOME/lerobot_datasets/tube_smoke \
  --dataset.video_backend=pyav \
  --policy.type=act \
  --policy.chunk_size=100 \
  --policy.n_action_steps=1 \
  --policy.push_to_hub=false \
  --output_dir=$HOME/act_runs/tube_v1 \
  --batch_size=8 \
  --num_workers=4 \
  --steps=3000 \
  --save_freq=1000 \
  --eval_freq=0 \
  --wandb.enable=false
```

### 3.2 两个**必需**的参数（不加会失败）

| 参数 | 不加的后果 |
|---|---|
| `--policy.push_to_hub=false` | `ValueError: 'policy.repo_id' argument missing`<br>默认 `true`，会校验推 HuggingFace Hub 的 repo_id |
| `--dataset.video_backend=pyav` | DataLoader worker 崩溃：`torchcodec` 加载 `libavutil.so.59` 失败 |

**FFmpeg 坑详解**：`torchcodec` 需要与系统 FFmpeg **严格匹配**的共享库
（libavutil 56/57/59）。本机 FFmpeg 版本对不上就加载失败，日志里会出现
一大串 `OSError: libavutil.so.5x: cannot open shared object file`，逐版本
（FFmpeg 4/5/6）试探 —— 那是探测噪音，不用管。指定 `pyav` 后端即可绕开。

### 3.3 实测曲线

```
cfg.steps=3000    dataset.num_frames=651    dataset.num_episodes=1
num_learnable_params=51650464 (52M)
```

| step | loss |
|---|---|
| 200 | 6.319 |
| 800 | 2.016 |
| 2000 | 1.049 |
| **3000** | **0.598** |

耗时 5:53，8.48 step/s，37 epoch。

> ⚠️ **这是 1 条轨迹反复 37 个 epoch 的结果。** loss 低只说明记住了这条路径，
> 换个初始位置大概率失败。真实数据量需要 50~200 条（ACT 论文）。

### 3.4 关键超参

| 参数 | 默认 | 建议 | 原因 |
|---|---|---|---|
| `batch_size` | 8 | 8~16 | 4090 24GB，3 路 640×480 够用 |
| `steps` | 100000 | 20000~50000 | 数据少易过拟合 |
| `save_freq` | 20000 | 1000~5000 | 想早点看到 ckpt |
| `eval_freq` | 20000 | **0** | 我们没有 test split |
| `num_workers` | 4 | 4~8 | 3 路视频解码是瓶颈 |
| `policy.chunk_size` | 100 | 100 | ACT 论文默认 |
| `policy.n_action_steps` | 100 | **1**（闭环最稳）/ 50（提速） | 每步重预测，避免开环漂 |

### 3.5 产物

```
~/act_runs/tube_v1/
├── checkpoints/
│   ├── 001000/pretrained_model/   ← model.safetensors + train_config.json + 归一化 stats
│   ├── 002000/pretrained_model/
│   ├── 003000/pretrained_model/
│   └── last/                      ← 指向最新
└── train_config.json
```

约 1.8G（52M 参数 + 优化器状态）。

---

## 4. 推理

推理**只在本仓库做**（不在 sdg），原因：

1. 需要 `lerobot`+`torch`（`.venv-train`，numpy 2.x），与 Isaac 进程的
   numpy 1.x / cv_bridge 冲突，装不进 Isaac 进程
2. 驱动应该**走运控**（OCS2 做 IK），而不是 Isaac 内部直接改关节 ——
   这样才与真实部署一致
3. 推理与「数采平台」职责无关

仓库里已有完整实现：

| 路径 | 作用 |
|---|---|
| `examples/IsaacSim/inference.py` | 统一推理启动器（LeRobot + ROS2） |
| `examples/IsaacSim/online_infer/core.py` | 策略循环 + ROS2 发送（707 行） |
| `examples/IsaacSim/online_infer/openloop_check.py` | **L2 开环验证**（只算不驱动） |
| `lerobot_robot_ros2/.../robot.py` | `ROS2Robot.send_action()` 发 `Pose` 给运控 |
| `examples/IsaacSim/robots/Unitree/G1D/` | G1D 的任务配置 |

### 4.1 手工加载 checkpoint（已验证）

```python
from lerobot.policies.act.modeling_act import ACTPolicy   # ← 类名是 ACTPolicy

p = ACTPolicy.from_pretrained(
    "/home/fiveages/act_runs/tube_v1/checkpoints/003000/pretrained_model"
)
print(p.config.chunk_size, p.config.n_action_steps)   # 100 1
```

> ⚠️ **别导错符号**：`modeling_act.py` 里有两个类
> - `ACTPolicy`（`PreTrainedPolicy` 子类，有 `from_pretrained` / `select_action`）← 用这个
> - `ACT`（纯 `nn.Module`，内部网络）← 没有 `from_pretrained`

### 4.2 推理链路

```
Isaac 渲染相机 + 关节状态
    ↓（ROS2 话题，由 lerobot_camera_ros2 / lerobot_robot_ros2 订阅）
obs 张量 → preprocessor 归一化 → policy.select_action() → postprocessor 反归一化
    ↓
ROS2Robot.send_action({"left_ee.pos.x": ..., ...})
    ↓ geometry_msgs/Pose → /left_target /right_target
OCS2 做 IK → 关节驱动
```

**关键：不需要自己写 IK** —— `ROS2Robot.send_action()` 把末端位姿发成
`Pose` 消息给运控，OCS2 负责解算关节。这与 sdg 的 `live` 回放
`drive="ee"` 模式（`live_replay.py:22`）是同一条链路。

核心代码在 `online_infer/core.py`：

```python
from lerobot.policies.factory import make_policy          # :36
train_cfg = TrainPipelineConfig.from_pretrained(...)      # :372
policy = make_policy(cfg.policy, ds_meta=meta_obj)        # :420
action = policy.select_action(batch)                      # :532
robot.ros2_interface.send_fsm_command(FSM_HOLD)           # :597
```

### 4.3 启动

```bash
cd ~/lerobot_ros2/examples/IsaacSim
source ~/lerobot_ros2/.venv-train/bin/activate

python inference.py --robot g1d --task <task_key> \
    --policy-path ~/act_runs/tube25/checkpoints/020000/pretrained_model
```

> 参数以 `python inference.py --help` 为准（此命令结构已核对，但未做端到端实跑）。

---

## 5. 三层验证

| 层 | 做什么 | 命令 | 能回答 |
|---|---|---|---|
| **L1** | 留出帧算 MSE / loss | 训练日志 | 拟合好坏（**不能说明能用**） |
| **L2** | 逐帧预测与真值对比，**不驱动** | `openloop_check.py` | 量级/方向/抖动对不对 |
| **L3** | 策略→运控→新观测→循环 | `inference.py` | **真实成功率** |

**顺序**：L1 → **L2（先做，成本低）** → L3。

### 5.1 L2 开环验证（已实测）

```bash
cd ~/lerobot_ros2 && source .venv-train/bin/activate

python examples/IsaacSim/online_infer/openloop_check.py \
    --dataset ~/lerobot_datasets/tube_25eps \
    --checkpoint ~/act_runs/tube25/checkpoints/020000/pretrained_model \
    --episode 0 --start 300 --max-frames 200
```

**`--start 300` 很重要**：跳过开头静止段。静止段动作接近 0，
预测再准也说明不了问题（会得到「真值 med=0、相关性 NaN」）。

实测（25 条 / 20k 步 / loss 0.044）：

| 指标 | 值 |
|---|---|
| MAE | **0.0057 rad**（≈0.33°） |
| 量级 | 真值 0.1564 / 预测 0.1595 |
| **逐维相关性** | **mean 0.996，中位 0.998，20/20 维 > 0.8** |
| 抖动比 | 0.94x |

**归一化陷阱（踩过）**：`select_action()` 只做前向，归一化在
`PolicyProcessorPipeline` 里。漏掉后预测值是真值的 **1000 倍**、
21.9% 饱和到 ±1。必须 `preprocessor(batch)` → `select_action` →
`postprocessor({"action": a})`。补上后 MAE 从 0.496 → 0.0057（**改善 87 倍**）。

### 5.2 L3 成功判定（还缺）

目前 L3 只能肉眼观察。要算成功率需要自动判定，可用 sdg 采集的
`tracked_prims`（物体位姿）：

```python
d = ||test_tube_pos - target_hole_pos||
ok = d < 0.02 and tube_z < rack_top_z
```

---

## 6. 常见问题

| 现象 | 原因 | 解决 |
|---|---|---|
| `No module named lerobot.scripts.train` | 0.5.1 改了脚本名 | 用 `lerobot.scripts.lerobot_train` |
| `'policy.repo_id' argument missing` | `push_to_hub` 默认 true | 加 `--policy.push_to_hub=false` |
| DataLoader worker 崩溃 + `libavutil.so.59` | torchcodec 缺匹配 FFmpeg | 加 `--dataset.video_backend=pyav` |
| `ACT has no attribute from_pretrained` | 导入错符号 | 用 `ACTPolicy`，不是 `ACT` |
| `.venv-train` 只有 100K | Cursor 沙箱重定向 UV_CACHE_DIR | `env UV_CACHE_DIR="$HOME/.cache/uv" ./init.sh ...` |
| `numpy 2.5.4` 导致装 lerobot 失败 | lerobot 0.5.1 要求 `<2.3` | 让 uv 自动解析，或手动 `numpy==2.2.6` |

---

## 7. 与 sdg 的分工

| 环节 | 工具 | 环境 |
|---|---|---|
| 采集 | `sdg.sh capture`（Isaac 内直采） | `.venv` + ROS |
| 打包 / 体检 | `sdg.sh pack` / `diag` / `trim` | `.venv` |
| 导出 LeRobot | `sdg.sh export` / `export-check` | `.venv` |
| 训练 | `lerobot.scripts.lerobot_train` | `.venv-train` |
| 推理 | `examples/IsaacSim/inference.py` | `.venv-train` |

sdg 侧的详细说明见
`~/FaSim-Isaac/exts/fasim.replicator.sdg/docs/ACT_TRAINING_ROADMAP.md`。
