# 从零配置

仓库里有两条互不绑定的使用路径。没配灵巧手时，任务编排、运控启动和录制可以照常用。

| 路径 | 做什么 | 不包含 |
|------|--------|--------|
| **任务编排** | `ros2-stack` 起运控，`motion-generation` 按 YAML 跑技能序列 | HUG、MANO、手型 retarget |
| **灵巧手抓取（可选）** | `grasp-generation` 手动 Viser，或技能 `dex_hand.auto_grasp` 接进同一条任务队列 | 不替换上面的编排 |

参考实现是 **FiveAges W2 + Wuji Hand2**：`examples/IsaacSim/robots/FiveAges_W2/`。换机器人时复制 `grasp_generation.yaml` / `robot.yaml`，改相机、坐标系和 `hand_profile`。没有这份抓取配置的机器人，仍然可以只用任务编排。

前置：Ubuntu 上的 ROS 2（本仓在 Jazzy 上用过）、Python 3.12、[uv](https://docs.astral.sh/uv/) 或 Conda。Isaac Sim、USD、运控工作空间不在本仓库里，见 [`examples/IsaacSim/README.md`](../examples/IsaacSim/README.md)。

## 1. 只要任务编排

```bash
git clone git@github.com:fiveages-sim/lerobot_ros2.git
cd lerobot_ros2
./init.sh all-motion
./init.sh ros2-workspace          # 把 ROS 2 工作空间写进激活脚本
source .venv/bin/activate         # backend=conda 时改为 conda activate
```

`all-motion` 只检出 `ros2_robot_interface` 和 `robot_action_composer`，并装进父仓环境。不会下载 HUG，也不会装 `wuji-sdk`。

```bash
cd examples/IsaacSim
ros2-stack launch --robot <robot_key>
# 另一个已激活同一环境的终端：
motion-generation --robot <robot_key>
```

检查缺什么（只打印，不安装）：

```bash
./init.sh doctor
```

## 2. 再加上灵巧手抓取

在第 1 步之后，或直接：

```bash
./init.sh all-grasp
source .venv/bin/activate
./init.sh doctor
```

`all-grasp` = 任务编排 + 检出 `submodules/hug` 与 `submodules/wuji-retargeting` + 父仓安装 `viser` 和 `wuji-sdk` + HUG 自己的 Python 3.10 环境。

也可以拆开，缺哪步补哪步：

| 命令 | 作用 |
|------|------|
| `./init.sh submodules-dex` | 只做 git 检出。不含权重、不含 MANO |
| `./init.sh install-grasp` | 父仓环境安装 `viser` 与 `wuji-sdk` |
| `./init.sh hug-env` | `submodules/hug/.venv`（Python 3.10）并尝试下载权重 |

**不要**把 `ros2-stack` / `grasp-generation` 装进 `submodules/hug/.venv`。那个环境只跑 HUG 推理；父仓 CLI 用子进程去调它。

### 2.1 git 里没有的两样东西

HUG 子模块的 `.gitignore` 排除了权重和 MANO。父仓 `.gitignore` 又写了一遍，避免 gitlink 异常时被误加进父仓库。

| 资源 | 路径 | 怎么得到 |
|------|------|----------|
| HUG 权重 | `submodules/hug/checkpoints/hug_full.safetensors` | `hug-env` 会调 `scripts/download_assets.sh`；失败就在 hug 环境里重跑该脚本。国内默认 `HF_ENDPOINT=https://hf-mirror.com` |
| DINOv2 | `~/.cache/huggingface/...dinov2-with-registers-base` | 同上脚本预拉。有缓存后应离线加载；直连 huggingface.co 会一直 Retry，Viser 起不来 |
| MANO | `submodules/hug/assets/mano_models/models/MANO_RIGHT.pkl` | **不能自动下载**。到 [mano.is.tue.mpg.de](https://mano.is.tue.mpg.de/) 注册，解压后保留 `models/` 与 `webuser/` |

布局和网络坑写在 [`submodules/hug/docs/SETUP_LOCAL.md`](../submodules/hug/docs/SETUP_LOCAL.md)。

`wuji-retargeting` 子模块默认 **不** 递归拉取里面的 `mujoco-sim` / `wuji-description`（上游历史里有过缺失的 LFS 对象）。运行时用的是 PyPI 上的 `wuji-sdk`。只有要读开源算法、开 tuning 工具时才：

```bash
bash scripts/setup_wuji_retargeting.sh --recursive-nested
```

其它手型不要求装 `wuji-sdk`。`grasp_generation.yaml` 里的 `hand_profile` 决定要哪套 retarget；参考实现的值是 `wuji_hand2`。

## 3. 参考实现怎么交互

终端都先 `source` 父仓环境，工作目录 `examples/IsaacSim`。

```bash
ros2-stack launch --robot fiveages_w2
```

```bash
grasp-generation
# 或 grasp-generation serve --robot fiveages_w2 --workspace .
```

任务队列里的同一套库（导航靠近后再抓）走 `motion-generation`，技能名 `dex_hand.auto_grasp`。参考任务：`examples/IsaacSim/robots/FiveAges_W2/task_configs/dex_hand/navigate_and_dex_grasp.yaml`。

Viser 操作、特权选物、`click_prim`、close 执行器 B0–B5，见 [`submodules/robot_action_composer/docs/GRASP_GENERATION.md`](../submodules/robot_action_composer/docs/GRASP_GENERATION.md)。

## 4. 常见坑（配置阶段）

- 系统 `pip` 在 Ubuntu 上会报 `externally-managed-environment`。用 `./init.sh`，或父仓环境里的 `uv pip`，本地包加 `--no-deps`（`rclpy` 来自 ROS，不是 PyPI）。
- `./init.sh all` / `install-lerobot` 会把 NumPy 升到 2.x。只做抓取和编排时用 `all-motion` 或 `all-grasp`，不要为了录制去污染这套环境。
- 激活脚本会 source `.fa-env.toml` 里的 ROS 2 工作空间。没配过就 `./init.sh ros2-workspace`。
- 缺模块时 CLI 会打印缺什么、用哪条 `init.sh` 补。这不改变抓取或编排的执行逻辑；补齐后按原来的命令再跑。

文档索引：[`docs/README.md`](README.md)。
