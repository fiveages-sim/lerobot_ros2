# 本工作区环境说明（grasp / ros2-stack）

**唯一开发目录：** 本仓库 clone 根目录。文档与脚本不再写死某一台机器的绝对路径。

抓取与运控共用 **父仓环境（Python 3.12 + 系统 rclpy）**。HUG 推理单独用 **`submodules/hug/.venv`（Python 3.10）**，不要混装 CLI。

从零步骤与「可以不装灵巧手」见 [`SETUP.md`](SETUP.md)。

## 一键

```bash
git clone git@github.com:fiveages-sim/lerobot_ros2.git
cd lerobot_ros2

# 只要任务编排：
./init.sh all-motion
./init.sh ros2-workspace

# 还要灵巧手抓取（含 HUG 环境；MANO 仍要自己放）：
./init.sh all-grasp

source .venv/bin/activate
./init.sh doctor
```

`scripts/setup_grasp_dev_env.sh` 等价于 `./init.sh all-grasp`。不要灵巧手时不要跑它，用 `all-motion`。

## 日常启动

```bash
source .venv/bin/activate

cd examples/IsaacSim
ros2-stack launch --robot <robot_key>

# 另开终端同样 activate 后：
motion-generation
# 该机器人配了 grasp_generation.yaml 时：
grasp-generation
```

参考机器人键：`fiveages_w2`。

## 禁止

- 用系统 `pip install`（会 `externally-managed-environment`）
- 在 `submodules/hug/.venv` 里装 `ros2-stack` / `grasp-generation` / `motion-generation`
- 把环境依赖指到其它 checkout 的 `.venv`
- 在本仓环境里为了 lerobot 强行装 **NumPy 2.x** 后再跑依赖系统 `cv2` 的抓取（`./init.sh all` / `install-lerobot` 会装 numpy 2；编排和抓取用 `all-motion` / `all-grasp`）
