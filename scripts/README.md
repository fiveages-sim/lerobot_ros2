# `scripts/` — 父仓辅助脚本

日常部署走仓库根目录 **`./init.sh`**，不要另开一条环境。

## 日常（保留，可给新人）

| 脚本 | 作用 | 怎么用 |
|------|------|--------|
| `lr-env.sh` | `init.sh` 的环境后端（uv/conda、ROS hook、pip） | **不要直接跑**，由 `init.sh` source |
| `setup_grasp_dev_env.sh` | 兼容入口 | 等价于 `./init.sh all-motion` |
| `setup_wuji_retargeting.sh` | 可选：OSS retarget 仓 + 嵌套 MuJoCo | 产品路径已是 `pip install wuji-sdk`（`init.sh install` 会装）。仅调试开源算法时用 |

## 维护者一次性（公司 remote / submodule 登记）

clone 部署 **不需要** 跑这些：

| 脚本 | 作用 |
|------|------|
| `push_hug_company_remote.sh` | 把本地 HUG 推到 `fiveages-sim/HUG`（无 force） |
| `merge_push_hug_company.sh` | 合并公司仓初始 commit 后再 push |
| `push_wuji_retargeting_company.sh` | 推 wuji-retargeting 公司仓 |
| `register_wuji_retargeting_submodule.sh` | 把 wuji-retargeting 登记进 `.gitmodules`（已登记则不必再跑） |

## 推荐部署

```bash
git clone --recursive git@github.com:fiveages-sim/lerobot_ros2.git
cd lerobot_ros2
./init.sh all-motion          # 运控 + grasp-generation CLI
./init.sh ros2-workspace      # 写入 ROS2 工作空间并挂 activate
# 需要 HUG 推理时：
./init.sh hug-env             # 或 ./init.sh all-grasp
source .venv/bin/activate
```
