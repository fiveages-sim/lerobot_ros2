# HUG 子模块策略（公司仓用法）

上游公开仓：[KevinyWu/hug](https://github.com/KevinyWu/hug) — **不要直接 push 我们的补丁**。

公司 fork：[fiveages-sim/hug](https://github.com/fiveages-sim/hug)  
Submodule URL（`.gitmodules`）：`git@github.com:fiveages-sim/hug.git`

本仓以 submodule 挂在 `submodules/hug`，并带有 **公司本地提交**（uv 环境、HF 离线加载、`pcl_rgb` 推理修复、W2 smoke 脚本等）；补丁应 push 到公司 fork 的 `main`。

## 别人 clone 后怎么拿到补丁

```bash
git submodule update --init submodules/hug
```

父仓记录的 gitlink SHA 指向公司 fork 上已推送的 commit（例如 `3a0e67d`）。

本地 `submodules/hug` 建议 remotes：

- `origin` → `git@github.com:fiveages-sim/hug.git`（日常 push）
- `upstream` → `https://github.com/KevinyWu/hug.git`（仅参考 / 同步上游）

## 不要提交进 git 的内容

| 路径 | 原因 |
|------|------|
| `submodules/hug/.venv/` | 本地环境 |
| `submodules/hug/checkpoints/*.safetensors` | 大权重 |
| `submodules/hug/assets/mano_models/` | MANO 许可，需各自注册下载 |
| `submodules/hug/data/*` | 采帧 / 临时数据（已 gitignore） |
| 父仓 `submodules/lerobot/` | 与本功能无关，勿误加 |
| 父仓 `.venv/` | 本地环境 |

## 本仓相关补丁摘要

- `src/inference.py`：checkpoint `pcl_use_rgb=true` 时传入 `pcl_rgb`
- `src/models/encoders.py`：DINOv2 优先本地 HF cache / 镜像
- `scripts/*`、`docs/SETUP_LOCAL.md`、`pyproject.toml`：uv 本地开发
