# HUG 子模块策略（公司仓用法）

上游公开仓：[KevinyWu/hug](https://github.com/KevinyWu/hug) — **不要直接 push 我们的补丁**。

本仓以 submodule 挂在 `submodules/hug`，并带有 **公司本地提交**（uv 环境、HF 离线加载、`pcl_rgb` 推理修复、W2 smoke 脚本等）。

## 别人 clone 后怎么拿到补丁

当前 `.gitmodules` 仍指向上游 URL。上游 **没有** 我们的 commit SHA，因此：

```bash
git submodule update --init submodules/hug
# 只会落到上游能看到的历史；公司补丁需另同步
```

**推荐（分享 / CI 前做一次）：**

1. 在 `fiveages-sim`（或公司 Git）建 fork：`fiveages-sim/hug`
2. 把本机 `submodules/hug` 的分支 push 到该 fork（含补丁 commit）
3. 改父仓 `.gitmodules`：

```gitconfig
[submodule "submodules/hug"]
	path = submodules/hug
	url = git@github.com:fiveages-sim/hug.git
```

4. 父仓记录的 gitlink SHA 指向 fork 上已推送的 commit

**临时（仅本机 / 内网拷贝）：** 直接拷贝整个 `submodules/hug`（含 `.git`）或 `git bundle` 传补丁 commit。

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
