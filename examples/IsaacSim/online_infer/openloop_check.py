#!/usr/bin/env python3
"""L2 开环验证 —— 用训练好的 ACT 在已有数据集上逐帧预测，与真值对比。

## 这个工具在什么位置
这是**训练后的第一道验证**，不是推理部署。它只算数字、不驱动机器人，
用来在花时间跑闭环之前，先排掉明显的低级错误：

  - 动作量级错（预测是真值的上千倍 → 归一化没做）
  - 大量饱和（±1 撑满）
  - 某几维完全不动（学漏了）
  - 抖动过大（相邻帧乱跳）

真正的闭环推理在 `../inference.py`（走运控、OCS2 做 IK）。

## 用法
```bash
cd ~/lerobot_ros2 && source .venv-train/bin/activate

python examples/IsaacSim/online_infer/openloop_check.py \
    --dataset ~/lerobot_datasets/tube_25eps \
    --checkpoint ~/act_runs/tube25/checkpoints/020000/pretrained_model \
    --episode 0 --start 300 --max-frames 200
```

**`--start` 很重要**：跳过开头静止段。静止段所有动作接近 0，
预测再准也说明不了问题（会得到「真值 med=0、相关性 NaN」）。

## 归一化（最容易踩的坑）
lerobot 0.5.x 把归一化抽到了 `PolicyProcessorPipeline`：
`policy.select_action()` **只做前向**，既不归一化输入，也不反归一化输出。
漏掉这一步的症状：预测值是真值的 **1000 倍**、21.9% 饱和到 ±1。

正确流程：
```python
batch = preprocessor(batch)                  # 归一化输入
a = policy.select_action(batch)              # 前向
a = postprocessor({"action": a})["action"]   # 反归一化输出
```

实测（25 条 / 20k 步 / loss 0.044，取运动段）：补上归一化后
MAE 从 0.496 → **0.0057**（改善 87 倍），逐维相关性 0.996。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _open_dataset(root: Path, episode: int, video_backend: str = "pyav"):
    """用官方 LeRobotDataset 打开（自动处理视频解码与归一化 stats）。"""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(
        repo_id=root.name,
        root=str(root),
        video_backend=video_backend,
    )
    ep_idx = np.asarray(ds.hf_dataset["episode_index"])
    idx = np.flatnonzero(ep_idx == episode)
    return ds, idx


def _load_policy(ckpt: Path):
    """加载 ACTPolicy 与 pre/post processor（归一化）。"""
    from lerobot.policies.act.modeling_act import ACTPolicy

    policy = ACTPolicy.from_pretrained(str(ckpt))
    policy.eval()

    pre = post = None
    try:
        from lerobot.processor.pipeline import PolicyProcessorPipeline

        for fname, tag in (("policy_preprocessor.json", "pre"),
                           ("policy_postprocessor.json", "post")):
            if (ckpt / fname).is_file():
                pipe = PolicyProcessorPipeline.from_pretrained(
                    pretrained_model_name_or_path=str(ckpt),
                    config_filename=fname,
                    to_transition=None,
                )
                if tag == "pre":
                    pre = pipe
                else:
                    post = pipe
    except Exception as exc:  # noqa: BLE001
        print(f"[l2] WARN processor 加载失败（预测值将不可比）: {exc}")
    return policy, pre, post


def main() -> int:
    ap = argparse.ArgumentParser(description="ACT 开环验证（只算不驱动）")
    ap.add_argument("--dataset", required=True, help="LeRobot 数据集根目录")
    ap.add_argument("--checkpoint", required=True, help="pretrained_model 目录")
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--out", default="", help="输出目录")
    ap.add_argument("--start", type=int, default=0,
                    help="从第几帧开始（跳过开头静止段，建议给几百）")
    ap.add_argument("--max-frames", type=int, default=0, help="只跑 N 帧")
    ap.add_argument("--video-backend", default="pyav",
                    help="视频后端（torchcodec 在部分机器上缺 libavutil）")
    args = ap.parse_args()

    root = Path(args.dataset).expanduser().resolve()
    ckpt = Path(args.checkpoint).expanduser().resolve()
    out = (Path(args.out).expanduser() if args.out
           else ckpt.parent.parent / "l2")
    out.mkdir(parents=True, exist_ok=True)

    ds, idx = _open_dataset(root, args.episode, args.video_backend)
    if idx.size == 0:
        raise SystemExit(f"episode {args.episode} 不在 {root}")
    if args.start > 0:
        idx = idx[args.start:]
    if args.max_frames > 0:
        idx = idx[: args.max_frames]
    print(f"[l2] episode {args.episode}: {len(idx)} 帧  from {root.name}")

    policy, pre, post = _load_policy(ckpt)
    import torch

    dev = next(policy.parameters()).device
    print(f"[l2] device: {dev}  preprocessor={'有' if pre else '无'} "
          f"postprocessor={'有' if post else '无'}")
    if pre is None:
        print("[l2] WARN 没有 preprocessor —— 输入未归一化，结果不可信")

    preds, gts = [], []
    with torch.no_grad():
        for k, i in enumerate(idx):
            item = ds[int(i)]
            batch = {}
            for key, val in item.items():
                if key == "task":
                    continue
                t = val if isinstance(val, torch.Tensor) else torch.as_tensor(val)
                batch[key] = t.unsqueeze(0).to(dev)

            if pre is not None:                    # 1) 归一化输入
                batch = pre(batch)
            a = policy.select_action(batch)        # 2) 前向
            if post is not None:                   # 3) 反归一化输出
                o = post({"action": a})
                a = o.get("action", a) if isinstance(o, dict) else o
            arr = np.asarray(a.squeeze(0).detach().cpu()).reshape(-1)
            preds.append(arr.astype(np.float32))
            gts.append(np.asarray(item["action"]).reshape(-1).astype(np.float32))
            if (k + 1) % 100 == 0:
                print(f"[l2]   {k+1}/{len(idx)}")

    pred = np.array(preds, dtype=np.float32)
    gt = np.array(gts, dtype=np.float32)
    if pred.shape != gt.shape:
        print(f"[l2] WARN 形状不一致 pred={pred.shape} gt={gt.shape}")

    mae_dim = np.abs(pred - gt).mean(axis=0)
    mae = float(np.abs(pred - gt).mean())
    print(f"\n[l2] 整体 MAE: {mae:.6f}")
    print(f"[l2] 量级: 真值 mean|a|={np.abs(gt).mean():.6f}  "
          f"预测 mean|a|={np.abs(pred).mean():.6f}")

    corrs = []
    for d in range(pred.shape[1]):
        x, y = pred[:, d], gt[:, d]
        corrs.append(np.nan if (x.std() < 1e-9 or y.std() < 1e-9)
                     else float(np.corrcoef(x, y)[0, 1]))
    corrs = np.array(corrs)
    valid = ~np.isnan(corrs)
    if valid.any():
        print(f"[l2] 逐维相关性: mean={np.nanmean(corrs):.3f} "
              f"中位={np.nanmedian(corrs):.3f} "
              f">0.8 的维度 {int((corrs[valid]>0.8).sum())}/{valid.sum()}")
    else:
        print("[l2] 逐维相关性: 全部 NaN（真值或预测方差为 0 —— 可能在静止段）")

    jp = np.linalg.norm(np.diff(pred, axis=0), axis=1) if len(pred) > 1 else np.array([0.0])
    jg = np.linalg.norm(np.diff(gt, axis=0), axis=1) if len(gt) > 1 else np.array([0.0])
    ratio = float(np.median(jp) / max(np.median(jg), 1e-12))
    print(f"[l2] 相邻帧变化: 真值 med={np.median(jg):.6f} 预测 med={np.median(jp):.6f} "
          f"({ratio:.2f}x)")

    sat = float((np.abs(pred) > 0.99).mean())
    dead = int((np.abs(pred).std(axis=0) < 1e-6).sum())

    issues = []
    if mae > 0.1:
        issues.append(f"MAE 偏大 ({mae:.4f})")
    if valid.any() and np.nanmean(corrs) < 0.5:
        issues.append(f"相关性低 ({np.nanmean(corrs):.2f})")
    if ratio > 3:
        issues.append(f"抖动 {ratio:.1f}x")
    if sat > 0.01:
        issues.append(f"{sat*100:.1f}% 饱和")
    print("\n[l2] 判定: " + ("✅ 无明显问题" if not issues else "⚠️ " + "; ".join(issues)))

    res = {
        "episode": args.episode, "frames": len(pred), "start": args.start,
        "mae": mae,
        "pred_abs_mean": float(np.abs(pred).mean()),
        "gt_abs_mean": float(np.abs(gt).mean()),
        "corr_mean": float(np.nanmean(corrs)) if valid.any() else None,
        "corr_median": float(np.nanmedian(corrs)) if valid.any() else None,
        "jitter_ratio": ratio,
        "saturated_frac": sat,
        "dead_dims": dead,
        "issues": issues,
        "per_dim_mae": mae_dim.tolist(),
        "per_dim_corr": [None if np.isnan(c) else float(c) for c in corrs],
    }
    p = out / f"l2_ep{args.episode}.json"
    p.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[l2] 结果: {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
