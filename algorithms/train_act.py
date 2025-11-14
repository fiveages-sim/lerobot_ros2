#!/usr/bin/env python3
"""
Thin wrapper around `lerobot.scripts.train` to train ACT on a local dataset.

Usage:
    python algorithms/train_act.py /path/to/dataset \
        --chunk-size 16 --steps 1000 --batch-size 8 --camera-key observation.images.wrist_camera
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_LEROBOT = REPO_ROOT.parent / "lerobot" / "src"
if UPSTREAM_LEROBOT.exists():
    sys.path.insert(0, str(UPSTREAM_LEROBOT))

from lerobot.configs.default import DatasetConfig, WandBConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.scripts.train import train as lerobot_train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ACT policy on a local LeRobot dataset.")
    parser.add_argument("dataset_path", type=Path, help="Path to the dataset directory (contains meta/data/videos)")
    parser.add_argument("--chunk-size", type=int, default=50, help="ACT chunk size")
    parser.add_argument("--n-action-steps", type=int, default=50, help="How many action steps to execute per chunk")
    parser.add_argument("--steps", type=int, default=10000, help="Number of optimizer steps")
    parser.add_argument("--batch-size", type=int, default=8, help="Training batch size")
    parser.add_argument("--num-workers", type=int, default=4, help="PyTorch DataLoader workers")
    parser.add_argument("--device", type=str, default="cuda", help="Training device (cuda|cpu|mps)")
    parser.add_argument("--output-dir", type=Path, default=None, help="Optional output directory for checkpoints")
    parser.add_argument("--log-freq", type=int, default=100, help="Frequency (in steps) for logging metrics")
    parser.add_argument("--save-freq", type=int, default=5000, help="Frequency for checkpointing (0 disables)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_path = args.dataset_path.expanduser().resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(dataset_path)

    dataset_cfg = DatasetConfig(
        repo_id=dataset_path.name,
        root=str(dataset_path),
        use_imagenet_stats=True,
    )

    policy_cfg = ACTConfig(
        chunk_size=args.chunk_size,
        n_action_steps=args.n_action_steps,
        push_to_hub=False,
        repo_id=None,
        device=args.device,
    )
    train_cfg = TrainPipelineConfig(
        dataset=dataset_cfg,
        policy=policy_cfg,
        output_dir=args.output_dir,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        steps=args.steps,
        log_freq=args.log_freq,
        eval_freq=0,
        save_checkpoint=args.save_freq > 0,
        save_freq=args.save_freq if args.save_freq > 0 else args.steps,
        wandb=WandBConfig(enable=False),
    )

    lerobot_train(train_cfg)


if __name__ == "__main__":
    main()
