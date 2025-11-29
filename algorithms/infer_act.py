#!/usr/bin/env python3
"""
Run offline inference with a trained ACT checkpoint on a dataset.

Usage:
    python algorithms/infer_act.py \
        --dataset /path/to/dataset \
        --checkpoint /path/to/checkpoint_dir \
        --num-episodes 1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]


from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_policy
from lerobot.configs.train import TrainPipelineConfig
from lerobot.configs.default import DatasetConfig
from lerobot.scripts.train import make_dataset
from lerobot.configs.parser import parser


def parse_args() -> argparse.Namespace:
    parser_ = argparse.ArgumentParser(description="ACT inference on recorded dataset")
    parser_.add_argument("--dataset", type=Path, required=True, help="Path to LeRobot dataset directory")
    parser_.add_argument("--train-config", type=Path, required=True, help="Path to train_config.json or directory")
    parser_.add_argument("--num-episodes", type=int, default=1, help="Number of episodes to replay")
    return parser_.parse_args()


def main() -> None:
    args = parse_args()
    dataset_path = args.dataset.expanduser().resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(dataset_path)

    # Load train pipeline config (contains policy + dataset settings).
    train_cfg = TrainPipelineConfig.from_pretrained(args.train_config)
    train_cfg.dataset.root = dataset_path
    train_cfg.dataset.repo_id = dataset_path.name
    ds = make_dataset(train_cfg)

    policy = make_policy(train_cfg.policy, ds_meta=ds.meta)

    episodes = min(args.num_episodes, ds.meta.total_episodes)
    for ep in range(episodes):
        start = ds.meta.episodes["dataset_from_index"][ep]
        end = ds.meta.episodes["dataset_to_index"][ep]
        print(f"Episode {ep}: frames {start} -> {end}")
        for idx in tqdm.trange(start, end):
            sample = ds[idx]
            batch = {"observation": sample}
            actions = policy.select_action(batch)
            # For offline inspection we simply print the first action vector.
            print(f"Frame {idx}: action={actions.squeeze(0).cpu().numpy()}")
        policy.reset()


if __name__ == "__main__":
    main()
