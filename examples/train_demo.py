#!/usr/bin/env python3
"""
Thin wrapper around `lerobot.scripts.train` to train ACT or Diffusion on a local dataset.

Usage:
    # ACT
    python algorithms/train_act.py /path/to/dataset --policy act --chunk-size 16 --n-action-steps 8 --steps 1000
    CUDA_VISIBLE_DEVICES=1 /home/gtengliu/miniconda3/envs/lerobot_ros2_act/bin/python   /home/gtengliu/lerobot_ros2/algorithms/train_demo.py   /data/sylcito/grasp_dataset_50   --chunk-size 16 --n-action-steps 8   --steps 20000 --batch-size 8   --output-dir outputs/act_run1 --device cuda
    #diffusion_policy
    CUDA_VISIBLE_DEVICES=4 /home/gtengliu/miniconda3/envs/lerobot_ros2_act/bin/python   /home/gtengliu/lerobot_ros2/algorithms/train_demo.py  /data/sylcito/grasp_dataset_50  --policy diffusion --diffusion-horizon 16 --diffusion-n-action-steps 8 --diffusion-n-obs-steps 2 --steps 2100   --output-dir outputs/diffusion_run1 --device cuda
    python algorithms/train_act.py /path/to/dataset --policy diffusion --diffusion-horizon 16 \
        --diffusion-n-action-steps 8 --diffusion-n-obs-steps 2 --steps 1000
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]


from lerobot.constants import CHECKPOINTS_DIR, LAST_CHECKPOINT_LINK
from lerobot.configs.default import DatasetConfig, WandBConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.scripts.train import train as lerobot_train
from lerobot.datasets.video_utils import get_safe_default_codec
from lerobot.utils.train_utils import get_step_checkpoint_dir
from lerobot.utils.utils import init_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ACT or Diffusion policy on a local LeRobot dataset.")
    parser.add_argument(
        "dataset_path",
        type=Path,
        default="/home/gtengliu/lerobot_ros2/dataset/grasp_dataset_50_no_depth",
        help="Path to the dataset directory (contains meta/data/videos)",
    )
    parser.add_argument(
        "--episodes",
        type=str,
        default=None,
        help="Optional subset of episodes to train on, e.g. '0:1' or '0,1,2' (useful to overfit a tiny set).",
    )
    parser.add_argument("--policy", choices=["act", "diffusion"], default="act", help="Which policy to train.")
    # ACT-specific knobs.
    parser.add_argument("--chunk-size", type=int, default=50, help="ACT chunk size")
    parser.add_argument("--n-action-steps", type=int, default=50, help="ACT: action steps executed per chunk")
    parser.add_argument(
        "--images-only",
        action="store_true",
        help="Use only image inputs (drop observation.state/environment_state) to mimic vision-only ACT.",
    )
    parser.add_argument(
        "--act-disable-vae",
        action="store_true",
        help="Disable VAE loss for ACT (useful when debugging/overfitting).",
    )
    parser.add_argument(
        "--act-lr",
        type=float,
        default=1e-5,
        help="Override ACT base learning rate (default from config if not set).",
    )
    parser.add_argument(
        "--act-lr-backbone",
        type=float,
        default=1e-5,
        help="Override ACT backbone learning rate (default from config if not set).",
    )
    # Diffusion-specific knobs.
    parser.add_argument("--diffusion-horizon", type=int, default=16, help="Diffusion: trajectory horizon length")
    parser.add_argument("--diffusion-n-action-steps", type=int, default=8, help="Diffusion: action steps to execute")
    parser.add_argument(
        "--diffusion-n-obs-steps", type=int, default=2, help="Diffusion: number of observation timesteps as context"
    )
    parser.add_argument("--steps", type=int, default=10000, help="Number of optimizer steps")
    parser.add_argument("--batch-size", type=int, default=8, help="Training batch size")
    parser.add_argument("--num-workers", type=int, default=4, help="PyTorch DataLoader workers")
    parser.add_argument("--device", type=str, default="cuda", help="Training device (cuda|cpu|mps)")
    parser.add_argument("--output-dir", type=Path, default=None, help="Optional output directory for checkpoints")
    parser.add_argument("--log-freq", type=int, default=100, help="Frequency (in steps) for logging metrics")
    parser.add_argument("--save-freq", type=int, default=5000, help="Frequency for checkpointing (0 disables)")
    parser.add_argument(
        "--tolerance-s",
        type=float,
        default=0.5,
        help=(
            "Timestamp tolerance (seconds) for video/trajectory alignment; "
            "defaults to 1/fps - 1e-4 when not set."
        ),
    )
    return parser.parse_args()


def _parse_episode_list(ep_str: str | None) -> list[int] | None:
    if ep_str is None:
        return None
    ep_str = ep_str.strip()
    if "," in ep_str:
        return [int(e) for e in ep_str.split(",") if e]
    if ":" in ep_str:
        start, end = ep_str.split(":")
        return list(range(int(start), int(end)))
    return [int(ep_str)]


def main() -> None:
    init_logging()
    args = parse_args()
    dataset_path = args.dataset_path.expanduser().resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(dataset_path)

    episodes = _parse_episode_list(args.episodes)
    video_backend = get_safe_default_codec()
    dataset_cfg = DatasetConfig(
        repo_id=dataset_path.name,
        root=str(dataset_path),
        use_imagenet_stats=True,
        tolerance_s=args.tolerance_s,
        video_backend=video_backend,
        episodes=episodes,
    )

    if args.policy == "act":
        lr = args.act_lr if args.act_lr is not None else None
        lr_backbone = args.act_lr_backbone if args.act_lr_backbone is not None else None
        policy_cfg = ACTConfig(
            chunk_size=args.chunk_size,
            n_action_steps=args.n_action_steps,
            push_to_hub=False,
            repo_id=None,
            device=args.device,
            use_vae=not args.act_disable_vae,
            optimizer_lr=lr if lr is not None else ACTConfig().optimizer_lr,
            optimizer_lr_backbone=lr_backbone if lr_backbone is not None else ACTConfig().optimizer_lr_backbone,
        )
        policy_cfg.images_only = args.images_only
        policy_desc = f"ACT (chunk_size={args.chunk_size}, n_action_steps={args.n_action_steps})"
    elif args.policy == "diffusion":
        policy_cfg = DiffusionConfig(
            n_obs_steps=args.diffusion_n_obs_steps,
            horizon=args.diffusion_horizon,
            n_action_steps=args.diffusion_n_action_steps,
            push_to_hub=False,
            repo_id=None,
            device=args.device,
        )
        policy_desc = (
            "Diffusion "
            f"(n_obs_steps={args.diffusion_n_obs_steps}, horizon={args.diffusion_horizon}, "
            f"n_action_steps={args.diffusion_n_action_steps})"
        )
    else:
        raise ValueError(f"Unsupported policy '{args.policy}'")

    train_cfg = TrainPipelineConfig(
        dataset=dataset_cfg,
        policy=policy_cfg,
        output_dir=args.output_dir,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        steps=args.steps,
        log_freq=max(args.log_freq, 100),
        eval_freq=0,
        save_checkpoint=args.save_freq > 0,
        save_freq=args.save_freq if args.save_freq > 0 else args.steps,
        wandb=WandBConfig(enable=False),
    )

    # Run validation here to surface the resolved output directory for logging.
    train_cfg.validate()

    checkpoints_root = train_cfg.output_dir / CHECKPOINTS_DIR
    final_checkpoint_dir = get_step_checkpoint_dir(train_cfg.output_dir, train_cfg.steps, train_cfg.steps)
    last_checkpoint_link = checkpoints_root / LAST_CHECKPOINT_LINK

    logging.info("Training %s on dataset: %s", policy_desc, dataset_path)
    logging.info("Output directory (logs/checkpoints): %s", train_cfg.output_dir)
    if train_cfg.save_checkpoint:
        logging.info("Checkpoints every %s steps under: %s", train_cfg.save_freq, checkpoints_root)
        logging.info("Latest checkpoint symlink will be: %s", last_checkpoint_link)
        logging.info("Final-step checkpoint will be: %s", final_checkpoint_dir)
    else:
        logging.info("Checkpointing disabled (--save-freq set to 0); weights will not be saved.")

    lerobot_train(train_cfg)

    if train_cfg.save_checkpoint:
        if last_checkpoint_link.exists():
            logging.info("Training finished. Latest checkpoint is at: %s", last_checkpoint_link.resolve())
        else:
            logging.info("Training finished. Checkpoints saved under: %s", checkpoints_root)
    else:
        logging.info("Training finished. No checkpoints were written.")


if __name__ == "__main__":
    main()
