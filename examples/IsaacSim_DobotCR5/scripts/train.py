#!/usr/bin/env python3
"""
Thin wrapper around `lerobot.scripts.train` to train ACT or Diffusion on a local dataset.

Usage:
    # ACT
    python examples/train_demo.py /path/to/dataset --policy act --chunk-size 16 --n-action-steps 8 --steps 1000
    CUDA_VISIBLE_DEVICES=1 python examples/train_demo.py  /data/sylcito/grasp_dataset_50  --policy act --chunk-size 16 --n-action-steps 8   --steps 20000 --batch-size 8   --output-dir outputs/act_run1 --device cuda
    CUDA_VISIBLE_DEVICES=1 python examples/train_demo.py /path/to/dataset --policy act --chunk-size 16 --n-action-steps 8 --steps 1000 --batch-size 8 --num-workers 4 --device cuda --images-only --drop-ee-state --act-disable-vae

    #diffusion_policy
    CUDA_VISIBLE_DEVICES=4 python   examples/train_demo.py  /data/sylcito/grasp_dataset_50  --policy diffusion --diffusion-horizon 16 --diffusion-n-action-steps 8 --diffusion-n-obs-steps 2 --steps 2100   --output-dir outputs/diffusion_run1 --device cuda
    python examples/train_demo.py /path/to/dataset --policy diffusion --diffusion-horizon 16 \
        --diffusion-n-action-steps 8 --diffusion-n-obs-steps 2 --steps 1000
"""

from __future__ import annotations

import argparse
from datetime import datetime
import inspect
import json
import logging
from pathlib import Path


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
        nargs="?",
        default=None,
        help=(
            "Path to the dataset directory (contains meta/data/videos). "
            "If omitted, auto-detect latest dataset under ./lerobot_dataset/"
        ),
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
        "--drop-ee-state",
        action="store_true",
        help="For ACT, drop end-effector/ gripper state dims (last 8 dims) from observation.state.",
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
        default=None,
        help=(
            "Timestamp tolerance (seconds) for video/trajectory alignment; "
            "only applied when current DatasetConfig supports this field."
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


def _load_state_names(dataset_root: Path) -> list[str] | None:
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.is_file():
        return None
    info = json.loads(info_path.read_text())
    state = info.get("features", {}).get("observation.state", {})
    names = state.get("names")
    return list(names) if isinstance(names, list) else None


def _resolve_dataset_path(dataset_path_arg: Path | None) -> Path:
    if dataset_path_arg is not None:
        dataset_path = dataset_path_arg.expanduser().resolve()
        if not dataset_path.exists():
            raise FileNotFoundError(dataset_path)
        return dataset_path

    dataset_parent = Path.cwd() / "lerobot_dataset"
    if not dataset_parent.is_dir():
        raise FileNotFoundError(
            f"dataset_path not provided and dataset parent not found: {dataset_parent}"
        )

    candidates = [
        p for p in dataset_parent.iterdir()
        if p.is_dir() and (p / "meta" / "info.json").is_file()
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No valid dataset found in {dataset_parent} (expected meta/info.json in subdirectories)"
        )

    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    logging.info("Auto-selected latest dataset: %s", latest)
    return latest.resolve()


def _resolve_output_dir(output_dir_arg: Path | None, dataset_path: Path, policy: str) -> Path:
    if output_dir_arg is not None:
        return output_dir_arg.expanduser().resolve()

    output_parent = Path.cwd() / "outputs"
    output_parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    auto_output_dir = output_parent / f"{dataset_path.name}_{policy}_{stamp}"
    logging.info("Auto-selected output directory: %s", auto_output_dir)
    return auto_output_dir.resolve()


def main() -> None:
    init_logging()
    args = parse_args()
    dataset_path = _resolve_dataset_path(args.dataset_path)
    output_dir = _resolve_output_dir(args.output_dir, dataset_path, args.policy)

    episodes = _parse_episode_list(args.episodes)
    video_backend = get_safe_default_codec()
    dataset_kwargs: dict[str, object] = {
        "repo_id": dataset_path.name,
        "root": str(dataset_path),
        "use_imagenet_stats": True,
    }
    supported_dataset_fields = set(inspect.signature(DatasetConfig).parameters.keys())
    if "video_backend" in supported_dataset_fields:
        dataset_kwargs["video_backend"] = video_backend
    if "episodes" in supported_dataset_fields:
        dataset_kwargs["episodes"] = episodes
    if args.tolerance_s is not None and "tolerance_s" in supported_dataset_fields:
        dataset_kwargs["tolerance_s"] = args.tolerance_s
    elif args.tolerance_s is not None and "tolerance_s" not in supported_dataset_fields:
        logging.warning(
            "Current DatasetConfig does not support 'tolerance_s'; ignoring --tolerance-s=%s",
            args.tolerance_s,
        )
    dataset_cfg = DatasetConfig(**dataset_kwargs)

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
        if args.drop_ee_state:
            # Drop gripper + end-effector pose/rotation dims based on dataset meta.
            state_names = _load_state_names(dataset_path)
            if state_names:
                drop_names = [
                    "gripper_joint.pos",
                    "end_effector.position.x",
                    "end_effector.position.y",
                    "end_effector.position.z",
                    "end_effector.orientation.x",
                    "end_effector.orientation.y",
                    "end_effector.orientation.z",
                    "end_effector.orientation.w",
                    "end_effector.rot6d_0",
                    "end_effector.rot6d_1",
                    "end_effector.rot6d_2",
                    "end_effector.rot6d_3",
                    "end_effector.rot6d_4",
                    "end_effector.rot6d_5",
                ]
                policy_cfg.drop_state_indices = [i for i, n in enumerate(state_names) if n in drop_names]
            else:
                # Fallback to dropping the tail if meta is missing.
                policy_cfg.drop_state_indices = list(range(12, 22))
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
        output_dir=output_dir,
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
