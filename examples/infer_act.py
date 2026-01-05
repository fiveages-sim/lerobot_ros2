#!/usr/bin/env python3
"""
Run offline inference with a trained ACT checkpoint on a dataset.

Usage:
    # Minimal: point to checkpoint dir (or train_config.json) and dataset
    python algorithms/infer_act.py /path/to/checkpoint_dir --dataset /path/to/dataset --num-episodes 1
    CUDA_VISIBLE_DEVICES=1 /home/gtengliu/miniconda3/envs/lerobot_ros2_act/bin/python /home/gten
    gliu/lerobot_ros2/algorithms/infer_act.py /home/gtengliu/outputs/act_overfit1/checkpoints/last \
  --dataset /home/gtengliu/lerobot_ros2/dataset/grasp_dataset_1 \
  --num-episodes 1
    # If dataset is omitted, we use the dataset path stored in train_config.json inside the checkpoint.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]


from lerobot.configs.train import TrainPipelineConfig
from lerobot.policies.factory import make_policy
from lerobot.constants import ACTION, OBS_IMAGES
from lerobot.scripts.train import make_dataset
from lerobot.datasets.sampler import EpisodeAwareSampler

def parse_args() -> argparse.Namespace:
    parser_ = argparse.ArgumentParser(description="ACT inference on recorded dataset")
    parser_.add_argument(
        "checkpoint",
        type=Path,
        help="Path to checkpoint dir (containing pretrained_model/train_config.json) or directly to train_config.json",
    )
    parser_.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Path to LeRobot dataset directory (defaults to the one recorded in train_config).",
    )
    parser_.add_argument("--num-episodes", type=int, default=1, help="Number of episodes to replay")
    parser_.add_argument(
        "--compare-actions",
        action="store_true",
        help="Compute L1/MSE between predicted actions and ground-truth actions from the dataset.",
    )
    parser_.add_argument(
        "--debug-stats",
        action="store_true",
        help="Run a one-off normalized/denormalized stats check and exit.",
    )
    return parser_.parse_args()


def main() -> None:
    args = parse_args()
    ckpt_path = args.checkpoint.expanduser().resolve()

    # Resolve train_config.json location from either a direct file or a checkpoint directory.
    candidates = []
    if ckpt_path.is_file():
        candidates.append(ckpt_path)
    candidates.append(ckpt_path / "train_config.json")
    candidates.append(ckpt_path / "pretrained_model" / "train_config.json")
    # Fallback: search one level down for train_config.json
    candidates.extend(ckpt_path.glob("*/train_config.json"))
    train_cfg_path = None
    for cand in candidates:
        if cand.is_file():
            train_cfg_path = cand
            break
    if train_cfg_path is None:
        raise FileNotFoundError(
            f"train_config.json not found under {ckpt_path}. "
            "Pass either the train_config.json itself or a checkpoint directory containing it."
        )

    # Locate pretrained model dir to load weights/normalizer from checkpoint.
    if train_cfg_path.parent.name == "pretrained_model":
        pretrained_dir = train_cfg_path.parent
    elif (train_cfg_path.parent / "pretrained_model").is_dir():
        pretrained_dir = train_cfg_path.parent / "pretrained_model"
    else:
        pretrained_dir = train_cfg_path.parent

    # Load train pipeline config (contains policy + dataset settings).
    train_cfg = TrainPipelineConfig.from_pretrained(train_cfg_path)
    train_cfg.policy.pretrained_path = str(pretrained_dir)

    # Override dataset path if provided; otherwise use the one stored in the config.
    dataset_root = args.dataset if args.dataset else Path(train_cfg.dataset.root)
    dataset_path = dataset_root.expanduser().resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(dataset_path)

    train_cfg.dataset.root = str(dataset_path)
    train_cfg.dataset.repo_id = dataset_path.name
    ds = make_dataset(train_cfg)

    policy = make_policy(train_cfg.policy, ds_meta=ds.meta)
    device = torch.device(train_cfg.policy.device)

    if args.debug_stats:
        policy.eval()
        # Build a dataloader similar to training.
        if hasattr(policy.config, "drop_n_last_frames"):
            shuffle = False
            sampler = EpisodeAwareSampler(
                ds.meta.episodes["dataset_from_index"],
                ds.meta.episodes["dataset_to_index"],
                drop_n_last_frames=policy.config.drop_n_last_frames,
                shuffle=True,
            )
        else:
            shuffle = True
            sampler = None
        loader = torch.utils.data.DataLoader(
            ds,
            batch_size=min(train_cfg.batch_size, len(ds)),
            shuffle=shuffle and not train_cfg.dataset.streaming,
            sampler=sampler,
            num_workers=train_cfg.num_workers,
            pin_memory=device.type == "cuda",
            drop_last=False,
            prefetch_factor=2,
        )
        batch = next(iter(loader))
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                if batch[key].dtype != torch.bool:
                    batch[key] = batch[key].type(torch.float32) if device.type == "mps" else batch[key]
                batch[key] = batch[key].to(device, non_blocking=device.type == "cuda")

        # Replicate training forward: normalize inputs/targets then model().
        batch_norm = policy.normalize_inputs(batch)
        if policy.config.image_features:
            batch_norm = dict(batch_norm)
            batch_norm[OBS_IMAGES] = [batch_norm[key] for key in policy.config.image_features]
        batch_norm = policy.normalize_targets(batch_norm)
        gt_actions_norm = batch_norm[ACTION]
        with torch.no_grad():
            actions_hat, _ = policy.model(batch_norm)

        gt_flat = gt_actions_norm.detach().reshape(-1, gt_actions_norm.shape[-1]).cpu()
        pr_flat = actions_hat.detach().reshape(-1, actions_hat.shape[-1]).cpu()
        print("========== DEBUG: normalized space stats ==========")
        for d in range(gt_flat.shape[-1]):
            print(
                f"Norm dim {d}: "
                f"gt[{gt_flat[:, d].min():.3f}, {gt_flat[:, d].max():.3f}], std={gt_flat[:, d].std():.3f} | "
                f"pred[{pr_flat[:, d].min():.3f}, {pr_flat[:, d].max():.3f}], std={pr_flat[:, d].std():.3f}"
            )

        gt_actions_phys = policy.unnormalize_outputs({ACTION: gt_actions_norm})[ACTION]
        pred_actions_phys = policy.unnormalize_outputs({ACTION: actions_hat})[ACTION]
        gt_phys_flat = gt_actions_phys.detach().reshape(-1, gt_actions_phys.shape[-1]).cpu()
        pr_phys_flat = pred_actions_phys.detach().reshape(-1, pred_actions_phys.shape[-1]).cpu()
        print("========== DEBUG: physical space (denorm) stats ==========")
        for d in range(gt_phys_flat.shape[-1]):
            print(
                f"Phys dim {d}: "
                f"gt[{gt_phys_flat[:, d].min():.4f}, {gt_phys_flat[:, d].max():.4f}] | "
                f"pred[{pr_phys_flat[:, d].min():.4f}, {pr_phys_flat[:, d].max():.4f}]"
            )
        mse = (actions_hat - gt_actions_norm).pow(2).mean().item()
        l1 = (actions_hat - gt_actions_norm).abs().mean().item()
        print(f"Norm-space loss on this batch: L1={l1:.4f}, MSE={mse:.4f}")
        sys.exit(0)

    total_l1 = 0.0
    total_mse = 0.0
    total_frames = 0

    episodes = min(args.num_episodes, ds.meta.total_episodes)
    for ep in range(episodes):
        start = ds.meta.episodes["dataset_from_index"][ep]
        end = ds.meta.episodes["dataset_to_index"][ep]
        print(f"Episode {ep}: frames {start} -> {end}")
        for idx in tqdm.trange(start, end):
            sample = ds[idx]
            # Build a batch with the same keys as training; drop targets if present.
            batch = dict(sample)
            target_action = batch.get("action")
            for key in ["action", "action_is_pad"]:
                batch.pop(key, None)
            # Move tensors to the policy device for normalization/inference.
            for k, v in list(batch.items()):
                if isinstance(v, torch.Tensor):
                    v = v.to(device)
                    # Add batch dimension since we're feeding a single sample.
                    if v.dim() > 0:
                        v = v.unsqueeze(0)
                    batch[k] = v
            actions = policy.select_action(batch)
            # For offline inspection we simply print the first action vector.
            print(f"Frame {idx}: action={actions.squeeze(0).cpu().numpy()}")

            if args.compare_actions and target_action is not None:
                pred = actions.squeeze(0).detach().cpu()
                tgt = target_action.squeeze(0).detach().cpu()
                l1 = torch.nn.functional.l1_loss(pred, tgt, reduction="mean").item()
                mse = torch.nn.functional.mse_loss(pred, tgt, reduction="mean").item()
                total_l1 += l1
                total_mse += mse
                total_frames += 1
        policy.reset()

    if args.compare_actions and total_frames > 0:
        avg_l1 = total_l1 / total_frames
        avg_mse = total_mse / total_frames
        print(f"Average L1: {avg_l1:.6f}, Average MSE: {avg_mse:.6f} over {total_frames} frames.")


if __name__ == "__main__":
    main()
