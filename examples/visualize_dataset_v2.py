#!/usr/bin/env python3
"""
LeRobot dataset visualizer with Rerun.

This tool loads a local LeRobot dataset (recorded via demo_grasp_record, demo_record, etc.),
prints metadata, and streams a selected episode to the Rerun viewer. Observation/action
scalars are logged per-dimension, and camera frames are displayed as RGB images.

Usage:
    python visualize_dataset_v2.py /path/to/dataset --episode-index 0 --batch-size 8
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import rerun as rr
import torch
import torch.utils.data
import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]


from lerobot.datasets.lerobot_dataset import LeRobotDataset


class EpisodeSampler(torch.utils.data.Sampler[int]):
    """Sampler that iterates over a single episode's frame indices."""

    def __init__(self, dataset: LeRobotDataset, episode_index: int):
        from_idx = dataset.meta.episodes["dataset_from_index"][episode_index]
        to_idx = dataset.meta.episodes["dataset_to_index"][episode_index]
        self.frame_ids = range(from_idx, to_idx)

    def __iter__(self):
        return iter(self.frame_ids)

    def __len__(self) -> int:
        return len(self.frame_ids)


def to_hwc_uint8(image_tensor: torch.Tensor | np.ndarray) -> np.ndarray:
    """Convert CHW float tensor or uint8 array to HWC uint8 numpy."""
    if isinstance(image_tensor, torch.Tensor):
        if image_tensor.dtype == torch.float32:
            img = (image_tensor.clamp(0.0, 1.0) * 255).to(torch.uint8)
        else:
            img = image_tensor.to(torch.uint8)
        if img.ndim == 4:
            img = img.squeeze(0)
        img = img.permute(1, 2, 0).cpu().numpy()
    else:
        img = image_tensor
        if img.dtype != np.uint8:
            img = np.clip(img, 0.0, 1.0)
            img = (img * 255).astype(np.uint8)
        if img.ndim == 3 and img.shape[0] in (1, 3):
            img = np.moveaxis(img, 0, -1)
    return img


def print_dataset_info(dataset: LeRobotDataset) -> None:
    print("=" * 70)
    print("Dataset summary")
    print("=" * 70)
    print(f"Repo ID        : {dataset.repo_id}")
    print(f"Local root     : {dataset.root}")
    print(f"Robot type     : {dataset.meta.robot_type}")
    print(f"FPS            : {dataset.meta.fps}")
    print(f"Episodes       : {dataset.meta.total_episodes}")
    print(f"Frames         : {dataset.meta.total_frames}")
    if dataset.meta.total_episodes:
        avg = dataset.meta.total_frames / dataset.meta.total_episodes
        print(f"Avg frames/ep  : {avg:.1f}")
    print("\nTasks:")
    for idx, task in enumerate(dataset.meta.tasks):
        print(f"  {idx}: {task}")
    print("\nFeatures:")
    for key, info in dataset.features.items():
        print(f"  {key}: {info}")
    print("\nCamera keys:")
    for cam in dataset.meta.camera_keys:
        print(f"  {cam}")
    print("=" * 70)


def log_scalar_vector(prefix: str, values: torch.Tensor | np.ndarray, names: list[str] | None):
    arr = values.cpu().numpy() if isinstance(values, torch.Tensor) else values
    if arr.ndim == 0:
        rr.log(prefix, rr.Scalar(float(arr)))
        return
    if names is None:
        names = [f"{prefix}_{i}" for i in range(len(arr))]
    for idx, name in enumerate(names):
        rr.log(f"{prefix}/{name}", rr.Scalar(float(arr[idx])))


def visualize_episode(
    dataset: LeRobotDataset,
    episode_index: int,
    batch_size: int,
    tolerance_s: float,
    log_images: bool,
) -> None:
    dataset.tolerance_s = tolerance_s

    sampler = EpisodeSampler(dataset, episode_index)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=0,
    )

    obs_feature = dataset.features.get("observation.state", {})
    obs_names = obs_feature.get("names")
    act_feature = dataset.features.get("action", {})
    act_names = act_feature.get("names")

    for batch in tqdm.tqdm(dataloader, desc="Processing frames"):
        batch_size_actual = batch["index"].shape[0]
        for i in range(batch_size_actual):
            timestamp = float(batch["timestamp"][i].item())
            rr.set_time_seconds("timeline", timestamp)

            if "observation.state" in batch:
                log_scalar_vector(
                    "observation/state",
                    batch["observation.state"][i],
                    obs_names,
                )

            if "action" in batch:
                log_scalar_vector(
                    "action",
                    batch["action"][i],
                    act_names,
                )

            if log_images:
                for cam_key in dataset.meta.camera_keys:
                    if cam_key in batch:
                        image = batch[cam_key][i]
                        img = to_hwc_uint8(image)
                        rr.log(f"camera/{cam_key}", rr.Image(img))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize LeRobot dataset with Rerun.")
    parser.add_argument(
        "dataset_path",
        nargs="?",
        default=None,
        help="Path to the dataset directory (default: latest under ./dataset)",
    )
    parser.add_argument("--episode-index", type=int, default=0, help="Episode to visualize")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for DataLoader",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.04,
        help="Tolerance (seconds) for video timestamp alignment",
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Skip logging camera images",
    )
    parser.add_argument(
        "--info-only",
        action="store_true",
        help="Only print dataset information",
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Run rerun viewer in server mode (no auto-spawn window)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dataset_path is None:
        dataset_root = REPO_ROOT / "dataset"
        candidates = sorted(dataset_root.glob("grasp_dataset_*"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not candidates:
            raise FileNotFoundError(f"No datasets found under {dataset_root}")
        dataset_path = candidates[0]
        print(f"[info] No dataset path provided, using latest: {dataset_path}")
    else:
        dataset_path = Path(args.dataset_path).expanduser().resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")

    dataset = LeRobotDataset(repo_id=dataset_path.name, root=str(dataset_path))
    print_dataset_info(dataset)

    if args.info_only:
        return

    if not (0 <= args.episode_index < dataset.meta.total_episodes):
        raise ValueError(
            f"Episode index {args.episode_index} out of range "
            f"(max {dataset.meta.total_episodes - 1})"
        )

    viewer_name = f"{dataset.repo_id}/episode_{args.episode_index}"
    rr.init(viewer_name, spawn=not args.serve)
    if args.serve:
        rr.serve()

    visualize_episode(
        dataset=dataset,
        episode_index=args.episode_index,
        batch_size=args.batch_size,
        tolerance_s=args.tolerance,
        log_images=not args.no_video,
    )
    print("✅ Visualization finished.")


if __name__ == "__main__":
    main()
