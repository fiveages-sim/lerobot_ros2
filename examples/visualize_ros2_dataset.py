#!/usr/bin/env python3
"""
ROS2 Dataset Visualization Script

This script visualizes ROS2 datasets recorded in the examples folder using LeRobot's visualization tools.
It provides both basic dataset information and interactive visualization using rerun.

Usage:
    python visualize_ros2_dataset.py [dataset_path] [--episode-index EPISODE] [--mode MODE]

Examples:
    # Visualize the latest dataset
    python visualize_ros2_dataset.py
    
    # Visualize specific dataset
    python visualize_ros2_dataset.py ./ros2_dataset_1761133403
    
    # Visualize specific episode
    python visualize_ros2_dataset.py ./ros2_dataset_1761133403 --episode-index 0
    
    # Save visualization to file
    python visualize_ros2_dataset.py ./ros2_dataset_1761133403 --save --output-dir ./viz_output
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import rerun as rr
import torch
import torch.utils.data
import tqdm

# Add lerobot to path
sys.path.insert(0, '/home/fiveages/PycharmProjects/lerobot/src')

from lerobot.datasets.lerobot_dataset import LeRobotDataset


class EpisodeSampler(torch.utils.data.Sampler):
    """Sampler for selecting frames from a specific episode."""
    def __init__(self, dataset: LeRobotDataset, episode_index: int):
        from_idx = dataset.meta.episodes["dataset_from_index"][episode_index]
        to_idx = dataset.meta.episodes["dataset_to_index"][episode_index]
        self.frame_ids = range(from_idx, to_idx)

    def __iter__(self):
        return iter(self.frame_ids)

    def __len__(self) -> int:
        return len(self.frame_ids)


def to_hwc_uint8_numpy(chw_float32_torch: torch.Tensor) -> np.ndarray:
    """Convert CHW float32 tensor to HWC uint8 numpy array."""
    assert chw_float32_torch.dtype == torch.float32
    assert chw_float32_torch.ndim == 3
    c, h, w = chw_float32_torch.shape
    assert c < h and c < w, f"expect channel first images, but instead {chw_float32_torch.shape}"
    hwc_uint8_numpy = (chw_float32_torch * 255).type(torch.uint8).permute(1, 2, 0).numpy()
    return hwc_uint8_numpy


def print_dataset_info(dataset: LeRobotDataset):
    """Print basic information about the dataset."""
    print("=" * 60)
    print("📊 Dataset Information")
    print("=" * 60)
    print(f"Repository ID: {dataset.repo_id}")
    print(f"Robot Type: {dataset.meta.robot_type}")
    print(f"FPS: {dataset.meta.fps}")
    print(f"Total Episodes: {dataset.meta.total_episodes}")
    print(f"Total Frames: {dataset.meta.total_frames}")
    print(f"Average Frames per Episode: {dataset.meta.total_frames / dataset.meta.total_episodes:.1f}")
    
    print(f"\n📋 Tasks:")
    for i, task in enumerate(dataset.meta.tasks):
        print(f"  {i}: {task}")
    
    print(f"\n🔧 Features:")
    for key, feature_info in dataset.features.items():
        print(f"  {key}: {feature_info}")
    
    print(f"\n📹 Camera Keys:")
    for cam_key in dataset.meta.camera_keys:
        print(f"  {cam_key}")
    
    print("=" * 60)


def visualize_dataset(
    dataset: LeRobotDataset,
    episode_index: int = 0,
    batch_size: int = 32,
    num_workers: int = 0,
    mode: str = "local",
    web_port: int = 9090,
    ws_port: int = 9087,
    save: bool = False,
    output_dir: Optional[Path] = None,
) -> Optional[Path]:
    """Visualize a specific episode of the dataset using rerun."""
    
    if save:
        assert output_dir is not None, "Set an output directory with --output-dir"
        output_dir.mkdir(parents=True, exist_ok=True)

    repo_id = dataset.repo_id
    episode_length = dataset.meta.episodes["dataset_to_index"][episode_index] - dataset.meta.episodes["dataset_from_index"][episode_index]
    
    print(f"\n🎬 Visualizing Episode {episode_index}")
    print(f"Episode Length: {episode_length} frames")
    print(f"Duration: {episode_length / dataset.meta.fps:.1f} seconds")

    # Create episode sampler
    episode_sampler = EpisodeSampler(dataset, episode_index)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        batch_size=batch_size,
        sampler=episode_sampler,
    )

    print("🚀 Starting Rerun Visualization...")

    if mode not in ["local", "distant"]:
        raise ValueError(f"Invalid mode: {mode}")

    spawn_local_viewer = mode == "local" and not save
    rr.init(f"{repo_id}/episode_{episode_index}", spawn=spawn_local_viewer)

    if mode == "distant":
        rr.serve(open_browser=False, web_port=web_port, ws_port=ws_port)

    print("📊 Logging data to Rerun...")

    for batch in tqdm.tqdm(dataloader, total=len(dataloader), desc="Processing frames"):
        # Iterate over the batch
        for i in range(len(batch["index"])):
            frame_index = batch["index"][i].item()
            timestamp = batch["timestamp"][i].item()
            
            # Log timestamp
            rr.log("timeline", rr.TimeSeriesScalar(timestamp))
            
            # Log observation state if available
            if "observation.state" in batch:
                obs_state = batch["observation.state"][i]
                rr.log("observation/state", rr.Scalar(obs_state.numpy()))
            
            # Log action if available
            if "action" in batch:
                action = batch["action"][i]
                rr.log("action", rr.Scalar(action.numpy()))
            
            # Log camera images
            for cam_key in dataset.meta.camera_keys:
                image_key = f"observation.images.{cam_key}"
                if image_key in batch:
                    image = batch[image_key][i]
                    if image.dim() == 3:  # Single image
                        hwc_image = to_hwc_uint8_numpy(image)
                        rr.log(f"camera/{cam_key}", rr.Image(hwc_image))
                    elif image.dim() == 4:  # Batch of images
                        for j in range(image.shape[0]):
                            hwc_image = to_hwc_uint8_numpy(image[j])
                            rr.log(f"camera/{cam_key}/frame_{j}", rr.Image(hwc_image))

    if save:
        output_file = output_dir / f"{repo_id}_episode_{episode_index}.rrd"
        rr.save(output_file)
        print(f"💾 Visualization saved to: {output_file}")
        return output_file
    
    print("✅ Visualization complete! Check the rerun viewer.")
    return None


def find_latest_dataset(examples_dir: Path) -> Optional[Path]:
    """Find the latest ROS2 dataset in the examples directory."""
    dataset_dirs = list(examples_dir.glob("ros2_dataset_*"))
    if not dataset_dirs:
        return None
    
    # Sort by modification time, newest first
    dataset_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return dataset_dirs[0]


def main():
    parser = argparse.ArgumentParser(description="Visualize ROS2 datasets")
    parser.add_argument(
        "dataset_path",
        nargs="?",
        help="Path to the dataset directory (default: latest dataset)"
    )
    parser.add_argument(
        "--episode-index",
        type=int,
        default=0,
        help="Episode index to visualize (default: 0)"
    )
    parser.add_argument(
        "--mode",
        choices=["local", "distant"],
        default="local",
        help="Visualization mode (default: local)"
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save visualization to file instead of opening viewer"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./viz_output"),
        help="Output directory for saved visualizations (default: ./viz_output)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for data loading (default: 32)"
    )
    parser.add_argument(
        "--info-only",
        action="store_true",
        help="Only print dataset information, don't visualize"
    )

    args = parser.parse_args()

    # Determine dataset path
    examples_dir = Path(__file__).parent
    if args.dataset_path:
        dataset_path = Path(args.dataset_path)
        if not dataset_path.is_absolute():
            dataset_path = examples_dir / dataset_path
    else:
        dataset_path = find_latest_dataset(examples_dir)
        if dataset_path is None:
            print("❌ No ROS2 datasets found in examples directory")
            print("Available datasets:")
            for item in examples_dir.iterdir():
                if item.is_dir() and item.name.startswith("ros2_dataset_"):
                    print(f"  {item.name}")
            return
        print(f"🔍 Using latest dataset: {dataset_path.name}")

    if not dataset_path.exists():
        print(f"❌ Dataset path does not exist: {dataset_path}")
        return

    try:
        # Load dataset
        print(f"📂 Loading dataset from: {dataset_path}")
        dataset = LeRobotDataset(str(dataset_path))
        
        # Print dataset information
        print_dataset_info(dataset)
        
        if args.info_only:
            return

        # Check episode index
        if args.episode_index >= dataset.meta.total_episodes:
            print(f"❌ Episode index {args.episode_index} is out of range (max: {dataset.meta.total_episodes - 1})")
            return

        # Visualize dataset
        visualize_dataset(
            dataset=dataset,
            episode_index=args.episode_index,
            batch_size=args.batch_size,
            mode=args.mode,
            save=args.save,
            output_dir=args.output_dir,
        )

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
