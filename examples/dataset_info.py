#!/usr/bin/env python3
"""
ROS2 Dataset Information Script

This script displays information about ROS2 datasets recorded in the examples folder.

Usage:
    python dataset_info.py [dataset_path]

Examples:
    # Show info for the latest dataset
    python dataset_info.py
    
    # Show info for specific dataset
    python dataset_info.py ./ros2_dataset_1761133403
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch



from lerobot.datasets.lerobot_dataset import LeRobotDataset


def print_dataset_info(dataset: LeRobotDataset):
    """Print comprehensive information about the dataset."""
    print("=" * 80)
    print("📊 ROS2 Dataset Information")
    print("=" * 80)
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
    
    print(f"\n📈 Episode Details:")
    for i in range(dataset.meta.total_episodes):
        from_idx = dataset.meta.episodes["dataset_from_index"][i]
        to_idx = dataset.meta.episodes["dataset_to_index"][i]
        episode_length = to_idx - from_idx
        duration = episode_length / dataset.meta.fps
        print(f"  Episode {i}: {episode_length} frames ({duration:.1f}s)")
    
    print("=" * 80)


def analyze_sample_data(dataset: LeRobotDataset, episode_index: int = 0, num_frames: int = 5):
    """Analyze sample data from the dataset."""
    print(f"\n🔍 Sample Data Analysis (Episode {episode_index})")
    print("-" * 50)
    
    from_idx = dataset.meta.episodes["dataset_from_index"][episode_index]
    to_idx = dataset.meta.episodes["dataset_to_index"][episode_index]
    
    # Sample a few frames
    sample_indices = np.linspace(from_idx, to_idx - 1, min(num_frames, to_idx - from_idx), dtype=int)
    
    for i, frame_idx in enumerate(sample_indices):
        print(f"\nFrame {i + 1} (Dataset Index: {frame_idx}):")
        
        try:
            frame_data = dataset[frame_idx]
            
            # Print observation state
            if "observation.state" in frame_data:
                obs_state = frame_data["observation.state"]
                print(f"  Observation State Shape: {obs_state.shape}")
                print(f"  Observation State Range: [{obs_state.min():.3f}, {obs_state.max():.3f}]")
                print(f"  Observation State Mean: {obs_state.mean():.3f}")
            
            # Print action
            if "action" in frame_data:
                action = frame_data["action"]
                print(f"  Action Shape: {action.shape}")
                print(f"  Action Range: [{action.min():.3f}, {action.max():.3f}]")
                print(f"  Action Mean: {action.mean():.3f}")
            
            # Print camera data
            for cam_key in dataset.meta.camera_keys:
                image_key = f"observation.images.{cam_key}"
                if image_key in frame_data:
                    image = frame_data[image_key]
                    print(f"  Camera {cam_key} Shape: {image.shape}")
                    print(f"  Camera {cam_key} Range: [{image.min():.3f}, {image.max():.3f}]")
            
            # Print timestamp
            if "timestamp" in frame_data:
                print(f"  Timestamp: {frame_data['timestamp']:.3f}")
                
        except Exception as e:
            print(f"  Error loading frame: {e}")


def find_latest_dataset(examples_dir: Path) -> Optional[Path]:
    """Find the latest ROS2 dataset in the examples directory."""
    dataset_dirs = list(examples_dir.glob("ros2_dataset_*"))
    if not dataset_dirs:
        return None
    
    # Sort by modification time, newest first
    dataset_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return dataset_dirs[0]


def main():
    parser = argparse.ArgumentParser(description="Display ROS2 dataset information")
    parser.add_argument(
        "dataset_path",
        nargs="?",
        help="Path to the dataset directory (default: latest dataset)"
    )
    parser.add_argument(
        "--episode-index",
        type=int,
        default=0,
        help="Episode index to analyze (default: 0)"
    )
    parser.add_argument(
        "--sample-frames",
        type=int,
        default=5,
        help="Number of sample frames to analyze (default: 5)"
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Also analyze sample data from the dataset"
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
        
        # Analyze sample data if requested
        if args.analyze:
            if args.episode_index >= dataset.meta.total_episodes:
                print(f"❌ Episode index {args.episode_index} is out of range (max: {dataset.meta.total_episodes - 1})")
                return
            
            analyze_sample_data(dataset, args.episode_index, args.sample_frames)

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
