#!/usr/bin/env python3
"""
Simple viewer for BridgeVLA point cloud PKL files using Open3D (plus a Z heatmap).

Assumptions:
- Point cloud PKL stores (H, W, 3) float32 XYZ in meters.
- Files live under:
    <root>/points/<pointcloud_key>/chunk-000/episode_XXXXXX/frame_XXXXXX.pkl

Usage:
    python examples/visualize_pcd.py --root /home/king/lerobot_ros2/dataset/grasp_dataset_1 --episode 0 --frame 0 --stride 1
 
It will render:
- Open3D interactive 3D point cloud (downsampled if requested)
- Z heatmap for quick inspection (Matplotlib)
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d


def load_pointcloud(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    with open(path, "rb") as f:
        arr = pickle.load(f)
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[-1] not in (3, 6):
        raise ValueError(f"Unexpected point cloud shape {arr.shape}, expected (H, W, 3) or (H, W, 6)")
    xyz = arr[..., :3]
    color = None
    if arr.shape[-1] >= 6:
        color = arr[..., 3:6]
        if color.max() > 1.0:
            color = color / 255.0
        color = np.clip(color, 0.0, 1.0)
    return xyz, color


def downsample(xyz: np.ndarray, stride: int, color: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray | None]:
    if stride <= 1:
        xyz_flat = xyz.reshape(-1, 3)
        color_flat = color.reshape(-1, 3) if color is not None else None
        return xyz_flat, color_flat
    xyz_ds = xyz[::stride, ::stride].reshape(-1, 3)
    color_ds = None
    if color is not None:
        color_ds = color[::stride, ::stride].reshape(-1, 3)
    return xyz_ds, color_ds


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize BridgeVLA point cloud PKL.")
    parser.add_argument("--root", type=Path, required=True, help="Dataset root containing points/")
    parser.add_argument("--episode", type=int, default=0, help="Episode index (default: 0)")
    parser.add_argument("--frame", type=int, default=0, help="Frame index (default: 0)")
    parser.add_argument(
        "--pointcloud-key",
        type=str,
        default="observation.points.zed_pcd",
        help="Point cloud key under points/ (default: observation.points.zed_pcd)",
    )
    parser.add_argument("--stride", type=int, default=4, help="Downsample stride for scatter (default: 4)")
    args = parser.parse_args()

    pcd_path = (
        args.root
        / "points"
        / args.pointcloud_key
        / "chunk-000"
        / f"episode_{args.episode:06d}"
        / f"frame_{args.frame:06d}.pkl"
    )

    if not pcd_path.exists():
        raise FileNotFoundError(f"Point cloud not found: {pcd_path}")

    xyz, color = load_pointcloud(pcd_path)
    flat_xyz, flat_color = downsample(xyz, args.stride, color)

    # Filter out invalid depths (<= 0) if present
    valid_mask = flat_xyz[:, 2] > 0
    if valid_mask.any():
        flat_xyz = flat_xyz[valid_mask]
        if flat_color is not None:
            flat_color = flat_color[valid_mask]

    # Open3D visualization
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(flat_xyz)
    if flat_color is not None:
        cloud.colors = o3d.utility.Vector3dVector(flat_color)
    else:
        # Fallback: color by normalized Z
        if flat_xyz.size > 0:
            z = flat_xyz[:, 2]
            z_norm = (z - z.min()) / (z.ptp() + 1e-8)
            colors = plt.cm.viridis(z_norm)[:, :3]
            cloud.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([cloud], window_name=f"ep={args.episode} frame={args.frame}")

    # Matplotlib Z heatmap
    fig = plt.figure(figsize=(6, 5))
    ax2 = fig.add_subplot(1, 1, 1)
    z_map = xyz[:, :, 2]
    im = ax2.imshow(z_map, cmap="viridis")
    ax2.set_title(f"Z heatmap ep={args.episode} frame={args.frame}")
    fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04, label="Z (m)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
