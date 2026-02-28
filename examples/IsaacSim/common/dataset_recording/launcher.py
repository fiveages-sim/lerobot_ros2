#!/usr/bin/env python3
"""Common interactive helpers for IsaacSim dataset record launchers."""

from __future__ import annotations

from typing import Any


def select_option(*, title: str, options: dict[str, dict[str, Any]], default_key: str) -> str:
    keys = list(options.keys())
    print(f"\n{title}")
    for idx, key in enumerate(keys, start=1):
        label = options[key].get("label", key)
        suffix = " (default)" if key == default_key else ""
        print(f"  {idx}. {label} [{key}]{suffix}")
    raw = input("Select option (press Enter for default): ").strip()
    if raw == "":
        return default_key
    if raw.isdigit():
        idx = int(raw) - 1
        if 0 <= idx < len(keys):
            return keys[idx]
    if raw in options:
        return raw
    print(f"[info] Invalid option '{raw}', using default '{default_key}'.")
    return default_key


def collect_runtime_options(
    *,
    pointcloud_supported: bool,
    default_enable_keypoint_pcd: bool = False,
) -> tuple[int, bool, bool]:
    loops = 1
    try:
        loops = max(1, int(input("How many episodes to record? ")))
    except Exception:
        print("[info] invalid input, defaulting to 1 episode")

    if pointcloud_supported:
        default_pcd = "Y" if default_enable_keypoint_pcd else "N"
        use_pcd_input = input(f"Capture depth+pointcloud at keypoints? [y/{default_pcd}]: ").strip().lower()
        if use_pcd_input == "":
            enable_keypoint_pcd = default_enable_keypoint_pcd
        else:
            enable_keypoint_pcd = use_pcd_input in {"y", "yes"}
    else:
        enable_keypoint_pcd = False
        print("[info] Pointcloud option hidden: no depth camera configured for this robot.")

    manual_check_input = input("Manually review each episode after return-to-home? [y/N]: ").strip().lower()
    enable_manual_episode_check = manual_check_input in {"y", "yes"}
    return loops, enable_keypoint_pcd, enable_manual_episode_check
