#!/usr/bin/env python3
"""Common interactive helpers for IsaacSim dataset record launchers."""

from __future__ import annotations

from typing import Any


def prompt_positive_int(
    message: str,
    *,
    default: int = 1,
    min_value: int = 1,
) -> int:
    """Parse a positive integer from stdin; invalid input falls back to ``default``.

    Used for "how many episodes" / "how many demo runs" style prompts so record and
    motion-generation launchers share one implementation.
    """
    if min_value < 1:
        raise ValueError("min_value must be >= 1")
    default = max(min_value, int(default))
    try:
        raw = input(message).strip()
        if raw == "":
            return default
        value = int(raw)
        return max(min_value, value)
    except Exception:
        print(f"[info] invalid input, using default {default}")
        return default


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
    loops = prompt_positive_int("How many episodes to record? ", default=1, min_value=1)

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
