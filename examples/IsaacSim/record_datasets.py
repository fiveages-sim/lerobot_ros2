#!/usr/bin/env python3
"""Unified IsaacSim dataset recording entrypoint."""

from __future__ import annotations

from dataclasses import replace
import sys
from pathlib import Path
from typing import Any


def _select_option(*, title: str, options: dict[str, dict[str, Any]], default_key: str) -> str:
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


def _collect_runtime_options(default_enable_keypoint_pcd: bool) -> tuple[int, bool, bool]:
    loops = 1
    try:
        loops = max(1, int(input("How many episodes to record? ")))
    except Exception:
        print("[info] invalid input, defaulting to 1 episode")

    default_pcd = "Y" if default_enable_keypoint_pcd else "N"
    use_pcd_input = input(f"Capture depth+pointcloud at keypoints? [y/{default_pcd}]: ").strip().lower()
    if use_pcd_input == "":
        enable_keypoint_pcd = default_enable_keypoint_pcd
    else:
        enable_keypoint_pcd = use_pcd_input in {"y", "yes"}

    manual_check_input = input("Manually review each episode after return-to-home? [y/N]: ").strip().lower()
    enable_manual_episode_check = manual_check_input in {"y", "yes"}
    return loops, enable_keypoint_pcd, enable_manual_episode_check


def main() -> None:
    # Extend module search paths.
    isaac_dir = Path(__file__).resolve().parent
    common_dir = isaac_dir / "common"
    for path in (isaac_dir, common_dir):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    from robots.registry_loader import load_robot_entries

    discovered = load_robot_entries(isaac_dir)
    registry: dict[str, dict[str, Any]] = {}
    for key, entry in discovered.items():
        record = entry["record"]
        if not record:
            continue
        base_cfg = record["record_cfg"]
        profiles = {}
        for profile in record["profiles"]:
            overrides = profile.get("overrides", {})
            cfg = replace(base_cfg, **overrides) if overrides else base_cfg
            profiles[profile["key"]] = {
                "label": profile.get("label", profile["key"]),
                "record_cfg": cfg,
            }
        registry[key] = {
            "label": entry["label"],
            "robot_cfg": entry["robot_cfg"],
            "tasks": record["tasks"],
            "record_profiles": profiles,
        }
    if not registry:
        raise RuntimeError("No record-capable robot configs found under examples/IsaacSim/robots")

    print("IsaacSim Record Datasets")
    print("=" * 70)
    robot_keys = list(registry.keys())
    default_robot = "dobot_cr5" if "dobot_cr5" in registry else robot_keys[0]
    robot_key = _select_option(title="Select robot", options=registry, default_key=default_robot)
    robot_entry = registry[robot_key]

    task_keys = list(robot_entry["tasks"].keys())
    default_task = "pick_place_flow" if "pick_place_flow" in robot_entry["tasks"] else task_keys[0]
    task_key = _select_option(
        title="Select task",
        options=robot_entry["tasks"],
        default_key=default_task,
    )
    task_entry = robot_entry["tasks"][task_key]
    task_cfg = task_entry["task_cfg"]
    if isinstance(task_cfg, dict) and "base_task_overrides" in task_cfg:
        from pick_place_flow_demo_common import PickPlaceFlowTaskConfig  # pyright: ignore[reportMissingImports]

        task_cfg = PickPlaceFlowTaskConfig(**task_cfg["base_task_overrides"])

    profile_keys = list(robot_entry["record_profiles"].keys())
    default_profile = "default" if "default" in robot_entry["record_profiles"] else profile_keys[0]
    profile_key = _select_option(
        title="Select record profile",
        options=robot_entry["record_profiles"],
        default_key=default_profile,
    )
    profile_entry = robot_entry["record_profiles"][profile_key]

    loops, enable_keypoint_pcd, enable_manual_episode_check = _collect_runtime_options(
        profile_entry["record_cfg"].enable_keypoint_pcd
    )
    print(
        f"[Selection] robot={robot_key}, task={task_key}, profile={profile_key}, "
        f"episodes={loops}, pointcloud={enable_keypoint_pcd}, manual_review={enable_manual_episode_check}"
    )
    task_entry["runner"](
        robot_cfg=robot_entry["robot_cfg"],
        task_cfg=task_cfg,
        record_cfg=profile_entry["record_cfg"],
        loops=loops,
        enable_keypoint_pcd=enable_keypoint_pcd,
        enable_manual_episode_check=enable_manual_episode_check,
    )


if __name__ == "__main__":
    main()
