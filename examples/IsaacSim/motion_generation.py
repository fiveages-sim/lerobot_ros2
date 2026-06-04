#!/usr/bin/env python3
"""Unified IsaacSim motion-generation launcher (delegates to ``robot_action_composer``)."""

from __future__ import annotations

import argparse
from pathlib import Path

from robot_action_composer.cli.motion_main import run_motion_generation


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified IsaacSim motion generation launcher")
    parser.add_argument("--robot", type=str, default=None, help="Robot key (e.g. FiveAges_W2)")
    parser.add_argument("--task-key", type=str, default=None, help="Task key (e.g. poc_bimanual_2box_rev)")
    parser.add_argument("--scene", type=str, default=None, help="Scene preset name (defaults to task default_scene)")
    parser.add_argument(
        "--object-resolution-json",
        dest="object_resolution_json",
        type=str,
        default=None,
        help="Optional override path for object-resolution JSON (sim: write; real: read)",
    )
    parser.add_argument("--no-reset", action="store_true", help="Skip environment reset (CLI mode only)")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    json_path = Path(args.object_resolution_json).expanduser() if args.object_resolution_json else None
    run_motion_generation(
        isaac_dir=Path(__file__).resolve().parent,
        robot_key=args.robot,
        task_key=args.task_key,
        scene=args.scene,
        object_resolution_json=json_path,
        no_reset=args.no_reset,
    )


if __name__ == "__main__":
    main()
