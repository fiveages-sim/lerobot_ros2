#!/usr/bin/env python3
"""Unified IsaacSim dataset recording entrypoint (delegates to ``robot_action_composer``)."""

from __future__ import annotations

import argparse
from pathlib import Path

from robot_action_composer.cli.record_main import run_record_datasets


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="IsaacSim dataset recording")
    parser.add_argument(
        "--workspace",
        type=str,
        default=None,
        help="Workspace root: legacy dir containing robots/, or a fa-project-usd project dir (env/)",
    )
    parser.add_argument(
        "--robots-root",
        type=str,
        default=None,
        help="Shared robot profile root (default: <workspace>/robots or the scene robot.yaml robots_root)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    workspace = (
        Path(args.workspace).expanduser().resolve()
        if args.workspace
        else Path(__file__).resolve().parent
    )
    run_record_datasets(isaac_dir=workspace, robots_root=args.robots_root)


if __name__ == "__main__":
    main()
