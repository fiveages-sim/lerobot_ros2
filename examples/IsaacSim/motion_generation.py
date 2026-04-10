#!/usr/bin/env python3
"""Unified IsaacSim motion-generation launcher (delegates to ``robot_action_composer``)."""

from __future__ import annotations

from pathlib import Path

from robot_action_composer.cli.motion_main import run_motion_generation


def main() -> None:
    run_motion_generation(isaac_dir=Path(__file__).resolve().parent)


if __name__ == "__main__":
    main()
