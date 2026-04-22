#!/usr/bin/env python3
"""Unified IsaacSim dataset recording entrypoint (delegates to ``robot_action_composer``)."""

from __future__ import annotations

from pathlib import Path

from robot_action_composer.cli.record_main import run_record_datasets


def main() -> None:
    run_record_datasets(isaac_dir=Path(__file__).resolve().parent)


if __name__ == "__main__":
    main()
