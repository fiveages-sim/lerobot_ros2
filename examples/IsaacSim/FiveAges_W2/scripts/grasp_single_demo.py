#!/usr/bin/env python3
"""FiveAges W2 handover demo (reuses Agibot G1 flow with W2 config)."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_g1_demo_module():
    g1_demo_path = (
        Path(__file__).resolve().parents[2]
        / "Agibot_G1"
        / "scripts"
        / "grasp_single_demo.py"
    )
    spec = importlib.util.spec_from_file_location("agibot_g1_grasp_demo", g1_demo_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load G1 demo module from: {g1_demo_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    # The imported module will resolve `from grasp_config import GRASP_CFG`
    # to this directory's `grasp_config.py`, so W2-specific config is used.
    demo_module = _load_g1_demo_module()
    demo_module.main()


if __name__ == "__main__":
    main()
