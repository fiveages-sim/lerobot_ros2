#!/usr/bin/env python3
"""Unified IsaacSim training launcher."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any


def _load_module(module_name: str, file_path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec: {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    isaac_dir = Path(__file__).resolve().parent
    common_dir = isaac_dir / "common"
    if str(common_dir) not in sys.path:
        sys.path.insert(0, str(common_dir))
    train_mod = _load_module("isaac_policy_train", common_dir / "policy_training" / "train.py")
    train_mod.main()


if __name__ == "__main__":
    main()
