"""Load IsaacSim per-robot task configs from ``.py`` or ``.yaml`` / ``.yml``."""

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


def _normalize_numeric_lists(obj: Any) -> Any:
    """YAML expresses tuples as lists; dataclasses often expect ``tuple`` for poses/orientations.

    Rule: a non-empty list made only of int/float becomes ``tuple``. Nested dict/list structures
    are traversed; lists of dicts (e.g. ``task_queue``, ``profiles``) are preserved as lists.
    """
    if isinstance(obj, dict):
        return {k: _normalize_numeric_lists(v) for k, v in obj.items()}
    if isinstance(obj, list):
        if obj and all(isinstance(x, (int, float)) for x in obj):
            return tuple(obj)
        return [_normalize_numeric_lists(x) for x in obj]
    return obj


def load_task_dict_from_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError as err:
        raise ImportError(
            "Loading task config from YAML requires PyYAML. Install with: pip install pyyaml"
        ) from err
    text = path.read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    if data is None:
        raise ValueError(f"YAML task config is empty: {path}")
    if not isinstance(data, dict):
        raise TypeError(f"YAML task config must be a mapping at root, got {type(data).__name__}: {path}")
    return _normalize_numeric_lists(data)  # type: ignore[return-value]


def discover_task_configs(task_cfg_dir: Path, *, robot_dir_name: str) -> dict[str, dict[str, Any]]:
    """Return ``task_key -> TASK_CONFIG`` for one robot's ``task_configs`` directory.

    For each basename (stem), **either** a ``.py`` **or** a ``.yaml``/``.yml`` may exist — not both.
    """
    stems: set[str] = set()
    for p in task_cfg_dir.glob("*.py"):
        stems.add(p.stem)
    for p in task_cfg_dir.glob("*.yaml"):
        stems.add(p.stem)
    for p in task_cfg_dir.glob("*.yml"):
        stems.add(p.stem)

    tasks: dict[str, dict[str, Any]] = {}
    for stem in sorted(stems):
        py_path = task_cfg_dir / f"{stem}.py"
        yaml_path = task_cfg_dir / f"{stem}.yaml"
        yml_path = task_cfg_dir / f"{stem}.yml"
        has_py = py_path.is_file()
        has_yaml = yaml_path.is_file() or yml_path.is_file()
        if has_py and has_yaml:
            raise ValueError(
                f"Robot {robot_dir_name!r}: task {stem!r} has both "
                f"{py_path.name} and a .yaml/.yml — keep only one."
            )
        if has_yaml:
            path = yaml_path if yaml_path.is_file() else yml_path
            raw = load_task_dict_from_yaml(path)
        elif has_py:
            task_mod = _load_module(f"{robot_dir_name}_{stem}_task_cfg", py_path)
            raw = getattr(task_mod, "TASK_CONFIG", None)
            if raw is None:
                raw = getattr(task_mod, "FLOW_CONFIG", None)
        else:
            continue

        if not isinstance(raw, dict):
            continue
        task_key = raw.get("task_key")
        if not isinstance(task_key, str):
            raise ValueError(f"Task config {stem!r} must define string task_key: {task_cfg_dir}")
        tasks[task_key] = dict(raw)
    return tasks
