#!/usr/bin/env python3
"""Shared preset-selection helpers for IsaacSim demos."""

from __future__ import annotations

import argparse
from dataclasses import fields, replace
from typing import Any


def _parse_scene_args(
    *,
    scene_names: list[str],
    cli_description: str,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=cli_description)
    parser.add_argument(
        "--scene",
        choices=tuple(scene_names),
        default=None,
        help="Select scene/object preset (omit to choose interactively)",
    )
    return parser.parse_args()


def _select_scene_interactively(scene_names: list[str], *, default_scene: str) -> str:
    print("[Select Scene]")
    for idx, scene in enumerate(scene_names, start=1):
        print(f"  {idx}. {scene}")
    prompt = f"Choose scene [1-{len(scene_names)}] (default: {default_scene}): "
    raw = input(prompt).strip()
    if raw == "":
        return default_scene
    if raw.isdigit():
        idx = int(raw) - 1
        if 0 <= idx < len(scene_names):
            return scene_names[idx]
    if raw in scene_names:
        return raw
    raise ValueError(f"Invalid scene selection: {raw}")


def resolve_dataclass_cfg_from_presets(
    *,
    base_cfg: Any,
    scene_presets: dict[str, dict[str, object]],
    cli_description: str,
    default_scene: str = "grab_medicine",
) -> tuple[str, Any]:
    """Resolve dataclass config from CLI/interactive scene choice and preset map."""
    scene_names = list(scene_presets.keys())
    if default_scene not in scene_names:
        raise ValueError(f"default_scene '{default_scene}' is not in scene presets")

    args = _parse_scene_args(scene_names=scene_names, cli_description=cli_description)
    scene = args.scene or _select_scene_interactively(scene_names, default_scene=default_scene)
    preset = scene_presets[scene]

    valid_fields = {f.name for f in fields(type(base_cfg))}
    unknown_keys = [k for k in preset if k not in valid_fields]
    if unknown_keys:
        raise ValueError(f"Unknown preset keys for {type(base_cfg).__name__}: {unknown_keys}")
    cfg = replace(base_cfg, **preset)
    return scene, cfg
