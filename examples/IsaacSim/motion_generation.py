#!/usr/bin/env python3
"""Unified IsaacSim motion-generation launcher."""

from __future__ import annotations

from dataclasses import fields, replace
import importlib.util
import sys
from pathlib import Path
from typing import Any


def _load_module(module_name: str, file_path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec: {file_path}")
    module = importlib.util.module_from_spec(spec)
    # Python 3.12 dataclass internals may consult sys.modules during class creation.
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _discover_task_registry(isaac_dir: Path) -> dict[str, dict[str, Any]]:
    from task_config_io import discover_task_configs  # pyright: ignore[reportMissingImports]

    robots_root = isaac_dir / "robots"
    if not robots_root.is_dir():
        return {}

    registry: dict[str, dict[str, Any]] = {}
    for robot_dir in sorted(p for p in robots_root.iterdir() if p.is_dir() and not p.name.startswith("__")):
        robot_cfg_file = robot_dir / "robot_config.py"
        task_cfg_dir = robot_dir / "task_configs"
        if not robot_cfg_file.is_file() or not task_cfg_dir.is_dir():
            continue

        robot_mod = _load_module(f"{robot_dir.name}_robot_cfg", robot_cfg_file)
        robot_key = getattr(robot_mod, "ROBOT_KEY", robot_dir.name.lower())
        robot_label = getattr(robot_mod, "ROBOT_LABEL", robot_dir.name)
        robot_cfg = getattr(robot_mod, "ROBOT_CFG")

        tasks = discover_task_configs(task_cfg_dir, robot_dir_name=robot_dir.name)

        if tasks:
            registry[robot_key] = {
                "label": robot_label,
                "robot_cfg": robot_cfg,
                "tasks": tasks,
            }

    return registry


def _select_option(*, title: str, options: list[str], default_value: str) -> str:
    print(f"\n{title}")
    for idx, name in enumerate(options, start=1):
        suffix = " (default)" if name == default_value else ""
        print(f"  {idx}. {name}{suffix}")
    raw = input("Select option (press Enter for default): ").strip()
    if raw == "":
        return default_value
    if raw.isdigit():
        index = int(raw) - 1
        if 0 <= index < len(options):
            return options[index]
    if raw in options:
        return raw
    print(f"[info] Invalid option '{raw}', using default '{default_value}'.")
    return default_value


def _apply_preset(base_cfg: Any, preset: dict[str, object]) -> Any:
    valid_fields = {f.name for f in fields(type(base_cfg))}
    unknown_keys = [k for k in preset if k not in valid_fields]
    if unknown_keys:
        raise ValueError(f"Unknown preset keys for {type(base_cfg).__name__}: {unknown_keys}")
    return replace(base_cfg, **preset)


def main() -> None:
    isaac_dir = Path(__file__).resolve().parent
    common_dir = isaac_dir / "common"
    for path in (isaac_dir, common_dir):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    from motion_generation.handover import (  # pyright: ignore[reportMissingImports]
        flatten_handover_task_overrides,
        format_handover_task_cfg_summary,
    )
    from motion_generation.pick_place import (  # pyright: ignore[reportMissingImports]
        PickPlaceFlowTaskConfig,
        flatten_pick_place_task_overrides,
        format_pick_place_cfg_summary,
        run_pick_place_demo,
    )
    from motion_generation.drawer import (  # pyright: ignore[reportMissingImports]
        DrawerPickPlaceTaskConfig,
        flatten_drawer_pick_place_task_overrides,
        format_drawer_task_cfg_summary,
        run_drawer_demo,
    )
    from motion_generation.handover import HandoverTaskConfig  # pyright: ignore[reportMissingImports]
    from motion_generation.bimanual_carry import (  # pyright: ignore[reportMissingImports]
        BimanualCarryTaskConfig,
        flatten_bimanual_carry_task_overrides,
        format_bimanual_carry_task_cfg_summary,
    )
    from dataset_recording.launcher import (  # pyright: ignore[reportMissingImports]
        prompt_positive_int,
        select_option as select_labeled_option,
    )

    registry = _discover_task_registry(isaac_dir)
    if not registry:
        raise RuntimeError("No motion-generation capable robot configs found under examples/IsaacSim/robots")

    print("IsaacSim Run Motion Generation")
    print("=" * 70)
    robot_keys = list(registry.keys())
    robot_key = _select_option(title="Select robot", options=robot_keys, default_value="dobot_cr5")
    robot_entry = registry[robot_key]

    tasks_map = robot_entry["tasks"]
    task_options = {
        key: {"label": str(meta.get("label", key))}
        for key, meta in tasks_map.items()
    }
    default_task_key = "pick_place" if "pick_place" in task_options else next(iter(task_options))
    task_key = select_labeled_option(
        title="Select task",
        options=task_options,
        default_key=default_task_key,
    )
    task_entry = robot_entry["tasks"][task_key]

    scene_presets: dict[str, dict[str, object]] = task_entry["scene_presets"]
    scene_names = list(scene_presets.keys())
    default_scene = task_entry["default_scene"]
    scene = _select_option(title="Select config", options=scene_names, default_value=default_scene)

    num_runs = prompt_positive_int(
        "How many motion runs? (Enter = 1): ",
        default=1,
        min_value=1,
    )

    # 多段运行时每段之间需要重置仿真/随机化物体；仅单段时可选择不重置。
    if num_runs > 1:
        reset_env = True
        print(
            "[info] Multiple motion runs: each run will reset the environment "
            "& randomize the object (same as record episodes)."
        )
    else:
        reset_env = _select_option(
            title="Reset environment & randomize object?",
            options=["yes", "no"],
            default_value="yes",
        ) == "yes"

    use_stamped = task_entry.get("use_stamped", True)

    if task_entry["kind"] == "pick_place":
        base_task_cfg = PickPlaceFlowTaskConfig(
            **flatten_pick_place_task_overrides(task_entry["base_task_overrides"])
        )
        task_cfg = _apply_preset(
            base_task_cfg,
            flatten_pick_place_task_overrides(scene_presets.get(scene, {})),
        )
        print(format_pick_place_cfg_summary(scene, task_cfg))
        task_queue = task_entry.get("task_queue")
        if task_queue:
            import task_runtime.skills  # noqa: F401 - register built-in skills

            from task_runtime.runner import run_single_arm_task_queue  # pyright: ignore[reportMissingImports]

            for run_idx in range(num_runs):
                print(f"\n{'=' * 70}\nMotion run {run_idx + 1}/{num_runs} (task queue)\n{'=' * 70}")
                run_single_arm_task_queue(
                    robot_cfg=robot_entry["robot_cfg"],
                    task_cfg=task_cfg,
                    robot_id=task_entry["robot_id"],
                    blocks=task_queue,
                    reset_env=reset_env,
                    use_stamped=use_stamped,
                )
            return
        for run_idx in range(num_runs):
            print(f"\n{'=' * 70}\nMotion run {run_idx + 1}/{num_runs}\n{'=' * 70}")
            run_pick_place_demo(
                robot_cfg=robot_entry["robot_cfg"],
                task_cfg=task_cfg,
                robot_id=task_entry["robot_id"],
                reset_env=reset_env,
                use_stamped=use_stamped,
            )
        return
    
    if task_entry["kind"] == "drawer":
        base_task_cfg = DrawerPickPlaceTaskConfig(
            **flatten_drawer_pick_place_task_overrides(task_entry["base_task_overrides"])
        )
        task_cfg = _apply_preset(
            base_task_cfg,
            flatten_drawer_pick_place_task_overrides(scene_presets.get(scene, {})),
        )
        print(format_drawer_task_cfg_summary(scene, task_cfg))
        task_queue = task_entry.get("task_queue")
        if task_queue:
            import task_runtime.skills  # noqa: F401 - register skills

            from task_runtime.runner import run_drawer_pick_place_task_queue  # pyright: ignore[reportMissingImports]

            for run_idx in range(num_runs):
                print(f"\n{'=' * 70}\nMotion run {run_idx + 1}/{num_runs} (drawer task queue)\n{'=' * 70}")
                run_drawer_pick_place_task_queue(
                    robot_cfg=robot_entry["robot_cfg"],
                    task_cfg=task_cfg,
                    robot_id=task_entry["robot_id"],
                    blocks=task_queue,
                    reset_env=reset_env,
                    use_stamped=use_stamped,
                )
            return
        for run_idx in range(num_runs):
            print(f"\n{'=' * 70}\nMotion run {run_idx + 1}/{num_runs}\n{'=' * 70}")
            run_drawer_demo(
                robot_cfg=robot_entry["robot_cfg"],
                task_cfg=task_cfg,
                robot_id=task_entry["robot_id"],
                reset_env=reset_env,
                use_stamped=use_stamped,
            )
        return

    if task_entry["kind"] == "handover":
        base_task_cfg = HandoverTaskConfig(
            **flatten_handover_task_overrides(task_entry["base_task_overrides"])
        )
        task_cfg = _apply_preset(
            base_task_cfg,
            flatten_handover_task_overrides(scene_presets.get(scene, {})),
        )
        print(format_handover_task_cfg_summary(scene, task_cfg))
        task_queue = task_entry.get("task_queue")
        if task_queue:
            import task_runtime.skills  # noqa: F401 - register skills

            from task_runtime.runner import run_handover_task_queue  # pyright: ignore[reportMissingImports]

            queue_for_run = list(task_queue)
            if num_runs == 1:
                queue_for_run = [
                    blk
                    for blk in queue_for_run
                    if str(blk.get("skill", "")).strip() != "handover.movej_return_initial"
                ]
                if len(queue_for_run) != len(task_queue):
                    print("[info] Single run: skip 'handover.movej_return_initial' by default.")

            for run_idx in range(num_runs):
                print(f"\n{'=' * 70}\nMotion run {run_idx + 1}/{num_runs} (handover task queue)\n{'=' * 70}")
                run_handover_task_queue(
                    robot_cfg=robot_entry["robot_cfg"],
                    task_cfg=task_cfg,
                    robot_id=task_entry["robot_id"],
                    blocks=queue_for_run,
                    reset_env=reset_env,
                    use_stamped=use_stamped,
                )
            return
        raise ValueError(
            "handover now requires task_queue in task config; "
            "legacy run_handover_demo path is disabled."
        )

    if task_entry["kind"] == "bimanual_carry":
        base_task_cfg = BimanualCarryTaskConfig(
            **flatten_bimanual_carry_task_overrides(task_entry["base_task_overrides"])
        )
        task_cfg = _apply_preset(
            base_task_cfg,
            flatten_bimanual_carry_task_overrides(scene_presets.get(scene, {})),
        )
        print(format_bimanual_carry_task_cfg_summary(scene, task_cfg))
        task_queue = task_entry.get("task_queue")
        if task_queue:
            import task_runtime.skills  # noqa: F401 - register skills

            from task_runtime.runner import run_bimanual_task_queue  # pyright: ignore[reportMissingImports]

            queue_for_run = list(task_queue)
            if num_runs == 1:
                def _is_movej_block(block: object) -> bool:
                    if isinstance(block, dict):
                        return str(block.get("skill", "")).strip() == "bimanual.movej_return_initial"
                    return getattr(block, "skill", "") == "bimanual.movej_return_initial"

                queue_for_run = [
                    blk
                    for blk in queue_for_run
                    if not _is_movej_block(blk)
                ]
                if len(queue_for_run) != len(task_queue):
                    print("[info] Single run: skip 'bimanual.movej_return_initial' by default.")

            for run_idx in range(num_runs):
                print(f"\n{'=' * 70}\nMotion run {run_idx + 1}/{num_runs} (bimanual task queue)\n{'=' * 70}")
                run_bimanual_task_queue(
                    robot_cfg=robot_entry["robot_cfg"],
                    task_cfg=task_cfg,
                    robot_id=task_entry["robot_id"],
                    blocks=queue_for_run,
                    reset_env=reset_env,
                    use_stamped=use_stamped,
                )
            return
        raise ValueError(
            "bimanual_carry now requires task_queue in task config; "
            "legacy run_bimanual_carry_demo has been removed."
        )

    raise ValueError(f"Unsupported task kind: {task_entry['kind']}")


if __name__ == "__main__":
    main()
