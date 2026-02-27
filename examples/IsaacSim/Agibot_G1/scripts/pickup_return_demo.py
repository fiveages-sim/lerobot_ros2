#!/usr/bin/env python3
"""Agibot G1 pickup-return demo entrypoint (uses IsaacSim common flow)."""

from __future__ import annotations

import sys
from pathlib import Path

from robot_config import ROBOT_CFG

COMMON_ISAAC_DIR = Path(__file__).resolve().parents[2] / "common"
if str(COMMON_ISAAC_DIR) not in sys.path:
    sys.path.append(str(COMMON_ISAAC_DIR))

from pickup_return_demo_common import (  # noqa: E402  # pyright: ignore[reportMissingImports]
    PickupReturnTaskConfig,
    format_task_cfg_summary,
    resolve_task_cfg_from_presets,
    run_pickup_return_demo,
)


PICKUP_RETURN_TASK_CFG = PickupReturnTaskConfig(
    mode="single_arm",
    initial_grasp_arm="right",
    object_xyz_random_offset=(0.0, 0.0, 0.0),
    approach_clearance=0.2,
    grasp_clearance=0.01,
    source_object_entity_path="/World/medicine_handover/FinasterideTablets/tablets/tablets",
    grasp_orientation=(-0.7, 0.7, 0.0, 0.0),
)

SCENE_PRESETS: dict[str, dict[str, object]] = {
    "grab_medicine": {
        "source_object_entity_path": "/World/medicine_handover/FinasterideTablets/tablets/tablets",
        "initial_grasp_arm": "right",
        "grasp_direction": "top",
    },
    "grab_bottle": {
        "source_object_entity_path": "/World/ConvenienceStore01/SM_Bottle_04_85/SM_Bottle_04",
        "initial_grasp_arm": "left",
        "grasp_direction": "front",
        "grasp_orientation": (-0.5, 0.5, -0.5, 0.5),
        "target_pose_offset": (0.0, 0.0, 0.1),
        "approach_clearance": 0.2,
        "grasp_clearance": -0.02,
        "retreat_direction_extra": 0.00,
        "retreat_raise_z": 0.1,
    },
}

def main() -> None:
    scene, task_cfg = resolve_task_cfg_from_presets(
        base_task_cfg=PICKUP_RETURN_TASK_CFG,
        scene_presets=SCENE_PRESETS,
        cli_description="Agibot G1 pickup-return demo",
        default_scene="grab_medicine",
    )
    print(format_task_cfg_summary(scene, task_cfg))
    run_pickup_return_demo(
        robot_cfg=ROBOT_CFG,
        task_cfg=task_cfg,
        robot_id="agibot_g1_pickup_return",
    )


if __name__ == "__main__":
    main()
