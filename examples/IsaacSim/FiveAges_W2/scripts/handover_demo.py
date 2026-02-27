#!/usr/bin/env python3
"""FiveAges W2 handover demo entrypoint (uses IsaacSim common flow)."""

from __future__ import annotations

import sys
from pathlib import Path

from robot_config import ROBOT_CFG

COMMON_ISAAC_DIR = Path(__file__).resolve().parents[2] / "common"
if str(COMMON_ISAAC_DIR) not in sys.path:
    sys.path.append(str(COMMON_ISAAC_DIR))

from handover_demo_common import (  # noqa: E402  # pyright: ignore[reportMissingImports]
    format_handover_task_cfg_summary,
    HandoverTaskConfig,
    resolve_handover_task_cfg_from_presets,
    run_handover_demo,
)


HANDOVER_TASK_CFG = HandoverTaskConfig(
    initial_grasp_arm="right",
    grasp_orientation=(1.0, 0.0, 0.0, 0.0),
    object_xyz_random_offset=(0.0, 0.0, 0.0),
    approach_clearance=0.2,
    grasp_clearance=0.05,
    source_object_entity_path="/World/medicine_handover/FinasterideTablets/tablets/tablets",
    handover_position=(0.55, 0.0, 1.35),
    source_handover_orientation=(0.7, 0.3, 0.7, -0.3),
    receiver_handover_orientation=(0.6, -0.3, 0.6, 0.3),
    receiver_place_position=(0.7, 0.3, 1.1),
    receiver_place_orientation=(
        0.6532920183548978,
        -0.12549174013187747,
        0.7390193620374562,
        0.10635668500949247,
    ),
)

SCENE_PRESETS: dict[str, dict[str, object]] = {
    "grab_medicine": {
        "source_object_entity_path": "/World/medicine_handover/FinasterideTablets/tablets/tablets",
    },
    "grab_bottle": {
        "source_object_entity_path": "/World/ConvenienceStore01/SM_Bottle_04_85",
    },
}


def main() -> None:
    scene, task_cfg = resolve_handover_task_cfg_from_presets(
        base_task_cfg=HANDOVER_TASK_CFG,
        scene_presets=SCENE_PRESETS,
        cli_description="FiveAges W2 handover demo",
        default_scene="grab_medicine",
    )
    print(format_handover_task_cfg_summary(scene, task_cfg))
    run_handover_demo(
        robot_cfg=ROBOT_CFG,
        handover_task_cfg=task_cfg,
        robot_id="fiveages_w2_bimanual_handover",
    )


if __name__ == "__main__":
    main()
