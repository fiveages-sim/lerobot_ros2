"""Agibot G1 pick-place task config.

Nested ``pick`` / ``place`` in presets are merged via
:func:`motion_generation.pick_place.flatten_pick_place_task_overrides`.
"""

from __future__ import annotations

_FLOW_DEFAULTS: dict[str, object] = {
    "mode": "single_arm",
    "max_stage_duration": 2.0,
    "pose_tol_pos": 0.025,
    "pose_tol_ori": 0.08,
    "require_orientation_reach": False,
    "use_object_orientation": False,
}

_PICK_TEMPLATE: dict[str, object] = {
    "initial_grasp_arm": "right",
    "object_xyz_random_offset": (0.0, 0.0, 0.0),
    "approach_clearance": 0.2,
    "grasp_clearance": 0.01,
    "source_object_entity_path": "/World/medicine_handover/FinasterideTablets/tablets/tablets",
    "grasp_orientation": (-0.7, 0.7, 0.0, 0.0),
    "grasp_direction": "top",
    "grasp_direction_vector": None,
}

TASK_CONFIG = {
    "task_key": "pick_place",
    "label": "Pick Place",
    "kind": "pick_place",
    "use_stamped": True,
    "robot_id": "agibot_g1_pick_place",
    "default_scene": "grab_medicine",
    "base_task_overrides": {
        **_FLOW_DEFAULTS,
        "pick": _PICK_TEMPLATE,
    },
    "scene_presets": {
        "grab_medicine": {
            "pick": {
                "source_object_entity_path": "/World/medicine_handover/FinasterideTablets/tablets/tablets",
                "initial_grasp_arm": "right",
                "grasp_direction": "top",
            },
        },
        "grab_bottle": {
            "pick": {
                "source_object_entity_path": "/World/ConvenienceStore01/SM_Bottle_04_85/SM_Bottle_04",
                "initial_grasp_arm": "left",
                "grasp_direction": "front",
                "grasp_orientation": (-0.5, 0.5, -0.5, 0.5),
                "target_pose_offset": (0.0, 0.0, 0.1),
                "approach_clearance": 0.2,
                "grasp_clearance": -0.02,
                "retreat_direction_extra": 0.0,
                "retreat_offset": (0.0, 0.0, 0.1),
            },
        },
    },
}
