"""FiveAges W2 handover task config.

Use nested ``pick`` / ``handover`` / ``place`` sections; merged by
:func:`motion_generation.handover.flatten_handover_task_overrides`.
"""

from __future__ import annotations

_PICK_TEMPLATE: dict[str, object] = {
    "initial_grasp_arm": "right",
    "grasp_orientation": (1.0, 0.0, 0.0, 0.0),
    "object_xyz_random_offset": (0.0, 0.0, 0.0),
    "approach_clearance": 0.2,
    "grasp_clearance": 0.01,
    "grasp_offset": (0.04, 0.0, 0.0),
    "source_object_entity_path": "/World/medicine_handover/FinasterideTablets/tablets/tablets",
    "grasp_direction": "top",
    "grasp_direction_vector": None,
}

_HANDOVER_TEMPLATE: dict[str, object] = {
    "handover_position": (0.6, 0.0, 1.35),
    "source_handover_orientation": (0.653, 0.271, 0.653, -0.271),
    "receiver_handover_orientation": (0.653, -0.271, 0.653, 0.271),
    "receiver_handover_offset": (0.0, 0.0, -0.04),
}

_PLACE_TEMPLATE: dict[str, object] = {
    "receiver_place_position": (0.7, 0.3, 1.1),
    "receiver_place_orientation": (
        0.6532920183548978,
        -0.12549174013187747,
        0.7390193620374562,
        0.10635668500949247,
    ),
    "run_place_after_handover": True,
}

TASK_CONFIG = {
    "task_key": "handover",
    "label": "Handover",
    "kind": "handover",
    "use_stamped": True,
    "robot_id": "fiveages_w2_bimanual_handover",
    "default_scene": "grab_medicine",
    "base_task_overrides": {
        "pick": _PICK_TEMPLATE,
        "handover": _HANDOVER_TEMPLATE,
        "place": _PLACE_TEMPLATE,
    },
    "scene_presets": {
        "grab_medicine": {
            "pick": {
                "source_object_entity_path": "/World/medicine_handover/FinasterideTablets/tablets/tablets",
            },
        },
        "grab_bottle": {
            "pick": {
                "source_object_entity_path": "/World/ConvenienceStore01/SM_Bottle_04_85",
            },
        },
    },
}
