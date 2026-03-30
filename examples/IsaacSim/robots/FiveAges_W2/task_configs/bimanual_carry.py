"""FiveAges W2 bimanual carry task config.

Robot-specific carry parameters live under the ``carry`` key; see
:func:`motion_generation.bimanual_carry.flatten_bimanual_carry_task_overrides`.
"""

from __future__ import annotations

_CARRY_TEMPLATE: dict[str, object] = {
    "source_object_entity_path": "/World/lift_box/LogisticBox_01/box/box",
    "lateral_offset": 0.29,
    "approach_offset": (-0.3, 0.0, 0.0),
    "lateral_clearance": 0.08,
    "grasp_offset": (0.0, 0.0, 0.155),
    "left_orientation": (-0.5, 0.5, -0.5, 0.5),
    "right_orientation": (-0.5, 0.5, -0.5, 0.5),
    "lift_offset": (0.0, 0.0, 0.1),
    "retreat_offset": (-0.2, 0.0, 0.0),
    "object_xyz_random_offset": (0.0, 0.0, 0.0),
}

TASK_CONFIG = {
    "task_key": "bimanual_carry",
    "label": "Bimanual Carry",
    "kind": "bimanual_carry",
    "use_stamped": True,
    "robot_id": "fiveages_w2_bimanual_carry",
    "default_scene": "box01",
    "base_task_overrides": {
        "carry": _CARRY_TEMPLATE,
    },
    "scene_presets": {
        "box01": {
            "carry": {
                "source_object_entity_path": "/World/lift_box/LogisticBox_01/box/box",
            },
        },
        "warehouse_box01": {
            "carry": {
                "source_object_entity_path": "/World/lift_box_warehouse/LogisticBox_01/box/box",
                "retreat_offset": (-0.3, 0.0, 0.0),
            },
        },
    },
}
