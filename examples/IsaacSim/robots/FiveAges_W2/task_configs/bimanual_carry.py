"""FiveAges W2 bimanual carry task config."""

TASK_CONFIG = {
    "task_key": "bimanual_carry",
    "label": "Bimanual Carry",
    "kind": "bimanual_carry",
    "use_stamped": True,
    "robot_id": "fiveages_w2_bimanual_carry",
    "default_scene": "grab_medicine",
    "base_task_overrides": {
        "source_object_entity_path": "/World/lift_box/LogisticBox_01/box/box",
        "lateral_offset": 0.29,
        "approach_offset": (-0.3, 0.0, 0.0),
        "lateral_clearance": 0.08,
        "grasp_offset": (0.0, 0.0, 0.155),
        "left_orientation": (-0.5, 0.5, -0.5, 0.5),
        "right_orientation": (-0.5, 0.5, -0.5, 0.5),
        "lift_offset": (0.0, 0.0, 0.1),
        "retreat_offset": (-0.2, 0.0, 0.0),
    },
    "scene_presets": {
        "box01": {
            "source_object_entity_path": "/World/lift_box/LogisticBox_01/box/box",
        },
        "warehouse_box01": {
            "source_object_entity_path": "/World/lift_box_warehouse/LogisticBox_01/box/box",
            "retreat_offset": (-0.3, 0.0, 0.0),
        },
    },
}
