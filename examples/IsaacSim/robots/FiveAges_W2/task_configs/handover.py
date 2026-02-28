"""FiveAges W2 handover task config."""

TASK_CONFIG = {
    "task_key": "handover",
    "label": "Handover",
    "kind": "handover",
    "robot_id": "fiveages_w2_bimanual_handover",
    "default_scene": "grab_medicine",
    "base_task_overrides": {
        "initial_grasp_arm": "right",
        "grasp_orientation": (1.0, 0.0, 0.0, 0.0),
        "object_xyz_random_offset": (0.0, 0.0, 0.0),
        "approach_clearance": 0.2,
        "grasp_clearance": 0.05,
        "source_object_entity_path": "/World/medicine_handover/FinasterideTablets/tablets/tablets",
        "handover_position": (0.55, 0.0, 1.35),
        "source_handover_orientation": (0.7, 0.3, 0.7, -0.3),
        "receiver_handover_orientation": (0.6, -0.3, 0.6, 0.3),
        "receiver_place_position": (0.7, 0.3, 1.1),
        "receiver_place_orientation": (
            0.6532920183548978,
            -0.12549174013187747,
            0.7390193620374562,
            0.10635668500949247,
        ),
    },
    "scene_presets": {
        "grab_medicine": {
            "source_object_entity_path": "/World/medicine_handover/FinasterideTablets/tablets/tablets",
        },
        "grab_bottle": {
            "source_object_entity_path": "/World/ConvenienceStore01/SM_Bottle_04_85",
        },
    },
}
