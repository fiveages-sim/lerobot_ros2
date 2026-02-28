"""Agibot G1 handover-flow config."""

FLOW_CONFIG = {
    "task_key": "handover",
    "label": "Handover",
    "kind": "handover",
    "robot_id": "agibot_g1_bimanual_handover",
    "default_scene": "grab_medicine",
    "base_task_overrides": {
        "initial_grasp_arm": "right",
        "grasp_orientation": (-0.7, 0.7, 0.0, 0.0),
        "object_xyz_random_offset": (0.0, 0.0, 0.0),
        "approach_clearance": 0.2,
        "grasp_clearance": 0.01,
        "source_object_entity_path": "/World/medicine_handover/FinasterideTablets/tablets/tablets",
        "handover_position": (0.55, 0.0, 1.4),
        "source_handover_orientation": (-0.77, 0.0, 0.0, 0.77),
        "receiver_handover_orientation": (0.0, 0.77, -0.77, 0.0),
        "receiver_place_position": (0.9, 0.22, 1.1),
        "receiver_place_orientation": (
            -0.33016505084249775,
            0.6534009839213628,
            -0.6175486972275431,
            0.2875618193803384,
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
