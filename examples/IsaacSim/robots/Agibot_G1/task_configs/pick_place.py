"""Agibot G1 pick-place task config."""

TASK_CONFIG = {
    "task_key": "pick_place",
    "label": "Pick Place",
    "kind": "pick_place",
    "robot_id": "agibot_g1_pick_place",
    "default_scene": "grab_medicine",
    "base_task_overrides": {
        "mode": "single_arm",
        "initial_grasp_arm": "right",
        "object_xyz_random_offset": (0.0, 0.0, 0.0),
        "approach_clearance": 0.2,
        "grasp_clearance": 0.01,
        "source_object_entity_path": "/World/medicine_handover/FinasterideTablets/tablets/tablets",
        "grasp_orientation": (-0.7, 0.7, 0.0, 0.0),
    },
    "scene_presets": {
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
            "retreat_direction_extra": 0.0,
            "retreat_raise_z": 0.1,
        },
    },
}
