"""FiveAges W2 pick-place-flow config."""

FLOW_CONFIG = {
    "task_key": "pick_place_flow",
    "label": "Pick Place Flow",
    "kind": "pick_place",
    "robot_id": "fiveages_w2_pick_place_flow",
    "default_scene": "grab_medicine",
    "base_task_overrides": {
        "mode": "single_arm",
        "initial_grasp_arm": "right",
        "object_xyz_random_offset": (0.0, 0.0, 0.0),
        "approach_clearance": 0.2,
        "grasp_clearance": 0.05,
        "source_object_entity_path": "/World/medicine_handover/FinasterideTablets/tablets/tablets",
        "grasp_orientation": (1.0, 0.0, 0.0, 0.0),
    },
    "scene_presets": {
        "grab_medicine": {
            "source_object_entity_path": "/World/medicine_handover/FinasterideTablets/tablets/tablets",
        },
        "grab_bottle": {
            "source_object_entity_path": "/World/ConvenienceStore01/SM_Bottle_04_85",
            "initial_grasp_arm": "left",
            "grasp_direction": "front",
            "grasp_orientation": (0.7, 0.0, 0.7, 0.0),
            "target_pose_offset": (0.0, 0.0, 0.1),
            "approach_clearance": 0.2,
            "grasp_clearance": -0.02,
            "retreat_direction_extra": 0.0,
            "retreat_raise_z": 0.1,
        },
    },
}
