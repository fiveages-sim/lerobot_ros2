"""FiveAges W2 pick-place task config."""

TASK_CONFIG = {
    "task_key": "pick_place",
    "label": "Pick Place",
    "kind": "pick_place",
    "use_stamped": True,
    "robot_id": "fiveages_w2_pick_place",
    "default_scene": "grab_medicine",
    "base_task_overrides": {
        "mode": "single_arm",
        "initial_grasp_arm": "right",
        "object_xyz_random_offset": (0.0, 0.0, 0.0),
        "approach_clearance": 0.2,
        "grasp_clearance": 0.01,
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
            "retreat_offset": (0.0, 0.0, 0.1),
        },
        "ptc-pick-24": {
            "source_object_entity_path": "/World/scene/trolley/sr15_24/SR15_10",
            "initial_grasp_arm": "right",
            "grasp_direction": "front",
            "grasp_orientation": (0.0, 0.91, 0.0, 0.39),
            "target_pose_offset": (-0.038, 0.028, 0.005),
            "approach_clearance": 0.3,
            "grasp_clearance": 0.00,
            "retreat_direction_extra": 0.0,
            "retreat_offset": (0.0, 0.0, 0.3),
        },
    },
}
