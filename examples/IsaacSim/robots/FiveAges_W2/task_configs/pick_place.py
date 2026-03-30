"""FiveAges W2 pick-place task config.

``base_task_overrides`` / ``scene_presets`` may use nested ``pick`` / ``place`` blocks;
see :func:`motion_generation.pick_place.flatten_pick_place_task_overrides`.
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
    "grasp_orientation": (1.0, 0.0, 0.0, 0.0),
    "grasp_direction": "top",
    "grasp_direction_vector": None,
}

TASK_CONFIG = {
    "task_key": "pick_place",
    "label": "Pick Place",
    "kind": "pick_place",
    "use_stamped": True,
    "robot_id": "fiveages_w2_pick_place",
    "default_scene": "grab_medicine",
    "base_task_overrides": {
        **_FLOW_DEFAULTS,
        "pick": _PICK_TEMPLATE,
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
                "initial_grasp_arm": "left",
                "grasp_direction": "front",
                "grasp_orientation": (0.7, 0.0, 0.7, 0.0),
                "target_pose_offset": (0.0, 0.0, 0.1),
                "approach_clearance": 0.2,
                "grasp_clearance": -0.02,
                "retreat_direction_extra": 0.0,
                "retreat_offset": (0.0, 0.0, 0.1),
            },
        },
        "ptc-pick-24": {
            "pick": {
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
        "ptc-pick-23": {
            "pick": {
                "source_object_entity_path": "/World/scene/trolley/sr15_23/SR15_10",
                "initial_grasp_arm": "right",
                "grasp_direction": "front",
                "grasp_orientation": (0.0, 0.91, 0.0, 0.39),
                "target_pose_offset": (-0.038, 0.01, 0.005),
                "approach_clearance": 0.3,
                "grasp_clearance": 0.00,
                "retreat_direction_extra": 0.0,
                "retreat_offset": (0.0, 0.0, 0.3),
            },
        },
    },
    "record": {
        "base_record_overrides": {
            "task_name": "pick_place",
            "fps": 30,
            "camera_info_timeout": 10.0,
            "enable_keypoint_pcd": False,
            "include_depth_feature": False,
            "image_writer_processes": 0,
            "image_writer_threads": 8,
            "video_encoding_batch_size": 5,
        },
        "profiles": [
            {
                "key": "default",
                "label": "Default",
                "overrides": {},
            },
            {
                "key": "fast",
                "label": "Fast Encode Queue",
                "overrides": {
                    "image_writer_threads": 12,
                    "video_encoding_batch_size": 10,
                },
            },
        ],
    },
}
