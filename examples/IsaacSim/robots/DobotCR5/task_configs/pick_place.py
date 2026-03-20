"""Dobot CR5 pick-place task config."""

TASK_CONFIG = {
    "task_key": "pick_place",
    "label": "Pick Place",
    "kind": "pick_place",
    "use_stamped": True,
    "robot_id": "dobot_cr5_pick_place",
    "default_scene": "apple",
    "base_task_overrides": {
        "mode": "single_arm",
        "initial_grasp_arm": "left",
        "object_xyz_random_offset": (0.5, 0.5, 0.0),
        "approach_clearance": 0.12,
        "grasp_clearance": -0.03,
        "source_object_entity_path": "/World/grasp_fruits/apple/apple/apple",
        "grasp_orientation": (0.7, 0.7, 0.0, 0.0),
        "grasp_direction": "top",
        "grasp_direction_vector": None,
        "max_stage_duration": 2.0,
        "pose_tol_pos": 0.025,
        "pose_tol_ori": 0.08,
        "require_orientation_reach": False,
        "use_object_orientation": False,
    },
    "scene_presets": {
        "apple": {
            "place_object_entity_path": "/World/grasp_fruits/shelf8",
            "place_pose_offset": (0.0, 0.0, 0.05),
            "place_direction": "left",
            "place_approach_clearance": -0.5,
            "place_orientation": (0.0, 0.7, 0.7, 0.0),
            "post_release_retract_offset": (0.0, -0.2, 0.0),
            "run_place_before_return": True,
        },
        "nailong": {
            "source_object_entity_path": "/World/nailong/PlushToy_3/material/material",
            "initial_grasp_arm": "left",
            "grasp_direction": "top",
            "grasp_orientation": (0.4, 0.9, 0.0, 0.0),
            "target_pose_offset": (0.0, 0.0, -0.035),
            "approach_clearance": 0.2,
            "grasp_clearance": 0.00,
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
