"""Dobot CR5 pick-place task config."""

TASK_CONFIG = {
    "task_key": "drawer_2",
    "label": "Open Drawer Pick Place",
    "kind": "drawer",
    "use_stamped": True,
    "robot_id": "dobot_cr5_pick_place",
    "default_scene": "default",
    "base_task_overrides": {
        "mode": "single_arm",
        "initial_grasp_arm": "left",
        "object_xyz_random_offset": (0.3, 0.3, 0.0),
        "object_xyz_random_offset_drawer": (0.05, 0.05, 0.0),
        "approach_clearance": 0.08,
        "grasp_clearance": -0.03,
        "grasp_clearance_drawer": -0.003,
        "source_object_entity_path": "/World/apple/apple/apple",
        "source_object_path_drawer_all": "/World/StorageFurniture135/StorageFurniture135",
        "source_object_path_drawer": "/World/StorageFurniture135/StorageFurniture135_Drawer002",
        "handle_extent_max": (0.016881, -0.256626, -0.003418),
        "handle_extent_min": (-0.019242, -0.265197, -0.022885),
        "drawer_scale": 0.85,
        "grasp_orientation": (0.7, 0.7, 0.0, 0.0),
        "grasp_direction": "top",
        "grasp_direction_vector": None,
        "run_place_before_return": True,
        "place_position": (-0.5211657941743888, 0.18339369774887115, 0.54),
        "place_orientation": (0.64, 0.64, -0.28, -0.28),
        "post_release_retract_offset": (0.0, -0.2, 0.0),
        "max_stage_duration": 2.0,
        "pose_tol_pos": 0.001,
        "pose_tol_ori": 0.08,
        "require_orientation_reach": False,
        "use_object_orientation": False,
        "retreat_direction_extra": 0.08,
    },
    "scene_presets": {
        "default": {}
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
