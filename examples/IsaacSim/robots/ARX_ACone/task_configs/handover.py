"""ARX AC_One handover task config."""

TASK_CONFIG = {
    "task_key": "handover",
    "label": "Handover",
    "kind": "handover",
    "use_stamped": True,
    "robot_id": "AC_ONE_bimanual_handover",
    "default_scene": "grab_medicine",
    "base_task_overrides": {
        "initial_grasp_arm": "right",
        "grasp_orientation": (1.0, 0.0, 0.0, 0.0),
        "object_xyz_random_offset": (0.0, 0.0, 0.0),
        "approach_clearance": 0.2,
        "grasp_clearance": 0.01,
        "grasp_offset": (0.04, 0.0, 0.0),
        "source_object_entity_path": "/World/apple_handover/apple/apple/apple",
        "handover_position": (0.4, 0.0, 1.0),
        "source_handover_orientation": (0.653, 0.271, 0.653, -0.271),
        "receiver_handover_orientation": (0.653, -0.271, 0.653, 0.271),
        "receiver_handover_offset": (0.0, 0.0, -0.04),
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
            "source_object_entity_path": "/World/apple_handover/apple/apple/apple",
        },
        "grab_bottle": {
            "source_object_entity_path": "/World/ConvenienceStore01/SM_Bottle_04_85",
        },
    },
    "record": {
        "base_record_overrides": {
            "task_name": "handover",
            "fps": 30,
            "camera_info_timeout": 10.0,
            "switch_to_hold_after_episode": True,
            "async_episode_save": True,
            "episode_save_queue_size": 2,
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
    }
}
