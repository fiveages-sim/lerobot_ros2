"""Marvin M6CCS handover task config.

Nested section keys: ``pick``, ``handover``, ``place`` (see
:func:`motion_generation.handover.flatten_handover_task_overrides`).
"""

from __future__ import annotations

_PICK_TEMPLATE: dict[str, object] = {
    "initial_grasp_arm": "right",
    "grasp_orientation": (1.0, 0.0, 0.0, 0.0),
    "object_xyz_random_offset": (0.0, 0.0, 0.0),
    "approach_clearance": 0.2,
    "grasp_clearance": 0.01,
    "grasp_offset": (0.02, 0.0, 0.0),
    "source_object_entity_path": "/World/medicine_handover/FinasterideTablets/tablets/tablets",
    "grasp_direction": "top",
    "grasp_direction_vector": None,
}

_HANDOVER_TEMPLATE: dict[str, object] = {
    "handover_position": (0.55, 0.0, 0.5),
    "source_handover_orientation": (0.5, 0.5, 0.5, -0.5),
    "receiver_handover_orientation": (0.5, -0.5, 0.5, 0.5),
    "receiver_handover_offset": (0.0, 0.0, -0.02),
}

_PLACE_TEMPLATE: dict[str, object] = {
    "receiver_place_position": (0.5, 0.25, 0.3),
    "receiver_place_orientation": (
        0.6532920183548978,
        -0.12549174013187747,
        0.7390193620374562,
        0.10635668500949247,
    ),
    "run_place_after_handover": True,
}

TASK_CONFIG = {
    "task_key": "handover",
    "label": "Handover",
    "kind": "handover",
    "use_stamped": True,
    "robot_id": "marvin_m6ccs_bimanual_handover",
    "default_scene": "grab_medicine",
    "base_task_overrides": {
        "pick": _PICK_TEMPLATE,
        "handover": _HANDOVER_TEMPLATE,
        "place": _PLACE_TEMPLATE,
    },
    "task_queue": [
        {"skill": "handover.pregrasp"},
        {"skill": "handover.pick"},
        {"skill": "handover.exchange_place"},
        {"skill": "handover.movej_return_initial"},
    ],
    "scene_presets": {
        "grab_medicine": {
            "pick": {
                "source_object_entity_path": "/World/medicine_handover/FinasterideTablets/tablets/tablets",
            },
        },
        "grab_bottle": {
            "pick": {
                "source_object_entity_path": "/World/ConvenienceStore01/SM_Bottle_04_85",
            },
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
    },
}
