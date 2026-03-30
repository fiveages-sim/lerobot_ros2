"""Realman RM75 pick-place task config.

``base_task_overrides`` / ``scene_presets`` may group fields under optional ``pick`` and
``place`` mappings; :func:`motion_generation.pick_place.flatten_pick_place_task_overrides`
merges them into the flat kwargs expected by :class:`PickPlaceFlowTaskConfig`.
"""

from __future__ import annotations

_FLOW_DEFAULTS: dict[str, object] = {
    "mode": "single_arm",
    "max_stage_duration": 2.0,
    "pose_tol_pos": 0.025,
    "pose_tol_ori": 0.08,
    "require_orientation_reach": False,
    "use_object_orientation": False,
    "run_place_before_return": True,
}

_PICK_TEMPLATE: dict[str, object] = {
    "initial_grasp_arm": "left",
    "object_xyz_random_offset": (0.15, 0.15, 0.0),
    "approach_clearance": 0.3,
    "grasp_clearance": 0.165,
    "source_object_entity_path": "/World/banana/banana/banana",
    "grasp_orientation": (0.0, 1.0, 0.0, 0.0),
    "grasp_direction": "top",
    "grasp_direction_vector": None,
}

_PLACE_TEMPLATE: dict[str, object] = {
    "place_position": (0.39015462527214956, -0.27559945312297013, 0.44),
    "place_orientation": (0.0, 0.8, 0.0, 0.6),
    "post_release_retract_offset": (0.0, 0.2, 0.0),
}

TASK_CONFIG = {
    "task_key": "pick_place",
    "label": "Pick Place",
    "kind": "pick_place",
    "use_stamped": False,
    "robot_id": "realman_rm75_pick_place",
    "default_scene": "default",
    "base_task_overrides": {
        **_FLOW_DEFAULTS,
        "pick": _PICK_TEMPLATE,
        "place": _PLACE_TEMPLATE,
    },
    "scene_presets": {
        "default": {},
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
