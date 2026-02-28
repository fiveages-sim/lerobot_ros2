"""Dobot CR5 pick-place-flow config."""

FLOW_CONFIG = {
    "task_key": "pick_place_flow",
    "label": "Pick Place Flow",
    "kind": "pick_place",
    "robot_id": "dobot_cr5_pick_place_flow",
    "default_scene": "default",
    "base_task_overrides": {
        "mode": "single_arm",
        "initial_grasp_arm": "left",
        "object_xyz_random_offset": (0.5, 0.5, 0.0),
        "approach_clearance": 0.12,
        "grasp_clearance": -0.03,
        "source_object_entity_path": "/World/apple/apple/apple",
        "grasp_orientation": (0.7, 0.7, 0.0, 0.0),
        "grasp_direction": "top",
        "grasp_direction_vector": None,
        "run_place_before_return": True,
        "place_position": (-0.5011657941743888, 0.36339369774887115, 0.34),
        "place_orientation": (0.64, 0.64, -0.28, -0.28),
        "post_release_retract_offset": (0.0, -0.2, 0.0),
        "max_stage_duration": 2.0,
        "pose_tol_pos": 0.025,
        "pose_tol_ori": 0.08,
        "require_orientation_reach": False,
        "use_object_orientation": False,
    },
    "scene_presets": {
        "default": {}
    },
}
