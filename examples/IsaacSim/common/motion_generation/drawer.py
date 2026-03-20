#!/usr/bin/env python3
"""IsaacSim **drawer + pick-place** composite flow (action template).

This extends :mod:`motion_generation.pick_place` with a fixed **phase** model:

1. **Open drawer** — grasp handle / pull (drawer-specific clearance & handle geometry).
2. **Pick & place** — same semantic stages as core pick-place (apple or general object).
3. **Close drawer** — push drawer / release.
4. **Retreat** — return arm toward home near the handle frame.

Configuration is a :class:`DrawerPickPlaceTaskConfig` — a **subclass** of
:class:`~motion_generation.pick_place.PickPlaceFlowTaskConfig` with extra fields
only for drawer phases. Task YAML/``TASK_CONFIG`` may use optional nested keys:

- ``pick`` / ``place`` — merged via :func:`motion_generation.pick_place.flatten_pick_place_task_overrides`
- ``drawer`` — merged last via :func:`flatten_drawer_pick_place_task_overrides`
"""

from __future__ import annotations

import signal
import sys
import time
import random
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from lerobot_robot_ros2.utils.pose_utils import (  # pyright: ignore[reportMissingImports]
    quat_conjugate,
    quat_multiply,
    quat_normalize,
    pose_from_tuple,
)

from ros2_robot_interface import (  # pyright: ignore[reportMissingImports]
    FSM_HOLD,
    FSM_OCS2,
    ArmSide,
    ArmStage,
    ArmTarget,
    SendMode,
    StageTarget,
    assign_to_arm,
    build_single_arm_pick_sequence,
    build_single_arm_place_sequence,
    build_single_arm_return_home_sequence,
    compose_bimanual_synchronized_sequence,
    execute_stage_sequence,
)

from lerobot_robot_ros2 import (
    ROS2Robot,
    ROS2RobotConfig,
)
from lerobot_robot_ros2.utils import obs_to_pose  # pyright: ignore[reportMissingImports]

from isaac_ros2_sim_common import (
    SimTimeHelper,
    get_entity_pose_world_service,
    get_object_pose_from_service,
    reset_simulation_and_randomize_object,
    randomize_object_xyz_after_reset,
    set_prim_orientation_local,
)
from demo_preset_utils import resolve_dataclass_cfg_from_presets

from motion_generation.pick_place import (  # pyright: ignore[reportMissingImports]
    PickPlaceFlowTaskConfig,
    flatten_pick_place_task_overrides,
)

def euler_to_quaternion(roll, pitch, yaw):
    """
    Convert Euler angles to quaternion.
    ZYX convention: yaw (z), pitch (y), roll (x)
    """
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    
    return (w, x, y, z)

def _resolve_gripper_open_close(mode: str) -> tuple[float, float]:
    if mode == "target_command":
        return 1.0, 0.0
    raise ValueError(
        "pick_place_flow only supports gripper_control_mode='target_command' without explicit gripper values"
    )


def _make_robot(robot_cfg: Any, robot_id: str) -> ROS2Robot:
    cfg = getattr(robot_cfg, "ros2_interface", None)
    if cfg is None:
        raise ValueError("robot_cfg must define ros2_interface (ROS2RobotInterfaceConfig)")
    return ROS2Robot(
        ROS2RobotConfig(
            id=robot_id,
            ros2_interface=cfg,
            gripper_control_mode=robot_cfg.gripper_control_mode,
        )
    )


def flatten_drawer_pick_place_task_overrides(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Like :func:`~motion_generation.pick_place.flatten_pick_place_task_overrides` but also
    merges an optional top-level ``drawer`` mapping (drawer-only field overrides).
    """
    if not isinstance(raw, Mapping):
        raise TypeError(f"task overrides must be a mapping, got {type(raw).__name__}")
    rest = {k: v for k, v in raw.items() if k != "drawer"}
    flat = flatten_pick_place_task_overrides(rest)
    drawer = raw.get("drawer")
    if drawer is not None:
        if not isinstance(drawer, Mapping):
            raise TypeError(f'"drawer" must be a mapping, got {type(drawer).__name__}')
        flat.update(dict(drawer))
    return flat


@dataclass(frozen=True)
class DrawerPickPlaceTaskConfig(PickPlaceFlowTaskConfig):
    """Pick-place task config plus parameters used only for open/close drawer phases."""

    object_xyz_random_offset_drawer: tuple[float, float, float] = (0.0, 0.0, 0.0)
    grasp_clearance_drawer: float = 0.01
    source_object_path_drawer_all: str = ""
    source_object_path_drawer: str = ""
    handle_extent_max: tuple[float, float, float] = (0.0, 0.0, 0.0)
    handle_extent_min: tuple[float, float, float] = (0.0, 0.0, 0.0)
    drawer_scale: float = 1.0


def resolve_drawer_task_cfg_from_presets(
    *,
    base_task_cfg: DrawerPickPlaceTaskConfig,
    scene_presets: dict[str, dict[str, object]],
    cli_description: str,
    default_scene: str = "default",
) -> tuple[str, DrawerPickPlaceTaskConfig]:
    scene, task_cfg = resolve_dataclass_cfg_from_presets(
        base_cfg=base_task_cfg,
        scene_presets=scene_presets,
        cli_description=cli_description,
        default_scene=default_scene,
    )
    return scene, task_cfg


def format_drawer_task_cfg_summary(scene: str, task_cfg: DrawerPickPlaceTaskConfig) -> str:
    return (
        f"[Drawer+PickPlace] scene={scene} object={task_cfg.source_object_entity_path} "
        f"drawer_prim={task_cfg.source_object_path_drawer} | "
        f"arm={task_cfg.initial_grasp_arm}, grasp_dir={task_cfg.grasp_direction}"
    )


def _apply_target_pose_offset(pose: Any, offset: tuple[float, float, float]) -> Any:
    ox, oy, oz = offset
    pose.position.x += ox
    pose.position.y += oy
    pose.position.z += oz
    return pose


def _resolve_arm_grasp_config(
    task_cfg: DrawerPickPlaceTaskConfig,
    *,
    is_right: bool,
) -> tuple[tuple[float, float, float, float], str, tuple[float, float, float] | None]:
    if is_right:
        orientation = task_cfg.right_grasp_orientation or task_cfg.grasp_orientation
        direction = task_cfg.right_grasp_direction or task_cfg.grasp_direction
        direction_vector = (
            task_cfg.right_grasp_direction_vector
            if task_cfg.right_grasp_direction_vector is not None
            else task_cfg.grasp_direction_vector
        )
    else:
        orientation = task_cfg.left_grasp_orientation or task_cfg.grasp_orientation
        direction = task_cfg.left_grasp_direction or task_cfg.grasp_direction
        direction_vector = (
            task_cfg.left_grasp_direction_vector
            if task_cfg.left_grasp_direction_vector is not None
            else task_cfg.grasp_direction_vector
        )
    return orientation, direction, direction_vector


def build_single_arm_pull_drawer_sequence(
    *,
    target_pose: Any,
    task_cfg: DrawerPickPlaceTaskConfig,
    arm_side: ArmSide,
    gripper_open: float,
    gripper_closed: float,
    grasp_orientation: tuple[float, float, float, float],
    grasp_direction_vector: tuple[float, float, float],
) -> list[StageTarget]:
    """Build a single-arm pick (+ optional place) + return-home sequence."""
    arm_seq: list[ArmStage] = list(build_single_arm_pick_sequence(
        target_pose=target_pose,
        approach_clearance=task_cfg.approach_clearance,
        grasp_clearance=task_cfg.grasp_clearance_drawer,
        grasp_orientation=grasp_orientation,
        grasp_direction_vector=grasp_direction_vector,
        grasp_offset=task_cfg.grasp_offset,
        retreat_direction_extra=task_cfg.retreat_direction_extra,
        retreat_offset=task_cfg.retreat_offset,
        gripper_open=gripper_open,
        gripper_closed=gripper_closed,
        stage_prefix="PickPlaceFlow",
    ))
    return assign_to_arm(arm_seq, arm_side)

def build_single_arm_close_drawer_sequence(
    *,
    target_pose: Any,
    home_pose: Any,
    task_cfg: DrawerPickPlaceTaskConfig,
    arm_side: ArmSide,
    gripper_open: float,
    gripper_closed: float,
    grasp_orientation: tuple[float, float, float, float],
    grasp_direction_vector: tuple[float, float, float],
) -> list[StageTarget]:
    """Build a single-arm pick (+ optional place) + return-home sequence."""
    arm_seq: list[ArmStage] = list(build_single_arm_pick_sequence(
        target_pose=target_pose,
        approach_clearance=0,
        grasp_clearance=task_cfg.grasp_clearance_drawer,
        grasp_orientation=grasp_orientation,
        grasp_direction_vector=grasp_direction_vector,
        grasp_offset=task_cfg.grasp_offset,
        retreat_direction_extra=0,
        retreat_offset=task_cfg.retreat_offset,
        gripper_open=gripper_open,
        gripper_closed=gripper_closed,
        stage_prefix="PickPlaceFlow",
    ))
    return assign_to_arm(arm_seq, arm_side)

def build_single_arm_back_home_sequence(
    *,
    place_position: Any,
    home_pose: Any,
    task_cfg: DrawerPickPlaceTaskConfig,
    arm_side: ArmSide,
    gripper_open: float,
    gripper_closed: float,
    grasp_orientation: tuple[float, float, float, float],
) -> list[StageTarget]:
    arm_seq: list[ArmStage] = list(
        build_single_arm_place_sequence(
            place_position=place_position,
            place_orientation=grasp_orientation,
            place_direction=task_cfg.place_direction,
            place_direction_vector=task_cfg.place_direction_vector,
            place_approach_clearance=task_cfg.place_approach_clearance,
            place_insert_clearance=task_cfg.place_insert_clearance,
            post_release_retract_offset=(0, 0, 0),
            gripper_open=gripper_open,
            gripper_closed=gripper_closed,
            stage_prefix="PickPlaceFlow",
            start_index=5,
        )
    )
    return_stage_name = "PickPlaceFlow-5-ReturnHomeHold"
    arm_seq.extend(
        build_single_arm_return_home_sequence(
            home_pose=home_pose,
            gripper=gripper_open,
            stage_name=return_stage_name,
        )
    )
    return assign_to_arm(arm_seq, arm_side)

def build_single_arm_pick_apple_sequence(
    *,
    target_pose: Any,
    task_cfg: DrawerPickPlaceTaskConfig,
    arm_side: ArmSide,
    gripper_open: float,
    gripper_closed: float,
    grasp_orientation: tuple[float, float, float, float],
    grasp_direction: str,
    grasp_direction_vector: tuple[float, float, float] | None,
    place_position: tuple[float, float, float]
) -> list[StageTarget]:
    """Build a single-arm pick (+ optional place) + return-home sequence."""
    arm_seq: list[ArmStage] = list(build_single_arm_pick_sequence(
        target_pose=target_pose,
        approach_clearance=task_cfg.approach_clearance,
        grasp_clearance=task_cfg.grasp_clearance,
        grasp_orientation=grasp_orientation,
        grasp_direction=grasp_direction,
        grasp_direction_vector=grasp_direction_vector,
        grasp_offset=task_cfg.grasp_offset,
        retreat_direction_extra=task_cfg.retreat_direction_extra,
        retreat_offset=task_cfg.retreat_offset,
        gripper_open=gripper_open,
        gripper_closed=gripper_closed,
        stage_prefix="PickPlaceFlow",
    ))
    if task_cfg.place_orientation is None:
        raise ValueError("place_orientation required for pick_apple place segment after drawer open")
    arm_seq.extend(
        build_single_arm_place_sequence(
            place_position=place_position,
            place_orientation=task_cfg.place_orientation,
            place_direction=task_cfg.place_direction,
            place_direction_vector=task_cfg.place_direction_vector,
            place_approach_clearance=task_cfg.place_approach_clearance,
            place_insert_clearance=task_cfg.place_insert_clearance,
            post_release_retract_offset=task_cfg.post_release_retract_offset,
            gripper_open=gripper_open,
            gripper_closed=gripper_closed,
            stage_prefix="PickPlaceFlow",
            start_index=5,
        )
    )
    return assign_to_arm(arm_seq, arm_side)

def run_drawer_demo(
    *,
    robot_cfg: Any,
    task_cfg: DrawerPickPlaceTaskConfig,
    robot_id: str = "isaac_pick_place_flow",
    reset_env: bool = True,
    use_stamped: bool = False,
) -> None:
    robot = _make_robot(robot_cfg, robot_id)
    sim_time = SimTimeHelper()
    robot_connected = False

    def shutdown_handler(sig, frame) -> None:
        try:
            if robot_connected:
                robot.disconnect()
        finally:
            sim_time.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown_handler)

    try:
        robot.connect()
        robot_connected = True
        print("[OK] Robot connected")
        mode = task_cfg.mode.lower()
        if mode not in {"single_arm", "dual_arm"}:
            raise ValueError("mode must be 'single_arm' or 'dual_arm'")
        if mode == "dual_arm" and task_cfg.run_place_before_return:
            raise ValueError("run_place_before_return is currently only supported in single_arm mode")

        gripper_open, gripper_closed = _resolve_gripper_open_close(robot_cfg.gripper_control_mode)
        frame_id = robot_cfg.base_link_entity_path.rsplit("/", 1)[-1] if use_stamped else "arm_base"

        robot.ros2_interface.send_fsm_command(FSM_HOLD)
        sim_time.sleep(robot_cfg.fsm_switch_delay)
        robot.ros2_interface.send_fsm_command(FSM_OCS2)

        obs0 = robot.get_observation()
        base_world_pos, base_world_quat = get_entity_pose_world_service(robot_cfg.base_link_entity_path)

        if mode == "single_arm":
            initial_arm = task_cfg.initial_grasp_arm.lower()
            if initial_arm not in {"left", "right"}:
                raise ValueError("initial_grasp_arm must be 'left' or 'right'")
            source_is_right = initial_arm == "right"
            arm_side = ArmSide.RIGHT if source_is_right else ArmSide.LEFT
            source_ee_prefix = "right_ee" if source_is_right else "left_ee"
            source_home_pose = obs_to_pose(obs0, source_ee_prefix)
            source_object_path = task_cfg.source_object_entity_path
            source_object_path_drawer = task_cfg.source_object_path_drawer
            source_object_path_drawer_all = task_cfg.source_object_path_drawer_all
            if not source_object_path:
                raise ValueError("source_object_entity_path is required for single_arm mode")

            if reset_env:
                reset_simulation_and_randomize_object(
                    source_object_path,
                    xyz_offset=task_cfg.object_xyz_random_offset,
                    post_reset_wait=robot_cfg.post_reset_wait,
                    sleep_fn=sim_time.sleep,
                )
                rot_z = random.uniform(-0.1, 0.1)
                quat = euler_to_quaternion(0, 0, rot_z)
                set_prim_orientation_local(source_object_path_drawer_all, quat)
                randomize_object_xyz_after_reset(source_object_path_drawer_all,xyz_offset=task_cfg.object_xyz_random_offset_drawer)
                sim_time.sleep(1)

            # Pre-grasp: open gripper at home position.
            # Use the frame_id from the ee pose topic (observation frame),
            # NOT from base_link_entity_path.
            source_handler = (
                robot.ros2_interface.right_arm_handler if source_is_right
                else robot.ros2_interface.left_arm_handler
            )
            ee_frame_id = (source_handler.frame_id if source_handler else None) or frame_id
            pregrasp_target = ArmTarget(pose=source_home_pose, gripper=gripper_open)
            pregrasp_stage = assign_to_arm(
                [("pregrasp", pregrasp_target)], arm_side,
            )
            execute_stage_sequence(
                interface=robot.ros2_interface,
                sequence=pregrasp_stage,
                send_mode=SendMode.STAMPED if use_stamped else SendMode.UNSTAMPED,
                frame_id=ee_frame_id,
                arrival_timeout=robot_cfg.arrival_timeout,
                arrival_poll=robot_cfg.arrival_poll,
                time_now_fn=sim_time.now_seconds,
                sleep_fn=sim_time.sleep,
                gripper_action_wait=robot_cfg.gripper_action_wait,
            )

            # step1: pull drawer

            source_target_pose_d = get_object_pose_from_service(
                base_world_pos,
                base_world_quat,
                source_object_path_drawer,
                include_orientation=True,
            )
            Handle1_offset = ((task_cfg.handle_extent_max[0]+task_cfg.handle_extent_min[0])/2*task_cfg.drawer_scale,
                              (task_cfg.handle_extent_max[1]+task_cfg.handle_extent_min[1])/2*task_cfg.drawer_scale,
                              (task_cfg.handle_extent_max[2]+task_cfg.handle_extent_min[2])/2*task_cfg.drawer_scale,)

            def rotate_vector_by_quat(
                vec: tuple[float, float, float], quat_xyzw: tuple[float, float, float, float]
            ) -> tuple[float, float, float]:
                q = quat_normalize(quat_xyzw)
                q_inv = quat_conjugate(q)
                v_quat = (vec[0], vec[1], vec[2], 0.0)
                rotated = quat_multiply(quat_multiply(q, v_quat), q_inv)
                return (rotated[0], rotated[1], rotated[2])

            Handle1_offset = rotate_vector_by_quat(Handle1_offset, (
                float(source_target_pose_d.orientation.x),
                float(source_target_pose_d.orientation.y),
                float(source_target_pose_d.orientation.z),
                float(source_target_pose_d.orientation.w),
            ))

            _apply_target_pose_offset(source_target_pose_d, Handle1_offset)
            place_pose_ref = (
                float(source_target_pose_d.position.x),
                float(source_target_pose_d.position.y),
                float(source_target_pose_d.position.z),
            )
            print(place_pose_ref)
            source_grasp_orientation_drawer = quat_multiply((
                float(source_target_pose_d.orientation.x),
                float(source_target_pose_d.orientation.y),
                float(source_target_pose_d.orientation.z),
                float(source_target_pose_d.orientation.w),
            ), (0.5, 0.5, 0.5, -0.5))

            source_grasp_direction_vector_drawer = rotate_vector_by_quat((0, -1, 0), (
                float(source_target_pose_d.orientation.x),
                float(source_target_pose_d.orientation.y),
                float(source_target_pose_d.orientation.z),
                float(source_target_pose_d.orientation.w),
            ))
            sequence = build_single_arm_pull_drawer_sequence(
                target_pose=source_target_pose_d,
                task_cfg=task_cfg,
                arm_side=arm_side,
                gripper_open=gripper_open,
                gripper_closed=gripper_closed,
                grasp_orientation=source_grasp_orientation_drawer,
                grasp_direction_vector=source_grasp_direction_vector_drawer,
                )
            execute_stage_sequence(
                interface=robot.ros2_interface,
                sequence=sequence,
                send_mode=SendMode.STAMPED if use_stamped else SendMode.UNSTAMPED,
                frame_id=frame_id,
                arrival_timeout=robot_cfg.arrival_timeout,
                arrival_poll=robot_cfg.arrival_poll,
                time_now_fn=sim_time.now_seconds,
                sleep_fn=sim_time.sleep,
                gripper_action_wait=robot_cfg.gripper_action_wait,
                warn_prefix="PickPlaceFlow timeout",
            )
            # step2: pick and place apple
            source_target_pose = get_object_pose_from_service(
                base_world_pos,
                base_world_quat,
                source_object_path,
                include_orientation=False,
            )
            source_target_pose = _apply_target_pose_offset(source_target_pose, task_cfg.target_pose_offset)
            source_grasp_orientation, source_grasp_direction, source_grasp_direction_vector = (
                _resolve_arm_grasp_config(task_cfg, is_right=source_is_right)
            )
            place_pose = (place_pose_ref[0] + source_grasp_direction_vector_drawer[0]*0.04,
                          place_pose_ref[1] + source_grasp_direction_vector_drawer[1]*0.04,
                          place_pose_ref[2] + 0.15)
            sequence = build_single_arm_pick_apple_sequence(
                target_pose=source_target_pose,
                task_cfg=task_cfg,
                arm_side=arm_side,
                gripper_open=gripper_open,
                gripper_closed=gripper_closed,
                grasp_orientation=source_grasp_orientation,
                grasp_direction=source_grasp_direction,
                grasp_direction_vector=source_grasp_direction_vector,
                place_position=place_pose
            )
            if use_stamped:
                for stage in sequence:
                    if "ReturnHome" in stage.name:
                        stage.frame_id = ee_frame_id
            execute_stage_sequence(
                interface=robot.ros2_interface,
                sequence=sequence,
                send_mode=SendMode.STAMPED if use_stamped else SendMode.UNSTAMPED,
                frame_id=frame_id,
                arrival_timeout=robot_cfg.arrival_timeout,
                arrival_poll=robot_cfg.arrival_poll,
                time_now_fn=sim_time.now_seconds,
                sleep_fn=sim_time.sleep,
                gripper_action_wait=robot_cfg.gripper_action_wait,
                warn_prefix="PickPlaceFlow timeout",
            )

            # step3: close drawer
            source_target_pose_d = get_object_pose_from_service(
                base_world_pos,
                base_world_quat,
                source_object_path_drawer,
                include_orientation=True,
            )
            # source_grasp_direction_vector_drawer = rotate_vector_by_quat((0, -0.9961947, -0.0871557), (
            #     float(source_target_pose_d.orientation.x),
            #     float(source_target_pose_d.orientation.y),
            #     float(source_target_pose_d.orientation.z),
            #     float(source_target_pose_d.orientation.w),
            # ))
            source_grasp_orientation_drawer = quat_multiply(source_grasp_orientation_drawer, (0, -0.2164396, 0, 0.976296))
            _apply_target_pose_offset(source_target_pose_d, Handle1_offset)
            left_target = pose_from_tuple((
                float(source_target_pose_d.position.x),
                float(source_target_pose_d.position.y),
                float(source_target_pose_d.position.z),
            ), source_grasp_orientation_drawer)
            robot.ros2_interface.left_arm_handler.send_target_stamped("base_link", left_target)
            time.sleep(3)
            sequence = build_single_arm_close_drawer_sequence(target_pose=source_target_pose_d,
                home_pose=source_home_pose,
                task_cfg=task_cfg,
                arm_side=arm_side,
                gripper_open=gripper_open,
                gripper_closed=gripper_closed,
                grasp_orientation=source_grasp_orientation_drawer,
                grasp_direction_vector=source_grasp_direction_vector_drawer,
                )
            execute_stage_sequence(
                interface=robot.ros2_interface,
                sequence=sequence,
                send_mode=SendMode.STAMPED if use_stamped else SendMode.UNSTAMPED,
                frame_id=frame_id,
                arrival_timeout=robot_cfg.arrival_timeout,
                arrival_poll=robot_cfg.arrival_poll,
                time_now_fn=sim_time.now_seconds,
                sleep_fn=sim_time.sleep,
                gripper_action_wait=robot_cfg.gripper_action_wait,
                warn_prefix="PickPlaceFlow timeout",
            )
            left_target = pose_from_tuple(place_pose_ref, source_grasp_orientation_drawer)
            robot.ros2_interface.left_arm_handler.send_target_stamped("base_link", left_target)
            time.sleep(3)

            # step4: back_home
            sequence = build_single_arm_back_home_sequence(
                home_pose=source_home_pose,
                place_position=place_pose_ref,
                task_cfg=task_cfg,
                arm_side=arm_side,
                gripper_open=gripper_open,
                gripper_closed=gripper_closed,
                grasp_orientation=source_grasp_orientation_drawer,
                )
            execute_stage_sequence(
                interface=robot.ros2_interface,
                sequence=sequence,
                send_mode=SendMode.STAMPED if use_stamped else SendMode.UNSTAMPED,
                frame_id=frame_id,
                arrival_timeout=robot_cfg.arrival_timeout,
                arrival_poll=robot_cfg.arrival_poll,
                time_now_fn=sim_time.now_seconds,
                sleep_fn=sim_time.sleep,
                gripper_action_wait=robot_cfg.gripper_action_wait,
                warn_prefix="PickPlaceFlow timeout",
            )

        robot.ros2_interface.send_fsm_command(FSM_HOLD)
        print("[OK] PickPlaceFlow demo completed")
    except Exception as err:
        print(f"[ERROR] PickPlaceFlow failed: {err}")
    finally:
        sim_time.shutdown()
        if robot_connected:
            robot.disconnect()
            print("[OK] Robot disconnected")


def run_drawer_task_entry(
    *,
    robot_cfg: Any,
    base_task_cfg: DrawerPickPlaceTaskConfig,
    scene_presets: dict[str, dict[str, object]],
    cli_description: str,
    default_scene: str,
    robot_id: str,
) -> None:
    """CLI-style entry: scene preset → :class:`DrawerPickPlaceTaskConfig` → :func:`run_drawer_demo`."""
    scene, task_cfg = resolve_drawer_task_cfg_from_presets(
        base_task_cfg=base_task_cfg,
        scene_presets=scene_presets,
        cli_description=cli_description,
        default_scene=default_scene,
    )
    print(format_drawer_task_cfg_summary(scene, task_cfg))
    run_drawer_demo(
        robot_cfg=robot_cfg,
        task_cfg=task_cfg,
        robot_id=robot_id,
    )


# Backward compatibility for older scripts
run_pick_place_entry = run_drawer_task_entry
