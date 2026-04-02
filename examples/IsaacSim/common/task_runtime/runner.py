"""Execute a linear single-arm task queue (vendor-agnostic skills: ``single_arm.*``)."""

from __future__ import annotations

import random
import signal
import sys
from typing import Any, Sequence

from ros2_robot_interface import (  # pyright: ignore[reportMissingImports]
    ArmSide,
    FSM_HOLD,
    FSM_OCS2,
    execute_stage_sequence,
)

from isaac_ros2_sim_common import (  # pyright: ignore[reportMissingImports]
    SimTimeHelper,
    get_entity_pose_world_service,
    randomize_object_xyz_after_reset,
    reset_simulation_and_randomize_object,
    set_prim_orientation_local,
)

from lerobot_robot_ros2 import ROS2Robot, ROS2RobotConfig  # pyright: ignore[reportMissingImports]
from lerobot_robot_ros2.utils import obs_to_pose  # pyright: ignore[reportMissingImports]

from motion_generation.movej_return import capture_initial_arm_joint_positions  # pyright: ignore[reportMissingImports]
from motion_generation.bimanual_carry import BimanualCarryTaskConfig  # pyright: ignore[reportMissingImports]
from motion_generation.drawer import DrawerPickPlaceTaskConfig, euler_to_quaternion  # pyright: ignore[reportMissingImports]
from motion_generation.handover import HandoverTaskConfig  # pyright: ignore[reportMissingImports]
from motion_generation.pick_place import PickPlaceFlowTaskConfig  # pyright: ignore[reportMissingImports]

from task_runtime.context import BimanualMotionContext, HandoverMotionContext, SingleArmMotionContext
from task_runtime.registry import get_skill
from task_runtime.types import BlockSpec, block_spec_from_mapping


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


def _resolve_gripper_open_close(mode: str) -> tuple[float, float]:
    if mode == "target_command":
        return 1.0, 0.0
    raise ValueError(
        "single-arm task queue only supports gripper_control_mode='target_command' "
        "without explicit gripper values"
    )


def run_single_arm_task_queue(
    *,
    robot_cfg: Any,
    task_cfg: PickPlaceFlowTaskConfig,
    robot_id: str,
    blocks: Sequence[BlockSpec | dict[str, Any]],
    reset_env: bool = True,
    use_stamped: bool = True,
) -> None:
    """Run ordered skills for one single-arm cycle (connect → FSM → blocks).

    ``task_cfg.mode`` must be ``single_arm``. Typical ``task_queue`` ends with
    Cartesian ``single_arm.return_home`` then optional ``single_arm.movej_return_initial``.

    Initial joint positions are cached on ``ctx`` at connect time for the MoveJ skill.
    """
    mode = task_cfg.mode.lower()
    if mode != "single_arm":
        raise ValueError(f"run_single_arm_task_queue requires mode='single_arm', got {task_cfg.mode!r}")

    specs: list[BlockSpec] = [b if isinstance(b, BlockSpec) else block_spec_from_mapping(b) for b in blocks]
    if not specs:
        raise ValueError("task queue blocks list is empty")

    robot = _make_robot(robot_cfg, robot_id)
    sim_time = SimTimeHelper()
    robot_connected = False

    def shutdown_handler(sig, frame) -> None:  # noqa: ARG001
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
        print("[OK] Robot connected (task queue)")
        left_initial_joint_positions, right_initial_joint_positions = capture_initial_arm_joint_positions(robot)

        gripper_open, gripper_closed = _resolve_gripper_open_close(robot_cfg.gripper_control_mode)
        frame_id = robot_cfg.base_link_entity_path.rsplit("/", 1)[-1] if use_stamped else "arm_base"

        robot.ros2_interface.send_fsm_command(FSM_HOLD)
        sim_time.sleep(robot_cfg.fsm_switch_delay)
        robot.ros2_interface.send_fsm_command(FSM_OCS2)

        obs0 = robot.get_observation()
        base_world_pos, base_world_quat = get_entity_pose_world_service(robot_cfg.base_link_entity_path)

        initial_arm = task_cfg.initial_grasp_arm.lower()
        if initial_arm not in {"left", "right"}:
            raise ValueError("initial_grasp_arm must be 'left' or 'right'")
        source_is_right = initial_arm == "right"
        arm_side = ArmSide.RIGHT if source_is_right else ArmSide.LEFT
        source_ee_prefix = "right_ee" if source_is_right else "left_ee"
        source_home_pose = obs_to_pose(obs0, source_ee_prefix)
        source_object_path = task_cfg.source_object_entity_path
        if not source_object_path:
            raise ValueError("source_object_entity_path is required for single-arm task queue")

        if reset_env:
            reset_simulation_and_randomize_object(
                source_object_path,
                xyz_offset=task_cfg.object_xyz_random_offset,
                post_reset_wait=robot_cfg.post_reset_wait,
                sleep_fn=sim_time.sleep,
            )

        source_handler = (
            robot.ros2_interface.right_arm_handler if source_is_right else robot.ros2_interface.left_arm_handler
        )
        ee_frame_id = (source_handler.frame_id if source_handler else None) or frame_id

        ctx = SingleArmMotionContext(
            robot=robot,
            robot_cfg=robot_cfg,
            sim_time=sim_time,
            task_cfg=task_cfg,
            gripper_open=gripper_open,
            gripper_closed=gripper_closed,
            use_stamped=use_stamped,
            frame_id=frame_id,
            ee_frame_id=ee_frame_id,
            arm_side=arm_side,
            source_is_right=source_is_right,
            source_home_pose=source_home_pose,
            base_world_pos=base_world_pos,
            base_world_quat=base_world_quat,
            gripper_for_return_home=gripper_closed,
            source_ee_prefix=source_ee_prefix,
            left_initial_joint_positions=left_initial_joint_positions,
            right_initial_joint_positions=right_initial_joint_positions,
        )

        for idx, spec in enumerate(specs):
            skill_fn = get_skill(spec.skill)
            stages, meta = skill_fn(ctx, spec.params)
            if not stages:
                print(f"[TaskQ] block {idx + 1}/{len(specs)} {spec.skill!r} -> (no stages / inline-only)")
                continue
            print(f"[TaskQ] block {idx + 1}/{len(specs)} {spec.skill!r} -> {len(stages)} stage(s)")
            execute_stage_sequence(
                interface=robot.ros2_interface,
                sequence=stages,
                send_mode=meta.send_mode,
                frame_id=meta.frame_id,
                arrival_timeout=robot_cfg.arrival_timeout,
                arrival_poll=robot_cfg.arrival_poll,
                time_now_fn=sim_time.now_seconds,
                sleep_fn=sim_time.sleep,
                gripper_action_wait=robot_cfg.gripper_action_wait,
                left_arrival_guard_stage=meta.left_arrival_guard_stage,
                warn_prefix=meta.warn_prefix,
            )

        robot.ros2_interface.send_fsm_command(FSM_HOLD)
        print("[OK] Task queue demo completed")
    except Exception as err:
        print(f"[ERROR] Task queue failed: {err}")
    finally:
        sim_time.shutdown()
        if robot_connected:
            robot.disconnect()
            print("[OK] Robot disconnected")


def run_drawer_pick_place_task_queue(
    *,
    robot_cfg: Any,
    task_cfg: DrawerPickPlaceTaskConfig,
    robot_id: str,
    blocks: Sequence[BlockSpec | dict[str, Any]],
    reset_env: bool = True,
    use_stamped: bool = True,
) -> None:
    """Drawer + apple pick/place as a task queue (``single_arm.drawer.*`` + shared pick/place).

    Environment reset matches :func:`motion_generation.drawer.run_drawer_demo`.
    Add ``single_arm.movej_return_initial`` last in ``task_queue`` for joint-space
    home (cached at connect on ``ctx``). Legacy ``run_drawer_demo`` has no MoveJ.
    """
    if not isinstance(task_cfg, DrawerPickPlaceTaskConfig):
        raise TypeError(f"run_drawer_pick_place_task_queue expects DrawerPickPlaceTaskConfig, got {type(task_cfg)}")
    mode = task_cfg.mode.lower()
    if mode != "single_arm":
        raise ValueError(f"drawer task queue currently requires mode='single_arm', got {task_cfg.mode!r}")

    specs: list[BlockSpec] = [b if isinstance(b, BlockSpec) else block_spec_from_mapping(b) for b in blocks]
    if not specs:
        raise ValueError("task queue blocks list is empty")

    robot = _make_robot(robot_cfg, robot_id)
    sim_time = SimTimeHelper()
    robot_connected = False

    def shutdown_handler(sig, frame) -> None:  # noqa: ARG001
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
        print("[OK] Robot connected (drawer task queue)")
        left_initial_joint_positions, right_initial_joint_positions = capture_initial_arm_joint_positions(robot)

        gripper_open, gripper_closed = _resolve_gripper_open_close(robot_cfg.gripper_control_mode)
        frame_id = robot_cfg.base_link_entity_path.rsplit("/", 1)[-1] if use_stamped else "arm_base"

        robot.ros2_interface.send_fsm_command(FSM_HOLD)
        sim_time.sleep(robot_cfg.fsm_switch_delay)
        robot.ros2_interface.send_fsm_command(FSM_OCS2)

        obs0 = robot.get_observation()
        base_world_pos, base_world_quat = get_entity_pose_world_service(robot_cfg.base_link_entity_path)

        initial_arm = task_cfg.initial_grasp_arm.lower()
        if initial_arm not in {"left", "right"}:
            raise ValueError("initial_grasp_arm must be 'left' or 'right'")
        source_is_right = initial_arm == "right"
        arm_side = ArmSide.RIGHT if source_is_right else ArmSide.LEFT
        source_ee_prefix = "right_ee" if source_is_right else "left_ee"
        source_home_pose = obs_to_pose(obs0, source_ee_prefix)
        apple_path = task_cfg.source_object_entity_path
        drawer_prim = task_cfg.source_object_path_drawer
        drawer_all = task_cfg.source_object_path_drawer_all
        if not apple_path or not drawer_prim or not drawer_all:
            raise ValueError(
                "drawer task queue requires source_object_entity_path, "
                "source_object_path_drawer, source_object_path_drawer_all"
            )

        if reset_env:
            reset_simulation_and_randomize_object(
                apple_path,
                xyz_offset=task_cfg.object_xyz_random_offset,
                post_reset_wait=robot_cfg.post_reset_wait,
                sleep_fn=sim_time.sleep,
            )
            rot_z = random.uniform(-0.1, 0.1)
            set_prim_orientation_local(drawer_all, euler_to_quaternion(0, 0, rot_z))
            randomize_object_xyz_after_reset(
                drawer_all,
                xyz_offset=task_cfg.object_xyz_random_offset_drawer,
            )
            sim_time.sleep(1)

        source_handler = (
            robot.ros2_interface.right_arm_handler if source_is_right else robot.ros2_interface.left_arm_handler
        )
        ee_frame_id = (source_handler.frame_id if source_handler else None) or frame_id

        ctx = SingleArmMotionContext(
            robot=robot,
            robot_cfg=robot_cfg,
            sim_time=sim_time,
            task_cfg=task_cfg,
            gripper_open=gripper_open,
            gripper_closed=gripper_closed,
            use_stamped=use_stamped,
            frame_id=frame_id,
            ee_frame_id=ee_frame_id,
            arm_side=arm_side,
            source_is_right=source_is_right,
            source_home_pose=source_home_pose,
            base_world_pos=base_world_pos,
            base_world_quat=base_world_quat,
            gripper_for_return_home=gripper_closed,
            source_ee_prefix=source_ee_prefix,
            left_initial_joint_positions=left_initial_joint_positions,
            right_initial_joint_positions=right_initial_joint_positions,
        )

        for idx, spec in enumerate(specs):
            skill_fn = get_skill(spec.skill)
            stages, meta = skill_fn(ctx, spec.params)
            if not stages:
                print(f"[TaskQ/drawer] block {idx + 1}/{len(specs)} {spec.skill!r} -> (no stages / inline-only)")
                continue
            print(f"[TaskQ/drawer] block {idx + 1}/{len(specs)} {spec.skill!r} -> {len(stages)} stage(s)")
            execute_stage_sequence(
                interface=robot.ros2_interface,
                sequence=stages,
                send_mode=meta.send_mode,
                frame_id=meta.frame_id,
                arrival_timeout=robot_cfg.arrival_timeout,
                arrival_poll=robot_cfg.arrival_poll,
                time_now_fn=sim_time.now_seconds,
                sleep_fn=sim_time.sleep,
                gripper_action_wait=robot_cfg.gripper_action_wait,
                left_arrival_guard_stage=meta.left_arrival_guard_stage,
                warn_prefix=meta.warn_prefix,
            )

        robot.ros2_interface.send_fsm_command(FSM_HOLD)
        print("[OK] Drawer task queue demo completed")
    except Exception as err:
        print(f"[ERROR] Drawer task queue failed: {err}")
    finally:
        sim_time.shutdown()
        if robot_connected:
            robot.disconnect()
            print("[OK] Robot disconnected")


def run_bimanual_task_queue(
    *,
    robot_cfg: Any,
    task_cfg: BimanualCarryTaskConfig,
    robot_id: str,
    blocks: Sequence[BlockSpec | dict[str, Any]],
    reset_env: bool = True,
    use_stamped: bool = True,
) -> None:
    """Run ordered skills for one bimanual cycle (connect → FSM → blocks)."""
    specs: list[BlockSpec] = [b if isinstance(b, BlockSpec) else block_spec_from_mapping(b) for b in blocks]
    if not specs:
        raise ValueError("task queue blocks list is empty")

    robot = _make_robot(robot_cfg, robot_id)
    sim_time = SimTimeHelper()
    robot_connected = False

    def shutdown_handler(sig, frame) -> None:  # noqa: ARG001
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
        print("[OK] Robot connected (bimanual task queue)")
        left_initial_joint_positions, right_initial_joint_positions = capture_initial_arm_joint_positions(robot)
        gripper_open, gripper_closed = _resolve_gripper_open_close(robot_cfg.gripper_control_mode)
        frame_id = robot_cfg.base_link_entity_path.rsplit("/", 1)[-1] if use_stamped else "arm_base"

        if reset_env:
            reset_simulation_and_randomize_object(
                task_cfg.source_object_entity_path,
                xyz_offset=task_cfg.object_xyz_random_offset,
                post_reset_wait=robot_cfg.post_reset_wait,
                sleep_fn=sim_time.sleep,
            )

        robot.ros2_interface.send_fsm_command(FSM_HOLD)
        sim_time.sleep(robot_cfg.fsm_switch_delay)
        robot.ros2_interface.send_fsm_command(FSM_OCS2)

        obs0 = robot.get_observation()
        left_home_pose = obs_to_pose(obs0, "left_ee")
        right_home_pose = obs_to_pose(obs0, "right_ee")
        base_world_pos, base_world_quat = get_entity_pose_world_service(robot_cfg.base_link_entity_path)
        left_handler = robot.ros2_interface.left_arm_handler
        ee_frame_id = (left_handler.frame_id if left_handler else None) or frame_id

        ctx = BimanualMotionContext(
            robot=robot,
            robot_cfg=robot_cfg,
            sim_time=sim_time,
            task_cfg=task_cfg,
            gripper_open=gripper_open,
            gripper_closed=gripper_closed,
            use_stamped=use_stamped,
            frame_id=frame_id,
            ee_frame_id=ee_frame_id,
            left_home_pose=left_home_pose,
            right_home_pose=right_home_pose,
            base_world_pos=base_world_pos,
            base_world_quat=base_world_quat,
            left_initial_joint_positions=left_initial_joint_positions,
            right_initial_joint_positions=right_initial_joint_positions,
        )

        for idx, spec in enumerate(specs):
            skill_fn = get_skill(spec.skill)
            stages, meta = skill_fn(ctx, spec.params)
            if not stages:
                print(
                    f"[TaskQ/bimanual] block {idx + 1}/{len(specs)} {spec.skill!r} "
                    "-> (no stages / inline-only)"
                )
                continue
            print(f"[TaskQ/bimanual] block {idx + 1}/{len(specs)} {spec.skill!r} -> {len(stages)} stage(s)")
            execute_stage_sequence(
                interface=robot.ros2_interface,
                sequence=stages,
                send_mode=meta.send_mode,
                frame_id=meta.frame_id,
                arrival_timeout=robot_cfg.arrival_timeout,
                arrival_poll=robot_cfg.arrival_poll,
                time_now_fn=sim_time.now_seconds,
                sleep_fn=sim_time.sleep,
                gripper_action_wait=robot_cfg.gripper_action_wait,
                left_arrival_guard_stage=meta.left_arrival_guard_stage,
                warn_prefix=meta.warn_prefix,
            )

        robot.ros2_interface.send_fsm_command(FSM_HOLD)
        print("[OK] Bimanual task queue demo completed")
    except Exception as err:
        print(f"[ERROR] Bimanual task queue failed: {err}")
    finally:
        sim_time.shutdown()
        if robot_connected:
            robot.disconnect()
            print("[OK] Robot disconnected")


def run_handover_task_queue(
    *,
    robot_cfg: Any,
    task_cfg: HandoverTaskConfig,
    robot_id: str,
    blocks: Sequence[BlockSpec | dict[str, Any]],
    reset_env: bool = True,
    use_stamped: bool = True,
) -> None:
    """Run ordered skills for one bimanual handover cycle (connect → FSM → blocks)."""
    specs: list[BlockSpec] = [b if isinstance(b, BlockSpec) else block_spec_from_mapping(b) for b in blocks]
    if not specs:
        raise ValueError("task queue blocks list is empty")

    robot = _make_robot(robot_cfg, robot_id)
    sim_time = SimTimeHelper()
    robot_connected = False

    def shutdown_handler(sig, frame) -> None:  # noqa: ARG001
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
        print("[OK] Robot connected (handover task queue)")
        left_initial_joint_positions, right_initial_joint_positions = capture_initial_arm_joint_positions(robot)
        initial_arm = task_cfg.initial_grasp_arm.lower()
        if initial_arm not in {"left", "right"}:
            raise ValueError("initial_grasp_arm must be 'left' or 'right'")
        source_is_right = initial_arm == "right"
        gripper_open, gripper_closed = _resolve_gripper_open_close(robot_cfg.gripper_control_mode)
        frame_id = robot_cfg.base_link_entity_path.rsplit("/", 1)[-1] if use_stamped else "arm_base"

        if reset_env:
            reset_simulation_and_randomize_object(
                task_cfg.source_object_entity_path,
                xyz_offset=task_cfg.object_xyz_random_offset,
                post_reset_wait=robot_cfg.post_reset_wait,
                sleep_fn=sim_time.sleep,
            )

        robot.ros2_interface.send_fsm_command(FSM_HOLD)
        sim_time.sleep(robot_cfg.fsm_switch_delay)
        robot.ros2_interface.send_fsm_command(FSM_OCS2)

        obs0 = robot.get_observation()
        left_home_pose = obs_to_pose(obs0, "left_ee")
        right_home_pose = obs_to_pose(obs0, "right_ee")
        source_home_pose = right_home_pose if source_is_right else left_home_pose
        receiver_home_pose = left_home_pose if source_is_right else right_home_pose
        source_handler = (
            robot.ros2_interface.right_arm_handler if source_is_right else robot.ros2_interface.left_arm_handler
        )
        ee_frame_id = (source_handler.frame_id if source_handler else None) or frame_id
        base_world_pos, base_world_quat = get_entity_pose_world_service(robot_cfg.base_link_entity_path)

        ctx = HandoverMotionContext(
            robot=robot,
            robot_cfg=robot_cfg,
            sim_time=sim_time,
            task_cfg=task_cfg,
            gripper_open=gripper_open,
            gripper_closed=gripper_closed,
            use_stamped=use_stamped,
            frame_id=frame_id,
            ee_frame_id=ee_frame_id,
            source_is_right=source_is_right,
            source_home_pose=source_home_pose,
            receiver_home_pose=receiver_home_pose,
            base_world_pos=base_world_pos,
            base_world_quat=base_world_quat,
            left_initial_joint_positions=left_initial_joint_positions,
            right_initial_joint_positions=right_initial_joint_positions,
        )

        for idx, spec in enumerate(specs):
            skill_fn = get_skill(spec.skill)
            stages, meta = skill_fn(ctx, spec.params)
            if not stages:
                print(
                    f"[TaskQ/handover] block {idx + 1}/{len(specs)} {spec.skill!r} "
                    "-> (no stages / inline-only)"
                )
                continue
            print(f"[TaskQ/handover] block {idx + 1}/{len(specs)} {spec.skill!r} -> {len(stages)} stage(s)")
            execute_stage_sequence(
                interface=robot.ros2_interface,
                sequence=stages,
                send_mode=meta.send_mode,
                frame_id=meta.frame_id,
                arrival_timeout=robot_cfg.arrival_timeout,
                arrival_poll=robot_cfg.arrival_poll,
                time_now_fn=sim_time.now_seconds,
                sleep_fn=sim_time.sleep,
                gripper_action_wait=robot_cfg.gripper_action_wait,
                left_arrival_guard_stage=meta.left_arrival_guard_stage,
                warn_prefix=meta.warn_prefix,
            )

        robot.ros2_interface.send_fsm_command(FSM_HOLD)
        print("[OK] Handover task queue demo completed")
    except Exception as err:
        print(f"[ERROR] Handover task queue failed: {err}")
    finally:
        sim_time.shutdown()
        if robot_connected:
            robot.disconnect()
            print("[OK] Robot disconnected")