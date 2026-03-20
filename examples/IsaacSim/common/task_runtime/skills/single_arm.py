"""Generic single-arm queue skills (pregrasp / pick / place / Cartesian home / MoveJ home).

Uses :class:`PickPlaceFlowTaskConfig` + :class:`SingleArmMotionContext`; any IsaacSim
单臂机器人在 ``motion_generation`` 中走 ``run_single_arm_task_queue`` 时均可复用
技能 ID ``single_arm.*``，与具体厂商无关。

``single_arm.movej_return_initial`` 使用 Runner 在连接时写入 ctx 的初始关节角缓存。
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ros2_robot_interface import (  # pyright: ignore[reportMissingImports]
    ArmTarget,
    SendMode,
    StageTarget,
    assign_to_arm,
    build_single_arm_pick_sequence,
    build_single_arm_place_sequence,
    build_single_arm_return_home_sequence,
)

from isaac_ros2_sim_common import get_object_pose_from_service  # pyright: ignore[reportMissingImports]

from motion_generation.pick_place import (  # pyright: ignore[reportMissingImports]
    _apply_target_pose_offset,
    _resolve_arm_grasp_config,
    resolve_pick_place_place_from_entity,
)
from motion_generation.movej_return import movej_return_to_initial_state  # pyright: ignore[reportMissingImports]

from task_runtime.context import SingleArmMotionContext
from task_runtime.registry import register_skill
from task_runtime.types import ExecutionMeta


def _stamped_mode(ctx: SingleArmMotionContext) -> SendMode:
    return SendMode.STAMPED if ctx.use_stamped else SendMode.UNSTAMPED


def skill_pregrasp(ctx: SingleArmMotionContext, _params: Mapping[str, Any]) -> tuple[list[StageTarget], ExecutionMeta]:
    pregrasp_target = ArmTarget(pose=ctx.source_home_pose, gripper=ctx.gripper_open)
    stages = assign_to_arm([("TaskQ-pregrasp", pregrasp_target)], ctx.arm_side)
    return stages, ExecutionMeta(
        send_mode=_stamped_mode(ctx),
        frame_id=ctx.ee_frame_id,
        warn_prefix="TaskQ pregrasp timeout",
    )


def skill_pick(ctx: SingleArmMotionContext, _params: Mapping[str, Any]) -> tuple[list[StageTarget], ExecutionMeta]:
    cfg = ctx.task_cfg
    path = cfg.source_object_entity_path
    if not path:
        raise ValueError("source_object_entity_path is required for skill single_arm.pick")
    target = get_object_pose_from_service(
        ctx.base_world_pos,
        ctx.base_world_quat,
        path,
        include_orientation=False,
    )
    target = _apply_target_pose_offset(target, cfg.target_pose_offset)
    grasp_orientation, grasp_direction, grasp_direction_vector = _resolve_arm_grasp_config(
        cfg, is_right=ctx.source_is_right
    )
    arm_seq = build_single_arm_pick_sequence(
        target_pose=target,
        approach_clearance=cfg.approach_clearance,
        grasp_clearance=cfg.grasp_clearance,
        grasp_orientation=grasp_orientation,
        grasp_direction=grasp_direction,
        grasp_direction_vector=grasp_direction_vector,
        grasp_offset=cfg.grasp_offset,
        retreat_direction_extra=cfg.retreat_direction_extra,
        retreat_offset=cfg.retreat_offset,
        gripper_open=ctx.gripper_open,
        gripper_closed=ctx.gripper_closed,
        stage_prefix="TaskQ-Pick",
    )
    stages = assign_to_arm(arm_seq, ctx.arm_side)
    ctx.gripper_for_return_home = ctx.gripper_closed
    return stages, ExecutionMeta(
        send_mode=_stamped_mode(ctx),
        frame_id=ctx.frame_id,
        warn_prefix="TaskQ pick timeout",
    )


def skill_place(ctx: SingleArmMotionContext, _params: Mapping[str, Any]) -> tuple[list[StageTarget], ExecutionMeta]:
    cfg = ctx.task_cfg
    if not cfg.run_place_before_return:
        return [], ExecutionMeta(
            send_mode=_stamped_mode(ctx),
            frame_id=ctx.frame_id,
            warn_prefix="TaskQ place skipped",
        )

    obs = ctx.robot.get_observation()
    ctx.task_cfg = resolve_pick_place_place_from_entity(
        ctx.task_cfg,
        base_world_pos=ctx.base_world_pos,
        base_world_quat=ctx.base_world_quat,
        current_obs=obs,
        ee_prefix_for_orientation_fallback=ctx.source_ee_prefix,
    )
    cfg = ctx.task_cfg
    if cfg.place_position is None or cfg.place_orientation is None:
        raise ValueError(
            "place_position and place_orientation are required when run_place_before_return=True "
            "(set explicitly or via place_object_entity_path)"
        )
    arm_seq = build_single_arm_place_sequence(
        place_position=cfg.place_position,
        place_orientation=cfg.place_orientation,
        place_direction=cfg.place_direction,
        place_direction_vector=cfg.place_direction_vector,
        place_approach_clearance=cfg.place_approach_clearance,
        place_insert_clearance=cfg.place_insert_clearance,
        post_release_retract_offset=cfg.post_release_retract_offset,
        gripper_open=ctx.gripper_open,
        gripper_closed=ctx.gripper_closed,
        stage_prefix="TaskQ-Place",
        start_index=1,
    )
    stages = assign_to_arm(arm_seq, ctx.arm_side)
    ctx.gripper_for_return_home = ctx.gripper_open
    return stages, ExecutionMeta(
        send_mode=_stamped_mode(ctx),
        frame_id=ctx.frame_id,
        warn_prefix="TaskQ place timeout",
    )


def skill_movej_return_initial(
    ctx: SingleArmMotionContext, _params: Mapping[str, Any]
) -> tuple[list[StageTarget], ExecutionMeta]:
    """关节空间回到 Runner 连接时缓存的初始臂角（无 StageTarget，在技能内执行 MoveJ）。"""
    rcfg = ctx.robot_cfg
    moved = movej_return_to_initial_state(
        robot=ctx.robot,
        left_initial_positions=ctx.left_initial_joint_positions,
        right_initial_positions=ctx.right_initial_joint_positions,
        arrival_timeout=rcfg.arrival_timeout,
        arrival_poll=rcfg.arrival_poll,
        sim_time=ctx.sim_time,
    )
    if not moved:
        print("[WARN] MoveJ return-to-initial skipped: no valid cached joints or motion failed.")
    return [], ExecutionMeta(
        send_mode=_stamped_mode(ctx),
        frame_id=ctx.frame_id,
        warn_prefix="TaskQ movej return",
    )


def skill_return_home(
    ctx: SingleArmMotionContext, _params: Mapping[str, Any]
) -> tuple[list[StageTarget], ExecutionMeta]:
    arm_seq = build_single_arm_return_home_sequence(
        home_pose=ctx.source_home_pose,
        gripper=ctx.gripper_for_return_home,
        stage_name="TaskQ-ReturnHomeHold",
    )
    stages = assign_to_arm(arm_seq, ctx.arm_side)
    if ctx.use_stamped:
        for st in stages:
            if "ReturnHome" in st.name:
                st.frame_id = ctx.ee_frame_id
    return stages, ExecutionMeta(
        send_mode=_stamped_mode(ctx),
        frame_id=ctx.frame_id,
        warn_prefix="TaskQ return home timeout",
    )


def register_single_arm_skills() -> None:
    register_skill("single_arm.pregrasp", skill_pregrasp)
    register_skill("single_arm.pick", skill_pick)
    register_skill("single_arm.place", skill_place)
    register_skill("single_arm.return_home", skill_return_home)
    register_skill("single_arm.movej_return_initial", skill_movej_return_initial)


register_single_arm_skills()
