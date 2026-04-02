"""Built-in handover queue skills (``handover.*``)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ros2_robot_interface import (  # pyright: ignore[reportMissingImports]
    ArmSide,
    ArmTarget,
    SendMode,
    StageTarget,
    assign_to_arm,
)

from isaac_ros2_sim_common import get_object_pose_from_service  # pyright: ignore[reportMissingImports]

from motion_generation.handover import build_handover_record_sequence  # pyright: ignore[reportMissingImports]
from motion_generation.movej_return import movej_return_to_initial_state  # pyright: ignore[reportMissingImports]

from task_runtime.context import HandoverMotionContext
from task_runtime.registry import register_skill
from task_runtime.types import ExecutionMeta


def _source_arm(ctx: HandoverMotionContext) -> ArmSide:
    return ArmSide.RIGHT if ctx.source_is_right else ArmSide.LEFT


def _single_mode(ctx: HandoverMotionContext) -> SendMode:
    return SendMode.STAMPED if ctx.use_stamped else SendMode.UNSTAMPED


def _dual_mode(ctx: HandoverMotionContext) -> SendMode:
    return SendMode.DUAL_ARM_STAMPED if ctx.use_stamped else SendMode.UNSTAMPED


def _full_sequence(ctx: HandoverMotionContext) -> list[StageTarget]:
    target = get_object_pose_from_service(
        ctx.base_world_pos,
        ctx.base_world_quat,
        ctx.task_cfg.source_object_entity_path,
        include_orientation=False,
    )
    return build_handover_record_sequence(
        handover_task_cfg=ctx.task_cfg,
        source_target_pose=target,
        source_home_pose=ctx.source_home_pose,
        receiver_home_pose=ctx.receiver_home_pose,
        source_is_right=ctx.source_is_right,
        gripper_open=ctx.gripper_open,
        gripper_closed=ctx.gripper_closed,
    )


def skill_pregrasp(
    ctx: HandoverMotionContext, _params: Mapping[str, Any]
) -> tuple[list[StageTarget], ExecutionMeta]:
    stage = assign_to_arm(
        [("TaskQ-Handover-pregrasp", ArmTarget(pose=ctx.source_home_pose, gripper=ctx.gripper_open))],
        _source_arm(ctx),
    )
    return stage, ExecutionMeta(
        send_mode=_single_mode(ctx),
        frame_id=ctx.ee_frame_id,
        warn_prefix="TaskQ handover pregrasp timeout",
    )


def skill_pick(
    ctx: HandoverMotionContext, _params: Mapping[str, Any]
) -> tuple[list[StageTarget], ExecutionMeta]:
    pickup = [s for s in _full_sequence(ctx) if s.name.startswith("Pickup-")]
    return pickup, ExecutionMeta(
        send_mode=_single_mode(ctx),
        frame_id=ctx.frame_id,
        warn_prefix="TaskQ handover pickup timeout",
    )


def skill_exchange_place(
    ctx: HandoverMotionContext, _params: Mapping[str, Any]
) -> tuple[list[StageTarget], ExecutionMeta]:
    stages = [s for s in _full_sequence(ctx) if not s.name.startswith("Pickup-")]
    if ctx.use_stamped:
        for stage in stages:
            if "ReturnHome" in stage.name:
                stage.frame_id = ctx.ee_frame_id
    return stages, ExecutionMeta(
        send_mode=_dual_mode(ctx),
        frame_id=ctx.frame_id,
        left_arrival_guard_stage="Handover-1-SyncMove" if ctx.source_is_right else None,
        warn_prefix="TaskQ handover exchange/place timeout",
    )


def skill_movej_return_initial(
    ctx: HandoverMotionContext, _params: Mapping[str, Any]
) -> tuple[list[StageTarget], ExecutionMeta]:
    moved = movej_return_to_initial_state(
        robot=ctx.robot,
        left_initial_positions=ctx.left_initial_joint_positions,
        right_initial_positions=ctx.right_initial_joint_positions,
        arrival_timeout=ctx.robot_cfg.arrival_timeout,
        arrival_poll=ctx.robot_cfg.arrival_poll,
        sim_time=ctx.sim_time,
    )
    if not moved:
        print("[WARN] MoveJ return-to-initial skipped: no valid cached joints or motion failed.")
    return [], ExecutionMeta(
        send_mode=_dual_mode(ctx),
        frame_id=ctx.frame_id,
        warn_prefix="TaskQ handover movej return",
    )


def register_handover_skills() -> None:
    register_skill("handover.pregrasp", skill_pregrasp)
    register_skill("handover.pick", skill_pick)
    register_skill("handover.exchange_place", skill_exchange_place)
    register_skill("handover.movej_return_initial", skill_movej_return_initial)


register_handover_skills()
