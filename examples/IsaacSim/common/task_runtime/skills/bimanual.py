"""Built-in bimanual queue skills (``bimanual.*``)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ros2_robot_interface import (  # pyright: ignore[reportMissingImports]
    ArmTarget,
    SendMode,
    StageTarget,
)

from isaac_ros2_sim_common import get_object_pose_from_service  # pyright: ignore[reportMissingImports]

from motion_generation.bimanual_carry import (  # pyright: ignore[reportMissingImports]
    build_bimanual_carry_record_sequence,
)
from motion_generation.movej_return import movej_return_to_initial_state  # pyright: ignore[reportMissingImports]

from task_runtime.context import BimanualMotionContext
from task_runtime.registry import register_skill
from task_runtime.types import ExecutionMeta


def _bimanual_mode(ctx: BimanualMotionContext) -> SendMode:
    return SendMode.DUAL_ARM_STAMPED if ctx.use_stamped else SendMode.UNSTAMPED


def skill_pregrasp(
    ctx: BimanualMotionContext, _params: Mapping[str, Any]
) -> tuple[list[StageTarget], ExecutionMeta]:
    stages = [
        StageTarget(
            name="TaskQ-Bimanual-pregrasp",
            left=ArmTarget(pose=ctx.left_home_pose, gripper=ctx.gripper_open),
            right=ArmTarget(pose=ctx.right_home_pose, gripper=ctx.gripper_open),
        )
    ]
    return stages, ExecutionMeta(
        send_mode=_bimanual_mode(ctx),
        frame_id=ctx.ee_frame_id,
        warn_prefix="TaskQ bimanual pregrasp timeout",
    )


def skill_carry(
    ctx: BimanualMotionContext, _params: Mapping[str, Any]
) -> tuple[list[StageTarget], ExecutionMeta]:
    cfg = ctx.task_cfg
    object_center = get_object_pose_from_service(
        ctx.base_world_pos,
        ctx.base_world_quat,
        cfg.source_object_entity_path,
        include_orientation=False,
    )
    stages = build_bimanual_carry_record_sequence(
        carry_task_cfg=cfg,
        object_center=object_center,
        gripper_open=ctx.gripper_open,
        gripper_closed=ctx.gripper_closed,
    )
    return stages, ExecutionMeta(
        send_mode=_bimanual_mode(ctx),
        frame_id=ctx.frame_id,
        warn_prefix="TaskQ bimanual carry timeout",
    )


def skill_movej_return_initial(
    ctx: BimanualMotionContext, _params: Mapping[str, Any]
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
        send_mode=_bimanual_mode(ctx),
        frame_id=ctx.frame_id,
        warn_prefix="TaskQ bimanual movej return",
    )


def register_bimanual_skills() -> None:
    register_skill("bimanual.pregrasp", skill_pregrasp)
    register_skill("bimanual.carry", skill_carry)
    register_skill("bimanual.movej_return_initial", skill_movej_return_initial)


register_bimanual_skills()
