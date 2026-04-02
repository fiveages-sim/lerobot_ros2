"""Runtime context passed between queue skills."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class DrawerPhaseState:
    """跨 ``single_arm.drawer.*`` 技能传递的会话状态（拉手 → 关抽屉 → 撤退）。

    与 **柜门、开关柜** 等同类任务：建议同样使用独立命名空间（例如未来的
    :class:`DoorPhaseState` + ``ctx.door``），避免往通用单臂上下文中平铺字段。
    """

    place_pose_ref: tuple[float, float, float]
    grasp_orientation_xyzw: tuple[float, float, float, float]
    grasp_direction_vector: tuple[float, float, float]
    handle_offset_rotated: tuple[float, float, float]


@dataclass
class SingleArmMotionContext:
    """Mutable state for a single-arm Isaac queue run (any robot using ``single_arm.*`` skills)."""

    robot: Any
    robot_cfg: Any
    sim_time: Any
    task_cfg: Any  # PickPlaceFlowTaskConfig once motion_generation.pick_place is loaded
    gripper_open: float
    gripper_closed: float
    use_stamped: bool
    frame_id: str
    ee_frame_id: str
    arm_side: Any  # ArmSide
    source_is_right: bool
    source_home_pose: Any
    base_world_pos: Any
    base_world_quat: Any
    # Gripper value to use for the final return-home segment (open after place, closed after pick-only).
    gripper_for_return_home: float = 0.0
    source_ee_prefix: str = "left_ee"
    # 仅抽屉任务队列使用；柜门等后续用独立子对象（如 ``ctx.door``）。
    drawer: DrawerPhaseState | None = None
    # 连接后由 Runner 写入，供 ``single_arm.movej_return_initial`` 使用。
    left_initial_joint_positions: list[float] | None = None
    right_initial_joint_positions: list[float] | None = None

    def __post_init__(self) -> None:
        if self.gripper_for_return_home == 0.0:
            self.gripper_for_return_home = float(self.gripper_closed)


@dataclass
class BaseBimanualMotionContext:
    """Shared mutable state for all bimanual Isaac task queue runs."""

    robot: Any
    robot_cfg: Any
    sim_time: Any
    task_cfg: Any
    gripper_open: float
    gripper_closed: float
    use_stamped: bool
    frame_id: str
    ee_frame_id: str
    base_world_pos: Any
    base_world_quat: Any
    left_initial_joint_positions: list[float] | None
    right_initial_joint_positions: list[float] | None


@dataclass
class BimanualMotionContext(BaseBimanualMotionContext):
    """Mutable state for bimanual carry task queue run."""

    task_cfg: Any  # BimanualCarryTaskConfig once motion_generation.bimanual_carry is loaded
    left_home_pose: Any
    right_home_pose: Any


@dataclass
class HandoverMotionContext(BaseBimanualMotionContext):
    """Mutable state for handover task queue run (a bimanual specialization)."""

    task_cfg: Any  # HandoverTaskConfig once motion_generation.handover is loaded
    source_is_right: bool
    source_home_pose: Any
    receiver_home_pose: Any
