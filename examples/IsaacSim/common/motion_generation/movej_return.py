#!/usr/bin/env python3
"""Shared MoveJ return-to-initial helpers for IsaacSim flows."""

from __future__ import annotations

import time
from typing import Any

from lerobot_robot_ros2 import ROS2Robot  # pyright: ignore[reportMissingImports]


def _extract_arm_positions_from_categorized(
    joint_state: dict[str, Any],
) -> tuple[list[float] | None, list[float] | None]:
    left_positions = joint_state.get("left_arm", {}).get("positions")
    right_positions = joint_state.get("right_arm", {}).get("positions")
    if left_positions is None:
        # Single-arm fallback.
        left_positions = joint_state.get("arm", {}).get("positions")
    left = list(left_positions) if left_positions else None
    right = list(right_positions) if right_positions else None
    return left, right


def _extract_arm_positions_from_raw(
    joint_state: dict[str, Any],
) -> tuple[list[float] | None, list[float] | None]:
    names = joint_state.get("names") or []
    positions = joint_state.get("positions") or []
    if not names or not positions or len(names) != len(positions):
        return None, None
    left: list[float] = []
    right: list[float] = []
    for name, pos in zip(names, positions):
        n = str(name).lower()
        # Exclude non-arm joints and grippers/hands.
        if "gripper" in n or "hand" in n or "head" in n or "body" in n:
            continue
        if not n.startswith("left_") and not n.startswith("right_"):
            continue
        if "joint" not in n:
            continue
        if n.startswith("left_"):
            left.append(float(pos))
        elif n.startswith("right_"):
            right.append(float(pos))
    return (left or None), (right or None)


def capture_initial_arm_joint_positions(
    robot: ROS2Robot,
    *,
    wait_timeout: float = 2.0,
    poll_period: float = 0.05,
) -> tuple[list[float] | None, list[float] | None]:
    """Capture current arm joint positions for post-task MoveJ return."""
    deadline = time.monotonic() + max(0.0, wait_timeout)
    while True:
        try:
            categorized = robot.ros2_interface.get_joint_state(categorized=True) or {}
            left, right = _extract_arm_positions_from_categorized(categorized)
            if left is not None or right is not None:
                return left, right
        except Exception:
            pass
        try:
            raw = robot.ros2_interface.get_joint_state(categorized=False) or {}
            left, right = _extract_arm_positions_from_raw(raw)
            if left is not None or right is not None:
                return left, right
        except Exception:
            pass
        if time.monotonic() >= deadline:
            return None, None
        time.sleep(max(0.0, poll_period))


def movej_return_to_initial_state(
    *,
    robot: ROS2Robot,
    left_initial_positions: list[float] | None,
    right_initial_positions: list[float] | None,
    arrival_timeout: float,
    arrival_poll: float,
    sim_time: Any,
) -> bool:
    """Best-effort MoveJ return to initial arm joint state."""
    interface = robot.ros2_interface
    moved = False
    left_count = len(left_initial_positions) if left_initial_positions else 0
    right_count = len(right_initial_positions) if right_initial_positions else 0
    if left_count == 0 and right_count == 0:
        print("[MoveJ] Skip: no cached initial arm joints.")
        return False

    # Reduce OCS2 interference before switching to MoveJ.
    try:
        interface.send_fsm_command(2)  # HOLD
        sim_time.sleep(0.1)
    except Exception as exc:
        print(f"[MoveJ] WARN: failed to switch HOLD before MoveJ: {exc}")

    print(f"[MoveJ] Returning to initial joints: left={left_count}, right={right_count}")

    # Try unified dual-arm command first, then gracefully fallback to per-arm commands.
    if left_initial_positions and right_initial_positions and interface.right_arm_handler is not None:
        try:
            interface.send_dual_arm_joint_positions(left_initial_positions, right_initial_positions)
            print("[MoveJ] Dual-arm return command sent.")
            moved = True
        except Exception as exc:
            print(f"[MoveJ] WARN: dual-arm MoveJ failed, fallback to per-arm: {exc}")

    if not moved and left_initial_positions and interface.left_arm_handler is not None:
        interface.left_arm_handler.send_joint_positions(left_initial_positions)
        moved = True
    if not moved and right_initial_positions and interface.right_arm_handler is not None:
        interface.right_arm_handler.send_joint_positions(right_initial_positions)
        moved = True
    if not moved and left_initial_positions and right_initial_positions:
        # Dual-arm publisher may be unavailable on some robots; fallback to per-arm both sides.
        if interface.left_arm_handler is not None:
            interface.left_arm_handler.send_joint_positions(left_initial_positions)
            moved = True
        if interface.right_arm_handler is not None:
            interface.right_arm_handler.send_joint_positions(right_initial_positions)
            moved = True
    if not moved:
        print("[MoveJ] WARN: no valid MoveJ publisher/handler available.")
        return False

    joint_wait = interface.wait_until_joint_arrive(
        left_target_positions=left_initial_positions,
        right_target_positions=right_initial_positions,
        timeout=max(1.0, float(arrival_timeout)),
        poll_period=max(0.02, float(arrival_poll)),
        joint_tolerance=0.03,
        time_now_fn=sim_time.now_seconds,
        sleep_fn=sim_time.sleep,
    )
    if joint_wait.get("arrived", False):
        print("[MoveJ] Joint arrival reached, switching to HOLD.")
    else:
        print(
            "[MoveJ] WARN: joint arrival timeout, force switching to HOLD. "
            f"left_err={joint_wait.get('left_error_max_abs')}, right_err={joint_wait.get('right_error_max_abs')}"
        )

    if moved and (left_initial_positions or right_initial_positions):
        try:
            interface.send_fsm_command(2)  # HOLD
            print("[MoveJ] Switched to HOLD after MoveJ.")
        except Exception as exc:
            print(f"[MoveJ] WARN: failed to switch HOLD after MoveJ: {exc}")
    if moved and (left_initial_positions and right_initial_positions):
        # already printed dual-arm message above
        pass
    elif moved:
        print("[MoveJ] Per-arm return command sent.")
    return moved
