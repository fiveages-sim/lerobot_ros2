"""Shared types for IsaacSim task queue runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from ros2_robot_interface import SendMode  # pyright: ignore[reportMissingImports]


@dataclass(frozen=True)
class ExecutionMeta:
    """Per-block execution options for :func:`execute_stage_sequence`."""

    send_mode: SendMode
    frame_id: str
    warn_prefix: str = "TaskQueue"
    left_arrival_guard_stage: str | None = None


@dataclass(frozen=True)
class BlockSpec:
    """One executable step in a task queue.

    ``params`` can override skill-specific options without changing global ``task_cfg``
    (future); current Dobot skills read mostly from :class:`PickPlaceFlowTaskConfig`.
    """

    skill: str
    params: Mapping[str, Any] = field(default_factory=dict)


def block_spec_from_mapping(raw: Mapping[str, Any]) -> BlockSpec:
    if not isinstance(raw, Mapping):
        raise TypeError(f"block spec must be a mapping, got {type(raw).__name__}")
    sk = raw.get("skill")
    if not isinstance(sk, str) or not sk.strip():
        raise ValueError('block spec requires non-empty string "skill"')
    params = raw.get("params", {})
    if params is not None and not isinstance(params, Mapping):
        raise TypeError('"params" must be a mapping when present')
    return BlockSpec(skill=sk.strip(), params=dict(params or {}))
