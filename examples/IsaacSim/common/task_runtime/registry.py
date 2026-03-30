"""Skill registry for task queue."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from task_runtime.context import SingleArmMotionContext

from task_runtime.types import ExecutionMeta

SkillFn = Callable[["SingleArmMotionContext", Mapping[str, Any]], tuple[list[Any], ExecutionMeta]]

_REGISTRY: dict[str, SkillFn] = {}


def register_skill(name: str, fn: SkillFn) -> None:
    if name in _REGISTRY:
        raise ValueError(f"Skill already registered: {name}")
    _REGISTRY[name] = fn


def get_skill(name: str) -> SkillFn:
    if name not in _REGISTRY:
        known = ", ".join(sorted(_REGISTRY)) or "(none)"
        raise KeyError(f"Unknown skill {name!r}. Registered: {known}")
    return _REGISTRY[name]


def registered_skill_names() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))
