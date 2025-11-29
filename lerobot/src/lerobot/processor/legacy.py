#!/usr/bin/env python
"""
Legacy processor shims to keep older policy code working with newer LeRobot.

The upstream library recently refactored the processor API; this module
reintroduces a small subset of the previous interfaces (e.g. AddBatchDimension,
Normalizer/Unnormalizer, RenameObservations, PolicyProcessorPipeline) so code
in this repository can continue to import them without modification.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Generic, Iterable, List, Optional, TypeVar

import numpy as np
import torch
from torch import Tensor

InT = TypeVar("InT")
OutT = TypeVar("OutT")

# Simple alias used throughout the codebase.
PolicyAction = Dict[str, Any]


def _to_tensor(val: Any, device: str | torch.device | None = None) -> Tensor:
    if isinstance(val, torch.Tensor):
        return val.to(device=device)
    if isinstance(val, np.ndarray):
        return torch.from_numpy(val).to(device=device)
    return torch.as_tensor(val, device=device)


class AddBatchDimensionProcessorStep:
    """Ensure tensors carry a leading batch dimension."""

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        out = {}
        for k, v in data.items():
            if isinstance(v, torch.Tensor):
                out[k] = v if v.ndim > 0 else v.unsqueeze(0)
            elif isinstance(v, np.ndarray):
                t = torch.from_numpy(v)
                out[k] = t if t.ndim > 0 else t.unsqueeze(0)
            else:
                out[k] = v
        return out


class DeviceProcessorStep:
    """Move all tensors to the specified device."""

    def __init__(self, device: str | torch.device) -> None:
        self.device = device

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        out = {}
        for k, v in data.items():
            if isinstance(v, torch.Tensor):
                out[k] = v.to(self.device)
            elif isinstance(v, np.ndarray):
                out[k] = torch.from_numpy(v).to(self.device)
            else:
                out[k] = v
        return out


@dataclass
class _NormConfig:
    features: dict[str, Any]
    norm_map: dict[str, Any]
    stats: dict[str, dict[str, Any]] | None
    device: str | torch.device | None = None
    eps: float = 1e-8


class NormalizerProcessorStep:
    """Apply mean/std or min/max normalization using provided stats."""

    def __init__(
        self,
        features: dict[str, Any],
        norm_map: dict[str, Any],
        stats: dict[str, dict[str, Any]] | None = None,
        device: str | torch.device | None = None,
        eps: float = 1e-8,
    ) -> None:
        self.cfg = _NormConfig(features=features, norm_map=norm_map, stats=stats or {}, device=device, eps=eps)
        self._tensor_stats: dict[str, dict[str, Tensor]] = {}
        for key, sub in self.cfg.stats.items():
            self._tensor_stats[key] = {name: _to_tensor(val, device) for name, val in sub.items()}

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        out = dict(data)
        for key, stats in self._tensor_stats.items():
            if key not in out:
                continue
            tensor = _to_tensor(out[key], self.cfg.device).float()
            if "mean" in stats and "std" in stats:
                out[key] = (tensor - stats["mean"]) / (stats["std"] + self.cfg.eps)
            elif "min" in stats and "max" in stats:
                out[key] = 2 * (tensor - stats["min"]) / (stats["max"] - stats["min"] + self.cfg.eps) - 1
        return out


class UnnormalizerProcessorStep:
    """Inverse of NormalizerProcessorStep."""

    def __init__(
        self,
        features: dict[str, Any],
        norm_map: dict[str, Any],
        stats: dict[str, dict[str, Any]] | None = None,
        device: str | torch.device | None = None,
        eps: float = 1e-8,
    ) -> None:
        self.cfg = _NormConfig(features=features, norm_map=norm_map, stats=stats or {}, device=device, eps=eps)
        self._tensor_stats: dict[str, dict[str, Tensor]] = {}
        for key, sub in self.cfg.stats.items():
            self._tensor_stats[key] = {name: _to_tensor(val, device) for name, val in sub.items()}

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        out = dict(data)
        for key, stats in self._tensor_stats.items():
            if key not in out:
                continue
            tensor = _to_tensor(out[key], self.cfg.device).float()
            if "mean" in stats and "std" in stats:
                out[key] = tensor * (stats["std"] + self.cfg.eps) + stats["mean"]
            elif "min" in stats and "max" in stats:
                out[key] = (tensor + 1) * (stats["max"] - stats["min"]) / 2 + stats["min"]
        return out


class RenameObservationsProcessorStep:
    """Rename observation keys according to a provided map."""

    def __init__(self, rename_map: dict[str, str]) -> None:
        self.rename_map = rename_map or {}

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        out = {}
        for k, v in data.items():
            new_key = self.rename_map.get(k, k)
            out[new_key] = v
        return out


class PolicyProcessorPipeline(Generic[InT, OutT]):
    """Lightweight pipeline that chains processor steps."""

    def __init__(
        self,
        steps: Iterable[Callable[[Any], Any]] | None = None,
        name: str | None = None,
        to_transition: Callable[[InT], Any] | None = None,
        to_output: Callable[[Any], OutT] | None = None,
    ) -> None:
        self.steps: List[Callable[[Any], Any]] = list(steps) if steps is not None else []
        self.name = name or "processor_pipeline"
        self.to_transition = to_transition
        self.to_output = to_output

    def __call__(self, data: InT) -> OutT:
        x = self.to_transition(data) if self.to_transition else data
        for step in self.steps:
            x = step(x)
        x = self.to_output(x) if self.to_output else x
        return x  # type: ignore[return-value]

    def reset(self) -> None:
        for step in self.steps:
            if hasattr(step, "reset"):
                step.reset()

    def to(self, device: str | torch.device) -> "PolicyProcessorPipeline[InT, OutT]":
        for step in self.steps:
            if hasattr(step, "to"):
                step.to(device)
        return self

    def state_dict(self) -> dict[str, Any]:
        states: list[Any] = []
        for step in self.steps:
            if hasattr(step, "state_dict"):
                states.append(step.state_dict())
            else:
                states.append(None)
        return {"steps": states}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        for step, st in zip(self.steps, state.get("steps", [])):
            if st is not None and hasattr(step, "load_state_dict"):
                step.load_state_dict(st)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        config_filename: str | None = None,
        overrides: Optional[dict[str, Any]] = None,
        to_transition: Callable[[InT], Any] | None = None,
        to_output: Callable[[Any], OutT] | None = None,
    ) -> "PolicyProcessorPipeline[InT, OutT]":
        """
        Minimal loader to satisfy legacy API. It does not reconstruct full
        processor graphs; instead it returns an empty pipeline that can be
        customised via ``overrides`` if needed.
        """
        _ = (pretrained_model_name_or_path, config_filename)  # kept for API compatibility
        steps: list[Callable[[Any], Any]] = []
        if overrides:
            # Allow injecting steps directly via overrides["steps"] to keep
            # tests or custom loaders working.
            steps = overrides.get("steps", steps)
        return cls(steps=steps, name=config_filename, to_transition=to_transition, to_output=to_output)

