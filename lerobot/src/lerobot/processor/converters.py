#!/usr/bin/env python
"""
Compatibility converters for legacy policy processor API.

Recent versions of LeRobot refactored the processor module, so downstream
code (including this repo) that expects ``lerobot.processor.converters`` will
otherwise fail to import. The helpers below provide a minimal bridge by
mapping between flat batch dicts and transition/policy-action dictionaries.
"""

from __future__ import annotations

from typing import Any, Dict

import torch

# Transition representation used by legacy policy processors.
Transition = Dict[str, Any]


def batch_to_transition(batch: dict[str, Any]) -> Transition:
    """Convert a flat batch dict (observation.*, action, etc.) into a transition dict."""
    obs = {k: v for k, v in batch.items() if k.startswith("observation.")}
    # Strip the prefix for readability in downstream code.
    obs = {k.replace("observation.", ""): v for k, v in obs.items()}

    transition: Transition = {
        "observation": obs if obs else None,
        "action": batch.get("action"),
        "reward": batch.get("next.reward"),
        "done": batch.get("next.done"),
        "truncated": batch.get("next.truncated"),
        "info": batch.get("info"),
    }
    return transition


def transition_to_batch(transition: Transition) -> dict[str, Any]:
    """Inverse of ``batch_to_transition``."""
    batch: dict[str, Any] = {}
    obs = transition.get("observation") or {}
    for k, v in obs.items():
        batch[f"observation.{k}"] = v
    if "action" in transition:
        batch["action"] = transition["action"]
    if "reward" in transition:
        batch["next.reward"] = transition["reward"]
    if "done" in transition:
        batch["next.done"] = transition["done"]
    if "truncated" in transition:
        batch["next.truncated"] = transition["truncated"]
    if "info" in transition:
        batch["info"] = transition["info"]
    return batch


def policy_action_to_transition(policy_action: dict[str, Any]) -> Transition:
    """
    Wrap a policy action dict into a transition. The legacy format places the
    action tensor under the "action" key.
    """
    if policy_action is None:
        return {"observation": None, "action": None, "reward": None, "done": None, "truncated": None, "info": None}
    if "action" in policy_action:
        action = policy_action["action"]
    else:
        # Allow passing a bare tensor.
        action = policy_action
    return {"observation": None, "action": action, "reward": None, "done": None, "truncated": None, "info": None}


def transition_to_policy_action(transition: Transition) -> dict[str, Any]:
    """Extract the action field from a transition and wrap it in a dict."""
    action = transition.get("action")
    return {"action": action}

