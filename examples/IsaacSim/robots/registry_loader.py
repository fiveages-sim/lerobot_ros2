#!/usr/bin/env python3
"""Backward-compatible re-exports; implementation lives in ``robot_action_composer``."""

from __future__ import annotations

from robot_action_composer.discovery.registry_loader import load_motion_entries, load_robot_entries

__all__ = ["load_motion_entries", "load_robot_entries"]
