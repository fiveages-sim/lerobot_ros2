"""
Compatibility shim for older imports.

Some downstream code (including lerobot_ros2) still imports ``lerobot.utils.constants``.
The constants live in ``lerobot.constants`` in recent versions, so we re-export them here.
"""

from lerobot.constants import *  # noqa: F401,F403

# Legacy defaults for processor pipeline names (not present in upstream constants).
POLICY_PREPROCESSOR_DEFAULT_NAME = "policy_preprocessor"
POLICY_POSTPROCESSOR_DEFAULT_NAME = "policy_postprocessor"
