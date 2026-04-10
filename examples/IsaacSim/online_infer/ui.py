"""Interactive CLI helpers for inference launcher."""

from __future__ import annotations

from typing import Any


def select_option(*, title: str, options: dict[str, dict[str, Any]], default_key: str) -> str:
    keys = list(options.keys())
    print(f"\n{title}")
    for idx, key in enumerate(keys, start=1):
        label = options[key].get("label", key)
        suffix = " (default)" if key == default_key else ""
        print(f"  {idx}. {label} [{key}]{suffix}")
    raw = input("Select option (press Enter for default): ").strip()
    if raw == "":
        return default_key
    if raw.isdigit():
        idx = int(raw) - 1
        if 0 <= idx < len(keys):
            return keys[idx]
    if raw in options:
        return raw
    print(f"[info] Invalid option '{raw}', using default '{default_key}'.")
    return default_key
