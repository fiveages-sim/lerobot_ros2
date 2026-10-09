#!/usr/bin/env bash
# Thin wrapper — canonical path is ./init.sh all-motion (same env as motion-generation).
# Kept so older docs / muscle memory still work.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
echo ">>> [setup_grasp_dev_env] delegating to ./init.sh all-motion"
exec bash "$ROOT/init.sh" all-motion
