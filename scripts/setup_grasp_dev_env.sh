#!/usr/bin/env bash
# Thin wrapper — full dex path is ./init.sh all-grasp.
# Motion-only (no HUG / no wuji-sdk): ./init.sh all-motion
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
echo ">>> [setup_grasp_dev_env] delegating to ./init.sh all-grasp"
echo ">>> 只要任务编排、不要灵巧手时，改跑: ./init.sh all-motion"
exec bash "$ROOT/init.sh" all-grasp
