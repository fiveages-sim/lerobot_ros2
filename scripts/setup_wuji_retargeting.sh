#!/usr/bin/env bash
# Prepare Wuji retargeting deps (like HUG local setup).
# - Ensure company submodule / clone under submodules/wuji-retargeting (NO nested recurse)
# - pip install wuji-sdk into parent .venv (product path)
# - Optional: --editable-oss / --recursive-nested
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OSS="${ROOT}/submodules/wuji-retargeting"
UPSTREAM_URL="${WUJI_RETARGETING_URL:-git@github.com:fiveages-sim/wuji-retargeting.git}"
EDITABLE_OSS=0
RECURSIVE=0

for arg in "$@"; do
  case "$arg" in
    --editable-oss) EDITABLE_OSS=1 ;;
    --recursive-nested) RECURSIVE=1 ;;
    -h|--help)
      echo "Usage: $0 [--editable-oss] [--recursive-nested]"
      exit 0
      ;;
  esac
done

if [[ ! -d "${OSS}/.git" && ! -f "${OSS}/.git" ]]; then
  echo "[setup] cloning ${UPSTREAM_URL} → ${OSS} (no recurse)"
  git clone "${UPSTREAM_URL}" "${OSS}"
else
  echo "[setup] OSS present @ $(git -C "${OSS}" rev-parse --short HEAD)"
fi

if [[ "${RECURSIVE}" -eq 1 ]]; then
  echo "[setup] init nested submodules (mujoco-sim / wuji-description)"
  git -C "${OSS}" submodule update --init --recursive
else
  echo "[setup] skip nested submodules (pass --recursive-nested for MuJoCo/tuning)"
fi

VENV="${ROOT}/.venv"
if [[ ! -x "${VENV}/bin/python" ]]; then
  echo "[setup] missing ${VENV}; create parent venv first" >&2
  exit 1
fi

# shellcheck disable=SC1091
source "${VENV}/bin/activate"
python -m pip install -U 'wuji-sdk>=0.10.0' numpy

python - <<'PY'
import numpy as np
from wuji_sdk import Handedness, HandModel, RetargetSession

kp = np.zeros((21, 3), dtype=np.float32)
kp[4] = [0.05, 0.05, 0.03]
for i, x in enumerate([0.0, 0.02, 0.04, 0.06]):
    base = 5 + i * 4
    for j, z in enumerate([0.03, 0.05, 0.07, 0.09]):
        kp[base + j] = [x, 0.0, z]
session = RetargetSession.for_hand(HandModel.WujiHand2, side=Handedness.Right)
q = session.step(kp)
assert q.shape == (20,), q.shape
print("[setup] wuji-sdk RetargetSession OK → q.shape", q.shape)
PY

if [[ "${EDITABLE_OSS}" -eq 1 ]]; then
  echo "[setup] editable OSS install (heavy)"
  python -m pip install -U pip
  python -m pip install -r "${OSS}/requirements.txt"
  python -m pip install -e "${OSS}"
fi

echo "[setup] done. See docs/WUJI_RETARGETING_SUBMODULE.md"
