#!/usr/bin/env bash
# After company main has the snapshot, register submodules/wuji-retargeting
# (non-recursive). Safe to re-run if already registered.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

URL="${WUJI_RETARGETING_URL:-git@github.com:fiveages-sim/wuji-retargeting.git}"

if git config -f .gitmodules --get-regexp path 2>/dev/null | grep -q 'submodules/wuji-retargeting'; then
  echo "already in .gitmodules"
  git submodule sync -- submodules/wuji-retargeting
  git submodule update --init -- submodules/wuji-retargeting
  exit 0
fi

# Replace local clone with proper gitlink
if [ -e submodules/wuji-retargeting ]; then
  echo "Moving local clone aside..."
  rm -rf /tmp/wuji-retargeting-bak-$$
  mv submodules/wuji-retargeting /tmp/wuji-retargeting-bak-$$
fi

git submodule add "$URL" submodules/wuji-retargeting
# Explicit: do NOT recurse nested mujoco-sim / wuji-description
git submodule update --init -- submodules/wuji-retargeting

echo
echo "Registered. Nested submodules left uninitialized (by design)."
echo "Optional: cd submodules/wuji-retargeting && git submodule update --init --recursive"
git status -sb | head -20
