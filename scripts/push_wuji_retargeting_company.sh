#!/usr/bin/env bash
# Push local company-main → fiveages-sim/wuji-retargeting main (FF from stub).
# Avoids full upstream LFS history (missing zhuliang.pkl). No --force.
set -euo pipefail
DIR="$(cd "$(dirname "$0")/../submodules/wuji-retargeting" && pwd)"
cd "$DIR"

git remote get-url origin | grep -q fiveages-sim/wuji-retargeting
git fetch origin
git checkout company-main

echo "Pushing company-main → origin/main (should FF from Initial commit)..."
git push -u origin company-main:main
echo
git ls-remote --heads origin
git log -2 --oneline
echo "OK. Then in parent repo:"
echo "  bash scripts/register_wuji_retargeting_submodule.sh"
