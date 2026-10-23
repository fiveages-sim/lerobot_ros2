#!/usr/bin/env bash
# Conservative: only ls-remote + plain push to company HUG. No --force.
set -euo pipefail
REPO="${1:-git@github.com:fiveages-sim/HUG.git}"
HUG_DIR="$(cd "$(dirname "$0")/../submodules/hug" && pwd)"
cd "$HUG_DIR"

echo "== remotes =="
git remote -v
echo
echo "== local HEAD =="
git log -2 --oneline
echo
echo "== ls-remote (read-only) =="
git ls-remote "$REPO"
echo

# Ensure origin matches company repo; keep upstream read-only if present
if git remote get-url origin >/dev/null 2>&1; then
  git remote set-url origin "$REPO"
else
  git remote add origin "$REPO"
fi
if ! git remote get-url upstream >/dev/null 2>&1; then
  git remote add upstream https://github.com/KevinyWu/hug.git || true
fi

echo "== push -u origin main (NO force) =="
git push -u origin main

echo
echo "== verify =="
git ls-remote --heads origin
git status -sb
echo "OK: company HUG has local main (incl. 3a0e67d)."
