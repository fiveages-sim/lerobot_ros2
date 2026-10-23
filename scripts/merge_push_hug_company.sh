#!/usr/bin/env bash
# Merge remote initial commit into local main; keep LOCAL README; plain push (no force).
set -euo pipefail
REPO="${1:-git@github.com:fiveages-sim/HUG.git}"
HUG_DIR="$(cd "$(dirname "$0")/../submodules/hug" && pwd)"
cd "$HUG_DIR"

git remote get-url origin >/dev/null 2>&1 || git remote add origin "$REPO"
git remote set-url origin "$REPO" || echo "warn: could not set-url (ok if already correct)"
echo "origin=$(git remote get-url origin)"
echo "== fetch origin =="
git fetch origin

echo "== remote main =="
git log --oneline origin/main -3
echo "== local main =="
git log --oneline main -3
echo

# Unrelated histories: prefer ours on conflicts (local company README / patches).
echo "== merge origin/main (--allow-unrelated-histories, -X ours) =="
if git merge-base --is-ancestor origin/main main 2>/dev/null; then
  echo "origin/main already ancestor of main; skip merge"
elif git merge-base --is-ancestor main origin/main 2>/dev/null; then
  echo "local main is behind; fast-forward"
  git merge --ff-only origin/main
else
  git merge origin/main --allow-unrelated-histories -X ours -m "$(cat <<'EOF'
Merge remote-tracking branch 'origin/main' into main.

Keep local README and company HUG patches; drop remote stub README content on conflict.
EOF
)" || true

  # Ensure README is ours even if merge auto-resolved oddly
  if git show-ref --verify --quiet refs/heads/main && test -f README.md; then
    if git rev-parse -q --verify MERGE_HEAD >/dev/null 2>&1; then
      git checkout --ours README.md
      git add README.md
      # Prefer not leaving other conflicts: take ours for any remaining
      if git diff --name-only --diff-filter=U | grep -q .; then
        git diff --name-only --diff-filter=U | while read -r f; do
          git checkout --ours -- "$f"
          git add -- "$f"
        done
      fi
      git commit --no-edit -m "$(cat <<'EOF'
Merge origin/main (initial repo README); keep local README.
EOF
)"
    fi
  fi
fi

# Explicit: README must match pre-merge local tip's README if we still have it in reflog
# (already handled by -X ours / checkout --ours)

echo "== HEAD after merge =="
git log -3 --oneline
echo
echo "== push -u origin main (NO force) =="
git push -u origin main
echo
git ls-remote --heads origin
git status -sb
echo "OK"
