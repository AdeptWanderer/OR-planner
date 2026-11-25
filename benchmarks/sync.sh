#!/usr/bin/env bash
# Every 30 min, push this machine's benchmark results + checkpoints to its own
# branch so nothing is lost if the container is reclaimed. Snapshots are
# committed through a separate git worktree (.sync-worktree), so the main
# checkout stays on its branch and stays clean while the solvers write logs.
# usage: benchmarks/sync.sh <branch>
cd "$(dirname "$0")/.."
BR="$1"; WT=.sync-worktree
if [ ! -d "$WT" ]; then
  if git ls-remote --exit-code origin "refs/heads/$BR" >/dev/null 2>&1; then
    git fetch -q origin "$BR" && git worktree add -q -B "$BR" "$WT" "origin/$BR"
  else
    git worktree add -q -B "$BR" "$WT" HEAD
  fi
fi
while true; do
  mkdir -p "$WT/benchmarks"
  for f in paper_budget.jsonl ckpt logs supervise.out; do
    [ -e "benchmarks/$f" ] && cp -r "benchmarks/$f" "$WT/benchmarks/"
  done
  ( cd "$WT" && git add -f benchmarks/paper_budget.jsonl benchmarks/ckpt benchmarks/logs benchmarks/supervise.out 2>/dev/null
    git add -f benchmarks/ckpt benchmarks/logs 2>/dev/null
    git commit -q -m "bench: checkpoint $(date -u +%FT%TZ)" && \
      for i in 1 2 3 4; do git push -q -u origin "$BR" 2>/dev/null && break || sleep $((2**i)); done )
  pgrep -f supervise.sh >/dev/null || { echo "supervisor finished"; exit 0; }
  sleep 1800
done
