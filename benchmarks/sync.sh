#!/usr/bin/env bash
# Every 30 min, push this machine's benchmark results + checkpoints to its own
# branch so nothing is lost if the container is reclaimed.
# usage: benchmarks/sync.sh <branch>
cd "$(dirname "$0")/.."
BR="$1"
git checkout -B "$BR" >/dev/null 2>&1 || true
while true; do
  for f in benchmarks/paper_budget.jsonl benchmarks/ckpt benchmarks/logs benchmarks/supervise.out; do
    [ -e "$f" ] && git add -f "$f"
  done
  git commit -q -m "bench: checkpoint $(date -u +%FT%TZ)" 2>/dev/null && \
    for i in 1 2 3 4; do git push -q -u origin "$BR" && break || sleep $((2**i)); done
  pgrep -f supervise.sh >/dev/null || { echo "supervisor finished"; exit 0; }
  sleep 1800
done
