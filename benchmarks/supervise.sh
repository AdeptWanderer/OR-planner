#!/usr/bin/env bash
# Keeps the paper-budget benchmark running: relaunches it (resuming from
# checkpoints) after a crash, memory error or container restart, until all
# 13 instances have a result line.
cd "$(dirname "$0")/.."
ONLY="${1:-}"            # optional subset, e.g. A80,B60,C100
TARGET=$( [ -n "$ONLY" ] && echo "$ONLY" | tr ',' '\n' | wc -l || echo 13 )
for attempt in $(seq 1 20); do
  done_n=$(grep -c . benchmarks/paper_budget.jsonl 2>/dev/null || echo 0)
  [ "$done_n" -ge "$TARGET" ] && { echo "all done"; exit 0; }
  echo "$(date -u +%FT%TZ) attempt $attempt, $done_n/$TARGET finished"
  python3 benchmarks/run_paper_budget.py 4 "$ONLY" >> benchmarks/paper_budget.out 2>&1
  sleep 5
done
