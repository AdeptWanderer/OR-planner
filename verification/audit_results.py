"""Audit reported benchmark results against their saved solver state.

For every reported instance this script
  1. regenerates the instance from its (set, size, seed),
  2. loads the final checkpoint written by the solver run and takes the
     schedule it actually produced,
  3. re-validates that schedule with the independent checker,
  4. recomputes the scheduled minutes, the heuristic value and the gap from
     scratch and compares them with the reported numbers,
  5. checks the upper bound for consistency (UB >= LB, UB <= root bound,
     UB <= trivial capacity bound, LB/UB trajectory monotone),
  6. optionally (--crosscheck SECONDS) runs an independent monolithic CP-SAT
     model: a plan it finds must never exceed our upper bound, and its proven
     bound must never be below our plan.

usage: python verification/audit_results.py RESULTS.jsonl CKPT_DIR [--crosscheck 120]
Exit code 1 if any check fails.
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from orsched import generate_instance  # noqa: E402
from orsched.heuristic import constructive_heuristic  # noqa: E402
from orsched.validate import validate_columns  # noqa: E402


def audit(res_path, ckpt_dir, crosscheck=0.0, seed=1):
    rows, failures = [], 0
    for line in open(res_path):
        if not line.strip():
            continue
        r = json.loads(line)
        S, n = r["set"], r["n"]
        inst = generate_instance(n, S, seed=seed)
        m = inst.slot_minutes
        checks = {}
        ck = Path(ckpt_dir) / f"{S}{n}.pkl"
        if not ck.exists():
            checks["checkpoint present"] = False
            rows.append((f"{S}{n}", checks)); failures += 1
            continue
        st = pickle.load(open(ck, "rb"))
        cols = st["best_cols"] or []
        lb = sum(c.value(inst) for c in cols) * m
        checks["final phase reached"] = st["phase"] == "done"
        checks["schedule valid (independent checker)"] = validate_columns(inst, cols) == []
        checks["plan minutes recomputed == reported"] = math.isclose(lb, r["lb"])
        hc, _ = constructive_heuristic(inst, np.random.default_rng(0))
        checks["heuristic recomputed == reported"] = math.isclose(sum(c.value(inst) for c in hc) * m, r["heuristic"])
        ub = r["ub"]
        checks["UB >= LB"] = ub >= lb - 1e-6
        checks["UB <= root bound"] = ub <= st["root"]["bound"] * m + 1e-6
        triv = min(sum(inst.rooms) * inst.slots_per_day, int(inst.surgeon_max.sum()),
                   sum(s.duration for s in inst.surgeries)) * m
        checks["UB <= trivial capacity bound"] = ub <= triv + 1e-6
        gap = 100 * (ub - lb) / ub if ub else 0.0
        checks["gap recomputed == reported"] = abs(gap - r["gap"]) < 0.01
        if r.get("lb_no_polish") is not None:
            checks["polish never worsens the plan"] = r["lb"] >= r["lb_no_polish"] - 1e-6
        tr = [t for t in st.get("trajectory", []) if t[1] is not None and t[1] > -math.inf]
        checks["best plan non-decreasing over time"] = all(b2 >= b1 - 1e-9 for (_, b1, _), (_, b2, _) in zip(tr, tr[1:]))
        checks["time limits respected (root <= 2h, branching <= 2h + 1 node)"] = (
            st["used"]["root"] <= 7200 + 120 and st["used"]["branch"] <= 7200 + 3700)
        if crosscheck > 0:
            from orsched.compact import solve_compact
            ref = solve_compact(inst, crosscheck, workers=4)
            if ref["lb"] is not None:
                checks[f"independent CP-SAT plan ({ref['lb']:.0f}) <= our UB"] = ref["lb"] <= ub + 1e-6
            checks[f"independent CP-SAT bound ({ref['ub']:.0f}) >= our plan"] = ref["ub"] >= lb - 1e-6
        ok = all(checks.values())
        failures += 0 if ok else 1
        rows.append((f"{S}{n}", checks))
    return rows, failures


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("results"); ap.add_argument("ckpt_dir")
    ap.add_argument("--crosscheck", type=float, default=0.0)
    a = ap.parse_args()
    rows, fails = audit(a.results, a.ckpt_dir, a.crosscheck)
    for name, checks in rows:
        bad = [k for k, v in checks.items() if not v]
        print(f"{name:6s} {'PASS' if not bad else 'FAIL'}  ({len(checks)} checks)" + ("" if not bad else "  failed: " + "; ".join(bad)))
    print(f"\n{len(rows) - fails}/{len(rows)} instances pass every check")
    sys.exit(1 if fails else 0)
