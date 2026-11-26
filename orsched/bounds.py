"""Day-level relaxation of the IORPS (extension; not in the paper).

Variables y[i,d] = 1 if surgery i is performed on day d (d <= deadline).
Constraints: mandatory exactly once / optional at most once; per surgeon-day
minutes <= A_ld; per-day minutes <= |K_d| * T.  Every feasible plan satisfies
them, so the optimum is a valid upper bound on scheduled minutes.  It ignores
rooms' sequencing, cleaning and surgeons' no-overlap, so it is optimistic.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from .model import Instance


def day_level_bound(inst: Instance, integer: bool = True, time_limit: float = 60.0,
                    fixed_out: Optional[set] = None) -> Tuple[float, Dict[int, int], bool]:
    """Return (upper bound in slots, case->day of the relaxation's solution, proven).

    ``fixed_out``: cases forced out (used for what-if diagnostics only).
    The bound is the MIP dual bound (valid even when not solved to optimality)
    and is floored to an integer number of slots."""
    import highspy
    INF = highspy.kHighsInf
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("threads", 1)
    h.setOptionValue("time_limit", float(time_limit))
    var, cols = {}, 0
    for s in inst.surgeries:
        if fixed_out and s.id in fixed_out:
            continue
        for d in range(min(s.deadline + 1, inst.n_days)):
            if s.duration <= inst.surgeon_max[s.surgeon, d]:
                var[(s.id, d)] = cols
                cols += 1
    cost = np.zeros(cols)
    for (i, d), j in var.items():
        cost[j] = -inst.surgeries[i].duration
    h.addVars(cols, np.zeros(cols), np.ones(cols))
    h.changeColsCost(cols, np.arange(cols, dtype=np.int32), cost)
    if integer:
        h.changeColsIntegrality(cols, np.arange(cols, dtype=np.int32),
                                np.array([highspy.HighsVarType.kInteger] * cols))

    def row(js, vals, lo, hi):
        h.addRow(lo, hi, len(js), np.array(js, dtype=np.int32), np.array(vals, dtype=float))

    for s in inst.surgeries:
        js = [var[(s.id, d)] for d in range(inst.n_days) if (s.id, d) in var]
        if not js:
            if inst.is_mandatory(s.id) and not (fixed_out and s.id in fixed_out):
                return -1.0, {}, True                   # infeasible: no day can host a mandatory case
            continue
        row(js, [1.0] * len(js), 1.0 if inst.is_mandatory(s.id) else -INF, 1.0)
    for d in range(inst.n_days):
        by_l: Dict[int, list] = {}
        for (i, dd), j in var.items():
            if dd == d:
                by_l.setdefault(inst.surgeries[i].surgeon, []).append((j, inst.surgeries[i].duration))
        for l, lst in by_l.items():
            row([j for j, _ in lst], [t for _, t in lst], -INF, float(inst.surgeon_max[l, d]))
        allj = [(j, inst.surgeries[i].duration) for (i, dd), j in var.items() if dd == d]
        if allj:
            row([j for j, _ in allj], [t for _, t in allj], -INF, float(inst.rooms[d] * inst.slots_per_day))
    h.run()
    st = h.getModelStatus()
    info = h.getInfo()
    if integer:
        bound = -info.mip_dual_bound if np.isfinite(info.mip_dual_bound) else float("inf")
    else:
        bound = -info.objective_function_value
    sol = h.getSolution().col_value
    assign = {}
    if len(sol) == cols:
        for (i, d), j in var.items():
            if sol[j] > 0.5:
                assign[i] = d
    return float(np.floor(bound + 1e-6)), assign, st == highspy.HighsModelStatus.kOptimal
