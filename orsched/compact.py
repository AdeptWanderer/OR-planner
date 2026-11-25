"""Monolithic CP-SAT model of the IORPS (baseline, in the spirit of the pure
CP model of the paper's online Appendix 3).  Used to validate the
branch-and-price-and-cut results on small instances and as a comparison
method in the dashboard."""
from __future__ import annotations

import time
from typing import List, Optional

from ortools.sat.python import cp_model

from .model import Column, Instance


def solve_compact(inst: Instance, time_limit: float = 60.0, workers: int = 4,
                  hint: Optional[List[Column]] = None, lower_bound: Optional[int] = None):
    """hint: warm start (list of room-day columns); lower_bound (slots): only
    accept solutions at least this good."""
    T, OCT = inst.slots_per_day, inst.cleaning
    m = cp_model.CpModel()
    y, s, iv = {}, {}, {}
    for sg in inst.surgeries:
        i = sg.id
        s[i] = m.new_int_var(0, T - sg.duration, f"s{i}")
        for d in range(min(sg.deadline + 1, inst.n_days)):
            if sg.duration > inst.surgeon_max[sg.surgeon, d]:
                continue
            for k in range(inst.rooms[d]):
                y[i, d, k] = m.new_bool_var(f"y{i}_{d}_{k}")
                iv[i, d, k] = m.new_optional_fixed_size_interval_var(s[i], sg.duration, y[i, d, k], "")
        lits = [v for (a, _, _), v in y.items() if a == i]
        if inst.is_mandatory(i):
            m.add_exactly_one(lits) if lits else m.add_bool_or([])
        else:
            m.add_at_most_one(lits)
    for d in range(inst.n_days):
        for k in range(inst.rooms[d]):
            room = [(i, v) for (i, dd, kk), v in y.items() if dd == d and kk == k]
            m.add_no_overlap([iv[i, d, k] for i, _ in room])
            if OCT:
                for a in range(len(room)):
                    for b in range(a + 1, len(room)):
                        i, yi = room[a]; j, yj = room[b]
                        if inst.surgeries[i].infection == inst.surgeries[j].infection:
                            continue
                        bf = m.new_bool_var("")
                        m.add(s[j] >= s[i] + inst.surgeries[i].duration + OCT).only_enforce_if([yi, yj, bf])
                        m.add(s[i] >= s[j] + inst.surgeries[j].duration + OCT).only_enforce_if([yi, yj, bf.Not()])
            # room symmetry breaking: rooms used in decreasing load order
            if k + 1 < inst.rooms[d]:
                load_k = sum(inst.surgeries[i].duration * v for (i, dd, kk), v in y.items() if dd == d and kk == k)
                load_k1 = sum(inst.surgeries[i].duration * v for (i, dd, kk), v in y.items() if dd == d and kk == k + 1)
                m.add(load_k >= load_k1)
        for l in range(inst.n_surgeons):
            mine = [(i, k, v) for (i, dd, k), v in y.items() if dd == d and inst.surgeries[i].surgeon == l]
            if not mine:
                continue
            m.add(sum(inst.surgeries[i].duration * v for i, _, v in mine) <= int(inst.surgeon_max[l, d]))
            m.add_no_overlap([iv[i, d, k] for i, k, _ in mine])
    obj = sum(inst.surgeries[i].duration * v for (i, _, _), v in y.items())
    m.maximize(obj)
    if lower_bound is not None:
        m.add(obj >= int(lower_bound))
    if hint:
        on = {}
        per_day = {}
        for c in sorted(hint, key=lambda c: (c.day, -c.value(inst))):
            k = per_day.get(c.day, 0); per_day[c.day] = k + 1
            for i, st in c.items:
                on[(i, c.day, k)] = 1
                m.add_hint(s[i], st)
        for key, v in y.items():
            m.add_hint(v, on.get(key, 0))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_workers = workers
    t0 = time.time()
    st = solver.solve(m)
    out = dict(status=solver.status_name(st), time=time.time() - t0,
               lb=None, ub=solver.best_objective_bound * inst.slot_minutes, columns=[])
    if st in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        out["lb"] = solver.objective_value * inst.slot_minutes
        cols = {}
        for (i, d, k), v in y.items():
            if solver.value(v):
                cols.setdefault((d, k), []).append((i, solver.value(s[i])))
        out["columns"] = [Column(d, tuple(sorted(it, key=lambda z: z[1]))) for (d, k), it in cols.items()]
    return out
