"""Day-decomposition start heuristic (extension; not in the paper).

1. Assign cases to days with the day-level optimum (orsched.bounds): it already
   respects deadlines, surgeon-day minutes and room minutes per day.
2. Walk the days in order.  Day d gets the cases assigned to it plus leftovers of
   earlier days whose deadline allows d.  A one-day CP-SAT model packs them into
   the day's rooms exactly (sequencing, cleaning, surgeon no-overlap, surgeon
   minutes).  Cases whose deadline is d (last chance) are mandatory in that model.
Unscheduled cases carry over.  The result is a set of feasible room-day columns;
it is a valid plan only if every mandatory case ended up scheduled.
"""
from __future__ import annotations

from typing import List, Tuple

from .bounds import day_level_bound
from .compact import solve_compact
from .model import Column, Instance, Surgery


def dayplan_heuristic(inst: Instance, time_per_day: float = 30.0, workers: int = 4,
                      log=None) -> Tuple[List[Column], dict]:
    _, assign, _ = day_level_bound(inst, integer=True, time_limit=120)
    by_day = {d: [i for i, dd in assign.items() if dd == d] for d in range(inst.n_days)}
    leftover: List[int] = []
    columns: List[Column] = []
    info = {"unscheduled_mandatory_last_chance": 0, "day_status": []}
    for d in range(inst.n_days):
        cand = sorted(set(by_day[d]) | {i for i in leftover if inst.surgeries[i].deadline >= d})
        if not cand:
            continue
        def make(force):
            subs = [Surgery(k, inst.surgeries[i].duration, inst.surgeries[i].surgeon, 0 if force(i) else 1,
                            inst.surgeries[i].infection, inst.surgeries[i].label) for k, i in enumerate(cand)]
            return subs, Instance(subs, 1, [inst.rooms[d]], inst.slots_per_day, inst.surgeon_max[:, d:d + 1].copy(),
                                  inst.cleaning, inst.slot_minutes, inst.surgeon_names)
        last = lambda i: inst.surgeries[i].deadline == d                      # last chance today
        planned = set(by_day[d])
        # 1st try: force last-chance cases AND the mandatory cases the relaxation put on this day,
        # so optional cases cannot crowd mandatory ones into a later, fuller day
        subs, sub = make(lambda i: last(i) or (i in planned and inst.is_mandatory(i)))
        r = solve_compact(sub, time_per_day, workers)
        if r["status"] in ("INFEASIBLE", "UNKNOWN") or not r["columns"]:
            subs, sub = make(last)
            r = solve_compact(sub, time_per_day, workers)
        if r["status"] in ("INFEASIBLE", "UNKNOWN") or not r["columns"]:
            # retry with mandatory candidates only, so optional cases cannot crowd out deadlines
            keep = [k for k, i in enumerate(cand) if inst.is_mandatory(i)]
            cand = [cand[k] for k in keep]
            subs = [Surgery(n, subs[k].duration, subs[k].surgeon, subs[k].deadline, subs[k].infection, subs[k].label)
                    for n, k in enumerate(keep)]
            sub = Instance(subs, 1, [inst.rooms[d]], inst.slots_per_day, inst.surgeon_max[:, d:d + 1].copy(),
                           inst.cleaning, inst.slot_minutes, inst.surgeon_names)
            r = solve_compact(sub, time_per_day, workers)
        if r["status"] == "INFEASIBLE":                  # last-chance cases do not all fit: relax them
            subs = [Surgery(s.id, s.duration, s.surgeon, 1, s.infection, s.label) for s in subs]
            sub = Instance(subs, 1, [inst.rooms[d]], inst.slots_per_day, inst.surgeon_max[:, d:d + 1].copy(),
                           inst.cleaning, inst.slot_minutes, inst.surgeon_names)
            r = solve_compact(sub, time_per_day, workers)
        done = set()
        for c in r["columns"]:
            items = tuple((cand[k], st) for k, st in c.items)
            columns.append(Column(day=d, items=items))
            done.update(cand[k] for k, _ in c.items)
        leftover = [i for i in cand if i not in done]
        miss = [i for i in leftover if inst.surgeries[i].deadline == d and inst.is_mandatory(i)]
        info["unscheduled_mandatory_last_chance"] += len(miss)
        info["day_status"].append((d, r["status"], len(cand), len(done)))
        if log:
            log(f"dayplan day {d + 1}: {r['status']} cand {len(cand)} scheduled {len(done)} missed-mandatory {len(miss)}")
    return columns, info
