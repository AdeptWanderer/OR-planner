"""Independent feasibility checker for a set of room-day schedules."""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable, List

from .model import Column, Instance


def validate_columns(inst: Instance, columns: Iterable[Column], require_mandatory: bool = True) -> List[str]:
    errs: List[str] = []
    columns = list(columns)
    seen = defaultdict(int)
    per_day = defaultdict(int)
    surgeon_time = defaultdict(int)
    surgeon_iv = defaultdict(list)
    T = inst.slots_per_day
    for c in columns:
        per_day[c.day] += 1
        items = sorted(c.items, key=lambda x: x[1])
        prev = None
        for i, s in items:
            sg = inst.surgeries[i]
            seen[i] += 1
            if c.day > sg.deadline:
                errs.append(f"{sg.label}: scheduled on day {c.day + 1} after its deadline (day {sg.deadline + 1})")
            if s < 0 or s + sg.duration > T:
                errs.append(f"{sg.label}: outside the room day (start {s}, dur {sg.duration})")
            if prev is not None:
                pi, ps = prev
                need = ps + inst.surgeries[pi].duration + inst.clean(pi, i)
                if s < need:
                    errs.append(f"{inst.surgeries[pi].label}->{sg.label}: overlap / missing cleaning on day {c.day + 1}")
            prev = (i, s)
            surgeon_time[(sg.surgeon, c.day)] += sg.duration
            surgeon_iv[(sg.surgeon, c.day)].append((s, s + sg.duration, sg.label))
    for i, k in seen.items():
        if k > 1:
            errs.append(f"{inst.surgeries[i].label}: scheduled {k} times")
    if require_mandatory:
        for i in inst.mandatory:
            if seen.get(i, 0) == 0:
                errs.append(f"{inst.surgeries[i].label}: mandatory surgery not scheduled")
    for d, k in per_day.items():
        if k > inst.rooms[d]:
            errs.append(f"day {d + 1}: {k} rooms used > {inst.rooms[d]} available")
    for (l, d), v in surgeon_time.items():
        if v > inst.surgeon_max[l, d]:
            errs.append(f"{inst.surgeon_name(l)} day {d + 1}: {v} slots > max {inst.surgeon_max[l, d]}")
    for (l, d), ivs in surgeon_iv.items():
        ivs.sort()
        for (a1, b1, n1), (a2, b2, n2) in zip(ivs, ivs[1:]):
            if a2 < b1:
                errs.append(f"{inst.surgeon_name(l)} day {d + 1}: overlapping cases {n1} and {n2}")
    return errs


def validate_solution(inst: Instance, sol) -> List[str]:
    return validate_columns(inst, sol.columns)
