"""Constructive heuristic of Section 6.1 (initial columns / instance repair).

Rooms are filled consecutively, from the first room of the first day to the
last room of the last day; each step appends one surgery to the current room
at the earliest feasible time (respecting cleaning, surgeon no-overlap across
rooms and surgeon daily limits).  Surgeries are prioritised by deadline, then
by a random surgeon-importance ranking, then by decreasing duration.  When a
mandatory deadline cannot be respected it is postponed to the next day.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from .model import Column, Instance


def _earliest_start(lb: int, dur: int, busy: List[Tuple[int, int]], horizon: int) -> Optional[int]:
    s = lb
    for a, b in sorted(busy):
        if s + dur <= a:
            break
        if s < b:
            s = max(s, b)
    return s if s + dur <= horizon else None


def constructive_heuristic(inst: Instance, rng: Optional[np.random.Generator] = None,
                           postpone: bool = False):
    """Return (columns, postponed_deadlines).

    ``postponed_deadlines`` maps surgery -> new (later) deadline for mandatory
    cases whose deadline could not be met (only filled if ``postpone``).
    """
    rng = rng or np.random.default_rng(0)
    L, D, T = inst.n_surgeons, inst.n_days, inst.slots_per_day
    importance = {l: r for r, l in enumerate(rng.permutation(L))}
    deadline = {s.id: s.deadline for s in inst.surgeries}
    unscheduled = set(range(inst.n))
    columns: List[Column] = []
    postponed: Dict[int, int] = {}

    for d in range(D):
        busy = {l: [] for l in range(L)}
        used = np.zeros(L, dtype=int)
        for _k in range(inst.rooms[d]):
            items: List[Tuple[int, int]] = []
            room_end, last = 0, None
            while True:
                cands = sorted(
                    (i for i in unscheduled if deadline[i] >= d),
                    key=lambda i: (deadline[i], importance[inst.surgeries[i].surgeon],
                                   -inst.surgeries[i].duration, i))
                placed = False
                for i in cands:
                    s_i = inst.surgeries[i]
                    l = s_i.surgeon
                    if used[l] + s_i.duration > inst.surgeon_max[l, d]:
                        continue
                    st = _earliest_start(room_end + inst.clean(last, i), s_i.duration, busy[l], T)
                    if st is None:
                        continue
                    items.append((i, st))
                    busy[l].append((st, st + s_i.duration))
                    used[l] += s_i.duration
                    room_end, last = st + s_i.duration, i
                    unscheduled.discard(i)
                    placed = True
                    break
                if not placed:
                    break
            if items:
                columns.append(Column(day=d, items=tuple(items)))
        # postpone deadlines that are violated
        for i in list(unscheduled):
            if deadline[i] == d:
                if postpone:
                    deadline[i] = d + 1
                    postponed[i] = d + 1
    return columns, postponed
