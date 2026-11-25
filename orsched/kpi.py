"""Healthcare KPIs and tabular views of a schedule."""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from .model import Column, Instance


def room_layout(columns: Sequence[Column]) -> List[tuple]:
    """Assign room numbers per day (sorted by first start, then load)."""
    out = []
    by_day: Dict[int, List[Column]] = {}
    for c in columns:
        by_day.setdefault(c.day, []).append(c)
    for d in sorted(by_day):
        cols = sorted(by_day[d], key=lambda c: (c.items[0][1] if c.items else 0, -len(c.items), c.items))
        for k, c in enumerate(cols):
            out.append((d, k, c))
    return out


def schedule_df(inst: Instance, columns: Sequence[Column]) -> pd.DataFrame:
    rows = []
    m = inst.slot_minutes
    for d, k, c in room_layout(columns):
        for i, s in sorted(c.items, key=lambda z: z[1]):
            sg = inst.surgeries[i]
            rows.append(dict(
                case=sg.label, case_id=i, day=d + 1, day_name=inst.day_name(d), room=f"OR {k + 1}",
                room_idx=k, surgeon=inst.surgeon_name(sg.surgeon), surgeon_idx=sg.surgeon,
                start_min=s * m, end_min=(s + sg.duration) * m, duration_min=sg.duration * m,
                infection=sg.infection, infectious=sg.infection > 0,
                deadline_day=(sg.deadline + 1) if sg.deadline < inst.n_days else None,
                mandatory=inst.is_mandatory(i),
                slack_days=(sg.deadline - d) if sg.deadline < inst.n_days else None,
            ))
    return pd.DataFrame(rows)


def cleaning_df(inst: Instance, columns: Sequence[Column]) -> pd.DataFrame:
    rows = []
    m = inst.slot_minutes
    for d, k, c in room_layout(columns):
        items = sorted(c.items, key=lambda z: z[1])
        for (a, sa), (b, sb) in zip(items, items[1:]):
            cl = inst.clean(a, b)
            if cl:
                e = sa + inst.surgeries[a].duration
                rows.append(dict(day=d + 1, room=f"OR {k + 1}", room_idx=k, start_min=e * m,
                                 end_min=(e + cl) * m, after=inst.surgeries[a].label,
                                 before=inst.surgeries[b].label))
    return pd.DataFrame(rows, columns=["day", "room", "room_idx", "start_min", "end_min", "after", "before"])


def room_utilization(inst: Instance, columns: Sequence[Column]) -> pd.DataFrame:
    m = inst.slot_minutes
    T = inst.slots_per_day * m
    used = {(d, k): c for d, k, c in room_layout(columns)}
    cl = cleaning_df(inst, columns)
    rows = []
    for d in range(inst.n_days):
        for k in range(inst.rooms[d]):
            c = used.get((d, k))
            busy = sum(inst.surgeries[i].duration for i, _ in c.items) * m if c else 0
            clean = float(((cl.day == d + 1) & (cl.room_idx == k)).sum() * inst.cleaning * m) if len(cl) else 0.0
            rows.append(dict(day=d + 1, day_name=inst.day_name(d), room=f"OR {k + 1}",
                             surgery_min=busy, cleaning_min=clean,
                             idle_min=T - busy - clean, cases=len(c.items) if c else 0,
                             utilization=100.0 * busy / T, opened=c is not None))
    return pd.DataFrame(rows)


def surgeon_utilization(inst: Instance, columns: Sequence[Column]) -> pd.DataFrame:
    m = inst.slot_minutes
    tot = np.zeros((inst.n_surgeons, inst.n_days))
    cnt = np.zeros((inst.n_surgeons, inst.n_days), dtype=int)
    for c in columns:
        for i, _ in c.items:
            sg = inst.surgeries[i]
            tot[sg.surgeon, c.day] += sg.duration
            cnt[sg.surgeon, c.day] += 1
    rows = []
    for l in range(inst.n_surgeons):
        for d in range(inst.n_days):
            mx = inst.surgeon_max[l, d] * m
            rows.append(dict(surgeon=inst.surgeon_name(l), surgeon_idx=l, day=d + 1, day_name=inst.day_name(d),
                             scheduled_min=tot[l, d] * m, max_min=mx, cases=cnt[l, d],
                             utilization=100.0 * tot[l, d] * m / mx if mx else 0.0))
    return pd.DataFrame(rows)


def waitlist_df(inst: Instance, columns: Sequence[Column]) -> pd.DataFrame:
    done = {i for c in columns for i, _ in c.items}
    rows = []
    for sg in inst.surgeries:
        if sg.id in done:
            continue
        rows.append(dict(case=sg.label, surgeon=inst.surgeon_name(sg.surgeon),
                         duration_min=sg.duration * inst.slot_minutes,
                         deadline=("day " + str(sg.deadline + 1)) if sg.deadline < inst.n_days
                         else f"after horizon (+{sg.deadline + 1 - inst.n_days} d)",
                         deadline_idx=sg.deadline, mandatory=inst.is_mandatory(sg.id),
                         infectious=sg.infection > 0))
    df = pd.DataFrame(rows, columns=["case", "surgeon", "duration_min", "deadline", "deadline_idx", "mandatory", "infectious"])
    return df.sort_values("deadline_idx").drop(columns="deadline_idx").reset_index(drop=True)


def summary(inst: Instance, columns: Sequence[Column]) -> Dict[str, float]:
    m = inst.slot_minutes
    sched = {i for c in columns for i, _ in c.items}
    cap = sum(inst.rooms) * inst.slots_per_day * m
    mins = sum(inst.surgeries[i].duration for i in sched) * m
    cl = cleaning_df(inst, columns)
    ru = room_utilization(inst, columns)
    opened = ru[ru.opened]
    mand = inst.mandatory
    return dict(
        scheduled_minutes=mins,
        scheduled_cases=len(sched),
        total_cases=inst.n,
        mandatory_scheduled=sum(1 for i in mand if i in sched),
        mandatory_total=len(mand),
        optional_scheduled=sum(1 for i in inst.optional if i in sched),
        optional_total=len(inst.optional),
        or_capacity_minutes=cap,
        utilization_all_rooms=100.0 * mins / cap if cap else 0.0,
        utilization_opened_rooms=float(opened.utilization.mean()) if len(opened) else 0.0,
        rooms_opened=int(ru.opened.sum()), rooms_available=int(sum(inst.rooms)),
        cleanings=len(cl), cleaning_minutes=len(cl) * inst.cleaning * m,
        demand_minutes=sum(s.duration for s in inst.surgeries) * m,
    )
