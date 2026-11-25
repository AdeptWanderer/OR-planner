"""CSV import / export so the solver can be fed with real hospital data.

surgeries.csv columns
    case (str), surgeon (str), duration_min (int), deadline_day (int, 1-based;
    blank or > horizon => optional / can be postponed), infection (int, 0 = none,
    k = infection type k; or yes/no)

surgeons.csv columns
    surgeon (str), then one column per day with the max operating minutes,
    e.g. Mon,Tue,Wed,Thu,Fri
"""
from __future__ import annotations

import math
from typing import List, Optional

import numpy as np
import pandas as pd

from .model import Instance, Surgery


def _infection(v) -> int:
    if isinstance(v, str):
        v = v.strip().lower()
        if v in ("", "no", "n", "false", "0", "none"):
            return 0
        if v in ("yes", "y", "true"):
            return 1
        return int(float(v))
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return 0
    return int(v)


def instance_from_frames(surg: pd.DataFrame, surgeons: pd.DataFrame, rooms: List[int],
                         day_minutes: int = 480, slot_minutes: int = 5,
                         cleaning_minutes: int = 30, name: str = "uploaded") -> Instance:
    days = [c for c in surgeons.columns if c != "surgeon"]
    D = len(days)
    if len(rooms) == 1:
        rooms = rooms * D
    names = [str(s) for s in surgeons["surgeon"]]
    idx = {n: k for k, n in enumerate(names)}
    A = np.zeros((len(names), D), dtype=int)
    for k, row in surgeons.reset_index(drop=True).iterrows():
        for d, c in enumerate(days):
            A[k, d] = min(day_minutes, int(row[c])) // slot_minutes
    out = []
    for j, row in surg.reset_index(drop=True).iterrows():
        s = str(row["surgeon"])
        if s not in idx:
            raise ValueError(f"case {row.get('case', j)}: surgeon '{s}' missing from surgeons table")
        dd = row.get("deadline_day")
        dd = D if dd is None or (isinstance(dd, float) and math.isnan(dd)) or str(dd).strip() == "" else int(dd) - 1
        out.append(Surgery(id=j, duration=int(math.ceil(float(row["duration_min"]) / slot_minutes)),
                           surgeon=idx[s], deadline=max(0, dd), infection=_infection(row.get("infection", 0)),
                           label=str(row.get("case", f"S{j + 1:03d}"))))
    return Instance(surgeries=out, n_days=D, rooms=[int(r) for r in rooms],
                    slots_per_day=day_minutes // slot_minutes, surgeon_max=A,
                    cleaning=int(math.ceil(cleaning_minutes / slot_minutes)), slot_minutes=slot_minutes,
                    surgeon_names=names, day_names=days, name=name)


def instance_to_frames(inst: Instance):
    m = inst.slot_minutes
    s = pd.DataFrame([dict(case=x.label, surgeon=inst.surgeon_name(x.surgeon), duration_min=x.duration * m,
                           deadline_day=x.deadline + 1, infection=x.infection) for x in inst.surgeries])
    cols = {inst.day_name(d): inst.surgeon_max[:, d] * m for d in range(inst.n_days)}
    g = pd.DataFrame(dict(surgeon=[inst.surgeon_name(l) for l in range(inst.n_surgeons)], **cols))
    return s, g
