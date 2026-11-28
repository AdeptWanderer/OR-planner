"""CSV import / export of instances (templates for real data)."""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd

from .model import Block, Costs, Instance, Surgery, SurgeryType

SURG_COLS = ["case", "type", "priority", "dur_mean", "dur_sd", "dur_lo", "dur_hi",
             "icu_mean", "icu_sd", "icu_lo", "icu_hi", "ward_mean", "ward_sd", "ward_lo", "ward_hi"]
BLOCK_COLS = ["day", "room", "minutes", "max_surgeries", "type"]       # day is 1-based; type optional (closed block)
BED_COLS = ["day", "icu_beds", "ward_beds"]


def instance_to_frames(inst: Instance) -> Dict[str, pd.DataFrame]:
    s = pd.DataFrame([dict(case=x.label or f"S{x.id}", type=inst.types[x.type].name, priority=x.priority,
                           dur_mean=x.mu_d, dur_sd=x.sd_d, dur_lo=x.lo_d, dur_hi=x.hi_d,
                           icu_mean=x.mu_c, icu_sd=x.sd_c, icu_lo=x.lo_c, icu_hi=x.hi_c,
                           ward_mean=x.mu_w, ward_sd=x.sd_w, ward_lo=x.lo_w, ward_hi=x.hi_w) for x in inst.surgeries],
                     columns=SURG_COLS)
    b = pd.DataFrame([dict(day=blk.day + 1, room=blk.room + 1, minutes=blk.length, max_surgeries=blk.K,
                           type=inst.types[blk.types[0]].name if blk.types else "") for blk in inst.blocks], columns=BLOCK_COLS)
    d = pd.DataFrame(dict(day=np.arange(1, inst.T + 1), icu_beds=inst.icu_beds, ward_beds=inst.ward_beds), columns=BED_COLS)
    return dict(surgeries=s, blocks=b, beds=d)


def instance_from_frames(surg: pd.DataFrame, blocks: pd.DataFrame, beds: pd.DataFrame, costs: Costs | None = None,
                         name: str = "uploaded") -> Instance:
    names = list(dict.fromkeys(surg["type"].astype(str)))
    types = []
    surgeries = []
    first = {}
    for n in names:
        r = surg[surg["type"].astype(str) == n].iloc[0]
        types.append(SurgeryType(n, float(r.dur_mean), float(r.dur_sd), float(r.icu_mean), float(r.icu_sd),
                                 float(r.ward_mean), float(r.ward_sd), "uploaded"))
    for i, r in enumerate(surg.itertuples(index=False)):
        surgeries.append(Surgery(i, names.index(str(r.type)), str(r.case), float(r.priority) if r.priority == r.priority else 1.0,
                                 float(r.dur_mean), float(r.dur_sd), float(r.dur_lo), float(r.dur_hi),
                                 float(r.icu_mean), float(r.icu_sd), int(r.icu_lo), int(r.icu_hi),
                                 float(r.ward_mean), float(r.ward_sd), int(r.ward_lo), int(r.ward_hi)))
    blks = []
    for j, r in enumerate(blocks.itertuples(index=False)):
        t = () if (r.type != r.type or str(r.type).strip() == "") else (names.index(str(r.type)),)
        blks.append(Block(j, int(r.day) - 1, int(r.room) - 1, float(r.minutes), int(r.max_surgeries), t))
    beds = beds.sort_values("day")
    return Instance(types, surgeries, blks, len(beds), beds["icu_beds"].to_numpy(float), beds["ward_beds"].to_numpy(float),
                    costs or Costs(), name)
