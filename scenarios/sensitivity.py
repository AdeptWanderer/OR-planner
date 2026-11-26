"""Sensitivity sweeps for Riverside with the day-level relaxation (exact MIP, seconds per point).
The relaxation was within 0.56-0.98% of the real plans in all three solved scenarios.
It ignores cleaning, so cleaning time is analysed with real day-decomposition plans instead
(scenarios/out/clean_<min>.json from solve_fast-like runs).  Writes scenarios/out/sensitivity.json."""
import json, sys, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import riverside as R
from orsched.bounds import day_level_bound
from orsched.model import Surgery

def bound(inst):
    b, _, ok = day_level_bound(inst, integer=False)   # LP relaxation: valid bound, solves in ms
    return None if b < 0 else b * R.SLOT          # None = mandatory cases cannot all be met
out = {}
base, meta = R.build()
B0 = bound(base); out["base"] = B0
# 1. rooms Mon-Thu and Fridays
out["rooms_mon_thu"] = []
for k in range(5, 11):
    inst, _ = R.build(); inst.rooms = [k, k, k, k, 6] * 2; out["rooms_mon_thu"].append([k, bound(inst)])
out["rooms_fri"] = []
for k in range(3, 9):
    inst, _ = R.build(friday_rooms=k); out["rooms_fri"].append([k, bound(inst)])
# 2. value of one extra full-day session for each surgeon (best free weekday in the fortnight)
out["extra_session"] = []
for l, (nm, spec, _) in enumerate(R.SURGEONS):
    best = (0, None)
    free = [d for d in range(10) if base.surgeon_max[l, d] == 0]
    for d in free:
        inst, _ = R.build(); inst.surgeon_max[l, d] = 480 // R.SLOT
        v = bound(inst)
        if v is not None and v - B0 > best[0]:
            best = (v - B0, inst.day_names[d])
    out["extra_session"].append(dict(surgeon=nm, spec=spec, gain=best[0], day=best[1], free_days=len(free)))
# 3. case durations overrun (all cases longer by x%)
out["overrun"] = []
for pct in (0, 5, 10, 15, 20, 25, 30):
    inst, _ = R.build()
    inst.surgeries = [Surgery(s.id, int(math.ceil(s.duration * (1 + pct / 100))), s.surgeon, s.deadline, s.infection, s.label)
                      for s in inst.surgeries]
    demand = sum(s.duration for s in inst.surgeries) * R.SLOT
    out["overrun"].append([pct, bound(inst), demand])
# 4. surgeon availability scaled (sessions shortened / lengthened)
out["availability"] = []
for pct in (-20, -15, -10, -5, 0, 5, 10, 15, 20):
    inst, _ = R.build()
    inst.surgeon_max = np.minimum(np.floor(inst.surgeon_max * (1 + pct / 100)).astype(int), 480 // R.SLOT)
    out["availability"].append([pct, bound(inst)])
json.dump(out, open("scenarios/out/sensitivity.json", "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "extra_session"}))
for x in out["extra_session"]: print(x)
