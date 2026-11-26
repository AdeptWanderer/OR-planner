"""Capacity analysis of the Riverside scenarios using the day-level relaxation (fast, exact LP/MIP).
Writes scenarios/out/analysis.json: bound per scenario, per-surgeon unmet backlog, per-day binding constraint."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parent))
import riverside as R
from orsched.bounds import day_level_bound
out = {"scenarios": [], "surgeons": [], "days": []}
variants = [("Baseline", {}), ("8 rooms on Fridays", dict(friday_rooms=8)),
            ("Neuro A extra Wed + Fri sessions", dict(neuro_extra_sessions=True)),
            ("Neuro A extra sessions + 8 Friday rooms", dict(neuro_extra_sessions=True, friday_rooms=8))]
for name, kw in variants:
    inst, _ = R.build(**kw)
    b, _, ok = day_level_bound(inst, integer=True, time_limit=120)
    out["scenarios"].append(dict(name=name, bound=b * R.SLOT, proved=ok, demand=sum(s.duration for s in inst.surgeries) * R.SLOT))
inst, _ = R.build()
b, assign, _ = day_level_bound(inst, integer=True, time_limit=120)
for l, nm in enumerate(inst.surgeon_names):
    dem = sum(s.duration for s in inst.surgeries if s.surgeon == l) * R.SLOT
    cap = int(inst.surgeon_max[l].sum()) * R.SLOT
    left = sum(s.duration for s in inst.surgeries if s.surgeon == l and s.id not in assign) * R.SLOT
    out["surgeons"].append(dict(name=nm, spec=R.SURGEONS[l][1], capacity=cap, demand=dem, unmet=left))
for d in range(inst.n_days):
    sc, rc = int(inst.surgeon_max[:, d].sum()) * R.SLOT, inst.rooms[d] * 480
    out["days"].append(dict(day=inst.day_names[d], surgeon_minutes=sc, room_minutes=rc, binding="rooms" if rc < sc else "surgeons"))
json.dump(out, open("scenarios/out/analysis.json", "w"), indent=1)
print(json.dumps(out["scenarios"], indent=1))
