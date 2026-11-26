"""Riverside: valid plans from the day-decomposition heuristic, certified by the
day-level relaxation bound (both extensions; see orsched/dayplan.py, orsched/bounds.py).
usage: python scenarios/solve_fast.py {baseline|neuro_sessions|ortho_leave}"""
import json, os, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import riverside as R
from orsched.bounds import day_level_bound
from orsched.dayplan import dayplan_heuristic
from orsched.heuristic import constructive_heuristic
from orsched.kpi import room_layout
from orsched.model import Surgery
from orsched.validate import validate_columns

variant = sys.argv[1]
kw = {"baseline": {}, "neuro_sessions": dict(neuro_extra_sessions=True), "ortho_leave": dict(extra_leave_week2=("Ortho B",))}[variant]
inst, meta = R.build(**kw)
at_risk, protected, breached = [], [], []
if variant == "ortho_leave":
    prot, brch = R.protect_at_risk(inst, "Ortho B")
    at_risk = [inst.surgeries[i].label for i in prot + brch]
    protected = [inst.surgeries[i].label for i in prot]; breached = [inst.surgeries[i].label for i in brch]
    inst.surgeries = [Surgery(s.id, s.duration, s.surgeon, 4, s.infection, s.label) if s.id in prot else
                      Surgery(s.id, s.duration, s.surgeon, inst.n_days, s.infection, s.label) if s.id in brch else s
                      for s in inst.surgeries]
t = time.time()
ub, _, proved = day_level_bound(inst, integer=True, time_limit=300)
cols, info = dayplan_heuristic(inst, time_per_day=30, workers=4, log=print)
errs = validate_columns(inst, cols)
# best valid greedy plan (paper's constructive heuristic, 200 random tie-breaks) for comparison
hbest, hcols = None, None
for k in range(200):
    c, _ = constructive_heuristic(inst, np.random.default_rng(1000 + k))
    if not validate_columns(inst, c):
        v = sum(x.value(inst) for x in c)
        if hbest is None or v > hbest: hbest, hcols = v, c
lb = sum(c.value(inst) for c in cols) if not errs else 0
m = R.SLOT
plan = [[i, d, k, s * m] for d, k, c in room_layout(cols) for i, s in c.items]
heur = [[i, d, k, s * m] for d, k, c in room_layout(hcols) for i, s in c.items] if hcols else []
out = dict(variant=variant, lb=lb * m, ub=ub * m, ub_proved=proved, gap=100 * (ub - lb) / ub if ub else 0,
           heuristic=(hbest or 0) * m, heuristic_errors=[] if hcols else ["no valid greedy plan in 200 starts"],
           errors=errs, time=time.time() - t, method="day-level bound + day-decomposition CP-SAT",
           rooms=inst.rooms, surgeon_max=(inst.surgeon_max * m).tolist(), cleaning=inst.cleaning * m,
           surgeons=inst.surgeon_names, days=inst.day_names, at_risk=at_risk, protected=protected, breached=breached,
           cases=[[s.label, s.surgeon, s.duration * m, (s.deadline + 1) if s.deadline < inst.n_days else None, s.infection]
                  for s in inst.surgeries], meta=meta, plan=plan, heuristic_plan=heur)
os.makedirs("scenarios/out", exist_ok=True)
json.dump(out, open(f"scenarios/out/{variant}.json", "w"))
print(f"FINAL {variant}: plan {lb*m} bound {ub*m} (proved {proved}) gap {out['gap']:.2f}% errors {len(errs)} greedy {(hbest or 0)*m} time {out['time']:.0f}s")
