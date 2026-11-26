"""Solve the Riverside two-week instance (and what-if variants).

usage: python scenarios/run_riverside.py {baseline|neuro_sessions|ortho_leave} [workers]
Identical settings for every variant: 60 min CG (parallel day pricing), 40 min
branching, 15 min final IP, 5 min polishing, 200-start heuristic, day-level bound.
SMOKE=1 runs a ~3-minute version to test the pipeline.
Writes scenarios/out/<variant>.json (plan, bounds, KPIs) and a checkpoint.
"""
import json, os, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import riverside as R
from orsched import BPCParams, solve
from orsched.kpi import room_layout
from orsched.model import Surgery
from orsched.validate import validate_columns

variant = sys.argv[1]
workers = int(sys.argv[2]) if len(sys.argv) > 2 else 4
smoke = os.environ.get("SMOKE") == "1"
kw = {"baseline": {}, "neuro_sessions": dict(neuro_extra_sessions=True),
      "ortho_leave": dict(extra_leave_week2=("Ortho B",))}[variant]
inst, meta = R.build(**kw)
at_risk, protected, breached = [], [], []
if variant == "ortho_leave":
    # Ortho B has no week-2 OR time. Their week-2-deadline mandatory cases cannot all be met:
    # a day-level knapsack picks which to protect (week-1 sessions), maximising how many
    # patients are protected; the rest are flagged as breached up front and become optional.
    ob = inst.surgeon_names.index("Ortho B")
    prot, brch = R.protect_at_risk(inst, "Ortho B")
    at_risk = [inst.surgeries[i].label for i in prot + brch]
    protected = [inst.surgeries[i].label for i in prot]
    breached = [inst.surgeries[i].label for i in brch]
    new = []
    for s_ in inst.surgeries:
        if s_.id in prot:
            s_ = Surgery(s_.id, s_.duration, s_.surgeon, 4, s_.infection, s_.label)        # must be done in week 1
        elif s_.id in brch:
            s_ = Surgery(s_.id, s_.duration, s_.surgeon, inst.n_days, s_.infection, s_.label)   # optional now
        new.append(s_)
    inst.surgeries = new
os.makedirs("scenarios/out", exist_ok=True)
S = 0.05 if smoke else 1.0
p = BPCParams(cg_time_limit=3600 * S, branch_time_limit=2400 * S, prove_fraction=0.25, final_ip_time=900 * S,
              polish_time=300 * S, cp_time_limit=2, cp_exact_time_limit=60 * (1 if not smoke else 0.2), workers=1,
              parallel_pricing=workers, heuristic_starts=200 if not smoke else 20, day_level_bound=True,
              checkpoint_path=f"scenarios/out/{variant}{'_smoke' if smoke else ''}.pkl", checkpoint_every=300,
              max_nodes=10**6)
tag = variant + ("_smoke" if smoke else "")
logf = open(f"scenarios/out/{tag}.log", "a", buffering=1)
t = time.time()
sol = solve(inst, p, log=lambda s: logf.write(s + "\n"))
errs = validate_columns(inst, sol.columns)
plan = [[i, d, k, s * R.SLOT] for d, k, c in room_layout(sol.columns) for i, s in c.items]
heur = [[i, d, k, s * R.SLOT] for d, k, c in room_layout(sol.stats["heuristic_columns"]) for i, s in c.items]
done = {x[0] for x in plan}
inversions = []
if variant == "ortho_leave":                 # urgent-vs-routine check required by the clinical logic
    ob = inst.surgeon_names.index("Ortho B")
    for i, m in enumerate(meta):
        if inst.surgeries[i].surgeon == ob and m["category"] == 3 and i in done:
            d_ = [x[1] for x in plan if x[0] == i][0]
            if d_ < 5 and breached and any(inst.surgeries[j].label in breached and j not in done for j in range(inst.n)):
                inversions.append(inst.surgeries[i].label)
out = dict(variant=variant, lb=sol.lower_bound, ub=sol.upper_bound, gap=sol.gap,
           lb_no_polish=sol.stats["lb_before_polish"], heuristic=sol.stats["heuristic_minutes"],
           heuristic_errors=sol.stats["heuristic_errors"], errors=errs, time=sol.stats["time"],
           nodes=sol.stats["n_nodes"], columns=sol.stats["n_columns"], cuts=sol.stats["n_cuts"],
           root=sol.stats["root"], polish=sol.stats["polish"], trajectory=sol.stats["trajectory"],
           rooms=inst.rooms, surgeon_max=(inst.surgeon_max * R.SLOT).tolist(), cleaning=inst.cleaning * R.SLOT,
           surgeons=inst.surgeon_names, days=inst.day_names, at_risk=at_risk, protected=protected,
           breached=breached, priority_inversions=inversions, smoke=smoke,
           cases=[[s.label, s.surgeon, s.duration * R.SLOT, (s.deadline + 1) if s.deadline < inst.n_days else None,
                   s.infection] for s in inst.surgeries], meta=meta, plan=plan, heuristic_plan=heur)
json.dump(out, open(f"scenarios/out/{tag}.json", "w"))
logf.write(f"FINAL {tag}: LB {sol.lower_bound} UB {sol.upper_bound} gap {sol.gap:.2f}% errors {len(errs)} wall {time.time()-t:.0f}s\n")
