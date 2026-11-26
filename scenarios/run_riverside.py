"""Solve the Riverside two-week instance (and what-if variants).

usage: python scenarios/run_riverside.py {baseline|friday8|ortho_leave} [workers]
Budget: 60 min CG (parallel day pricing), 40 min branching, 15 min final IP,
5 min polishing; 200-start constructive heuristic.
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
from orsched.validate import validate_columns

variant = sys.argv[1]
workers = int(sys.argv[2]) if len(sys.argv) > 2 else 4
kw = {"baseline": {}, "friday8": dict(friday_rooms=8), "ortho_leave": dict(extra_leave_week2=("Ortho B",))}[variant]
inst, meta = R.build(**kw)
at_risk = []
if variant == "ortho_leave":
    # Ortho B loses week 2: their mandatory cases due in week 2 cannot all be met
    # (2,105 min due vs 1,440 min left). Keep week-1 deadlines hard; make the
    # week-2-deadline cases optional ("at risk") and let the optimizer choose.
    from orsched.model import Surgery
    ob = inst.surgeon_names.index("Ortho B")
    new = []
    for s_ in inst.surgeries:
        if s_.surgeon == ob and inst.is_mandatory(s_.id) and s_.deadline >= 5:
            at_risk.append(s_.label)
            s_ = Surgery(s_.id, s_.duration, s_.surgeon, inst.n_days, s_.infection, s_.label)
        new.append(s_)
    inst.surgeries = new
os.makedirs("scenarios/out", exist_ok=True)
p = BPCParams(cg_time_limit=3600, branch_time_limit=2400, prove_fraction=0.25, final_ip_time=900,
              polish_time=300, cp_time_limit=2, cp_exact_time_limit=60, workers=1,
              parallel_pricing=workers, heuristic_starts=200,
              checkpoint_path=f"scenarios/out/{variant}.pkl", checkpoint_every=300, max_nodes=10**6)
logf = open(f"scenarios/out/{variant}.log", "a", buffering=1)
t = time.time()
sol = solve(inst, p, log=lambda s: logf.write(s + "\n"))
errs = validate_columns(inst, sol.columns)
heur_cols = sol.stats["heuristic_columns"]
plan = [[i, d, k, s * R.SLOT] for d, k, c in room_layout(sol.columns) for i, s in c.items]
heur = [[i, d, k, s * R.SLOT] for d, k, c in room_layout(heur_cols) for i, s in c.items]
out = dict(variant=variant, lb=sol.lower_bound, ub=sol.upper_bound, gap=sol.gap,
           lb_no_polish=sol.stats["lb_before_polish"], heuristic=sol.stats["heuristic_minutes"],
           heuristic_errors=sol.stats["heuristic_errors"], errors=errs, time=sol.stats["time"],
           nodes=sol.stats["n_nodes"], columns=sol.stats["n_columns"], cuts=sol.stats["n_cuts"],
           root=sol.stats["root"], polish=sol.stats["polish"], trajectory=sol.stats["trajectory"],
           rooms=inst.rooms, surgeon_max=(inst.surgeon_max * R.SLOT).tolist(), cleaning=inst.cleaning * R.SLOT,
           surgeons=inst.surgeon_names, days=inst.day_names,
           at_risk=at_risk, cases=[[s.label, s.surgeon, s.duration * R.SLOT, (s.deadline + 1) if s.deadline < inst.n_days else None,
                   s.infection] for s in inst.surgeries], meta=meta, plan=plan, heuristic_plan=heur)
json.dump(out, open(f"scenarios/out/{variant}.json", "w"))
logf.write(f"FINAL {variant}: LB {sol.lower_bound} UB {sol.upper_bound} gap {sol.gap:.2f}% errors {len(errs)} wall {time.time()-t:.0f}s\n")
