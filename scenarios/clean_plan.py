"""Real plans (day-decomposition CP-SAT) for the baseline with a different cleaning time.
usage: python scenarios/clean_plan.py MINUTES"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parent))
import riverside as R
from orsched.bounds import day_level_bound
from orsched.dayplan import dayplan_heuristic
from orsched.validate import validate_columns
m = int(sys.argv[1])
inst, _ = R.build(cleaning_min=m)
cols, _ = dayplan_heuristic(inst, time_per_day=30, workers=4)
errs = validate_columns(inst, cols)
ub, _, _ = day_level_bound(inst, integer=True)
plan = sum(c.value(inst) for c in cols) * R.SLOT if not errs else None
cleans = 0
for c in cols:
    it = sorted(c.items, key=lambda z: z[1])
    cleans += sum(1 for (a, _), (b, _) in zip(it, it[1:]) if inst.clean(a, b))
json.dump(dict(cleaning=m, plan=plan, bound=ub * R.SLOT, errors=len(errs), cleans=cleans), open(f"scenarios/out/clean_{m}.json", "w"))
print("FINAL", m, plan, ub * R.SLOT, len(errs), cleans)
