"""Reproduce the structure of Tables 3/6 of the paper on generated instances.

usage: python benchmarks/run_benchmark.py [time_scale]
"""
import json, sys, time
sys.path.insert(0, ".")
from orsched import BPCParams, generate_instance, solve, validate_solution

scale = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
cases = [("A", n) for n in (40, 60, 80, 100, 120)] + [("B", n) for n in (40, 60, 80, 100, 120)] + \
        [("C", n) for n in (60, 80, 100)]
out = []
for S, n in cases:
    inst = generate_instance(n, S, seed=1)
    p = BPCParams(cg_time_limit=120 * scale, branch_time_limit=120 * scale, final_ip_time=60 * scale)
    t = time.time()
    sol = solve(inst, p)
    r = dict(set=S, n=n, rooms=inst.rooms[0], heuristic=sol.stats["heuristic_minutes"],
             lb=sol.lower_bound, ub=sol.upper_bound, gap=round(sol.gap, 2),
             root_lp=sol.stats["root"]["lp"] * inst.slot_minutes,
             root_converged=sol.stats["root"]["converged"], columns=sol.stats["n_columns"],
             cuts=sol.stats["n_cuts"], nodes=sol.stats["n_nodes"], time=round(time.time() - t, 1),
             valid=not validate_solution(inst, sol))
    print(json.dumps(r), flush=True)
    out.append(r)
json.dump(out, open("benchmarks/results.json", "w"), indent=1)
