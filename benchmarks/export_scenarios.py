"""Run a grid of scenarios and export compact JSON for the static web dashboard.

usage: python benchmarks/export_scenarios.py <part> <n_parts> <out.jsonl>
"""
import json, sys, time
sys.path.insert(0, ".")
import numpy as np
from orsched import BPCParams, generate_instance, solve, validate_solution
from orsched.generator import room_sizing_heuristic

part, nparts, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]

grid = []
for S in "ABC":
    for n in (30, 50, 70):
        for oct_ in (0, 30, 60):
            for rooms in ("default", "tight"):
                grid.append(dict(set=S, n=n, oct=oct_, rooms=rooms, variant="full"))
for S in "ABC":
    for v in ("no_dominance", "no_cuts", "no_branching", "no_polish"):
        grid.append(dict(set=S, n=50, oct=30, rooms="default", variant=v))

BASE = dict(cg_time_limit=30, branch_time_limit=30, final_ip_time=15, polish_time=15, workers=2,
            cp_time_limit=2.0, cp_exact_time_limit=15.0)
VAR = {"full": {}, "no_dominance": dict(use_dominance=False, use_fid=False), "no_cuts": dict(use_cuts=False),
       "no_branching": dict(branching=False), "no_polish": dict(polish_time=0)}


def cols_items(inst, cols):
    out = []
    per_day = {}
    for c in sorted(cols, key=lambda c: (c.day, c.items[0][1] if c.items else 0, c.items)):
        k = per_day.get(c.day, 0); per_day[c.day] = k + 1
        for i, s in c.items:
            out.append([i, c.day, k, s * inst.slot_minutes])
    return out


with open(out, "w") as f:
    for idx, g in enumerate(grid):
        if idx % nparts != part:
            continue
        rooms = None
        if g["rooms"] == "tight":
            base = generate_instance(g["n"], g["set"], seed=1, cleaning_minutes=g["oct"])
            rooms = max(1, base.rooms[0] - 1)
        inst = generate_instance(g["n"], g["set"], seed=1, cleaning_minutes=g["oct"], rooms=rooms)
        p = BPCParams(**{**BASE, **VAR[g["variant"]]})
        t = time.time()
        sol = solve(inst, p)
        m = inst.slot_minutes
        st = sol.stats
        it = [r for r in st["iter_log"] if r["node"] == 0]
        pr = st["pricing_log"]
        cand = [max(1, r["candidates"]) for r in pr]
        def avg(xs): return float(np.mean(xs)) if xs else 0.0
        rec = dict(
            key=f"{g['set']}-{g['n']}-{g['oct']}-{g['rooms']}-{g['variant']}", **g,
            rooms_per_day=inst.rooms, day_minutes=inst.slots_per_day * m, slot=m,
            cases=[[s.label, s.surgeon, s.duration * m, (s.deadline + 1) if s.deadline < inst.n_days else None,
                    s.infection] for s in inst.surgeries],
            surgeon_max=(inst.surgeon_max * m).tolist(),
            schedule=cols_items(inst, sol.columns),
            heuristic=cols_items(inst, st["heuristic_columns"]),
            lb=sol.lower_bound, ub=sol.upper_bound, gap=round(sol.gap, 3), heur=st["heuristic_minutes"],
            root_lp=round(st["root"]["lp"] * m, 1), root_converged=st["root"]["converged"],
            root_iters=st["root"]["iterations"], time=round(st["time"], 1), columns=st["n_columns"],
            cuts=st["n_cuts"], nodes=st["n_nodes"],
            iters=[[r["iteration"], round(r["rmp_obj"] * m, 1), round(min(r["lagrangian_bound"] * m, 1e7), 1),
                    r["n_columns"], r["new_columns"], r["phase"], round(r["artificial"], 4)] for r in it],
            node_log=[[n["id"], n["parent"], n["label"], n["status"],
                       None if n["bound"] is None else round(n["bound"] * m, 1), n["depth"]] for n in st["node_log"][:80]],
            pricing=dict(calls=len(pr), cp_time=round(sum(r["cp_time"] for r in pr), 2),
                         dr1=avg([r["dr1_removed"] for r in pr]),
                         dr2=100 * avg([r["dr2"] / c for r, c in zip(pr, cand)]),
                         dr3=100 * avg([r["dr3"] / c for r, c in zip(pr, cand)]),
                         dr4=100 * avg([r["dr4"] / c for r, c in zip(pr, cand)]),
                         cond=100 * avg([r["conditioned"] / c for r, c in zip(pr, cand)]),
                         fid1=100 * avg([float(r["fid_infeasible"]) for r in pr]),
                         fid2=100 * avg([float(r["fid2_infeasible"]) for r in pr]),
                         knap=100 * avg([float(r["hit_knapsack_bound"]) for r in pr]),
                         cols_by_day=[sum(r["n_columns"] for r in pr if r["day"] == d) for d in range(inst.n_days)]),
            polish=st.get("polish"), errors=len(validate_solution(inst, sol)),
        )
        f.write(json.dumps(rec, separators=(",", ":")) + "\n"); f.flush()
        print(rec["key"], rec["lb"], rec["ub"], rec["gap"], rec["time"], flush=True)
