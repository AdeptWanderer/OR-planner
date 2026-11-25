"""Paper-budget benchmark: 2 h CG + 2 h branching (15 min proving) + 1 h IP,
CP fast-phase limit 10 s, one core per instance (as in the paper, Sec. 6.2).
Polishing (not in the paper) runs afterwards for 10 min; gaps are reported
before and after it.  Resumable: finished instances are skipped.

usage: python benchmarks/run_paper_budget.py [parallel=4] [only=A80,B60,...]
"""
import json, math, os, sys, time
from multiprocessing import Pool
sys.path.insert(0, ".")

OUT = "benchmarks/paper_budget.jsonl"
CASES = [("A", n) for n in (40, 60, 80, 100, 120)] + [("B", n) for n in (40, 60, 80, 100, 120)] + \
        [("C", n) for n in (60, 80, 100)]


MEM_LIMIT_GB = float(os.environ.get("MEM_LIMIT_GB", "3.5"))   # per worker (4 x 3.5 GB < 16 GB)


def run(case):
    import resource
    resource.setrlimit(resource.RLIMIT_AS, (int(MEM_LIMIT_GB * 2**30),) * 2)   # fail fast instead of OOM-killing the box
    from orsched import BPCParams, generate_instance, solve, validate_solution
    S, n = case
    inst = generate_instance(n, S, seed=1)
    p = BPCParams(cp_time_limit=10, cp_exact_time_limit=600, cg_time_limit=7200, branch_time_limit=7200,
                  prove_fraction=900 / 7200, final_ip_time=3600, polish_time=600, max_nodes=10**6, workers=1,
                  checkpoint_path=f"benchmarks/ckpt/{S}{n}.pkl", checkpoint_every=300, log_cap=20000)
    logf = open(f"benchmarks/logs/{S}{n}.log", "a", buffering=1)
    t = time.time()
    sol = solve(inst, p, log=lambda s: logf.write(s + "\n"))
    st = sol.stats
    lb0 = st["lb_before_polish"]
    gap0 = 100 * (sol.upper_bound - lb0) / sol.upper_bound if sol.upper_bound else 0
    r = dict(set=S, n=n, rooms=inst.rooms[0], heuristic=st["heuristic_minutes"],
             lb_no_polish=lb0, gap_no_polish=round(gap0, 2),
             lb=sol.lower_bound, ub=sol.upper_bound, gap=round(sol.gap, 2),
             root_lp=round(st["root"]["lp"] * inst.slot_minutes, 1), root_converged=st["root"]["converged"],
             root_time=round(st["root"]["time"], 1), columns=st["n_columns"], cuts=st["n_cuts"],
             nodes=st["n_nodes"], time=round(st["time"], 1), wall_this_process=round(time.time() - t, 1),
             time_used=st["time_used"], trajectory=st["trajectory"], valid=not validate_solution(inst, sol))
    logf.write("RESULT " + json.dumps(r) + "\n")
    return r


if __name__ == "__main__":
    par = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    os.makedirs("benchmarks/logs", exist_ok=True)
    os.makedirs("benchmarks/ckpt", exist_ok=True)
    done = set()
    if os.path.exists(OUT):
        for l in open(OUT):
            if l.strip():
                r = json.loads(l); done.add((r["set"], r["n"]))
    only = None
    if len(sys.argv) > 2 and sys.argv[2]:
        only = {(x[0], int(x[1:])) for x in sys.argv[2].split(",")}
    todo = [c for c in CASES if c not in done and (only is None or c in only)]
    # longest first so the waves pack well
    todo.sort(key=lambda c: (c[0] == "A" and c[1] <= 40) or (c[0] == "C" and c[1] == 80))
    with Pool(par, maxtasksperchild=1) as pool:
        futs = [pool.apply_async(run, (c,)) for c in todo]
        failed = 0
        for c, fu in zip(todo, futs):
            try:
                r = fu.get()
            except Exception as e:          # e.g. MemoryError: checkpoint kept, supervisor reruns
                print(f"FAILED {c}: {e!r}", flush=True); failed += 1; continue
            with open(OUT, "a") as f:
                f.write(json.dumps(r) + "\n")
            print(json.dumps(r), flush=True)
    sys.exit(1 if failed else 0)
