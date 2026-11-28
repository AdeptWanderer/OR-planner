"""Paper-scale run: ONE model on ONE instance with the paper's budget (N = 50, up to 2 h).

usage: python benchmarks/run_paper_scale.py <model: SP|DRO-M|DRO-W> <kind: R|Case|M> <I> <beds: 10,7|5,2> <cost 1|2> <seed> <seconds> [downstream 0|1] [scheduling 0|1]
Writes benchmarks/paper_scale/<tag>.json (final record) and <tag>.log (progress, flushed). One thread per process (HiGHS' MIP is mostly serial).
"""
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from esassp.milp import set_threads
set_threads(1)
from esassp import Params, paper_instance, solve
from esassp.experiment import instance_record, solution_record

model, kind, I, beds, cost, seed, secs = sys.argv[1], sys.argv[2], int(sys.argv[3]), "(" + sys.argv[4] + ")", int(sys.argv[5]), int(sys.argv[6]), float(sys.argv[7])
down = bool(int(sys.argv[8])) if len(sys.argv) > 8 else True
sched = bool(int(sys.argv[9])) if len(sys.argv) > 9 else True
tag = f"{kind}{I}_{sys.argv[4].replace(',', '-')}_c{cost}_s{seed}_{model}" + ("" if down else "_nodown") + ("" if sched else "_nosched")
out = Path(__file__).resolve().parent / "paper_scale"
logf = open(out / f"{tag}.log", "a")
def log(s):
    logf.write(f"{time.strftime('%H:%M:%S')} {s}\n"); logf.flush()
inst = paper_instance(kind, I, beds=beds, cost=cost, seed=seed)
# seed controls the sampled scenarios (and type counts are deterministic for Case/R); N = 50 as in the paper
p = Params(n_scenarios=50, time_limit=secs, master_time=min(1800.0, secs / 2), gap=1e-3, threads=1, seed=seed, eps_d=50.0, eps_l=5.0, log=log)
log(f"start {tag}: I={inst.I} B={inst.B} time_limit={secs}")
t = time.time()
res = solve(inst, model, p, downstream=down, scheduling=sched)
rec = dict(tag=tag, model=model, kind=kind, I=I, beds=beds, cost=cost, seed=seed, time_limit=secs, downstream=down, scheduling=sched,
           instance=instance_record(inst), run=solution_record(inst, res), wall=time.time() - t)
(out / f"{tag}.json").write_text(json.dumps(rec))
log(f"done: obj {res.objective:.1f} lb {res.lower_bound:.1f} gap {100 * res.gap:.2f}% status {res.status} {time.time() - t:.0f}s")
