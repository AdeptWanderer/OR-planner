"""Run the experiment grid (reduced analogue of the paper's §6) and checkpoint after every instance.

usage: python benchmarks/run_experiments.py <part> <n_parts> <out.jsonl> [time_scale]
Instances are generated with the paper's recipe but are SYNTHETIC and far smaller than R-70 / Case-100 (HiGHS, not CPLEX).
"""
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from esassp import Costs, Params, generate_instance
from esassp.experiment import run_experiment

GRID = [  # key, library, I, ORs, beds, cost, seed
    ("cardiac-12-b52-c1", "cardiac", 12, 1, "(5,2)", 1, 1),
    ("cardiac-16-b52-c1", "cardiac", 16, 1, "(5,2)", 1, 1),
    ("cardiac-20-b52-c1", "cardiac", 20, 2, "(5,2)", 1, 1),
    ("cardiac-20-b107-c1", "cardiac", 20, 2, "(10,7)", 1, 1),
    ("cardiac-20-b52-c2", "cardiac", 20, 2, "(5,2)", 2, 1),
    ("cardiac-24-b52-c1", "cardiac", 24, 2, "(5,2)", 1, 1),
    ("general-20-b52-c1", "general", 20, 2, "(5,2)", 1, 2),
    ("general-30-b107-c1", "general", 30, 2, "(10,7)", 1, 2),
]


def main():
    part, nparts, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    scale = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
    done = set()
    if Path(out).exists():
        done = {json.loads(l)["key"] for l in open(out) if l.strip()}
    for idx, (key, lib, n, ors, beds, cost, seed) in enumerate(GRID):
        if idx % nparts != part or key in done:
            continue
        inst = generate_instance(n, lib, seed, n_ors=ors, beds=beds, costs=Costs(c_o=26.0 if cost == 1 else 100.0), name=key)
        p = Params(n_scenarios=10, time_limit=100 * scale, master_time=50 * scale, gap=2e-3, threads=2, seed=1)
        t = time.time()
        rec = run_experiment(inst, p, n_oos=5000, baseline_time_limit=40 * scale, log=lambda s: print(f"[{key}] {s}", flush=True))
        rec["key"] = key
        with open(out, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"[{key}] done in {time.time() - t:.0f}s", flush=True)


if __name__ == "__main__":
    main()
