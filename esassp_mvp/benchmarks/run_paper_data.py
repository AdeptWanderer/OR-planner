"""Benchmark on instances built from the paper's REAL supplement data (Appendix G).
usage: python benchmarks/run_paper_data.py <part> <n_parts> <out.jsonl> [time_scale]"""
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from esassp import Params, paper_instance
from esassp.experiment import run_experiment

GRID = [("Case-15-b52-c1", "Case", 15, "(5,2)", 1), ("Case-20-b52-c1", "Case", 20, "(5,2)", 1),
        ("R-20-b52-c1", "R", 20, "(5,2)", 1), ("Case-25-b107-c1", "Case", 25, "(10,7)", 1),
        ("M-24-b52-c1", "M", 24, "(5,2)", 1), ("R-30-b107-c1", "R", 30, "(10,7)", 1)]

def main():
    part, nparts, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    scale = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
    done = {json.loads(l)["key"] for l in open(out)} if Path(out).exists() else set()
    for idx, (key, kind, n, beds, cost) in enumerate(GRID):
        if idx % nparts != part or key in done:
            continue
        inst = paper_instance(kind, n, beds=beds, cost=cost); inst.name = key
        p = Params(n_scenarios=10, time_limit=100 * scale, master_time=50 * scale, gap=2e-3, threads=2, seed=1)
        rec = run_experiment(inst, p, n_oos=5000, baseline_time_limit=40 * scale, log=lambda s: print(f"[{key}] {s}", flush=True))
        rec["key"] = key; rec["data"] = "paper supplement Appendix G"
        open(out, "a").write(json.dumps(rec) + "\n")
        print(f"[{key}] done", flush=True)

if __name__ == "__main__":
    main()
