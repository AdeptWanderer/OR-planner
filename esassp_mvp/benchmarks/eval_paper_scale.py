"""Evaluate paper-scale runs out of sample (N' = 10 000, as in the paper) and compare with the paper's published numbers.
usage: python benchmarks/eval_paper_scale.py [tag_prefix e.g. R70_10-7_c1_s1]"""
import glob, json, sys
from pathlib import Path
import numpy as np
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from esassp import paper_instance, sample_scenarios, evaluate
from esassp.model import Solution
from esassp.evaluate import rel_tc

# Paper, Table 1 (R-70 column): optimal surgery volume;  Table I.1: CPU seconds;  Table 5: RelTC (R-70, Cost 2, (10,7)/(5,2))
PAPER_VOLUME = {("10,7", 1): dict(SP=65, **{"DRO-M": 54, "DRO-W": 57}), ("5,2", 1): dict(SP=50, **{"DRO-M": 44, "DRO-W": 44}),
                ("10,7", 2): dict(SP=62, **{"DRO-M": 54, "DRO-W": 54}), ("5,2", 2): dict(SP=50, **{"DRO-M": 43, "DRO-W": 43})}
PAPER_CPU = {("10,7", 1): dict(SP=2932, **{"DRO-M": 610, "DRO-W": 1622}), ("5,2", 1): dict(SP=1218, **{"DRO-M": 264, "DRO-W": 1100}),
             ("10,7", 2): dict(SP=2344, **{"DRO-M": 536, "DRO-W": 3158}), ("5,2", 2): dict(SP=1185, **{"DRO-M": 326, "DRO-W": 1562})}
PAPER_RELTC_IS = {("10,7", 2): {"DRO-M": 0.64, "DRO-W": 0.65}, ("5,2", 2): {"DRO-M": 0.69, "DRO-W": 0.53}}

def load(prefix):
    recs = {}
    for f in sorted(glob.glob(str(root / "benchmarks" / "paper_scale" / f"{prefix}*.json"))):
        r = json.load(open(f)); recs[r["tag"]] = r
    return recs

def solution_of(inst, run):
    block = -np.ones(inst.I, int); pos = -np.ones(inst.I, int); start = {}
    for i, b, k, s in run["plan"]:
        block[i], pos[i] = b, k; start[(b, k)] = s
    for b, blk in enumerate(inst.blocks):
        for k in range(blk.K):
            start.setdefault((b, k), 0.0)
    return Solution(block, pos, start)

def main():
    prefix = sys.argv[1] if len(sys.argv) > 1 else "R70"
    recs = load(prefix)
    if not recs:
        print("no finished runs for", prefix); return
    out = {}
    for tag, r in recs.items():
        inst = paper_instance(r["kind"], r["I"], beds=r["beds"], cost=r["cost"], seed=r["seed"])
        sol = solution_of(inst, r["run"])
        row = dict(model=r["model"], scheduled=r["run"]["scheduled"], solve_s=round(r["run"]["time"]), gap=r["run"]["gap"], status=r["run"]["status"],
                   iters=r["run"]["stats"].get("iters"), obj=r["run"]["objective"], lb=r["run"]["lb"])
        for dl in (0.0, 0.2, 0.5):
            sc = sample_scenarios(inst, 10000, np.random.default_rng(777), delta=dl)
            ev = evaluate(inst, sol, sc)
            row[f"d{dl}"] = dict(mean_total=float(ev["mean_total"]), q90=float(ev["q90_total"]), pe_icu=np.asarray(ev["pe_icu"]).tolist(),
                                 pe_ward=np.asarray(ev["pe_ward"]).tolist(), me_icu=np.asarray(ev["me_icu"]).tolist(), me_ward=np.asarray(ev["me_ward"]).tolist(),
                                 wait=float(np.mean(ev["wait"])), idle=float(np.mean(ev["idle"])), over=float(np.mean(ev["over"])), tc=ev["total"].tolist())
        out[tag] = (r, row)
    # report
    print(f"{'model':8} {'sched':>5} {'paper':>5} {'solve s':>8} {'paper s':>8} {'gap %':>7} {'iters':>5} | mean cost Δ0 / Δ0.2 / Δ0.5 | maxPE icu/ward Δ0 | wait min | idle h")
    base = None
    for tag, (r, row) in sorted(out.items(), key=lambda kv: kv[1][1]["model"]):
        key = (r["beds"].strip("()"), r["cost"])
        pv = PAPER_VOLUME.get(key, {}).get(r["model"], "-"); pc = PAPER_CPU.get(key, {}).get(r["model"], "-")
        d0, d2, d5 = row["d0.0"], row["d0.2"], row["d0.5"]
        print(f"{r['model']:8} {row['scheduled']:5d} {pv!s:>5} {row['solve_s']:8d} {pc!s:>8} {100*row['gap']:7.2f} {row['iters'] or 0:5d} | "
              f"{d0['mean_total']:9.0f} {d2['mean_total']:9.0f} {d5['mean_total']:9.0f} | {max(d0['pe_icu']):.2f}/{max(d0['pe_ward']):.2f} | {d0['wait']:6.0f} | {d0['idle']/60:6.1f}")
        if r["model"] == "SP":
            base = tag
    if base:
        sp = np.array(out[base][1]["d0.0"]["tc"])
        for tag, (r, row) in out.items():
            if r["model"] != "SP":
                print(f"RelTC SP vs {r['model']} (Δ=0, out-of-sample): {rel_tc(sp, np.array(row['d0.0']['tc'])):.3f}   "
                      f"paper (Cost 2): {PAPER_RELTC_IS.get((r['beds'].strip('()'), r['cost']), {}).get(r['model'], '-')}")
    json.dump({t: row for t, (r, row) in out.items()}, open(root / "benchmarks" / f"paper_scale_eval_{prefix}.json", "w"))

if __name__ == "__main__":
    main()
