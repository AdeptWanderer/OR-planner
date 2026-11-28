"""Markdown tables from benchmarks/results_*part*.jsonl (used in the README)."""
import glob, json
from pathlib import Path
root = Path(__file__).resolve().parents[1]
recs = []
for f in sorted(glob.glob(str(root / "benchmarks" / "results_*part*.jsonl"))):
    recs += [json.loads(l) for l in open(f) if l.strip()]
recs.sort(key=lambda r: r["key"])
print("| Instance | Model | Scheduled | Solve s | Gap % | Mean cost Δ=0 | Mean cost Δ=0.5 | max ICU PE (Δ=0.2) | RelTC vs SP (Δ=0) |")
print("|---|---|---|---|---|---|---|---|---|")
for r in recs:
    for m in ("SP", "DRO-M", "DRO-W"):
        if m not in r["runs"]:
            continue
        run, e0, e5, e2 = r["runs"][m], r["eval"][m]["0.0"], r["eval"][m]["0.5"], r["eval"][m]["0.2"]
        rt = e0.get("rel_tc_vs_sp")
        print(f"| {r['key']} (I={len(r['instance']['surgeries'])}) | {m} | {run['scheduled']} | {run['time']:.0f} | {100 * run['gap']:.2f} | "
              f"{e0['mean_total_cost']:,.0f} | {e5['mean_total_cost']:,.0f} | {e2['max_pe_icu']:.2f} | {'' if rt is None or m == 'SP' else f'{rt:.2f}'} |")
print()
print("| Instance | Variant | Scheduled | mean cost Δ=0 | RelTC w/o vs integrated (Δ=0) |")
print("|---|---|---|---|---|")
for r in recs:
    for m, e in r["eval"].items():
        if " w/o" in m:
            print(f"| {r['key']} | {m} | {r['runs'][m]['scheduled']} | {e['0.0']['mean_total_cost']:,.0f} | {e['0.0'].get('rel_tc_vs_integrated', float('nan')):.2f} |")
