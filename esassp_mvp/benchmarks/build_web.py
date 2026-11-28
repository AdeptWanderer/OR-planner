"""Embed benchmark results into web/template.html -> web/index.html (self-contained)."""
import glob, json, sys
from pathlib import Path
root = Path(__file__).resolve().parents[1]
recs = []
for f in sorted(glob.glob(str(root / "benchmarks" / "results_part*.jsonl"))):
    recs += [json.loads(l) for l in open(f) if l.strip()]
order = [json.loads(json.dumps(r)) for r in recs]
order.sort(key=lambda r: (r["key"]))
html = (root / "web" / "template.html").read_text().replace("/*DATA*/null", json.dumps(order, separators=(",", ":")))
(root / "web" / "index.html").write_text(html)
print(f"{len(order)} instances, {len(html) / 1e6:.2f} MB")
