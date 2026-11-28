"""Build the publishable page: web/artifact.html (no doctype/head/body, fetches results.json) + web/results.json."""
import glob, json
from pathlib import Path
root = Path(__file__).resolve().parents[1]
recs = []
for f in sorted(glob.glob(str(root / "benchmarks" / "results_*part*.jsonl"))):
    recs += [json.loads(l) for l in open(f) if l.strip()]
recs.sort(key=lambda r: r["key"])
json.dump(recs, open(root / "web" / "results.json", "w"), separators=(",", ":"))
t = (root / "web" / "template.html").read_text()
t = t.replace('<!doctype html>\n', '').replace('<html lang="en"><head>', '').replace('</head><body>', '').replace('</body></html>', '')
t = t.replace('<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">\n', '<meta name="claude:viewport-fit-sides" content="cover">\n')
t = t.replace('.wrap{max-width:1120px;margin:0 auto;padding:20px 16px 60px}', '.wrap{max-width:1120px;margin:0 auto;padding-block:20px 60px;padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px))}')
t = t.replace('const DATA = /*DATA*/null;', 'let DATA=[];')
t = t.replace('\ninit();\n</script>', '\nfetch("results.json").then(r=>r.json()).then(d=>{DATA=d;init()}).catch(()=>{document.getElementById("view").textContent="Could not load results.json"});\n</script>')
t = t.replace('.cards{display:grid', '.panel,.card{min-width:0}.cards{display:grid')
(root / "web" / "artifact.html").write_text(t)
print(len(recs), "instances")
