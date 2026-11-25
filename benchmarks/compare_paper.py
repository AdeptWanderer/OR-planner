"""Build the paper-vs-ours comparison table from a results file.

usage: python benchmarks/compare_paper.py verification/reports/paper_budget_results.jsonl

Paper numbers are copied verbatim from Table 6 (branch-and-price-and-cut) of
Hashemi Doulabi, Rousseau & Pesant (2016).  Table 6 reports objective values
in 5-minute time slots; they are multiplied by 5 here to give minutes.
"""
import json
import sys

# set, n : (initial heuristic, UB, LB, gap %, total time s)  -- Table 6
PAPER = {
    ("A", 40): (1321, 1403, 1403, 0.00, 84), ("A", 60): (1814, 1971, 1960, 0.55, 8659),
    ("A", 80): (1894, 2203, 2174, 1.30, 8503), ("A", 100): (1968, 2297, 2255, 1.81, 9603),
    ("A", 120): (2014, 2298, 2266, 1.40, 8295),
    ("B", 40): (1193, 1361, 1361, 0.00, 2469), ("B", 60): (1601, 1853, 1843, 0.55, 8177),
    ("B", 80): (1824, 2161, 2132, 1.32, 9784), ("B", 100): (1993, 2401, 2337, 2.65, 14676),
    ("B", 120): (2004, 2457, 2355, 4.13, 13390),
    ("C", 60): (507, 576, 571, 0.94, 11333), ("C", 80): (688, 792, 713, 9.92, 15580),
    ("C", 100): (853, 978, 860, 12.01, 15545),
}
DUPLICATES = {("B", 100): ("A", 100), ("B", 120): ("A", 120)}   # same cases, same rooms in our generator


def main(path):
    rows = {}
    for l in open(path):
        if l.strip():
            r = json.loads(l); rows[(r["set"], r["n"])] = r
    keys = [k for k in PAPER if k in rows]
    out = ["| Set | Cases | Rooms | Paper gap % | Our gap %, paper method | Our gap %, + polishing | "
           "Paper gain over heuristic | Our gain over heuristic | Paper time (s) | Our time (s) | B&B nodes |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    for k in keys:
        p, r = PAPER[k], rows[k]
        pg = 100 * (p[2] - p[0]) / p[0]
        og = 100 * (r["lb_no_polish"] - r["heuristic"]) / r["heuristic"]
        dup = " (= A%d)" % k[1] if k in DUPLICATES else ""
        out.append(f"| {k[0]} | {k[1]}{dup} | {r['rooms']} | {p[3]:.2f} | **{r['gap_no_polish']:.2f}** | {r['gap']:.2f} | "
                   f"+{pg:.1f}% | +{og:.1f}% | {p[4]:,} | {r['time']:,.0f} | {r['nodes']:,} |")

    def avg(ks, f):
        v = [f(k) for k in ks]
        return sum(v) / len(v)
    uniq = [k for k in keys if k not in DUPLICATES]
    lines = ["", "| Average over | Instances | Paper gap % | Our gap %, paper method | Our gap %, + polishing |",
             "|---|---|---|---|---|"]
    for label, ks in [("all 13 sizes", keys), ("11 distinct instances", uniq),
                      ("sets A + B", [k for k in keys if k[0] in "AB"]),
                      ("set C", [k for k in keys if k[0] == "C"])]:
        lines.append(f"| {label} | {len(ks)} | {avg(ks, lambda k: PAPER[k][3]):.2f} | "
                     f"{avg(ks, lambda k: rows[k]['gap_no_polish']):.2f} | {avg(ks, lambda k: rows[k]['gap']):.2f} |")
    print("\n".join(out + lines))
    proven = [k for k in keys if rows[k]["gap_no_polish"] < 0.005]
    print(f"\nProven optimal with the paper's method: {len(proven)}/{len(keys)} "
          f"(paper: {sum(1 for k in keys if PAPER[k][3] == 0)}/{len(keys)})")
    print("All schedules valid:", all(rows[k]["valid"] for k in keys))


if __name__ == "__main__":
    main(sys.argv[1])
