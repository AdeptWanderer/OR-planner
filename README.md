# OR Planner — integrated operating-room planning & scheduling

An MVP implementation of

> S. H. Hashemi Doulabi, L.-M. Rousseau, G. Pesant (2016). *A Constraint-Programming-Based
> Branch-and-Price-and-Cut Approach for Operating Room Planning and Scheduling.*
> INFORMS Journal on Computing 28(3):432–448.

It decides, for a planning horizon (usually a week), **which surgeries are performed, on which day, in which
operating room and at what time**, maximising total scheduled surgery time. It respects:

* mandatory cases (deadline inside the horizon) are done by their deadline; optional cases may be postponed;
* surgeons' maximum operating minutes per day;
* no surgeon in two rooms at once (open scheduling, rooms shared between surgeons);
* obligatory room cleaning (OCT) when switching between infection types;
* rooms available per day and the room day length.

## Quick start

```bash
pip install -r requirements.txt
```

```bash
streamlit run app.py
```

Open http://localhost:8501. The dashboard has six tabs:

| Tab | What you get |
|---|---|
| ① How it works | Diagram and short explanation of the algorithm, the key formulas, how this implementation differs from the paper, and a trace of your last run |
| ② Instance & data | Editable tables of cases, surgeon limits and rooms per day; CSV import/export; demand vs. capacity |
| ③ Solver run | Lower/upper bound, gap, convergence per B&B node, pricing statistics (paper Table 2), branch-and-bound tree, log |
| ④ Schedule | Gantt chart per day (rooms, cleaning blocks, hatched infectious cases), surgeon timelines, downloadable schedule |
| ⑤ Hospital KPIs | OR utilisation heat map, where OR time goes, surgeon workload, deadline slack, waiting list, comparison with the heuristic |
| ⑥ Compare methods | Heuristic vs. B&P&C vs. a monolithic CP-SAT model, plus the paper's reported gaps |

The sidebar controls the instance (paper sets A/B/C, size, horizon, surgeons, rooms, infection mix, OCT,
deadlines, seed) and the algorithm (each enhancement can be switched on or off, plus every time limit).

## Using real data

Upload two CSV files (templates can be downloaded from the app):

`surgeries.csv`

| case | surgeon | duration_min | deadline_day | infection |
|---|---|---|---|---|
| P-0012 hip | Dr. A | 150 | 2 | 0 |
| P-0044 lap chole | Dr. B | 95 |  | 1 |

`deadline_day` is 1-based. Leave it empty, or set it past the horizon, for a case that may be postponed.
`infection` is 0 for non-infectious and k for infection type k.

`surgeons.csv`: one row per surgeon, one column per day giving the maximum operating minutes, e.g.
`surgeon,Mon,Tue,Wed,Thu,Fri`.

Durations are rounded **up** to the time slot (default 5 min).

## Using the solver from Python

```python
from orsched import generate_instance, solve, BPCParams, validate_solution
from orsched.kpi import schedule_df

inst = generate_instance(60, "B", seed=1)
sol = solve(inst, BPCParams(cg_time_limit=120, branch_time_limit=120), log=print)
print(sol.lower_bound, sol.upper_bound, f"{sol.gap:.2f}%")
assert validate_solution(inst, sol) == []
print(schedule_df(inst, sol.columns))
```

## Architecture

```
orsched/
  model.py      Instance, Surgery, Column (a room-day schedule), Solution
  generator.py  instance generator of §6.1 (sets A, B, C + room-sizing heuristic)
  heuristic.py  constructive heuristic of §6.1 (initial columns, deadline repair)
  master.py     restricted master problem (1)–(7) on HiGHS, duals → pricing data
  pricing.py    CP-SAT pricing subproblem (17)–(19), dominance rules 1–4, knapsack bound (§4.2)
  cuts.py       lifted minimal cover inequalities (§5.3)
  bpc.py        price-and-cut (Algorithm 1) and branch-and-price-and-cut (Algorithm 2)
  compact.py    monolithic CP-SAT baseline (validation / comparison)
  validate.py   independent feasibility checker
  kpi.py        schedule tables and hospital KPIs
  dataio.py     CSV import / export
app.py          Streamlit dashboard
tests/          pytest suite (includes exact cross-checks against the monolithic model)
benchmarks/     reproduction of the paper's experimental setup
```

### Algorithm, as implemented

1. **Initial columns** come from the constructive heuristic.
2. **Master LP** (HiGHS). Its duals give a value `G_di` for every case and day, and a cost for each start time,
   from the surgeon time-slot (coloring) duals.
3. **Lifted cover cuts** on the surgeon capacity knapsacks are separated and the LP is re-solved (up to 5 rounds).
4. **Pricing**: one CP-SAT model per day finds room schedules with positive reduced cost. Before the search:
   * DR1: start times in `[1, SD_d−1]` are removed;
   * DR2: cases with `Δbest < 0` are removed;
   * DR3a/b/c: dominated cases are removed, or made conditional on the dominating case (`C_i ≤ C_i'`);
   * DR4: cases whose conditioning chain cannot fit are removed;
   * FID: a greedy multidimensional knapsack bound (exact for this laminar structure) proves the
     subproblem useless, or caps the objective (29).
   
   All improving solutions found during the search become columns. They are also copied to other days
   when that is feasible and still improving.
5. **Two phases**: a short CP time limit first, then an optimality-proof phase once nothing more is found.
6. **Valid bounds at every iteration** (Lagrangian bound `z_RMP + Σ_d |K_d| · max(0, σ̄_d)`). A run stopped by
   a time limit still reports a true optimality gap.
7. **Branching**, best-first: (1) total number of cases, (2) number of cases per surgeon,
   (3) Ryan–Foster *same room / different rooms*. After `prove_fraction` of the branching time, nodes are
   processed in fast mode, as in the paper.
8. **Final IP** over all generated columns (HiGHS MIP), warm-started with the incumbent.
9. *(Extension, `polish_time`)* **CP-SAT polishing**: a monolithic CP-SAT model warm-started with the
   incumbent, constrained to be at least as good. On short-case instances (set C) the time-indexed columns
   are often hard to combine into an integer schedule: the LP bound is tight but the final IP is weak.
   The paper reports the same issue for set C. Polishing fixes the primal side; the gap is still certified
   by the B&P&C bound.

### Differences from the paper (deliberate)

* **Solvers**: HiGHS and OR-Tools CP-SAT (open source) replace CPLEX and CP Optimizer.
* **Pricing model**: optional interval variables and pairwise cleaning precedences. Requiring OCT between every
  pair of cases with different infection types is equivalent to requiring it between consecutive ones. A
  redundant makespan constraint (`Σ t + OCT·(#infection groups − 1) ≤ T`) makes pricing much faster on
  short-case instances.
* **Cover lifting**: equation (39) as printed sorts the cover by non-decreasing durations. That can produce
  invalid cuts (example: C = {3, 3, 8}, b = 13, t_i = 3). We use the Balas form, which uses the largest
  weights and is provably valid. It is checked by enumeration in the tests.
* **Dominance rules** are applied at the root of each CP search (p* = 1), not dynamically during it.
  DR3's λ uses the conservative value `2·OCT` whenever the two infection types differ.
* **Artificial variables** keep the master feasible for mandatory cases and for ≥ branching rows. Without
  them, a child node would be wrongly declared infeasible before pricing had a chance to add columns.

## Validation

`pytest -q tests` checks that:

* the B&P&C optimum equals the exact optimum of the monolithic CP-SAT model on small instances, and the
  upper bound is valid;
* every schedule passes the independent validator;
* the master's duals reproduce the reduced costs, and pricing columns respect the knapsack bound;
* lifted cover cuts never cut off a feasible point (exhaustive enumeration);
* the validator detects violations, and the CSV round trip preserves the instance.

## Benchmarks

`python benchmarks/run_benchmark.py [time_scale]` runs the paper's instance grid (A/B: 40–120 cases,
C: 60–100) with 120 s CG + 120 s branching + 60 s IP per instance (scale 1). The paper used 2 h + 2 h + 1 h.
Results go to `benchmarks/results.json`.

Results (seed 1, scale 1, CP-SAT polishing on):

| Set | Cases | Rooms/day | Heuristic (min) | Best schedule (min) | Upper bound | Gap % | Time (s) | Paper B&P&C gap %* |
|---|---|---|---|---|---|---|---|---|
| A | 40 | 6 | 6,940 | 7,010 | 7,010 | 0.00 | 4 | 0.00 |
| A | 60 | 6 | 8,525 | 8,910 | 8,910 | 0.00 | 147 | 0.55 |
| A | 80 | 6 | 10,070 | 10,910 | 11,475 | 4.92 | 225 | 1.30 |
| A | 100 | 6 | 10,455 | 11,275 | 11,930 | 5.49 | 272 | 1.81 |
| A | 120 | 6 | 9,450 | 10,990 | 11,730 | 6.31 | 323 | 1.40 |
| B | 40 | 3 | 5,925 | 6,940 | 6,980 | 0.57 | 172 | 0.00 |
| B | 60 | 4 | 7,900 | 8,895 | 9,190 | 3.21 | 224 | 0.55 |
| B | 80 | 5 | 9,735 | 10,790 | 11,555 | 6.62 | 233 | 1.32 |
| B | 100 | 6 | 10,455 | 11,275 | 11,985 | 5.92 | 275 | 2.65 |
| B | 120 | 6 | 9,450 | 10,930 | 11,750 | 6.98 | 315 | 4.13 |
| C | 60 | 2 | 3,900 | 4,170 | 4,170 | 0.00 | 192 | 0.94 |
| C | 80 | 3 | 5,515 | 5,515 | 5,515 | 0.00 | 16 | 9.92 |
| C | 100 | 3 | 5,700 | 6,350 | 6,985 | 9.09 | 282 | 12.01 |

Average gap: **3.78%**, every schedule passes the validator. The run used 4 CPU threads and ≤ 5.5 min per instance.

*Paper averages over 5 instances each, with up to ~4–5 h per instance on 12 cores using CPLEX/CP Optimizer. Our instances are generated with the same recipe but are not the same instances, so the comparison is indicative only.

## Limitations and next steps for production

* **Deterministic durations**, as in the paper. Add buffers, or a robust/stochastic extension, for real use.
* **Identical rooms.** Room-specific equipment needs one pricing problem per room type
  (paper Appendix 1, not implemented).
* **No emergency cases** and no recovery-bed capacity.
* CP-SAT is multi-threaded, so runs are not bit-for-bit reproducible. Use `workers=1` for determinism.
* Large instances (≥ 100 long cases) need minutes to hours for small gaps, as in the paper.
