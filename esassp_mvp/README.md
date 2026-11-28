# ESASSP — OR-to-downstream elective surgery planning under uncertainty

An independent MVP implementation of

> K. S. Shehadeh, M. Y. Tsang, R. Padman, A. Kilic (2026). *Operating room-to-downstream elective surgery planning
> under uncertainty.* European Journal of Operational Research 329(1):288–307.

For a planning week it decides **which elective surgeries to perform and which to postpone, in which OR block, in what
order and at what scheduled start time** — while accounting for the **ICU and ward beds** the patients need afterwards
and for **uncertain surgery durations and lengths of stay (LOS)**. Three models are implemented:

| Model | Idea | Solved by |
|---|---|---|
| **SP** | expected cost over N sampled scenarios (SAA) | one extensive-form MILP (HiGHS) |
| **DRO-M** | worst case over *all* distributions with the given mean and support | column-and-constraint generation (C&CG) |
| **DRO-W** | worst case over a 1-Wasserstein ball around N historical samples | C&CG (derived here, see below) |

plus the paper's **non-integrated baselines** (ignore the downstream units / ignore the scheduling problem).

## Quick start

```bash
cd esassp_mvp
pip install -r requirements.txt
streamlit run app.py              # dashboard at http://localhost:8501
python -m pytest -q tests         # 401 tests, ~1 min
```

Open `web/index.html` in a browser for the static dashboard built from the precomputed benchmark runs.

```python
from esassp import generate_instance, solve, Params, sample_scenarios, evaluate, validate_solution
inst = generate_instance(20, "cardiac", seed=1, n_ors=2, beds="(5,2)")
res = solve(inst, "DRO-M", Params(time_limit=120, gap=1e-3))
assert validate_solution(inst, res.solution) == []
ev = evaluate(inst, res.solution, sample_scenarios(inst, 5000, rng, delta=0.2))   # mis-specified stays
print(res.objective, res.lower_bound, res.gap, ev["pe_icu"], ev["mean_total"])
```

## Dashboard tabs

| Tab | Content |
|---|---|
| ① How it works | problem, C&CG loop diagram, key formulas, honest list of differences from the paper, trace of the last run |
| ② Instance & data | editable surgeries / blocks / beds, CSV import-export, demand vs capacity |
| ③ Solver run | table of all runs, LB/UB convergence, scenarios generated, per-iteration table |
| ④ Schedule | Gantt per block (allocated time vs expected duration), bed occupancy vs capacity, postponed list, CSV |
| ⑤ Robustness | probability / mean of exceedance per day, OR waiting distribution — under mis-specified Δ |
| ⑥ Compare models | cost, wait, idle, exceedance, RelTC vs SP and vs integrated, cost-vs-Δ curve |

## Using real data

`surgeries.csv`: `case,type,priority,dur_mean,dur_sd,dur_lo,dur_hi,icu_mean,icu_sd,icu_lo,icu_hi,ward_mean,ward_sd,ward_lo,ward_hi`
(minutes for durations, whole days for stays; *lo/hi* = the support, the paper uses the 20 % / 80 % data quantiles).
`blocks.csv`: `day,room,minutes,max_surgeries,type` (1-based day; empty `type` = open block). `beds.csv`: `day,icu_beds,ward_beds`.
Templates can be downloaded in the app.

## Architecture

```
esassp/
  model.py        Instance, Surgery, Block, Costs, Solution (+ mean-duration allocation)
  generator.py    synthetic instances (paper §6.1 recipe, 20/80 % quantile supports, bed presets)
  evaluate.py     closed-form Q^BC and Q^down, scenario sampling, out-of-sample metrics (the ORACLE)
  milp.py         tiny MILP layer over HiGHS (dual bound, process-wide threads)
  formulations.py first stage X (eq. 2), symmetry breaking (27)-(29), scenario copies of (4)-(5)
  subproblems.py  separation: worst-case durations (enumeration / dual MILP), worst-case LOS (MILP)
  sp.py           SAA stochastic program (6)
  dro.py          C&CG master + loop for DRO-M (17)-(18) and DRO-W (20)-(22)
  moment.py       brute-force PRIMAL worst-case expectations (independent oracle for the tests)
  experiment.py   one-instance experiment (models x baselines x mis-specification) -> JSON record
  kpi.py, dataio.py, validate.py
app.py            Streamlit dashboard          web/index.html  static dashboard
tests/            401 tests                    benchmarks/     experiment grid, table + web builders
```

### Algorithm, as implemented

1. **First stage**: x[i,b,k] (surgery i is the k-th in block b), x[i,b'] (postponed), s[b,k] start times; constraints (2); optional
   symmetry breaking (27)–(29) for identical surgeries in closed blocks.
2. **Second stage in closed form**: waiting `w_k = max(0, w_{k-1}+s_{k-1}+d_{k-1}−s_k)`, overtime from the last completion, idle `L−Σd+o`; ICU/ward
   beds needed minus beds. Verified against an LP of (4b)–(4e) and against the MILP copies (tests).
3. **DRO-M master**: multipliers ρ (durations), φ, ζ (ICU/ward LOS), McCormick products with the assignment, per-block cuts (24), bound (26),
   one recourse copy per scenario generated so far.
4. **Separation**: `Q^BC` is convex piecewise-linear in `d` ⇒ the worst case is a **vertex** of the duration box (DRO-W: `{lo, d^n, hi}` per surgery).
   Blocks up to 11 surgeries are enumerated; larger blocks use the **dual MILP** of (4b)–(4e) whose bounds on the dual variables are derived in
   `docs/NOTES.md`. The worst-case LOS is a MILP over monotone ICU-exit / ward-exit binaries (Prop. 3).
5. **Bounds**: LB = dual bound of the master, UB = master objective with the δ's replaced by the true separation values; stop at the target gap.
6. **DRO-W**: one multiplier ν for the duration ball and one for the LOS ball; cuts for every (sample n, scenario r) share the recourse copy of r.

## Validation (what the 401 tests check)

* closed-form `Q^BC` equals the LP of (4b)–(4e) on random data; hand-computed waiting / overtime / idle / bed examples;
* first-stage MILP solutions always pass the independent validator; recourse copies in the MILP equal the oracle for fixed (x, s);
* **separation MILP == enumeration == brute force** (box vertices; `{lo, d^n, hi}`; full integer LOS lattice);
* **strong duality**: for fixed (x, s) the C&CG value equals the *primal* moment LP (DRO-M) / transport LP (DRO-W) to 1e-6 — this also shows the
  Prop. 5 bounds and inequality (26) never cut off an optimum;
* Wasserstein radius 0 ⇒ sample-average value; value non-decreasing in the radius; LB monotone, UB monotone, LB ≤ UB;
* toggles (symmetry, per-block cuts, ρ-bound) do not change optima; SB exact with exchangeable scenarios;
* SP objective equals the oracle SAA objective; ignoring downstream capacity never helps out of sample;
* experiment records are JSON-serialisable and consistent; the Streamlit app renders and solves headlessly.

## Benchmarks (synthetic instances, HiGHS, 2 threads per run, 100 s per model)

`python benchmarks/run_experiments.py <part> <n_parts> <out.jsonl>` then `python benchmarks/build_web.py`. Out-of-sample: 5 000 simulated weeks.
Instances follow the paper's recipe but are **synthetic and much smaller** than R-70 / Case-100 (the paper used CPLEX on 16 cores).

| Instance | Model | Scheduled | Solve s | Gap % | Mean cost Δ=0 | Mean cost Δ=0.5 | max ICU PE (Δ=0.2) | RelTC vs SP (Δ=0) |
|---|---|---|---|---|---|---|---|---|
| cardiac-12-b52-c1 (I=12) | SP | 7 | 0 | 0.00 | 28,436 | 31,112 | 0.24 |  |
| cardiac-12-b52-c1 (I=12) | DRO-M | 5 | 1 | 0.04 | 28,404 | 29,725 | 0.00 | 0.00 |
| cardiac-12-b52-c1 (I=12) | DRO-W | 5 | 3 | 0.02 | 28,076 | 29,383 | 0.00 | 0.01 |
| cardiac-16-b52-c1 (I=16) | SP | 5 | 0 | 0.00 | 28,136 | 29,439 | 0.00 |  |
| cardiac-16-b52-c1 (I=16) | DRO-M | 5 | 1 | 0.02 | 28,496 | 29,782 | 0.00 | -0.01 |
| cardiac-16-b52-c1 (I=16) | DRO-W | 5 | 2 | 0.11 | 28,226 | 29,524 | 0.00 | -0.00 |
| cardiac-20-b107-c1 (I=20) | SP | 18 | 2 | 0.00 | 35,595 | 41,599 | 0.02 |  |
| cardiac-20-b107-c1 (I=20) | DRO-M | 16 | 100 | 16.06 | 37,250 | 41,561 | 0.00 | -0.04 |
| cardiac-20-b107-c1 (I=20) | DRO-W | 16 | 92 | 0.11 | 36,453 | 41,433 | 0.00 | -0.02 |
| cardiac-20-b52-c1 (I=20) | SP | 12 | 1 | 0.13 | 53,629 | 58,334 | 0.84 |  |
| cardiac-20-b52-c1 (I=20) | DRO-M | 10 | 100 | 3.58 | 54,602 | 56,041 | 0.75 | -0.02 |
| cardiac-20-b52-c1 (I=20) | DRO-W | 11 | 100 | 1.94 | 53,263 | 55,877 | 0.84 | 0.01 |
| cardiac-20-b52-c2 (I=20) | SP | 18 | 1 | 0.01 | 141,435 | 163,973 | 0.99 |  |
| cardiac-20-b52-c2 (I=20) | DRO-M | 18 | 100 | 5.74 | 146,155 | 168,259 | 1.00 | -0.03 |
| cardiac-20-b52-c2 (I=20) | DRO-W | 18 | 45 | 0.17 | 142,839 | 165,407 | 0.99 | -0.01 |
| cardiac-24-b52-c1 (I=24) | SP | 12 | 3 | 0.17 | 54,126 | 58,540 | 0.98 |  |
| cardiac-24-b52-c1 (I=24) | DRO-M | 10 | 100 | 4.01 | 54,127 | 55,342 | 0.74 | -0.00 |
| cardiac-24-b52-c1 (I=24) | DRO-W | 12 | 100 | 2.90 | 53,807 | 57,850 | 0.99 | 0.01 |
| general-20-b52-c1 (I=20) | SP | 19 | 1 | 0.15 | 57,675 | 66,118 | 0.00 |  |
| general-20-b52-c1 (I=20) | DRO-M | 17 | 100 | 6.29 | 58,115 | 63,821 | 0.00 | -0.01 |
| general-20-b52-c1 (I=20) | DRO-W | 18 | 68 | 0.18 | 57,958 | 65,365 | 0.00 | -0.00 |
| general-30-b107-c1 (I=30) | SP | 30 | 4 | 0.14 | 36,213 | 47,026 | 0.00 |  |
| general-30-b107-c1 (I=30) | DRO-M | 30 | 100 | 27.67 | 35,774 | 45,942 | 0.00 | 0.01 |
| general-30-b107-c1 (I=30) | DRO-W | 30 | 70 | 0.00 | 35,445 | 46,699 | 0.00 | 0.02 |

| Instance | Variant | Scheduled | mean cost Δ=0 | RelTC w/o vs integrated (Δ=0) |
|---|---|---|---|---|
| cardiac-12-b52-c1 | SP w/o downstream | 11 | 32,874 | 0.16 |
| cardiac-12-b52-c1 | SP w/o scheduling | 8 | 29,497 | 0.04 |
| cardiac-12-b52-c1 | DRO-M w/o downstream | 11 | 34,020 | 0.20 |
| cardiac-12-b52-c1 | DRO-M w/o scheduling | 5 | 28,071 | -0.01 |
| cardiac-16-b52-c1 | SP w/o downstream | 11 | 32,926 | 0.17 |
| cardiac-16-b52-c1 | SP w/o scheduling | 5 | 28,129 | -0.00 |
| cardiac-16-b52-c1 | DRO-M w/o downstream | 11 | 34,240 | 0.20 |
| cardiac-16-b52-c1 | DRO-M w/o scheduling | 5 | 28,129 | -0.01 |
| cardiac-20-b107-c1 | SP w/o downstream | 19 | 40,218 | 0.13 |
| cardiac-20-b107-c1 | SP w/o scheduling | 18 | 36,114 | 0.01 |
| cardiac-20-b107-c1 | DRO-M w/o downstream | 19 | 39,358 | 0.06 |
| cardiac-20-b107-c1 | DRO-M w/o scheduling | 18 | 36,114 | -0.03 |
| cardiac-20-b52-c1 | SP w/o downstream | 19 | 69,591 | 0.30 |
| cardiac-20-b52-c1 | SP w/o scheduling | 12 | 53,799 | 0.00 |
| cardiac-20-b52-c1 | DRO-M w/o downstream | 19 | 68,104 | 0.25 |
| cardiac-20-b52-c1 | DRO-M w/o scheduling | 10 | 54,218 | -0.01 |
| cardiac-20-b52-c2 | SP w/o downstream | 19 | 144,703 | 0.02 |
| cardiac-20-b52-c2 | SP w/o scheduling | 19 | 140,859 | -0.00 |
| cardiac-20-b52-c2 | DRO-M w/o downstream | 19 | 148,390 | 0.02 |
| cardiac-20-b52-c2 | DRO-M w/o scheduling | 18 | 143,986 | -0.01 |
| cardiac-24-b52-c1 | SP w/o downstream | 21 | 71,618 | 0.32 |
| cardiac-24-b52-c1 | SP w/o scheduling | 12 | 56,329 | 0.04 |
| cardiac-24-b52-c1 | DRO-M w/o downstream | 21 | 73,841 | 0.36 |
| cardiac-24-b52-c1 | DRO-M w/o scheduling | 10 | 54,350 | 0.00 |
| general-20-b52-c1 | SP w/o downstream | 20 | 58,170 | 0.01 |
| general-20-b52-c1 | SP w/o scheduling | 19 | 61,390 | 0.06 |
| general-20-b52-c1 | DRO-M w/o downstream | 20 | 60,526 | 0.04 |
| general-20-b52-c1 | DRO-M w/o scheduling | 19 | 59,871 | 0.03 |
| general-30-b107-c1 | SP w/o downstream | 30 | 35,568 | -0.02 |
| general-30-b107-c1 | SP w/o scheduling | 30 | 40,551 | 0.12 |
| general-30-b107-c1 | DRO-M w/o downstream | 30 | 35,774 | 0.00 |
| general-30-b107-c1 | DRO-M w/o scheduling | 30 | 40,850 | 0.14 |

**How to read this (measured, not the paper's numbers)**

* **Integration matters** — ignoring the ICU/ward capacity ("w/o downstream") schedules far more surgeries and costs **+13 % to +36 %** more out of
  sample on the tight-bed cardiac instances (paper: 24–60 % on much larger instances); it is harmless when beds are ample (`general-*`).
* **Robustness shows under distribution shift**, not in-sample: at Δ = 0 SP, DRO-M and DRO-W cost about the same (within about 5 %), but at Δ = 0.5
  DRO-M plans are cheaper than SP on 6 of 8 instances (e.g. cardiac-24: 55.3 k vs 58.5 k; DRO-W 57.9 k) and have lower or equal worst-day ICU exceedance
  (cardiac-12: 0.00 vs 0.24 at Δ = 0.2). They schedule the same or fewer surgeries, as in the paper.
* **"w/o scheduling"** (start times = cumulative mean durations) is only mildly worse here (0–14 %); the paper's large waiting-time effect needs bigger blocks.
* **Solver performance (HiGHS, 100 s, 2 threads)**: SP solves to < 0.2 % gap in seconds; DRO-W reaches ≤ 3 % on every instance; **DRO-M did not close the
  gap within 100 s for I ≥ 20 (3.6–28 %)**, the opposite of the paper's finding that DRO-M is the fastest DRO model (CPLEX, 16 cores, hours). The master
  MILP dominates run time (separation < 0.3 s); the DRO-M gap is an upper-bound-side effect of the time limit (the returned plans are feasible and evaluated out of sample).
* Differences between plans at Δ = 0 are small relative to Monte-Carlo noise on these small instances; do not over-interpret single rows.


## Differences from the paper (deliberate or forced)

* **Data**: hospital data and supplement Appendices B–J are not available. Types are a synthetic reconstruction; only ENT/OPHTH durations and
  VASCULAR LOS appear in the text. Never read these results as the paper's numbers.
* **Solver**: HiGHS instead of CPLEX ⇒ interactive sizes I ≈ 10–30 (paper: 50–100). Gaps are reported as measured.
* **DRO-W C&CG** is my own derivation from (20)–(22) (Appendix D is missing) and is verified against the primal transport LP.
* **Separation by enumeration** for blocks ≤ 11 surgeries (exact); the paper solves MILP (10)/(16) for everything. Our dual MILP is equivalent and tested.
* **Downstream sub-problem** uses a simpler exact linearisation (`p ≤ e_t, p ≤ occupancy`), not (16c)–(16e).
* **Stays beyond day T are not costed** (paper mixes [T] and [T+2]).
* **Products ρ·x** are aggregated per (surgery, block), LOS multipliers per surgery (equivalent, smaller).
* **Initial scenario** = nominal (mean) scenario instead of an empty scenario set.
* **No warm start** of the master: HiGHS accepted a partial hint as an invalid incumbent (documented in `docs/NOTES.md`).
* **μ in the DRO-M set** is the mean of the *clipped* in-sample distribution.
* **Symmetry breaking** assumes interchangeable surgeries: exact for DRO-M, a modelling choice for SAA-based SP / DRO-W.
* **Ward/ICU beds "(10,7)"** is read as (weekday, weekend) beds for both units.

## Limitations / next steps

Single-specialty blocks only; no emergency arrivals; no OR master-schedule redesign; no staff resources; Wasserstein radii are inputs (the paper
cross-validates them); larger R-70-scale instances need hours with HiGHS (or a commercial solver) and are not benchmarked here.
