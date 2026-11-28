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
python -m pytest -q tests         # 248 tests, ~1 min
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
tests/            248 tests                    benchmarks/     experiment grid, table + web builders
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

## Validation (what the 248 tests check)

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

<!--TABLES-->

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
