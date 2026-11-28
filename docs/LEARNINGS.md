# Learnings — how we turn an OR paper into an MVP (living document)

This file is the persistent memory for the "paper → MVP" workflow in this repo. Read it first in any new
session; append to it whenever something non-obvious is learned. The container is ephemeral — commit it.

## A. Reusable workflow (established with MVP #1, repeated for MVP #2)

Each paper becomes a **self-contained package + dashboard + tests + benchmarks**, with the paper's own numbers
used as the yardstick.

| Piece | MVP #1 (`orsched/`) | MVP #2 (`esassp/`) |
|---|---|---|
| Paper | Hashemi Doulabi, Rousseau, Pesant, IJOC 2016 — CP-based B&P&C for OR planning & scheduling | Shehadeh, Tsang, Padman, Kilic, EJOR 329 (2026) 288–307 — OR-to-downstream elective surgery planning under uncertainty (ESASSP, DRO) |
| Package | `orsched/` (model, generator, master, pricing, cuts, bpc, compact, validate, kpi, dataio) | `esassp/` (model, generator, evaluate, milp, formulations, subproblems, sp, dro, validate, kpi, dataio) |
| Dashboard | `app.py` Streamlit, 6 tabs | `app_esassp.py` Streamlit, 6 tabs |
| Static artifact | `web/index.html` + precomputed `scenarios.json` | `web_esassp/index.html` + precomputed `results.json` |
| Tests | `tests/test_orsched.py` — B&P&C == monolithic CP-SAT optimum | `tests/test_esassp.py` — DRO value == brute-force moment LP |
| Benchmarks | `benchmarks/run_*.py`, `supervise.sh` + `sync.sh` checkpoint to a branch via a separate git worktree | `benchmarks/esassp_*.py`, same supervisor/sync scripts |
| README | quick start, data format, architecture, algorithm *as implemented*, **Differences from the paper**, validation, benchmark table vs. paper, limitations | same sections |

Rules of thumb that paid off:

1. **Read the paper completely before coding**; list the equations you will implement and their numbers.
2. **Derive duals yourself** and then compare to the printed formulas. Printed formulas have typos
   (MVP #1: cover-lifting eq. (39) can produce invalid cuts → use Balas form). Never trust a printed formula that you
   have not checked numerically.
3. **Build an independent oracle first** (simulator / validator / brute force). Every solver result is checked
   against it. In MVP #1 the oracle was a monolithic CP-SAT model; in MVP #2 it is a closed-form evaluator of the
   recourse problems plus an enumeration of worst-case distributions.
4. **Exact cross-check test** at tiny scale (optimum == optimum; bound valid) beats any amount of eyeballing.
5. **Honest reporting**: put a "Differences from the paper" section in the README, and report gaps as measured,
   with the machine/time budget. Open-source solvers (HiGHS, CP-SAT) replace CPLEX/CP Optimizer and are slower.
6. **Time-limit everything**; every long job runs in the background with checkpoints (`supervise.sh`, `sync.sh`)
   and the container may be reclaimed at any time → commit/push at every milestone.
7. The paper's data is usually **not public** → generate synthetic instances with the same recipe and *label them
   as reconstructed*; never present them as the paper's data.
8. Dashboard conventions: sidebar = instance + algorithm switches; main tabs = how it works / data / solver run /
   schedule / KPIs / comparison. The static web artifact is built from precomputed runs (no solver in the browser).

## B. Paper-specific lessons — Shehadeh et al. (EJOR 2026), ESASSP

### Problem
Integrated **assignment + sequencing + scheduling** of elective surgeries over B surgical blocks (OR-days) with
**downstream ICU / ward bed capacity**. First stage: x[i,b,k] (surgery i is k-th in block b, or postponed to dummy
block b'), s[b,k] (scheduled start). Second stage (random): block-capacity recourse Q^BC(x,s,d) (waiting, overtime,
idle) and downstream recourse Q^down(x,l) (ICU / ward excess bed-days). Uncertainty: surgery duration d, ICU LOS l^c,
ward LOS l^w. Models: **SP** (SAA), **DRO-M** (mean-support ambiguity), **DRO-W** (1-Wasserstein ambiguity).

### Key mathematical facts (verified numerically in our tests)
* `Q^BC` is a convex piecewise-linear function of d ⇒ the worst case of `Q^BC − ρ·d` over a box sits on a **vertex**
  (2^m candidates per block of m surgeries). For Wasserstein, each coordinate is in `{lo, d^n, hi}` (3^m).
* DRO-M duality (Prop. 1/2): `sup_P E Q = min_ρ μ·ρ + max_d [Q(d) − ρ·d]`, with ρ multiplying the **scheduled**
  indicator (ρ_i x_ibk), because an unscheduled surgery does not influence Q.
* Dual of Q^BC (my derivation; matches printed (10a)–(10c)): for rows (4c),(4d),(4e) with duals α,β,γ ≥ 0:
  `α_k − α_{k+1} − β_k ≤ c^w`, `Σ_k β_k − γ ≤ c^o`, `γ ≤ c^g`. Hence
  `γ ≤ c^g`, `β_k ≤ c^o + c^g`, `α_k ≤ c^w (m−k+1) + c^o + c^g` — the bounds the paper never prints but needs for
  the τ·dual products.
* Big-M bounds of Prop. 5: `ρ ∈ [−c^g, 2c^o + c^g + c^w K]`, `ζ ∈ [0, c^ward]`, `φ ∈ [−c^ward, c^icu + c^ward]`.
  For Wasserstein the multiplier is bounded by the Lipschitz constant of Q (L1 norm): ν ≤ 2c^o + c^g + c^w K,
  φ_W ≤ c^icu + c^ward — tighter than Prop. 8's ratio.
* LOS characterisation (Prop. 3): monotone binaries r (left ICU by day t), h (left ward by day t);
  ICU occupancy = 1[t ≥ a] − r, ward occupancy = r − h, LOS bounds are sums of these. Max over integer LOS of a
  convex piecewise-linear occupancy cost is a small MILP (our own, simpler linearisation: occupancy product
  `p ≤ e_t, p ≤ y` with `e_t ∈ [0,1]` exact because the optimum is at e ∈ {0,1}).
* Costs: `c_ib = α c^o` (scheduled), `1.5 α c^o` (postponed), `c^g = c^o/1.5`, `c^w = c^o/1.5`,
  `c^icu = c^ward = 1000` per bed-day. An empty block still costs `c^g L_b` (idle), so scheduling a surgery
  *saves* `c^g·d`; volume is capped by overtime and bed excess, not by first-stage costs.
* C&CG needs no feasibility cuts (complete recourse). Terminates finitely (finite vertex set / lattice).
  UB = master objective with δ's replaced by the true sub-problem values; LB = master (dual) bound.
* Symmetry breaking (27)–(29) is only valid for **closed block scheduling** (a block hosts a single class) and
  classes of identical surgeries; (29) forces consecutive positions inside a block.
* Multi-cut (24)/(25): δ^BC = Σ_b δ_b^BC, one cut per block per scenario — tightens the master a lot.

### Conventions / ambiguities I had to settle (documented in the README too)
* Appendices B–J (supplementary) are **not in the PDF**: DRO-W C&CG (Appendix D), data statistics (G), extra
  experiments (H, J), CPU tables (I) are unavailable → DRO-W C&CG is **my own derivation** from (20)–(22).
* Hospital data is private → synthetic reconstruction. Only ENT (μ 74, σ 37 min), OPHTH (μ 38, σ 19 min) and
  VASCULAR (avg LOS 3.5 d) values are quoted in the text; every other type parameter is invented and labelled.
* "(10,7) beds" is read as (weekday, weekend) beds; the text calls them "beds in the ICU and ward during
  (week, weekends)". We expose ICU and ward weekday/weekend separately.
* Stays running past day T: (5b),(5d) sum over [T] while DRO-W uses [T+2]. **Our convention: only days < T are
  costed** (consistent in MILP, oracle and simulator).
* Duration bounds = 20 / 80 % quantiles of the data (as in the paper); we use the mean of the *clipped* in-sample
  distribution as μ so that the ambiguity set contains the sampling distribution.
* Mis-specified distributions (Δ ∈ {0, 0.2, 0.5}): duration gamma(μ, σ²) clipped to `[(1−Δ) lo, (1+Δ) hi]`;
  LOS gamma(mean (1+Δ)μ, σ²) clipped to `[lo, hi]`.
* Printed (4c) uses a big-M for empty positions; M_bk = L_b + (k−1)·d̄_max is valid and tight enough.

### Computational reality
* The paper used CPLEX 12.6 on 16 cores: R-70 takes 1.6–59 min. HiGHS is considerably slower, so interactive
  instances are I ≈ 15–30; bigger ones run as background benchmarks with honest gaps.
* The paper itself says SP solved directly by the solver beats decomposition; DRO-M C&CG is the fastest DRO.

(Sections below are appended as implementation progresses.)

## C. Implementation log / pitfalls
*(filled in while building MVP #2 — see git history)*

### Pitfalls found while building MVP #2 (append-only)
* **HiGHS `setSolution` with a partial hint is dangerous**: it was accepted as an incumbent and returned an objective
  *below* the true optimum (C&CG reported UB < LB). Never warm-start a MILP from a partially specified vector unless
  the result is re-validated. (Symptom to remember: UB < LB or UB below an independent oracle value.)
* **HiGHS thread scheduler is process-global**: setting `threads` to a different value in a later `Highs()` object raises
  an error and the solve silently returns "Not Set". Initialise once (`esassp.milp.set_threads`).
* Bilinear terms `tau_i * dual` linearised with the *dual* bounds derived from the dual feasible set; the resulting MILP
  matched brute-force enumeration to 1e-9 for both mean-support and Wasserstein — good template for "derive, then check
  against enumeration".
* Exact strong-duality test: with (x, s) fixed, C&CG value == moment-LP value over box vertices x integer LOS lattice to
  1e-4 absolute (34850.4195 both) — this also proves Prop. 5 bounds and inequality (26) do not cut off the optimum.
