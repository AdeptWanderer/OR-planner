# Verification: is this the paper's method?

This document checks the implementation against

> S. H. Hashemi Doulabi, L.-M. Rousseau, G. Pesant (2016). *A Constraint-Programming-Based
> Branch-and-Price-and-Cut Approach for Operating Room Planning and Scheduling.*
> INFORMS Journal on Computing 28(3):432–448.

It does this element by element, in three ways:

1. **Traceability.** Every equation, rule and algorithm step of the paper maps to the code that implements it.
   Each is classified as
   * **Faithful**: implemented as written.
   * **Adapted**: same mathematics, different machinery. Equivalence is tested.
   * **Extension**: not in the paper. It is reported separately and never mixed into paper-comparable numbers.
2. **Paper-literal reference implementations** (`verification/paper_models.py`), written straight from the
   equations and independent of `orsched/`. The conformance tests compare the production code against them.
3. **Independent exact solvers.** A monolithic CP-SAT model and a time-indexed MIP solved by HiGHS confirm
   that the method returns true optima on instances small enough to solve exactly.

## How to run

```bash
python -m pytest -q tests                      # unit tests (~40 s)
```

```bash
python -m pytest -q tests/conformance          # paper conformance + exactness (~1.5 h, CPU heavy)
```

```bash
python verification/audit_results.py RESULTS.jsonl CKPT_DIR --crosscheck 120   # audit reported numbers
```

## 1. Problem definition (Section 2)

| Paper | Implementation | Verified by | Status |
|---|---|---|---|
| Planning horizon of days `D`, rooms `K_d`, slots `T_d` | `model.Instance` | `test_generator_matches_section_6_1` | Faithful |
| Mandatory `I1` (deadline in horizon), optional `I2` | `Instance.mandatory/optional` | master tests | Faithful |
| Assumption 1: deterministic durations | `Surgery.duration` | n/a | Faithful |
| Assumption 2: surgeon max time per day `A_ld` | master row (5), pricing (14) | `test_validator_enforces_every_constraint[surgeon_max]` | Faithful |
| Assumption 3: cleaning `OCT` when switching to a different infection type or to a non-infectious case | `Instance.clean` | `test_cleaning_rule_assumption3` (matches Table 1 case 2) | Faithful |
| Assumption 4: surgeries pre-assigned to surgeons | `Surgery.surgeon` | n/a | Faithful |
| Assumption 5: recovery not a bottleneck | not modelled | n/a | Faithful |
| Assumption 6: 8 h rooms, no overtime | `day_minutes=480` default; finish ≤ day end | `validator[horizon]` | Faithful |
| Assumption 7: identical rooms | room index only orders schedules | n/a | Faithful |
| Objective: maximise total scheduled surgery time | `Column.value` = `B_j` | `test_objective_is_total_scheduled_surgery_time` | Faithful |
| Constraints 1–4 (mandatory done, surgeon max, one patient at a time for room and surgeon, cleaning) | master + pricing + independent `validate.py` | `test_validator_enforces_every_constraint` (9 violation types) | Faithful |

## 2. Master problem (Section 3.1)

| Paper | Implementation | Verified by | Status |
|---|---|---|---|
| Columns = feasible single-room, single-day schedules | `model.Column` | pricing validity checks | Faithful |
| (1) `max Σ B_j x_j` | `Master` (HiGHS, minimises `−B`) | `test_master_lp_equals_formulas` (vs SciPy LP built from the equations) | Faithful |
| (2) mandatory `= 1`, (3) optional `≤ 1` | `r_assign` rows | same | Faithful |
| (4) `Σ_{j∈J'_d} x_j ≤ |K_d|` | `r_rooms` | `test_column_coefficients_match_definitions` | Faithful |
| (5) surgeon minutes per day | `r_surg` | same | Faithful |
| (6) coloring: no surgeon in two rooms in a slot | `r_color[(d,l,t)]` | same, plus validator | Faithful |
| (7) `x_j ∈ {0,1}` | LP: `x ≥ 0`, with `x ≤ 1` implied by (2)/(3); IP: integer | `test_reduced_cost_is_the_papers_sigma`, `test_lp_optimality…` | Faithful. The bound is dropped as redundant, which keeps LP duality exact. |
| Artificial variables on (2) and on `≥` branching rows | `Master` (big-M = total case minutes) | `test_bpc_vs_*` exactness tests | Adapted. These keep the restricted master feasible before pricing. At any integer solution they are 0, and nodes are pruned only by bound, never by "artificial > 0". |

## 3. Pricing subproblem (Section 3.2)

| Paper | Implementation | Verified by | Status |
|---|---|---|---|
| Reduced cost `σ_j` and day decomposition | `Master.day_duals` gives `G_di`, `π*`, `π⁽⁴⁾` | `test_reduced_cost_is_the_papers_sigma` | Faithful |
| Position model (8)–(15), objective (17), positivity cut (19), solved by CP Optimizer | CP-SAT model in `pricing.solve_pricing`: optional intervals, NoOverlap, pairwise cleaning precedences, element start-costs, objective ≥ 1/SCALE | `test_pricing_optimum_equals_position_model`: same optimum as the literal position model in `paper_models.position_pricing` on 68 realistic and adversarial dual vectors | Adapted. Equivalent model. Requiring OCT between every pair of cases with different infection types equals requiring it between consecutive ones. |
| (8) only cases with `dd_i ≥ d` | candidate filter | `test_pricing_respects_deadlines_eq8` | Faithful |
| Redundant makespan constraint `Σt + OCT·(#types used − 1) ≤ T` | `pricing.py` | covered by the equivalence test | Extension (speed only). It is implied by the model, so it cannot change the optimum. |
| Multiple columns per subproblem, all positive solutions | solution callback | `_exact_prod` validity checks | Faithful |
| Column copying to other days (Alg. 1, line 12) | `column_value_on_day` | used in all runs | Faithful |

## 4. Enhancements (Section 4)

| Paper | Implementation | Verified by | Status |
|---|---|---|---|
| DR1: `V_1 = 0` or `V_1 ≥ SD_d` | start-time domains | `test_dr1_removes_exactly_1_to_SD_minus_1`; equivalence test against the paper model **with DR1 imposed** | Faithful. **Finding:** DR1 is not an exact pruning rule for the subproblem. For some dual vectors the best column starts at slot 1…SD−1 (e.g. slot 2), and DR1 removes it. The paper's proof shows that an optimal *IORPS* solution satisfies DR1, not that every pricing optimum does. DR1 therefore restricts the column set while keeping an optimal integer solution, so all bounds remain valid; the suite counts how often it changes the pricing optimum (`dr1_changed`). |
| DR2: remove `i` if `Δ^best_id < 0` | yes | equivalence test: optimum unchanged relative to the DR1-restricted problem; `test_enhancements_actually_fired` | Faithful |
| DR3(a)(b)(c): dominated cases removed or conditioned (`C_i ≤ C_i'`) | yes | same | Adapted. Rules are applied at the root of each CP search (p* = 1), not re-applied dynamically as `W_p` variables are fixed. λ uses the conservative `2·OCT` when infection types differ, because Table 1 needs the predecessor in position p*−1, which is unknown at p* = 1. Only surgeries that stay in the domain can dominate. This prunes less than the paper, never more. |
| DR4: conditioning chains that can't fit | transitive closure | same | Faithful (applied at p* = 1) |
| Fast infeasibility detection (24)–(29), steps 1–5 | `knapsack_bound` (greedy over the laminar room/surgeon capacities) + objective cap (29) | `test_knapsack_bound_matches_paper_steps_and_is_valid`: equals the literal steps 1–5 and bounds the true optimum | Faithful at p* = 1. Dynamic re-evaluation for p* > 1 is not done (adapted). |

## 5. Branch-and-price-and-cut (Section 5)

| Paper | Implementation | Verified by | Status |
|---|---|---|---|
| Algorithm 1: CP time limit phase 0 (10 s), optimality phase 1, switching rules, cut loop, final RMP-IP | `Solver.price_and_cut`, `run` | `test_algorithm1_phase_logic` | Faithful. The "inf" limit of phase 1 is capped at 600 s in the paper-budget run. |
| Algorithm 2: best-first, prove for the first 15 min then fast mode, leaf RMP-IP, final IP over all columns | `Solver.branch_and_bound`, `run` | `test_algorithm2_best_first_and_child_bounds` | Faithful |
| Branch (1) total #cases, (2) #cases per surgeon closest to 0.5, (3) Ryan–Foster pair closest to 0.5, in that order | `choose_branch`, `children` | `test_branching_rule_order_1_2_3`, `test_branching_children_partition_integer_solutions` | Faithful |
| Duals of (30)–(33) in the subproblem (`g_i(h)`) | generic `ExtraRow`; duals folded into `G_di` | master and pricing tests | Faithful |
| (34)/(35) in the subproblem for Ryan–Foster | `x_i = x_i'` / `x_i + x_i' ≤ 1` in CP-SAT | `test_pricing_respects_ryan_foster_restrictions` | Faithful |
| Lifted minimal cover cuts on (36) with `z_id` reformulation, duals λ_di in pricing (40) | `cuts.py`, `ExtraRow(day=d)` | `test_lifted_cover_cuts_valid_by_enumeration`, `test_cuts_do_not_cut_the_integer_optimum` | Adapted. Separation uses a greedy minimal cover instead of the Gu et al. (1998) routine. |
| Lifting coefficients, eq. (39) | Balas lifting with the largest cover weights | `test_eq39_as_printed_is_invalid…`: proves the printed formula cuts off a feasible point | Adapted (correction). Eq. (39) as printed, with the cover sorted nondecreasing, is invalid; we use the standard valid form. |
| Upper bound when CG is not converged: paper reports a trivial bound | Lagrangian bound `z_RMP + Σ_d |K_d|·max(0, σ̄_d)` | `test_bounds_are_monotone_and_consistent`; exactness tests (UB ≥ optimum) | Extension. A standard valid bound that only tightens what is reported. |
| (none) | CP-SAT polishing of the incumbent | results report both `gap_no_polish` and `gap` | Extension. Excluded from paper-comparable numbers. |
| (none) | checkpoint/resume, memory caps | `rt.py` kill/resume test; audit | Infrastructure. It does not change the algorithm or the time limits. |

## 6. Experimental setup (Section 6)

| Paper | Ours | Status |
|---|---|---|
| CPLEX 12.4 / CP Optimizer, 1 core per instance, Xeon X5675 | HiGHS 1.16 / OR-Tools CP-SAT 9.15, 1 core per instance (`workers=1`) | Adapted (solvers) |
| CG 2 h, CPtimelimit 10 s, branching 2 h (15 min proving), final IP 1 h | identical (`benchmarks/run_paper_budget.py`) | Faithful |
| Sets A, B, C; 5 days; 8 surgeons; deadlines U[1,14]; half infectious; OCT 30 min; 5-min slots | `generator.py` | Faithful |
| `A_ld` from Fei et al. (2009, Table 1) with modifications (mean ≈ 320 min) | normal(320, 60) rounded to 30 min, capped to [120, 480], at least the surgeon's longest case | **Adapted: the source table was not available.** |
| Room-sizing heuristic for B and C (online Appendix 6) | reconstruction from the text of §6.1 | **Adapted: appendix not available.** |
| Set C Pearson III durations (Fei et al. 2009) | 30 + Gamma(2, 22) min, clipped to [30, 240] | **Adapted: parameters not given in the paper.** |
| 5 instances per size | 1 instance per size (seed 1) | **Smaller sample.** |
| Sets A and B independent | same seed → identical cases; sets differ only in rooms per day. At 100 and 120 cases the room heuristic also gives 6 rooms, so **B100 = A100 and B120 = A120** | **Known flaw; disclosed in the results report.** |
| Constructive heuristic for initial columns and deadline repair | `heuristic.py` | Faithful |

**Not implemented from the paper:**
* online Appendix 1: non-identical rooms
* Appendix 2: the Marques et al. IP. We use our own time-indexed MIP for verification instead.
* Appendix 3: the paper's pure CP model. We use our own monolithic CP-SAT baseline.
* clique and odd-cycle cuts, which the paper tested and discarded.

## 7. Results provenance

* All numbers come from `benchmarks/run_paper_budget.py` (paper budget) or `benchmarks/run_benchmark.py`
  (short budget). Nothing is typed in by hand.
* Each run leaves a checkpoint with the final schedule. `verification/audit_results.py`:
  1. regenerates each instance;
  2. re-validates the stored schedule with the independent checker;
  3. recomputes plan minutes, heuristic value and gap and compares them with the reported numbers;
  4. checks bound consistency and time limits;
  5. optionally re-solves with an independent CP-SAT model to check that no plan it finds exceeds our upper bound.
* Paper figures quoted in the comparison are copied from Tables 3–8 of the paper.

## 8. Verification results

See `verification/reports/` for the raw test output; the summary is in the benchmark report.
