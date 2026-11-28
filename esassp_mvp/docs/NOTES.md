# ESASSP — technical notes (derivations, conventions, pitfalls)

Companion of the README; the cross-project workflow lives in `../docs/LEARNINGS.md`.

## 1. Notation (0-based in code)
I surgeries, B blocks (OR-days) with length L_b and K_b positions, T days. x[i,b,k] ∈ {0,1}, x[i,b'] postponed, s[b,k] ∈ [0, L_b].
d_i ∈ [lo_i, hi_i] minutes; l^c_i ∈ [lc_lo, lc_hi], l^w_i ∈ [lw_lo, lw_hi] days (integers).

## 2. Second-stage problems in closed form (oracle)
Q^BC for a block with ordered surgeries 1..m (waiting w, overtime o, idle g):
`w_1 = 0, w_k = max(0, w_{k-1} + s_{k-1} + d_{k-1} − s_k)`, `o = max(0, w_m + s_m + d_m − L)`, `g = L − Σd + o` (≥ 0).
Empty block: `g = L`. Q = c^w Σw + c^o o + c^g g. Verified against the LP (4b)–(4e).
Q^down: ICU days a..a+l^c−1, ward days a+l^c .. a+l^c+l^w−1 (a = surgery day); u_t = (ICU_t − Bed_t)^+, only t < T costed.

## 3. Dual of Q^BC and the bounds the paper does not print
Rows: (4c) w_k − w_{k−1} ≥ s_{k−1} − s_k + d_{k−1}[α_k]; (4d) o − w_k ≥ s_k + d_k − L [β_k]; (4e) g − o ≥ L − Σd [γ].
Dual feasibility: `α_k − α_{k+1} − β_k ≤ c^w`, `Σ_k β_k − γ ≤ c^o`, `γ ≤ c^g`, all ≥ 0 (α_{m+1}=0; w_1 is fixed so no row).
Hence `γ ≤ c^g`, `β_k ≤ c^o + c^g`, `α_k ≤ c^w (m−k) + c^o + c^g`.
The coefficient of d_i in the dual objective is `E_i = α_{k+1} + β_k − γ ∈ [−c^g, ᾱ_{k+1}+β̄_k]` (k = position of i) — the bilinear
term τ_i E_i is linearised with these bounds. Verified: MILP == enumeration (mean-support and Wasserstein).

## 4. DRO-M (mean-support)
`sup_P E Q = min_ρ μ·ρ x + max_d [Q(d) − ρ·d x]`. Prop. 5 bounds: ρ ∈ [−c^g, 2c^o + c^g + c^w K], ζ ∈ [0, c^ward], φ ∈ [−c^ward, c^icu + c^ward].
The multiplier is attached to the *scheduled* indicator because an unscheduled surgery does not influence Q.
Test: with (x,s) fixed, C&CG == moment LP over the 2^m box vertices × integer LOS lattice (1e-6).

## 5. DRO-W (derived; Appendix D missing from the PDF)
`sup_{W1(P,P_N) ≤ ε} E Q = min_{ν ≥ 0} εν + (1/N) Σ_n max_d [Q(d) − ν‖d − d^n‖_1]`; per coordinate the objective is convex-kink + concave-kink,
so the maximiser is in `{lo_i, d^n_i, hi_i}`. ν ≤ Lipschitz constant of Q (L1) = ρ̄; φ_W ≤ c^icu + c^ward.
Master: one recourse copy per scenario r, cuts for every n: `δ_{n,b} ≥ Q_b(d^r) − ν Σ_{i∈b} |d^r_i − d^n_i|`, `δ^down_n ≥ Q^down(l^r) − φ_W Σ_i (|Δl^c| + |Δl^w|)`.
Test: fixed (x,s): C&CG == transport LP over {lo, d^n, hi}^m candidates; ε = 0 gives the SAA value; value is non-decreasing in ε.

## 6. LOS characterisation used in the separation MILP (Prop. 3 variant)
R_t = 1 if the patient has left the ICU by day t (monotone), H_t = 1 if left the ward (monotone, H ≤ R).
ICU occupancy `1[t ≥ a] − R_t`, ward occupancy `R_t − H_t`, `l^c = l^c_lo + Σ_{free t}(1 − R_t)`, `l^w = Σ_t (R_t − H_t)`.
Variables are fixed outside the windows `[a+lcLo, a+lcHi)` and `[a+lcLo+lwLo, a+lcHi+lwHi)`. Excess product: `p ≤ e_t`, `p ≤ occupancy`, `e_t ∈ [0,1]`
(exact: for binary occupancy the optimum has e_t ∈ {0,1}). Timeline extended to `T_ext` so stays never get truncated; only days < T cost.

## 7. Conventions and ambiguities
See README "Differences from the paper". Additional: weekend beds apply to days ≥ 5; surgeries run Mon–Fri (5 blocks days).
Blocks are closed (one type per block); K_b = min(max_positions, I_c, ⌊L / lo_d⌋).

## 8. Pitfalls found (chronological)
1. HiGHS `setSolution` with a partial vector acted as an *invalid incumbent* (UB < LB, UB below the oracle). Removed; never warm-start without re-validation.
2. HiGHS thread scheduler is process-global; per-solve `threads` changes error out ("Not Set" model status).
3. SB (27)–(29) is not exact for SAA (each surgery has its own draw): the sampled optimum moved 0.5 %. Test uses exchangeable scenarios.
4. A failing cross-check can be the *reference* (sign error in my own test LP).
5. Test tolerances: MILP feasibility tolerance gives ~3e-6 absolute differences on values ~8 000 — compare with 1e-3 absolute.
6. Streamlit ≥ 1.50: `use_container_width` is deprecated → `width="stretch"`.
7. Parallel benchmark workers share cores: two runs × 2 threads each are ~2× slower than a solo 4-thread run; gaps in the table are for this setting.

## 9. Computational observations (HiGHS)
SP extensive form (N=50, I=40) solves in < 1 min. DRO-M C&CG: I=24 converged in ≈ 4 min (49 iterations, master 5–9 s each); DRO-W needs ~3× more
recourse copies (many (n,r) cuts) and stalls around 1 % gap in 5 min. Master solve time, not separation, dominates (separation < 0.3 s).

## 10. What the paper's online supplement (mmc1.pdf, 44 pp.) confirmed or changed
Read after the first version was built; the DOI of the article carries the supplement (page 305 of the PDF).
* **Algorithm 2 (Appendix D) = the DRO-W algorithm derived here**: master (D.3) with scalar ν, φ and ψ = ν·x, φ·x; per-sample
  (n) sub-problems (C.26) / (C.37); UB = master − mean(δ) + mean(sub-problem values); scenarios added for *every* n. Differences in detail: the paper starts with
  an empty scenario set (we start from the nominal scenario), adds all N scenarios per iteration (we add the most violated N/3, de-duplicated), and uses
  crude bounds for ν̄, φ̄ (D.4); ours are the Lipschitz constants 2c^o + c^g + c^w K and c^icu + c^ward (tighter, tested not to cut optima).
* **Dual of Q^BC (C.4) is identical to the one derived here** (including the printed `∀k ∈ [K_b]` typo-free form) and Prop. 11's three-way choice
  `d_i ∈ {lo, d^n, hi}` with the penalty weighted by x (C.28) matches the enumeration used in `subproblems.py`.
* **Horizon convention (Remark 3)**: ICU/ward needs beyond T are *not* part of the objective — same as the convention chosen here.
* **"(10,7) beds" = (beds on weekdays, beds on weekend)** (Appendix J.1: "(20,20), (15,15), …, (5,5)"), for ICU and ward alike. Confirmed.
* **Real data** (Appendix G): Table G.1 (5 cardiac types), G.5 (Min & Yih, 9 specialties + shares), G.7 (R-I, 10 types) and the block schedules
  G.2 / G.4 / G.6 are now in `generator.py` (`paper_instance("Case"|"R"|"M", I)`). The 16,613-case hospital data itself stays private; the tables give
  mean/sd only, so durations/LOS are still **gamma-distributed with those moments, clipped at the 20/80 % quantiles** (the paper uses empirical quantiles).
  Case-I: I/5 surgeries per type, one OR per type Mon–Fri. R-I: I/10 per type. M-I: type shares of G.5.
* **Cost structures 3 and 4** appear in Tables G.1/I.1 but are defined in the main text only partially; we implement Costs 1 and 2.
* **Paper CPU times (Table I.1, CPLEX 12.6, 16 cores)**: R-70, (10,7) beds, Cost 1: SP 2932 s, DRO-M 610 s, DRO-W 1622 s; (5,2): 1218 / 264 / 1100 s.
  R-100: 1999 / 1526 / 1750 s. Symmetry breaking cut R-70 times from 1120 → 610 s (Cost 1); valid inequalities cut DRO-M from 21 → 10 iterations.
* **Appendix F**: Benders decomposition of the SP is *worse* than the direct MILP (gaps 11–23 % after 2 h) — consistent with solving SP as one MILP here.
* **Sensitivity (App. J, R-130)**: SP schedules more surgeries than DRO-M, the gap widening as beds shrink (e.g. (5,5): 83 vs 68); raising c^icu = c^ward from
  100 to 1200 cuts DRO-M volume 127 → 60. Qualitatively reproducible with `paper_instance` and the app sliders.
