"""Paper-literal reference implementations used ONLY for verification.

Each function below is written directly from the equations of
Hashemi Doulabi, Rousseau & Pesant (2016), independently of the production
code in ``orsched``.  The conformance tests compare the production code
against these references.

    position_pricing      - CP subproblem (8)-(15), objective (17), cut (19)
    paper_fid_bound       - fast infeasibility detection, Section 4.2 steps 1-5
    paper_lifting_eq39    - lifted cover coefficients, eq. (39) AS PRINTED
    master_lp_from_formulas - master LP relaxation (1)-(6) built from the formulas
    reduced_cost_formula  - reduced cost sigma_j (Section 3.2)
    time_indexed_mip      - an independent compact MIP of the whole IORPS (HiGHS)
"""
from __future__ import annotations

import itertools
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from ortools.sat.python import cp_model

SCALE = 1000.0


# --------------------------------------------------------------------------
# Section 3.2: position-based CP model of the pricing subproblem
# --------------------------------------------------------------------------
def position_pricing(inst, d: int, G: np.ndarray, pi6: np.ndarray, const: float,
                     time_limit: float = 60.0, n_positions: Optional[int] = None, dr1: bool = False):
    """Solve  max { sum_p G[W_p] - sum_p pi*[W_p][V_p] - pi4_d }  s.t. (8)-(15).

    G      : G_di for i = 0..n-1 (the paper's G_d[i]; position value 0 -> 0)
    pi6    : coloring duals pi^(6)_{d,l,t} >= 0, shape (L, T)
    const  : -pi^(4)_d  (we pass the constant term directly)

    dr1    : also impose Dominance Rule 1, (V_1 == 0) or (V_1 >= SD_d).

    Returns (value of best schedule or None, items, status, proven upper bound).
    The surgery index in the paper is 1-based with 0 = empty; we map surgery i
    to value i+1.
    """
    T, OCT = inst.slots_per_day, inst.cleaning
    surg = inst.surgeries
    elig = [s.id for s in surg if s.deadline >= d]                   # (8)
    if n_positions is None:
        sd = min([surg[i].duration for i in elig] or [T])
        n_positions = max(1, T // sd)
    n = n_positions
    vals = [0] + [i + 1 for i in elig]
    N1 = inst.n + 1
    t = [0] + [s.duration for s in surg]                               # t[0] = 0
    sl = [0] + [s.surgeon for s in surg]
    inf = [None] + [s.infection for s in surg]

    def CL(a, b):                                                      # CL_0i = CL_i0 = 0
        if a == 0 or b == 0:
            return 0
        return OCT if inf[a] != inf[b] else 0

    def pistar(a, v):                                                  # pi*_{d,i,v}
        if a == 0:
            return 0.0
        l = sl[a]
        return float(pi6[l, v:min(T, v + t[a])].sum())

    m = cp_model.CpModel()
    W = [m.new_int_var_from_domain(cp_model.Domain.from_values(vals), f"W{p}") for p in range(n)]
    V = [m.new_int_var(0, T, f"V{p}") for p in range(n)]                # (9)
    tW = [m.new_int_var(0, T, f"tW{p}") for p in range(n)]
    for p in range(n):
        m.add_element(W[p], t, tW[p])
        m.add(V[p] + tW[p] <= T)        # finish inside the day (implicit in the paper)
    isz = [m.new_bool_var(f"z{p}") for p in range(n)]
    for p in range(n):
        m.add(W[p] == 0).only_enforce_if(isz[p])
        m.add(W[p] != 0).only_enforce_if(isz[p].Not())
    for p in range(n - 1):                                             # (10)
        m.add(W[p + 1] == 0).only_enforce_if(isz[p])
    cl_tab = [CL(a, b) for a in range(N1) for b in range(N1)]
    for p in range(n - 1):                                             # (11)
        idx = m.new_int_var(0, N1 * N1 - 1, "")
        m.add(idx == W[p] * N1 + W[p + 1])
        cl = m.new_int_var(0, OCT, "")
        m.add_element(idx, cl_tab, cl)
        m.add(V[p + 1] >= V[p] + tW[p] + cl)
    for i in elig:                                                     # (12)-(13)
        bs = []
        for p in range(n):
            b = m.new_bool_var("")
            m.add(W[p] == i + 1).only_enforce_if(b)
            m.add(W[p] != i + 1).only_enforce_if(b.Not())
            bs.append(b)
        m.add(sum(bs) <= 1)
    for l in range(inst.n_surgeons):                                   # (14)
        tl = [t[a] if (a and sl[a] == l) else 0 for a in range(N1)]
        terms = []
        for p in range(n):
            v = m.new_int_var(0, T, "")
            m.add_element(W[p], tl, v)
            terms.append(v)
        m.add(sum(terms) <= int(inst.surgeon_max[l, d]))
    if dr1:                                                            # Dominance Rule 1
        sd = min([surg[i].duration for i in elig] or [T])
        b = m.new_bool_var("v1zero")
        m.add(V[0] == 0).only_enforce_if(b)
        m.add(V[0] >= sd).only_enforce_if(b.Not())
    for p in range(1, n):                                              # (15)
        m.add(V[p] == V[p - 1] + tW[p - 1]).only_enforce_if(isz[p])
    # objective (17)
    Gtab = [0] + [int(round(G[i] * SCALE)) for i in range(inst.n)]
    ps_tab = [int(round(pistar(a, v) * SCALE)) for a in range(N1) for v in range(T + 1)]
    obj = []
    for p in range(n):
        g = m.new_int_var(min(Gtab), max(Gtab), "")
        m.add_element(W[p], Gtab, g)
        idx = m.new_int_var(0, N1 * (T + 1) - 1, "")
        m.add(idx == W[p] * (T + 1) + V[p])
        c = m.new_int_var(0, max(ps_tab), "")
        m.add_element(idx, ps_tab, c)
        obj += [g, -c]
    m.maximize(sum(obj) + int(round(const * SCALE)))
    s = cp_model.CpSolver()
    s.parameters.max_time_in_seconds = time_limit
    s.parameters.num_workers = 4
    st = s.solve(m)
    if st not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None, [], s.status_name(st), None
    items = [(s.value(W[p]) - 1, s.value(V[p])) for p in range(n) if s.value(W[p]) != 0]
    # exact float value of the returned schedule
    val = const + sum(G[i] - pistar(i + 1, v) for i, v in items)
    bound = s.best_objective_bound / SCALE + (2 * n + 1) * 0.5 / SCALE     # undo integer scaling, conservatively
    return val, items, s.status_name(st), bound


# --------------------------------------------------------------------------
# Section 4.2: fast infeasibility detection, literally steps 1-5
# --------------------------------------------------------------------------
def paper_fid_bound(inst, d: int, cand: Sequence[int], dbest: Dict[int, float]) -> float:
    UB = 0.0                                                            # step 1 (p* = 1)
    RAO = float(inst.slots_per_day)
    RAS = {l: float(inst.surgeon_max[l, d]) for l in range(inst.n_surgeons)}
    remaining = [i for i in cand if dbest[i] > 0]
    while remaining:
        i = max(remaining, key=lambda k: dbest[k] / inst.surgeries[k].duration)   # step 2
        remaining.remove(i)
        ti, l = inst.surgeries[i].duration, inst.surgeries[i].surgeon
        if ti <= RAO and ti <= RAS[l]:                                  # step 3
            UB += dbest[i]; RAS[l] -= ti; RAO -= ti
            continue
        if ti > RAS[l] and RAS[l] < RAO:                                # step 4
            UB += RAS[l] * dbest[i] / ti
            RAO -= RAS[l]; RAS[l] = 0
            continue
        if ti > RAO and RAO <= RAS[l]:                                  # step 5
            UB += RAO * dbest[i] / ti
            break
    return UB


# --------------------------------------------------------------------------
# Section 5.3: lifting coefficients, eq. (39) exactly as printed
# --------------------------------------------------------------------------
def paper_lifting_eq39(durations_C: Sequence[int], t_i: int) -> int:
    """theta_r = sum of the r SMALLEST cover weights (C sorted nondecreasing)."""
    c = sorted(durations_C)
    theta = list(itertools.accumulate(c))
    C = len(c)
    if t_i >= theta[-1]:
        return C
    for r in range(1, C):
        if theta[r - 1] <= t_i <= theta[r] - 1:
            return r
    return 0


# --------------------------------------------------------------------------
# Section 3.1: master LP (1)-(6) built directly from the formulas
# --------------------------------------------------------------------------
def master_lp_from_formulas(inst, columns):
    """Return (objective, x, duals dict) of the LP relaxation using scipy.

    Duals are returned in the paper's convention for a MAXIMISATION problem:
    pi >= 0 for <= rows, free for = rows.
    """
    from scipy.optimize import linprog
    from scipy.sparse import lil_matrix
    J = len(columns)
    D, L, T = inst.n_days, inst.n_surgeons, inst.slots_per_day
    B = np.array([sum(inst.surgeries[i].duration for i, _ in c.items) for c in columns], float)
    a1 = {}  # a'_ij
    for j, c in enumerate(columns):
        for i, _ in c.items:
            a1[(i, j)] = 1
    rows_eq, rows_ub, b_eq, b_ub, tags_eq, tags_ub = [], [], [], [], [], []
    for i in range(inst.n):
        r = np.zeros(J)
        for j in range(J):
            r[j] = a1.get((i, j), 0)
        if inst.is_mandatory(i):
            rows_eq.append(r); b_eq.append(1); tags_eq.append(("2", i))          # (2)
        else:
            rows_ub.append(r); b_ub.append(1); tags_ub.append(("3", i))          # (3)
    for d in range(D):                                                            # (4)
        rows_ub.append(np.array([1.0 if c.day == d else 0 for c in columns])); b_ub.append(inst.rooms[d]); tags_ub.append(("4", d))
    for l in range(L):                                                            # (5)
        for d in range(D):
            r = np.array([sum(inst.surgeries[i].duration for i, _ in c.items if inst.surgeries[i].surgeon == l)
                          if c.day == d else 0 for c in columns], float)
            rows_ub.append(r); b_ub.append(inst.surgeon_max[l, d]); tags_ub.append(("5", l, d))
    for d in range(D):                                                            # (6)
        for l in range(L):
            for t in range(T):
                r = np.zeros(J)
                for j, c in enumerate(columns):
                    if c.day != d:
                        continue
                    for i, s in c.items:
                        if inst.surgeries[i].surgeon == l and s <= t < s + inst.surgeries[i].duration:
                            r[j] += 1
                if r.any():
                    rows_ub.append(r); b_ub.append(1); tags_ub.append(("6", d, l, t))
    res = linprog(-B, A_ub=np.array(rows_ub), b_ub=b_ub, A_eq=np.array(rows_eq) if rows_eq else None,
                  b_eq=b_eq if rows_eq else None, bounds=[(0, None)] * J, method="highs")
    assert res.status == 0, res.message
    duals = {}
    for tg, v in zip(tags_ub, res.ineqlin.marginals):
        duals[tg] = -v          # scipy marginals are for min(-B): pi = -marginal >= 0
    if rows_eq:
        for tg, v in zip(tags_eq, res.eqlin.marginals):
            duals[tg] = -v
    return -res.fun, res.x, duals


def reduced_cost_formula(inst, col, duals) -> float:
    """sigma_j of Section 3.2 (paper's notation), for a column on day col.day."""
    d = col.day
    sig = sum(inst.surgeries[i].duration for i, _ in col.items)
    for i, s in col.items:
        sg = inst.surgeries[i]
        sig -= duals.get(("2", i), duals.get(("3", i), 0.0))
        sig -= sg.duration * duals.get(("5", sg.surgeon, d), 0.0)
        for t in range(s, s + sg.duration):
            sig -= duals.get(("6", d, sg.surgeon, t), 0.0)
    sig -= duals.get(("4", d), 0.0)
    return sig


# --------------------------------------------------------------------------
# An independent compact MIP of the full problem (time-indexed, HiGHS)
# --------------------------------------------------------------------------
def time_indexed_mip(inst, time_limit: float = 120.0):
    """x[i,d,k,s] = 1 if surgery i starts at slot s in room k on day d.

    Constraints: each mandatory case exactly once / optional at most once,
    deadlines, room capacity per slot with cleaning (a case i' of a different
    infection type may not start within [s, s+t_i+OCT) of i in the same room,
    in either order), surgeon one-place-at-a-time per slot, surgeon daily
    minutes.  Returns (optimal minutes or None, status).
    """
    import highspy
    T, OCT = inst.slots_per_day, inst.cleaning
    h = highspy.Highs(); h.setOptionValue("output_flag", False)
    h.setOptionValue("time_limit", float(time_limit)); h.setOptionValue("threads", 1)
    var = {}
    k_cols = 0
    for sg in inst.surgeries:
        for d in range(min(sg.deadline + 1, inst.n_days)):
            for k in range(inst.rooms[d]):
                for s in range(0, T - sg.duration + 1):
                    var[(sg.id, d, k, s)] = k_cols; k_cols += 1
    h.addVars(k_cols, np.zeros(k_cols), np.ones(k_cols))
    cost = np.zeros(k_cols)
    for (i, d, k, s), j in var.items():
        cost[j] = -inst.surgeries[i].duration
    h.changeColsCost(k_cols, np.arange(k_cols, dtype=np.int32), cost)
    h.changeColsIntegrality(k_cols, np.arange(k_cols, dtype=np.int32),
                            np.array([highspy.HighsVarType.kInteger] * k_cols))
    INF = highspy.kHighsInf

    def row(idx, lo, hi):
        idx = sorted(set(idx))
        h.addRow(lo, hi, len(idx), np.array(idx, dtype=np.int32), np.ones(len(idx)))

    by_i = {}
    for key, j in var.items():
        by_i.setdefault(key[0], []).append(j)
    for sg in inst.surgeries:
        js = by_i.get(sg.id, [])
        if inst.is_mandatory(sg.id):
            if not js:
                return None, "infeasible (no slot before deadline)"
            row(js, 1, 1)
        elif js:
            row(js, -INF, 1)
    for d in range(inst.n_days):
        for k in range(inst.rooms[d]):
            items = [(i, s, j) for (i, dd, kk, s), j in var.items() if dd == d and kk == k]
            # room occupancy per slot (no overlap)
            for t in range(T):
                js = [j for i, s, j in items if s <= t < s + inst.surgeries[i].duration]
                if len(js) > 1:
                    row(js, -INF, 1)
            # cleaning: i at s and i' (different type) at s' with s' in [s, s+t_i+OCT)
            if OCT:
                for (i, s, j) in items:
                    ti = inst.surgeries[i].duration
                    for (i2, s2, j2) in items:
                        if i2 == i or inst.surgeries[i2].infection == inst.surgeries[i].infection:
                            continue
                        if s <= s2 < s + ti + OCT:
                            row([j, j2], -INF, 1)
        for l in range(inst.n_surgeons):
            mine = [(i, k, s, j) for (i, dd, k, s), j in var.items() if dd == d and inst.surgeries[i].surgeon == l]
            if not mine:
                continue
            for t in range(T):
                js = [j for i, k, s, j in mine if s <= t < s + inst.surgeries[i].duration]
                if len(js) > 1:
                    row(js, -INF, 1)
            idx = [j for *_, j in mine]
            vals = [inst.surgeries[i].duration for i, k, s, j in mine]
            h.addRow(-INF, float(inst.surgeon_max[l, d]), len(idx), np.array(idx, dtype=np.int32), np.array(vals, float))
        # rooms used per day is implicit (k ranges over the K_d rooms)
    h.run()
    st = h.getModelStatus()
    if st == highspy.HighsModelStatus.kOptimal:
        return -h.getInfo().objective_function_value * inst.slot_minutes, "OPTIMAL"
    if st == highspy.HighsModelStatus.kInfeasible:
        return None, "INFEASIBLE"
    return None, str(st)
