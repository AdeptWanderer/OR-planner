"""Pricing subproblem (Section 3.2) with dominance rules (Section 4.1) and
the fast knapsack-based infeasibility detection (Section 4.2).

For each day d we look for a single-room schedule maximising

    value = sum_{i in S} G_di  -  sum_{i in S} pi*_{d,i,V_i}  +  y4_d     (17)/(18)

subject to: deadlines, room horizon, sequence-dependent cleaning, surgeon
daily maximum (14) and value > 0 (19).

The paper solves a position-based CP model with IBM CP Optimizer.  Here the
same subproblem is solved with OR-Tools CP-SAT using optional interval
variables (an equivalent model): presence literals play the role of C_i,
start variables the role of V_p, a NoOverlap constraint plus pairwise
precedence literals encode (10)-(11) with cleaning, and element constraints
encode the irregular start-time costs pi*.  Because cleaning is only needed
between cases of *different* infection types, requiring OCT between every
pair of different-type cases is equivalent to requiring it between
consecutive ones (any two different-type cases have a type switch between
them), which avoids an explicit sequencing variable.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from ortools.sat.python import cp_model

from .model import Column, Instance

SCALE = 1000.0
EPS = 1e-6


@dataclass
class PricingStats:
    candidates: int = 0
    dr1_removed: float = 0.0       # % of V1 domain values removed by DR1
    dr2: int = 0
    dr3: int = 0
    dr4: int = 0
    conditioned: int = 0
    fid_infeasible: bool = False   # detected by fast infeasibility detection
    fid2_infeasible: bool = False  # all surgeries removed by dominance rules
    hit_knapsack_bound: bool = False
    cp_status: str = ""
    cp_time: float = 0.0
    n_columns: int = 0
    best_value: float = 0.0
    value_bound: float = 0.0       # valid upper bound on the subproblem value
    proven: bool = False


@dataclass
class NodeRestrictions:
    """Ryan-Foster branching decisions (branch type 3)."""
    together: List[Tuple[int, int]] = field(default_factory=list)
    apart: List[Tuple[int, int]] = field(default_factory=list)

    def involved(self) -> Set[int]:
        s = set()
        for a, b in self.together + self.apart:
            s.add(a); s.add(b)
        return s

    def compatible(self, col: Column) -> bool:
        S = set(col.surgeries())
        for a, b in self.together:
            if (a in S) != (b in S):
                return False
        for a, b in self.apart:
            if a in S and b in S:
                return False
        return True


def start_costs(inst: Instance, pi6: np.ndarray, i: int) -> np.ndarray:
    """pi*_{d,i,v} for v = 0..T-t_i (sum of coloring duals covered)."""
    sg = inst.surgeries[i]
    T = inst.slots_per_day
    row = pi6[sg.surgeon]
    cs = np.concatenate([[0.0], np.cumsum(row)])
    v = np.arange(0, T - sg.duration + 1)
    return cs[v + sg.duration] - cs[v]


def knapsack_bound(inst: Instance, d: int, cand: List[int], delta_best: Dict[int, float]) -> float:
    """Fast infeasibility detection (Section 4.2) at p* = 1.

    Greedy on Delta_best / t over a 'room + per-surgeon' capacity structure.
    The capacity family is laminar, so the greedy fractional solution is
    optimal for the LP relaxation of (24)-(27) and is a valid upper bound.
    """
    RAO = float(inst.slots_per_day)
    RAS = {l: float(inst.surgeon_max[l, d]) for l in range(inst.n_surgeons)}
    ub = 0.0
    items = sorted((i for i in cand if delta_best[i] > 0),
                   key=lambda i: -delta_best[i] / inst.surgeries[i].duration)
    for i in items:
        if RAO <= 0:
            break
        t = inst.surgeries[i].duration
        l = inst.surgeries[i].surgeon
        take = min(t, RAS[l], RAO)
        if take <= 0:
            continue
        ub += delta_best[i] * take / t
        RAS[l] -= take
        RAO -= take
    return ub


class _Collector(cp_model.CpSolverSolutionCallback):
    def __init__(self, x, s, cand, limit):
        super().__init__()
        self.x, self.s, self.cand, self.limit = x, s, cand, limit
        self.solutions: List[Tuple[Tuple[int, int], ...]] = []

    def on_solution_callback(self):
        items = tuple(sorted(((i, int(self.value(self.s[i]))) for i in self.cand if self.value(self.x[i])),
                             key=lambda z: z[1]))
        self.solutions.append(items)
        if len(self.solutions) >= self.limit:
            self.stop_search()


def solve_pricing(inst: Instance, d: int, G: np.ndarray, pi6: np.ndarray, const: float,
                  restrictions: Optional[NodeRestrictions] = None, time_limit: float = 10.0,
                  use_dominance: bool = True, use_fid: bool = True, max_columns: int = 20,
                  workers: int = 4, seed: int = 0) -> Tuple[List[Column], PricingStats]:
    st = PricingStats()
    restrictions = restrictions or NodeRestrictions()
    T, OCT = inst.slots_per_day, inst.cleaning
    A = inst.surgeon_max[:, d]
    cand = [i for i in inst.eligible(d) if inst.surgeries[i].duration <= min(A[inst.surgeries[i].surgeon], T)]
    # Ryan-Foster 'together': partner unavailable -> both unavailable
    changed = True
    cset = set(cand)
    while changed:
        changed = False
        for a, b in restrictions.together:
            if (a in cset) != (b in cset):
                cset.discard(a); cset.discard(b); changed = True
    cand = [i for i in cand if i in cset]
    st.candidates = len(cand)
    if not cand:
        st.fid2_infeasible = True
        st.value_bound = const
        st.proven = True
        return [], st

    dur = {i: inst.surgeries[i].duration for i in cand}
    surg = {i: inst.surgeries[i].surgeon for i in cand}
    inf = {i: inst.surgeries[i].infection for i in cand}
    SD = min(inst.surgeries[i].duration for i in inst.eligible(d))  # SD_d

    # ---------------- start domains (DR1) & start costs --------------------
    costs = {i: start_costs(inst, pi6, i) for i in cand}
    dom: Dict[int, np.ndarray] = {}
    for i in cand:
        v = np.arange(0, T - dur[i] + 1)
        if use_dominance:
            v = v[(v == 0) | (v >= SD)]
        dom[i] = v
    if use_dominance:
        st.dr1_removed = 100.0 * max(0, SD - 1) / T
    dbest = {i: G[i] - float(costs[i][dom[i]].min()) for i in cand}
    dworst = {i: G[i] - float(costs[i][dom[i]].max()) for i in cand}

    protected = restrictions.involved()
    alive = list(cand)
    cond: Dict[int, Set[int]] = {i: set() for i in cand}   # i scheduled => all of cond[i]

    if use_dominance:
        # ---- DR2: surgeries that can only worsen the objective
        keep = []
        for i in alive:
            if dbest[i] < -EPS and i not in protected:
                st.dr2 += 1
            else:
                keep.append(i)
        alive = keep
        # ---- DR3 (a)(b)(c): dominated surgeries
        removed: Set[int] = set()
        order = sorted(alive, key=lambda i: dbest[i])
        for i in order:
            if i in protected:
                continue
            for ip in alive:
                if ip == i or ip in removed or ip in protected:
                    continue
                lam = 0 if inf[i] == inf[ip] else 2 * OCT
                c1 = dworst[ip] > dbest[i] + EPS
                c2 = dur[i] >= dur[ip] + lam
                if not (c1 and c2):
                    continue
                if surg[i] == surg[ip]:
                    if dur[i] + dur[ip] > A[surg[i]]:          # (3): cannot coexist -> remove i
                        removed.add(i); st.dr3 += 1
                        break
                    cond[i].add(ip)                            # DR3(b)
                else:
                    if A[surg[ip]] >= T:                        # DR3(c)
                        cond[i].add(ip)
        alive = [i for i in alive if i not in removed]
        aset = set(alive)
        for i in alive:
            cond[i] &= aset
        # ---- DR4: conditioning chains that cannot fit
        def closure(i):
            seen, stack = set(), list(cond[i])
            while stack:
                k = stack.pop()
                if k in seen:
                    continue
                seen.add(k); stack.extend(cond[k])
            seen.discard(i)
            return seen
        changed = True
        while changed:
            changed = False
            aset = set(alive)
            for i in list(alive):
                P = closure(i)
                if not P <= aset:
                    alive.remove(i); aset.discard(i); st.dr4 += 1; changed = True
                    continue
                tot = dur[i] + sum(dur[k] for k in P)
                tl = dur[i] + sum(dur[k] for k in P if surg[k] == surg[i])
                if tot > T or tl > A[surg[i]]:
                    alive.remove(i); aset.discard(i); st.dr4 += 1; changed = True
        st.conditioned = sum(1 for i in alive if cond[i])
        if not alive:
            st.fid2_infeasible = True
            st.value_bound = const if const > 0 else 0.0
            st.value_bound = max(0.0, const)
            st.proven = True
            return [], st

    # ---------------- fast infeasibility detection --------------------------
    kb = knapsack_bound(inst, d, alive, dbest)
    ub_value = kb + const
    st.value_bound = ub_value
    if use_fid and ub_value <= EPS:
        st.fid_infeasible = True
        st.proven = True
        return [], st

    # ---------------- CP-SAT model -----------------------------------------
    m = cp_model.CpModel()
    x, s, iv = {}, {}, {}
    for i in alive:
        x[i] = m.new_bool_var(f"x{i}")
        s[i] = m.new_int_var_from_domain(cp_model.Domain.from_values(dom[i].tolist()), f"s{i}")
        iv[i] = m.new_optional_fixed_size_interval_var(s[i], dur[i], x[i], f"I{i}")
    m.add_no_overlap(list(iv.values()))
    # redundant aggregated capacity: total duration + one cleaning per extra
    # infection-type group used (exact makespan of a single room schedule)
    types = sorted(set(inf[i] for i in alive))
    if OCT > 0 and len(types) > 1:
        u = {}
        for tau in types:
            u[tau] = m.new_bool_var(f"u{tau}")
            members = [x[i] for i in alive if inf[i] == tau]
            for xi in members:
                m.add_implication(xi, u[tau])
            m.add_bool_or(members).only_enforce_if(u[tau])
        m.add(sum(dur[i] * x[i] for i in alive) + OCT * sum(u.values()) <= T + OCT)
    else:
        m.add(sum(dur[i] * x[i] for i in alive) <= T)
    for l in set(surg[i] for i in alive):
        members = [i for i in alive if surg[i] == l]
        if sum(dur[i] for i in members) > A[l]:
            m.add(sum(dur[i] * x[i] for i in members) <= int(A[l]))
    # cleaning between different infection types
    if OCT > 0:
        for a_idx, a in enumerate(alive):
            for b in alive[a_idx + 1:]:
                if inf[a] == inf[b]:
                    continue
                if dur[a] + dur[b] + OCT > T:
                    m.add_bool_or([x[a].Not(), x[b].Not()])
                    continue
                before = m.new_bool_var("")
                m.add(s[b] >= s[a] + dur[a] + OCT).only_enforce_if([x[a], x[b], before])
                m.add(s[a] >= s[b] + dur[b] + OCT).only_enforce_if([x[a], x[b], before.Not()])
    # dominance conditioning (22)
    for i in alive:
        for k in cond[i]:
            m.add_implication(x[i], x[k])
    # Ryan-Foster
    for a, b in restrictions.together:
        if a in x and b in x:
            m.add(x[a] == x[b])
        elif a in x:
            m.add(x[a] == 0)
        elif b in x:
            m.add(x[b] == 0)
    for a, b in restrictions.apart:
        if a in x and b in x:
            m.add_bool_or([x[a].Not(), x[b].Not()])

    obj_terms = []
    for i in alive:
        gi = int(round(G[i] * SCALE))
        obj_terms.append(gi * x[i])
        c = costs[i]
        if c.max() > EPS:
            arr = [int(round(v * SCALE)) for v in c]
            cv = m.new_int_var(0, max(arr), f"c{i}")
            m.add_element(s[i], arr, cv)
            e = m.new_int_var(0, max(arr), f"e{i}")
            m.add(e == cv).only_enforce_if(x[i])
            m.add(e == 0).only_enforce_if(x[i].Not())
            obj_terms.append(-e)
    cst = int(round(const * SCALE))
    objective = sum(obj_terms) + cst
    m.add(objective >= 1)                              # (19): positive value
    if use_fid:
        m.add(objective <= int(np.floor(ub_value * SCALE)) + len(alive))  # (29)
    m.maximize(objective)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit)
    solver.parameters.num_workers = workers
    solver.parameters.random_seed = seed
    cb = _Collector(x, s, alive, max_columns * 3)
    t0 = time.time()
    status = solver.solve(m, cb)
    st.cp_time = time.time() - t0
    st.cp_status = solver.status_name(status)

    cols: List[Column] = []
    seen = set()
    best = 0.0
    for items in reversed(cb.solutions):
        if not items or items in seen:
            continue
        seen.add(items)
        val = sum(G[i] - costs[i][v] for i, v in items) + const
        if val > EPS:
            cols.append(Column(day=d, items=items))
            best = max(best, val)
        if len(cols) >= max_columns:
            break
    st.n_columns = len(cols)
    st.best_value = best
    if status == cp_model.OPTIMAL:
        st.proven = True
        st.value_bound = best
        st.hit_knapsack_bound = use_fid and best * SCALE >= np.floor(ub_value * SCALE) - 1
    elif status == cp_model.INFEASIBLE:
        st.proven = True
        st.value_bound = 0.0
    else:
        bound = solver.best_objective_bound / SCALE if status == cp_model.FEASIBLE else ub_value
        st.value_bound = min(ub_value, bound + len(alive) / SCALE)
        if not cols:
            st.value_bound = ub_value
    return cols, st


def column_value_on_day(inst: Instance, col: Column, d: int, G: np.ndarray, pi6: np.ndarray,
                        const: float, restrictions: Optional[NodeRestrictions] = None) -> Optional[float]:
    """Value of copying a column to another day d (Algorithm 1, line 12)."""
    used: Dict[int, int] = {}
    for i, _ in col.items:
        sg = inst.surgeries[i]
        if sg.deadline < d:
            return None
        used[sg.surgeon] = used.get(sg.surgeon, 0) + sg.duration
    for l, v in used.items():
        if v > inst.surgeon_max[l, d]:
            return None
    c2 = Column(day=d, items=col.items)
    if restrictions and not restrictions.compatible(c2):
        return None
    val = const
    for i, v in col.items:
        val += G[i] - start_costs(inst, pi6, i)[v]
    return val
