"""Restricted master problem (model (1)-(7)) on top of HiGHS.

Internally HiGHS minimises  sum_j (-B_j) x_j.  With HiGHS' convention the
reduced cost of a column is c_j - y^T a_j, so a column *improves* the RMP iff

        value(j) := B_j + y^T a_j  > 0 .

The pricing problem therefore maximises ``value`` and only needs the row duals
``y`` (no sign book-keeping of the paper's pi's).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import highspy
import numpy as np

from .model import Column, Instance

INF = highspy.kHighsInf


@dataclass
class ExtraRow:
    """A row whose column coefficient is  sum_{i in column} coef[i]
    (restricted to columns of ``day`` if given).  Used for branching rows
    (30)-(33) and lifted-cover cuts (38)."""
    coef: Dict[int, float]
    lo: float
    hi: float
    day: Optional[int] = None
    kind: str = "cut"
    tag: tuple = ()

    def coefficient(self, col: Column) -> float:
        if self.day is not None and col.day != self.day:
            return 0.0
        return float(sum(self.coef.get(i, 0.0) for i, _ in col.items))


class Master:
    def __init__(self, inst: Instance, extra_rows: Sequence[ExtraRow] = (), penalty: Optional[float] = None):
        self.inst = inst
        self.h = highspy.Highs()
        self.h.setOptionValue("output_flag", False)
        self.h.setOptionValue("threads", 1)
        self.columns: List[Column] = []
        self.col_index: Dict[tuple, int] = {}
        # big-M on artificials: at least the total case duration, so a node whose
        # master can only be 'feasible' through artificials gets a bound below any
        # real schedule and is pruned by bound (never pruned on art > 0 alone)
        self.penalty = penalty if penalty is not None else float(
            max(10 * inst.slots_per_day, sum(s.duration for s in inst.surgeries) + 1))
        self.extra: List[ExtraRow] = []
        self._build_base_rows()
        # extra rows known at construction (cuts, branching rows) - before any
        # column so that '>=' rows can get an artificial variable
        for er in extra_rows:
            self.h.addRow(er.lo, er.hi, 0, np.array([], dtype=np.int32), np.array([]))
            self.extra.append(er)
            if er.lo > -INF and np.isfinite(er.lo):
                r = self.n_base_rows + len(self.extra) - 1
                self.h.addCol(self.penalty, 0.0, INF, 1, np.array([r], dtype=np.int32), np.array([1.0]))
                self.art_cols.append(("row", r))
        self.n_art = len(self.art_cols)

    # ------------------------------------------------------------- structure
    def _build_base_rows(self):
        inst, h = self.inst, self.h
        D, L, T = inst.n_days, inst.n_surgeons, inst.slots_per_day
        r = 0
        self.r_assign = {}
        for i in range(inst.n):
            lo = 1.0 if inst.is_mandatory(i) else -INF
            h.addRow(lo, 1.0, 0, np.array([], dtype=np.int32), np.array([]))
            self.r_assign[i] = r; r += 1
        self.r_rooms = {}
        for d in range(D):
            h.addRow(-INF, float(inst.rooms[d]), 0, np.array([], dtype=np.int32), np.array([]))
            self.r_rooms[d] = r; r += 1
        self.r_surg = {}
        for l in range(L):
            for d in range(D):
                h.addRow(-INF, float(inst.surgeon_max[l, d]), 0, np.array([], dtype=np.int32), np.array([]))
                self.r_surg[(l, d)] = r; r += 1
        # coloring constraints (6) only for surgeons that have cases on day d
        self.r_color = {}
        has = {(s.surgeon, d) for s in inst.surgeries for d in range(min(s.deadline + 1, D))}
        for (l, d) in sorted(has):
            for t in range(T):
                h.addRow(-INF, 1.0, 0, np.array([], dtype=np.int32), np.array([]))
                self.r_color[(d, l, t)] = r; r += 1
        self.n_base_rows = r
        # artificial variables for mandatory rows (keeps RMP feasible)
        self.art_cols = []
        for i in inst.mandatory:
            h.addCol(self.penalty, 0.0, 1.0, 1, np.array([self.r_assign[i]], dtype=np.int32), np.array([1.0]))
            self.art_cols.append(i)
        self.n_art = len(self.art_cols)

    def column_entries(self, col: Column) -> Tuple[List[int], List[float]]:
        inst = self.inst
        rows: Dict[int, float] = {}
        d = col.day
        rows[self.r_rooms[d]] = 1.0
        for i, s in col.items:
            sg = inst.surgeries[i]
            rows[self.r_assign[i]] = rows.get(self.r_assign[i], 0.0) + 1.0
            rs = self.r_surg[(sg.surgeon, d)]
            rows[rs] = rows.get(rs, 0.0) + sg.duration
            for t in range(s, s + sg.duration):
                rows[self.r_color[(d, sg.surgeon, t)]] = 1.0
        for k, er in enumerate(self.extra):
            v = er.coefficient(col)
            if v != 0.0:
                rows[self.n_base_rows + k] = v
        idx = sorted(rows)
        return idx, [rows[r] for r in idx]

    def add_columns(self, cols: Sequence[Column]) -> int:
        added = 0
        for c in cols:
            if c.key() in self.col_index:
                continue
            idx, val = self.column_entries(c)
            self.h.addCol(-float(c.value(self.inst)), 0.0, INF, len(idx),
                          np.array(idx, dtype=np.int32), np.array(val, dtype=float))
            self.col_index[c.key()] = len(self.columns)
            self.columns.append(c)
            added += 1
        return added

    def add_row(self, er: ExtraRow):
        idx, val = [], []
        for j, c in enumerate(self.columns):
            v = er.coefficient(c)
            if v != 0.0:
                idx.append(self.n_art + j); val.append(v)
        self.h.addRow(er.lo, er.hi, len(idx), np.array(idx, dtype=np.int32), np.array(val, dtype=float))
        self.extra.append(er)

    # --------------------------------------------------------------- solving
    def solve_lp(self):
        self.h.setOptionValue("time_limit", INF)
        self.h.run()
        st = self.h.getModelStatus()
        if st != highspy.HighsModelStatus.kOptimal:
            raise RuntimeError(f"RMP LP not optimal: {st}")
        sol = self.h.getSolution()
        y = np.array(sol.row_dual)
        x_all = np.array(sol.col_value)
        obj = -self.h.getInfo().objective_function_value      # max form
        return obj, x_all[self.n_art:], x_all[:self.n_art], y

    def solve_ip(self, time_limit: float = 60.0, start: Optional[Sequence[Column]] = None):
        """Solve the RMP with binary variables (gives a lower bound)."""
        h = self.h
        n = h.getNumCol()
        h.changeColsIntegrality(n, np.arange(n, dtype=np.int32),
                                np.array([highspy.HighsVarType.kInteger] * n))
        h.setOptionValue("time_limit", float(time_limit))
        h.setOptionValue("mip_rel_gap", 1e-6)
        if start:
            keys = {c.key() for c in start}
            xs = np.zeros(n)
            for j, c in enumerate(self.columns):
                if c.key() in keys:
                    xs[self.n_art + j] = 1.0
            try:
                sol0 = highspy.HighsSolution()
                sol0.col_value = list(xs)
                sol0.value_valid = True
                h.setSolution(sol0)
            except Exception:
                pass
        h.run()
        info = h.getInfo()
        sol = h.getSolution()
        h.changeColsIntegrality(n, np.arange(n, dtype=np.int32),
                                np.array([highspy.HighsVarType.kContinuous] * n))
        h.setOptionValue("time_limit", INF)
        if info.primal_solution_status < 2:   # no feasible solution
            return None, None
        x = np.array(sol.col_value)
        art = x[:self.n_art]
        cols = [c for j, c in enumerate(self.columns) if x[self.n_art + j] > 0.5]
        obj = sum(c.value(self.inst) for c in cols)
        if art.sum() > 0.5:
            return cols, -obj  # infeasible w.r.t. mandatory cases
        return cols, obj

    # ----------------------------------------------------------- dual access
    def day_duals(self, y: np.ndarray, d: int):
        """Return (G[i] array, start-cost function data, constant) for day d.

        value(col) = sum_i G[i] - sum_i startcost_i(s_i) + const
        """
        inst = self.inst
        G = np.zeros(inst.n)
        for i, sg in enumerate(inst.surgeries):
            g = sg.duration + y[self.r_assign[i]] + sg.duration * y[self.r_surg[(sg.surgeon, d)]]
            for k, er in enumerate(self.extra):
                if er.day is None or er.day == d:
                    c = er.coef.get(i)
                    if c:
                        g += y[self.n_base_rows + k] * c
            G[i] = g
        L, T = inst.n_surgeons, inst.slots_per_day
        pi6 = np.zeros((L, T))   # >= 0 (= -y)
        for l in range(L):
            for t in range(T):
                r = self.r_color.get((d, l, t))
                if r is not None:
                    pi6[l, t] = max(0.0, -y[r])
        const = y[self.r_rooms[d]]
        return G, pi6, const

    def column_value(self, col: Column, y: np.ndarray) -> float:
        idx, val = self.column_entries(col)
        return col.value(self.inst) + float(np.dot(y[idx], val))
