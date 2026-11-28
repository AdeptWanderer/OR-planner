"""Tiny MILP modelling layer over HiGHS (highspy).  Variables/rows are accumulated in numpy-friendly lists and
passed to HiGHS in one call; solves return primal values, objective and the *dual bound* (valid lower bound)."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import highspy
import numpy as np

INF = highspy.kHighsInf
_THREADS: Optional[int] = None      # HiGHS' global scheduler can only be initialised once per process


def set_threads(n: int) -> None:
    """Call before the first solve; later calls with another value are ignored (HiGHS limitation)."""
    global _THREADS
    if _THREADS is None:
        _THREADS = int(n)
Terms = Union[Dict[int, float], Sequence[Tuple[int, float]]]


@dataclass
class MilpResult:
    status: str
    obj: float
    bound: float
    x: Optional[np.ndarray]
    time: float
    gap: float

    @property
    def ok(self) -> bool:
        return self.x is not None

    def __getitem__(self, j):
        return self.x[j]


class Model:
    def __init__(self):
        self.lb: List[float] = []; self.ub: List[float] = []; self.cost: List[float] = []
        self.integer: List[bool] = []; self.names: List[str] = []
        self._r_start = [0]; self._r_idx: List[int] = []; self._r_val: List[float] = []
        self._r_lo: List[float] = []; self._r_hi: List[float] = []

    # ------------------------------------------------------------ building
    @property
    def n_vars(self) -> int: return len(self.lb)
    @property
    def n_rows(self) -> int: return len(self._r_lo)

    def var(self, lb: float = 0.0, ub: float = INF, obj: float = 0.0, integer: bool = False, name: str = "") -> int:
        self.lb.append(lb); self.ub.append(ub); self.cost.append(obj); self.integer.append(integer); self.names.append(name)
        return len(self.lb) - 1

    def binary(self, obj: float = 0.0, name: str = "") -> int:
        return self.var(0.0, 1.0, obj, True, name)

    def add(self, terms: Terms, lo: float = -INF, hi: float = INF) -> int:
        items = terms.items() if isinstance(terms, dict) else terms
        acc: Dict[int, float] = {}
        for j, c in items:
            acc[j] = acc.get(j, 0.0) + c
        for j, c in acc.items():
            if c != 0.0:
                self._r_idx.append(j); self._r_val.append(c)
        self._r_start.append(len(self._r_idx)); self._r_lo.append(lo); self._r_hi.append(hi)
        return len(self._r_lo) - 1

    def set_obj(self, j: int, c: float):
        self.cost[j] = c

    def fix(self, j: int, v: float):
        self.lb[j] = v; self.ub[j] = v

    # ------------------------------------------------------------ solving
    def solve(self, time_limit: float = 60.0, gap: float = 1e-4, threads: int = 1, log: bool = False, abs_gap: float = 1e-6) -> MilpResult:
        h = highspy.Highs()
        h.setOptionValue("output_flag", bool(log))
        global _THREADS
        if _THREADS is None:
            _THREADS = int(threads)
        h.setOptionValue("threads", _THREADS)
        h.setOptionValue("time_limit", float(max(time_limit, 0.05)))
        h.setOptionValue("mip_rel_gap", float(gap))
        h.setOptionValue("mip_abs_gap", float(abs_gap))
        lp = highspy.HighsLp()
        n, m = self.n_vars, self.n_rows
        lp.num_col_, lp.num_row_ = n, m
        lp.col_cost_ = np.asarray(self.cost, float)
        lp.col_lower_ = np.asarray(self.lb, float); lp.col_upper_ = np.asarray(self.ub, float)
        lp.row_lower_ = np.asarray(self._r_lo, float); lp.row_upper_ = np.asarray(self._r_hi, float)
        lp.a_matrix_.format_ = highspy.MatrixFormat.kRowwise
        lp.a_matrix_.start_ = np.asarray(self._r_start, np.int32)
        lp.a_matrix_.index_ = np.asarray(self._r_idx, np.int32)
        lp.a_matrix_.value_ = np.asarray(self._r_val, float)
        if any(self.integer):
            lp.integrality_ = [highspy.HighsVarType.kInteger if z else highspy.HighsVarType.kContinuous for z in self.integer]
        h.passModel(lp)
        t0 = time.time()
        h.run()
        dt = time.time() - t0
        st = h.modelStatusToString(h.getModelStatus())
        info = h.getInfo()
        has = info.primal_solution_status == 2
        x = np.array(h.getSolution().col_value) if has else None
        if any(self.integer):
            obj = info.objective_function_value if has else float("inf")
            bound = info.mip_dual_bound if np.isfinite(info.mip_dual_bound) else -float("inf")
            g = info.mip_gap if has else float("inf")
        else:
            obj = info.objective_function_value if has else float("inf")
            bound = obj if has else -float("inf"); g = 0.0
        return MilpResult(st, float(obj), float(bound), x, dt, float(g))
