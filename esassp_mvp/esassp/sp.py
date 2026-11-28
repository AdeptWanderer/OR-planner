"""Stochastic programming model (6) via sample average approximation — solved as one extensive-form MILP."""
from __future__ import annotations

import time

import numpy as np

from .evaluate import Scenarios, sample_scenarios
from .formulations import FirstStage, add_bc_recourse, add_down_recourse
from .milp import Model
from .model import Instance, mean_allocation
from .result import Params, Result


def solve_sp(inst: Instance, params: Params = Params(), scenarios: Scenarios | None = None,
             downstream: bool = True, scheduling: bool = True) -> Result:
    t0 = time.time()
    rng = np.random.default_rng(params.seed)
    sc = scenarios or sample_scenarios(inst, params.n_scenarios, rng)
    N = sc.n
    m = Model()
    fs = FirstStage(m, inst, params.symmetry)
    for r in range(N):
        add_bc_recourse(fs, sc.d[r], prob_obj=1.0 / N, c_w=None if scheduling else 0.0)
        if downstream:
            add_down_recourse(fs, sc.lc[r], sc.lw[r], prob_obj=1.0 / N)
    res = m.solve(time_limit=params.time_limit, gap=params.gap, threads=params.threads, log=params.solver_log)
    if not res.ok:
        raise RuntimeError(f"SP infeasible / no solution: {res.status}")
    sol = fs.extract(res.x)
    if not scheduling:
        sol = mean_allocation(inst, sol)
    gap = max(0.0, (res.obj - res.bound) / max(abs(res.obj), 1e-9))
    return Result("SP", sol, res.obj, res.bound, gap, time.time() - t0, res.status, downstream, scheduling,
                  history=[], stats=dict(n_vars=m.n_vars, n_rows=m.n_rows, n_sb=fs.n_sb, N=N))
