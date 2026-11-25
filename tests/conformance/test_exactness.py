"""End-to-end exactness against two independent exact solvers.

* monolithic CP-SAT model (orsched.compact) on small instances
* time-indexed MIP solved by HiGHS (verification.paper_models) on tiny ones
The B&P&C lower bound must equal the optimum, its upper bound must be >= the
optimum, and every schedule must pass the independent validator.
"""
import pytest

from orsched import BPCParams, generate_instance, solve, validate_solution
from orsched.compact import solve_compact
from verification.paper_models import time_indexed_mip

P = BPCParams(cg_time_limit=60, branch_time_limit=90, final_ip_time=20, polish_time=0, workers=2)

SMALL = [dict(n_surgeries=n, instance_set=S, seed=sd, n_days=dd, n_surgeons=ns, rooms=r,
              cleaning_minutes=oc, n_infection_types=it)
         for (n, S, sd, dd, ns, r, oc, it) in [
             (12, "A", 1, 2, 4, 2, 30, 1), (14, "A", 2, 3, 4, 2, 30, 2), (16, "A", 3, 2, 5, None, 60, 1),
             (12, "B", 4, 2, 4, None, 30, 1), (16, "B", 5, 3, 4, None, 30, 2), (18, "B", 6, 2, 5, None, 0, 1),
             (14, "C", 7, 2, 4, 1, 30, 1), (18, "C", 8, 2, 4, None, 30, 3), (20, "C", 9, 3, 4, None, 60, 2),
             (16, "C", 10, 2, 3, 2, 0, 1), (20, "A", 11, 2, 5, 3, 30, 1), (20, "B", 12, 3, 5, None, 30, 1)]]


@pytest.mark.parametrize("spec", SMALL, ids=lambda s: f"{s['instance_set']}{s['n_surgeries']}-s{s['seed']}")
def test_bpc_vs_monolithic_cpsat(spec):
    inst = generate_instance(**spec)
    sol = solve(inst, P)
    assert validate_solution(inst, sol) == []
    ref = solve_compact(inst, 300, workers=2)
    assert ref["status"] == "OPTIMAL", "reference solver did not prove optimality"
    assert sol.upper_bound >= ref["lb"] - 1e-6, "B&P&C upper bound is below the true optimum"
    assert sol.lower_bound <= ref["lb"] + 1e-6, "B&P&C plan better than the optimum: invalid plan"
    assert sol.lower_bound == pytest.approx(ref["lb"]), f"suboptimal: {sol.lower_bound} vs {ref['lb']}"


TINY = [dict(n_surgeries=n, instance_set="C", seed=sd, n_days=dd, n_surgeons=ns, rooms=r,
             day_minutes=240, slot_minutes=15, cleaning_minutes=30, n_infection_types=it, max_deadline=3)
        for (n, sd, dd, ns, r, it) in [(6, 1, 1, 2, 1, 1), (7, 2, 2, 2, 1, 2), (8, 3, 2, 3, 2, 1),
                                       (8, 4, 2, 2, 1, 2), (9, 5, 2, 3, 2, 1), (10, 6, 3, 3, 1, 1)]]


@pytest.mark.parametrize("spec", TINY, ids=lambda s: f"tiny{s['n_surgeries']}-s{s['seed']}")
def test_bpc_vs_time_indexed_mip(spec):
    inst = generate_instance(**spec)
    sol = solve(inst, P)
    assert validate_solution(inst, sol) == []
    opt, st = time_indexed_mip(inst, 300)
    assert st == "OPTIMAL"
    assert sol.lower_bound == pytest.approx(opt)
    assert sol.upper_bound >= opt - 1e-6
    ref = solve_compact(inst, 120, workers=2)          # the two references agree too
    assert ref["lb"] == pytest.approx(opt)
