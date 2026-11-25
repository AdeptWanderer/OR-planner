"""Section 5.2 (branching) and 5.3 (lifted minimal cover cuts)."""
import itertools
import math

import numpy as np
import pytest

from conftest import small_instances
from orsched import BPCParams, Column, generate_instance
from orsched.bpc import Solver
from orsched.compact import solve_compact
from orsched.cuts import lifted_cover_row, separate
from orsched.master import Master
from orsched.pricing import NodeRestrictions, solve_pricing
from verification.paper_models import paper_lifting_eq39


# ------------------------------------------------------------------ cuts
def test_eq39_as_printed_is_invalid_and_documented_deviation_is_needed():
    """Eq. (39) with the cover sorted NONDECREASING yields alpha_i = 1 for
    t_i = 3 on C = {3,3,8}, b = 13, i.e. z_i + z_c1 + z_c2 + z_c3 <= 2,
    which cuts off the feasible packing {i, c1, c2} (3+3+3 = 9 <= 13)."""
    C, b, t_i = [3, 3, 8], 13, 3
    alpha = paper_lifting_eq39(C, t_i)
    assert alpha == 1
    packing = [t_i, 3, 3]                          # i, c1, c2
    assert sum(packing) <= b
    lhs = alpha * 1 + 1 + 1                        # alpha*z_i + z_c1 + z_c2
    assert lhs > len(C) - 1, "printed eq. (39) would be valid here (test is wrong)"


def _all_feasible_subsets(dur, cap):
    items = list(dur)
    for r in range(len(items) + 1):
        for S in itertools.combinations(items, r):
            if sum(dur[i] for i in S) <= cap:
                yield S


def test_lifted_cover_cuts_valid_by_enumeration():
    """Every cut generated in real runs is valid for (36): enumerate all
    capacity-feasible subsets of the surgeon's cases on that day."""
    checked = 0
    for inst in small_instances()[:4]:
        s = Solver(inst, BPCParams(cg_time_limit=20, branching=False, final_ip_time=5, polish_time=0, workers=2))
        s.run()
        for row in s.cuts:
            d = row.day
            l = inst.surgeries[next(iter(row.coef))].surgeon
            dur = {i: inst.surgeries[i].duration for i in inst.by_surgeon(l) if inst.surgeries[i].deadline >= d}
            cap = inst.surgeon_max[l, d]
            if len(dur) > 16:
                continue
            for S in _all_feasible_subsets(dur, cap):
                assert sum(row.coef.get(i, 0) for i in S) <= row.hi + 1e-9
            checked += 1
    assert checked > 0, "no cut was generated - test is vacuous"


def test_cuts_do_not_cut_the_integer_optimum():
    inst = generate_instance(14, "B", seed=2, n_days=2, n_surgeons=3)
    opt = solve_compact(inst, 60)
    assert opt["status"] == "OPTIMAL"
    s = Solver(inst, BPCParams(cg_time_limit=30, branching=False, final_ip_time=5, polish_time=0, workers=2))
    s.run()
    x = np.ones(len(opt["columns"]))
    for row in s.cuts:
        lhs = sum(row.coefficient(c) for c in opt["columns"])
        assert lhs <= row.hi + 1e-9


# ------------------------------------------------------------------ branching
def _solver_with_master(inst):
    s = Solver(inst, BPCParams(workers=2))
    return s


def test_branching_rule_order_1_2_3():
    """(1) total #cases, then (2) #cases per surgeon (closest to .5), then (3) Ryan-Foster pair."""
    inst = generate_instance(16, "C", seed=1, n_days=1, n_surgeons=2, rooms=4, max_deadline=1)
    s = _solver_with_master(inst)
    A0, A1 = inst.by_surgeon(0)[:2]
    B0, B1 = inst.by_surgeon(1)[:2]
    t = lambda i: inst.surgeries[i].duration + inst.cleaning
    one = lambda i: Column(0, ((i, 0),))
    two = lambda i, j: Column(0, ((i, 0), (j, t(i))))
    m = Master(inst); m.add_columns([one(A0), one(A1), one(B0)])
    assert s.choose_branch(m, np.array([0.5, 1.0, 1.0]))[0] == "count"
    m = Master(inst); m.add_columns([one(A0), one(B0)])          # total 1 (integral), surgeon 0: 0.5
    assert s.choose_branch(m, np.array([0.5, 0.5]))[0] == "surgeon"
    m = Master(inst); m.add_columns([two(A0, B0), two(A0, B1), two(A1, B0), two(A1, B1)])
    br = s.choose_branch(m, np.array([0.5, 0.5, 0.5, 0.5]))      # counts integral, pairs fractional
    assert br[0] == "pair"


def test_branching_children_partition_integer_solutions():
    """Every integer solution satisfies exactly one child of each branch."""
    inst = generate_instance(14, "C", seed=3, n_days=2, n_surgeons=4)
    opt = solve_compact(inst, 60)
    cols = opt["columns"]
    s = _solver_with_master(inst)
    m_count = sum(len(c.items) for c in cols)
    for kind, what, val in [("count", None, m_count + 0.5), ("count", None, m_count - 0.5),
                            ("surgeon", 0, 1.5), ("surgeon", 1, 0.5)]:
        kids = s.children((kind, what, val), NodeRestrictions(), [])
        ok = [all(r.lo - 1e-9 <= sum(r.coefficient(c) for c in cols) <= r.hi + 1e-9 for r in rows)
              for _, rows, _ in kids]
        assert sum(ok) == 1
    pairs = list(itertools.combinations(range(inst.n), 2))[:40]
    for a, b in pairs:
        kids = s.children(("pair", (a, b), 0.5), NodeRestrictions(), [])
        ok = [all(r.compatible(c) for c in cols) for r, _, _ in kids]
        assert sum(ok) == 1


def test_pricing_respects_ryan_foster_restrictions():
    inst = small_instances()[2]
    from conftest import real_duals
    m, y = real_duals(inst, iterations=2)[-1]
    a, b = inst.eligible(0)[:2]
    for restr in [NodeRestrictions(together=[(a, b)]), NodeRestrictions(apart=[(a, b)])]:
        for d in range(inst.n_days):
            cols, _ = solve_pricing(inst, d, *m.day_duals(y, d), restrictions=restr, time_limit=10,
                                    max_columns=10_000)
            assert all(restr.compatible(c) for c in cols)
