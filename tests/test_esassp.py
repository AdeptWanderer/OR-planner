"""Exact cross-checks for the ESASSP implementation (every dual / reformulation against an independent primal oracle)."""
import itertools
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import linprog

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from esassp import (Params, Solution, generate_instance, sample_scenarios, solve, validate_solution)  # noqa: E402
from esassp.dataio import instance_from_frames, instance_to_frames  # noqa: E402
from esassp.dro import solve_dro  # noqa: E402
from esassp.evaluate import (bc_cost, down_cost, evaluate, first_stage_cost, nominal_scenario, saa_objective)  # noqa: E402
from esassp.model import mean_allocation  # noqa: E402
from esassp.moment import wc_bc_mean, wc_bc_wass, wc_down_mean, wc_down_wass  # noqa: E402
from esassp.sp import solve_sp  # noqa: E402
from esassp.subproblems import _block_cost_rows, bc_sub, down_sub  # noqa: E402

P = Params(time_limit=60, master_time=30, gap=1e-6, threads=2, n_scenarios=4, eps_d=40.0, eps_l=3.0)


def tiny(seed=2, n=8, beds=((1, 1), (2, 1))):
    return generate_instance(n, "general", seed, n_ors=1, beds=beds)


@pytest.fixture(scope="module")
def inst():
    return tiny()


@pytest.fixture(scope="module")
def dro_m(inst):
    return solve_dro(inst, "M", P)


# ------------------------------------------------------------------ oracle vs. LP of (4b)-(4e)
def test_closed_form_bc_equals_lp_of_constraints_4b_4e(inst):
    rng = np.random.default_rng(0)
    c = inst.costs
    for trial in range(20):
        m = int(rng.integers(1, 5))
        L = 480.0
        d = rng.uniform(20, 200, m)
        s = np.sort(rng.uniform(0, L, m))
        # variables: w_2..w_m, o, g
        nv = (m - 1) + 2
        cost = np.r_[np.full(m - 1, c.w), c.o, c.g]
        A, b = [], []
        for k in range(1, m):                        # w_k >= w_{k-1} + s_{k-1} + d_{k-1} - s_k
            row = np.zeros(nv); row[k - 1] = -1
            if k - 1 >= 1: row[k - 2] = 1
            A.append(row); b.append(-(s[k - 1] + d[k - 1] - s[k]))
        for k in range(m):                           # o >= w_k + s_k + d_k - L
            row = np.zeros(nv); row[-2] = -1
            if k >= 1: row[k - 1] = 1
            A.append(row); b.append(L - s[k] - d[k])
        row = np.zeros(nv); row[-1] = -1; row[-2] = 1    # g >= L - sum d + o
        A.append(row); b.append(-(L - d.sum()))
        r = linprog(cost, A_ub=np.array(A), b_ub=b, bounds=(0, None), method="highs")
        sub = tiny(1)                                 # closed form through the block evaluator
        blk_inst = sub
        sol = Solution(np.full(blk_inst.I, -1), np.full(blk_inst.I, -1), {})
        order = list(range(m))
        for k in order:
            sol.block[k], sol.pos[k] = 0, k
            sol.start[(0, k)] = float(s[k])
        dm = d[None, :]
        val = _block_cost_rows(blk_inst, sol, 0, order, dm)[0]
        assert val == pytest.approx(r.fun, rel=1e-8, abs=1e-6)


def test_sp_milp_objective_equals_oracle(inst):
    for kw in (dict(), dict(downstream=False), dict(scheduling=False)):
        r = solve_sp(inst, Params(n_scenarios=10, time_limit=60, gap=1e-9, threads=2), **kw)
        assert validate_solution(inst, r.solution) == []
        sc = sample_scenarios(inst, 10, np.random.default_rng(1))
        sol = r.solution if kw.get("scheduling", True) else r.solution
        if kw.get("scheduling", True):
            expect = first_stage_cost(inst, sol) + bc_cost(inst, sol, sc.d)[0].mean() \
                + (down_cost(inst, sol, sc.lc, sc.lw)[0].mean() if kw.get("downstream", True) else 0.0)
            assert r.objective == pytest.approx(expect, rel=1e-7)


# ------------------------------------------------------------------ separation problems
def test_bc_sub_milp_equals_enumeration(dro_m, inst):
    sol = dro_m.solution
    rng = np.random.default_rng(3)
    rho = rng.uniform(-inst.costs.g, 2 * inst.costs.o, inst.I)
    emp = sample_scenarios(inst, 1, rng)
    for b in range(inst.B):
        if not sol.order(b):
            continue
        assert bc_sub(inst, sol, b, rho=rho, method="enum")[0] == pytest.approx(bc_sub(inst, sol, b, rho=rho, method="milp")[0], abs=1e-6)
        assert bc_sub(inst, sol, b, nu=3.0, dn=emp.d[0], method="enum")[0] == \
            pytest.approx(bc_sub(inst, sol, b, nu=3.0, dn=emp.d[0], method="milp")[0], abs=1e-6)


def _lattice_best(inst, sol, penalty):
    Pn = [int(i) for i in np.where(sol.scheduled)[0]]
    ranges = [[(c, w) for c in range(inst.surgeries[i].lo_c, inst.surgeries[i].hi_c + 1)
               for w in range(inst.surgeries[i].lo_w, inst.surgeries[i].hi_w + 1)] for i in Pn]
    nc, nw = np.rint(inst.mu_c).astype(int), np.rint(inst.mu_w).astype(int)
    best = -1e18
    for combo in itertools.product(*ranges):
        lc, lw = nc.copy(), nw.copy()
        for i, (c, w) in zip(Pn, combo):
            lc[i], lw[i] = c, w
        q, _ = down_cost(inst, sol, lc[None, :], lw[None, :])
        best = max(best, q[0] - penalty(Pn, lc, lw))
    return best


def test_down_sub_milp_equals_lattice_enumeration(dro_m, inst):
    sol = dro_m.solution
    if len(list(np.where(sol.scheduled)[0])) > 4:
        sol = sol.copy()
        for i in list(np.where(sol.scheduled)[0])[4:]:
            sol.block[i] = -1
    rng = np.random.default_rng(5)
    phi, zeta = rng.uniform(-300, 900, inst.I), rng.uniform(0, 900, inst.I)
    v, lc, lw = down_sub(inst, sol, phi=phi, zeta=zeta)
    assert v == pytest.approx(_lattice_best(inst, sol, lambda Pn, lc_, lw_: sum(phi[i] * lc_[i] + zeta[i] * lw_[i] for i in Pn)), abs=1e-6)
    emp = sample_scenarios(inst, 1, rng)
    v, _, _ = down_sub(inst, sol, phiW=40.0, lcn=emp.lc[0], lwn=emp.lw[0])
    assert v == pytest.approx(_lattice_best(inst, sol, lambda Pn, lc_, lw_: 40.0 * sum(abs(lc_[i] - emp.lc[0, i]) + abs(lw_[i] - emp.lw[0, i]) for i in Pn)), abs=1e-6)


# ------------------------------------------------------------------ strong duality (the paper's Propositions 1-2, 6)
def test_dro_m_dual_value_equals_primal_moment_lp(dro_m, inst):
    sol = dro_m.solution
    fixed = solve_dro(inst, "M", P, fixed=sol)
    primal = first_stage_cost(inst, sol) + wc_bc_mean(inst, sol) + wc_down_mean(inst, sol)
    assert fixed.objective == pytest.approx(primal, rel=1e-6)
    assert fixed.lower_bound == pytest.approx(primal, rel=1e-6)
    # the full C&CG optimum can only be better than (or equal to) the value of its own returned solution
    assert dro_m.objective <= primal + 1e-4 * abs(primal)
    assert dro_m.lower_bound <= dro_m.objective + 1e-6 * abs(dro_m.objective)


def test_dro_w_dual_value_equals_primal_transport_lp():
    inst = tiny(2, 8, beds=((1, 1), (1, 1)))         # tight beds so that the downstream part is active
    r = solve_dro(inst, "W", P)
    sol = r.solution
    emp = sample_scenarios(inst, P.n_scenarios, np.random.default_rng(P.seed))
    fixed = solve_dro(inst, "W", P, fixed=sol, empirical=emp)
    primal = first_stage_cost(inst, sol) + wc_bc_wass(inst, sol, emp, P.eps_d) + wc_down_wass(inst, sol, emp, P.eps_l)
    assert fixed.objective == pytest.approx(primal, rel=1e-6)
    assert r.objective <= primal + 1e-4 * abs(primal)


def test_bounds_valid_every_iteration(inst):
    r = solve_dro(inst, "M", Params(time_limit=60, gap=1e-6, threads=2))
    lbs = [h["lb"] for h in r.history]
    assert all(b2 >= b1 - 1e-6 for b1, b2 in zip(lbs, lbs[1:]))                 # LB monotone
    assert all(h["lb"] <= r.objective + 1e-6 for h in r.history)                # every LB below final UB


# ------------------------------------------------------------------ toggles must not change the optimum
@pytest.mark.parametrize("kw", [dict(symmetry=False), dict(multicut=False), dict(rho_bound=False)])
def test_toggles_do_not_change_dro_m_optimum(dro_m, inst, kw):
    r = solve_dro(inst, "M", Params(time_limit=60, gap=1e-6, threads=2, **kw))
    assert r.objective == pytest.approx(dro_m.objective, rel=1e-4)


def test_symmetry_breaking_keeps_optimum_when_scenarios_are_exchangeable():
    """SB (27)-(29) assumes identical surgeries are interchangeable.  In an SAA sample every surgery gets its own
    draw, so the *sampled* objective is not symmetric; with exchangeable scenarios (same draw for a whole class)
    the optimum must be identical.  (DRO-M has no sample and is exactly symmetric: see the toggle test above.)"""
    from esassp.evaluate import Scenarios
    inst = generate_instance(10, "cardiac", 4, n_ors=1, beds="(5,2)", type_weights=[3, 2, 1, 3, 1])
    sc = sample_scenarios(inst, 8, np.random.default_rng(2))
    first = {}
    for i, s_ in enumerate(inst.surgeries):
        j = first.setdefault(s_.klass, i)
        sc.d[:, i], sc.lc[:, i], sc.lw[:, i] = sc.d[:, j], sc.lc[:, j], sc.lw[:, j]
    a = solve_sp(inst, Params(n_scenarios=8, time_limit=60, gap=1e-9, threads=2, symmetry=True), scenarios=sc)
    b = solve_sp(inst, Params(n_scenarios=8, time_limit=60, gap=1e-9, threads=2, symmetry=False), scenarios=sc)
    assert a.objective == pytest.approx(b.objective, rel=1e-7)
    assert a.stats["n_sb"] > 0 and b.stats["n_sb"] == 0


# ------------------------------------------------------------------ misc
def test_integration_changes_decisions():
    inst = generate_instance(16, "cardiac", 1, n_ors=1, beds="(5,2)")
    with_down = solve_sp(inst, Params(n_scenarios=10, time_limit=60, threads=2), downstream=True)
    without = solve_sp(inst, Params(n_scenarios=10, time_limit=60, threads=2), downstream=False)
    assert without.solution.n_scheduled() >= with_down.solution.n_scheduled()
    sc = sample_scenarios(inst, 500, np.random.default_rng(9))
    e1, e0 = evaluate(inst, with_down.solution, sc), evaluate(inst, without.solution, sc)
    assert e1["mean_total"] <= e0["mean_total"] + 1e-6       # ignoring the downstream units never helps out of sample


def test_validator_and_dataio_roundtrip(inst):
    sol = solve_sp(inst, Params(n_scenarios=5, time_limit=30, threads=2)).solution
    assert validate_solution(inst, sol) == []
    bad = sol.copy()
    i = int(np.where(bad.scheduled)[0][0])
    bad.pos[i] = 99
    assert validate_solution(inst, bad)
    f = instance_to_frames(inst)
    j = instance_from_frames(f["surgeries"], f["blocks"], f["beds"], inst.costs)
    assert j.I == inst.I and j.B == inst.B and np.allclose(j.mu_d, inst.mu_d) and np.allclose(j.icu_beds, inst.icu_beds)


def test_sample_scenarios_respect_bounds_and_misspecification(inst):
    sc = sample_scenarios(inst, 2000, np.random.default_rng(0))
    assert (sc.d >= inst.lo_d - 1e-9).all() and (sc.d <= inst.hi_d + 1e-9).all()
    assert (sc.lc >= inst.lo_c).all() and (sc.lw <= inst.hi_w).all()
    assert abs(sc.d.mean(0) - inst.mu_d).max() < 0.15 * inst.hi_d.max()   # mu is the clipped-distribution mean
    sc2 = sample_scenarios(inst, 2000, np.random.default_rng(0), delta=0.5)
    assert sc2.lw.mean() >= sc.lw.mean()
