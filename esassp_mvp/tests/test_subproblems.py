import itertools

import numpy as np
import pytest
from conftest import hand_instance, sol_from, tiny
from test_formulations import random_feasible

from esassp import sample_scenarios
from esassp.evaluate import bc_cost, down_cost
from esassp.subproblems import _block_cost_rows, bc_sub, down_sub


def blocks_with_surgeries(inst, sol):
    return [b for b in range(inst.B) if sol.order(b)]


@pytest.mark.parametrize("seed", range(8))
def test_bc_enum_equals_milp_mean_support(seed):
    inst = tiny(seed % 4 + 1, 9, ors=1)
    sol = random_feasible(inst, seed)
    rng = np.random.default_rng(seed)
    rho = rng.uniform(-inst.costs.g, 2 * inst.costs.o + inst.costs.g, inst.I)
    for b in blocks_with_surgeries(inst, sol):
        v1, d1 = bc_sub(inst, sol, b, rho=rho, method="enum")
        v2, d2 = bc_sub(inst, sol, b, rho=rho, method="milp")
        assert v1 == pytest.approx(v2, abs=1e-3)


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("nu", [0.0, 1.0, 8.0, 80.0])
def test_bc_enum_equals_milp_wasserstein(seed, nu):
    inst = tiny(seed % 4 + 1, 9)
    sol = random_feasible(inst, seed + 3)
    emp = sample_scenarios(inst, 1, np.random.default_rng(seed))
    for b in blocks_with_surgeries(inst, sol):
        v1, _ = bc_sub(inst, sol, b, nu=nu, dn=emp.d[0], method="enum")
        v2, _ = bc_sub(inst, sol, b, nu=nu, dn=emp.d[0], method="milp")
        assert v1 == pytest.approx(v2, abs=1e-3)


def test_returned_worst_case_attains_value_and_lies_in_box():
    inst = tiny(2, 9)
    sol = random_feasible(inst, 4)
    rho = np.random.default_rng(1).uniform(-10, 50, inst.I)
    for b in blocks_with_surgeries(inst, sol):
        v, d = bc_sub(inst, sol, b, rho=rho, method="enum")
        order = sol.order(b)
        assert (d[order] >= inst.lo_d[order] - 1e-9).all() and (d[order] <= inst.hi_d[order] + 1e-9).all()
        q = _block_cost_rows(inst, sol, b, order, d[order][None, :])[0]
        assert q - float(rho[order] @ d[order]) == pytest.approx(v)


def test_bc_sub_dominates_random_points():
    inst = tiny(3, 9)
    sol = random_feasible(inst, 2)
    rng = np.random.default_rng(0)
    rho = rng.uniform(-10, 50, inst.I)
    for b in blocks_with_surgeries(inst, sol):
        order = sol.order(b)
        v, _ = bc_sub(inst, sol, b, rho=rho)
        pts = rng.uniform(inst.lo_d[order], inst.hi_d[order], size=(200, len(order)))
        vals = _block_cost_rows(inst, sol, b, order, pts) - pts @ rho[order]
        assert vals.max() <= v + 1e-7


def test_wasserstein_zero_radius_multiplier_large_returns_empirical_cost():
    inst = tiny(2, 8)
    sol = random_feasible(inst, 5)
    emp = sample_scenarios(inst, 1, np.random.default_rng(1))
    big = 1e4                                   # moving is never worthwhile
    tot = sum(bc_sub(inst, sol, b, nu=big, dn=emp.d[0])[0] for b in range(inst.B))
    assert tot == pytest.approx(bc_cost(inst, sol, emp.d)[0][0])


def test_empty_block_sub_is_idle_cost():
    inst = hand_instance()
    sol = sol_from(inst, {}, {})
    v, d = bc_sub(inst, sol, 0, rho=np.zeros(3))
    assert v == pytest.approx(inst.costs.g * 480) and np.isnan(d).all()


def test_auto_method_switches_to_milp_for_big_blocks():
    inst = hand_instance(n=3)
    sol = sol_from(inst, {0: (0, 0), 1: (0, 1), 2: (0, 2)}, {(0, 0): 0, (0, 1): 100, (0, 2): 200})
    rho = np.array([5.0, -3.0, 20.0])
    a = bc_sub(inst, sol, 0, rho=rho, method="auto", enum_limit=1)[0]
    b = bc_sub(inst, sol, 0, rho=rho, method="enum")[0]
    assert a == pytest.approx(b, abs=1e-3)


# ------------------------------------------------------------------ downstream
def _lattice(inst, sol, penalty):
    P = [int(i) for i in np.where(sol.scheduled)[0]]
    rng = [[(c, w) for c in range(inst.surgeries[i].lo_c, inst.surgeries[i].hi_c + 1)
            for w in range(inst.surgeries[i].lo_w, inst.surgeries[i].hi_w + 1)] for i in P]
    nc, nw = np.rint(inst.mu_c).astype(int), np.rint(inst.mu_w).astype(int)
    best = -1e18
    for combo in itertools.product(*rng):
        lc, lw = nc.copy(), nw.copy()
        for i, (c, w) in zip(P, combo):
            lc[i], lw[i] = c, w
        best = max(best, down_cost(inst, sol, lc[None, :], lw[None, :])[0][0] - penalty(P, lc, lw))
    return best


def _cap(sol, k=3):
    s = sol.copy()
    for i in list(np.where(s.scheduled)[0])[k:]:
        s.block[i] = -1; s.pos[i] = -1
    return s


@pytest.mark.parametrize("seed", range(6))
def test_down_sub_mean_support_equals_lattice(seed):
    inst = tiny(seed % 3 + 1, 8, beds=((1, 1), (1, 1)))
    sol = _cap(random_feasible(inst, seed))
    rng = np.random.default_rng(seed)
    phi, zeta = rng.uniform(-300, 900, inst.I), rng.uniform(0, 900, inst.I)
    v, lc, lw = down_sub(inst, sol, phi=phi, zeta=zeta)
    assert v == pytest.approx(_lattice(inst, sol, lambda P, a, b: sum(phi[i] * a[i] + zeta[i] * b[i] for i in P)), abs=1e-3)
    P = np.where(sol.scheduled)[0]
    assert ((lc[P] >= inst.lo_c[P]) & (lc[P] <= inst.hi_c[P]) & (lw[P] >= inst.lo_w[P]) & (lw[P] <= inst.hi_w[P])).all()


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("phiw", [0.0, 25.0, 400.0])
def test_down_sub_wasserstein_equals_lattice(seed, phiw):
    inst = tiny(seed % 3 + 1, 8, beds=((1, 1), (1, 1)))
    sol = _cap(random_feasible(inst, seed + 7))
    emp = sample_scenarios(inst, 1, np.random.default_rng(seed))
    v, _, _ = down_sub(inst, sol, phiW=phiw, lcn=emp.lc[0], lwn=emp.lw[0])
    ref = _lattice(inst, sol, lambda P, a, b: phiw * sum(abs(a[i] - emp.lc[0, i]) + abs(b[i] - emp.lw[0, i]) for i in P))
    assert v == pytest.approx(ref, abs=1e-3)


def test_down_sub_returned_scenario_attains_value():
    inst = tiny(2, 8, beds=((1, 1), (1, 1)))
    sol = _cap(random_feasible(inst, 1))
    rng = np.random.default_rng(2)
    phi, zeta = rng.uniform(-100, 600, inst.I), rng.uniform(0, 600, inst.I)
    v, lc, lw = down_sub(inst, sol, phi=phi, zeta=zeta)
    P = np.where(sol.scheduled)[0]
    q = down_cost(inst, sol, lc[None, :], lw[None, :])[0][0]
    assert q - sum(phi[i] * lc[i] + zeta[i] * lw[i] for i in P) == pytest.approx(v, abs=1e-3)


def test_down_sub_no_patients():
    inst = hand_instance()
    sol = sol_from(inst, {}, {})
    v, lc, lw = down_sub(inst, sol, phi=np.zeros(3), zeta=np.zeros(3))
    assert v == pytest.approx(0.0, abs=1e-6)
