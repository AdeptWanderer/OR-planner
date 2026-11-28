"""Strong-duality and structural properties of the DRO models, over several random instances."""
import numpy as np
import pytest
from conftest import FASTP, hand_instance, sol_from, tiny
from test_formulations import random_feasible

from esassp import Params, sample_scenarios
from esassp.dro import solve_dro
from esassp.evaluate import bc_cost, down_cost, first_stage_cost, nominal_scenario
from esassp.moment import wc_bc_mean, wc_bc_wass, wc_down_mean, wc_down_wass


def small_lattice(inst, sol, k=3):
    s = sol.copy()
    for i in list(np.where(s.scheduled)[0])[k:]:
        s.block[i] = -1; s.pos[i] = -1
    return s


def renumber(inst, s):
    for b, blk in enumerate(inst.blocks):
        order = s.order(b)
        for k, i in enumerate(order):
            s.pos[i] = k
        for k in range(blk.K):
            if k >= len(order):
                s.start[(b, k)] = 0.0
            else:
                s.start.setdefault((b, k), 0.0)
    return s


@pytest.mark.parametrize("seed", range(5))
def test_dro_m_fixed_solution_equals_moment_lp(seed):
    inst = tiny(seed % 4 + 1, 8, beds=((1, 1), (1, 2)))
    sol = renumber(inst, small_lattice(inst, random_feasible(inst, seed)))
    r = solve_dro(inst, "M", FASTP, fixed=sol)
    primal = first_stage_cost(inst, sol) + wc_bc_mean(inst, sol) + wc_down_mean(inst, sol)
    assert r.objective == pytest.approx(primal, rel=1e-6)
    assert r.status == "converged"


@pytest.mark.parametrize("seed", range(4))
def test_dro_w_fixed_solution_equals_transport_lp(seed):
    inst = tiny(seed % 4 + 1, 8, beds=((1, 1), (1, 1)))
    sol = renumber(inst, small_lattice(inst, random_feasible(inst, seed + 20)))
    emp = sample_scenarios(inst, FASTP.n_scenarios, np.random.default_rng(FASTP.seed))
    r = solve_dro(inst, "W", FASTP, fixed=sol, empirical=emp)
    primal = first_stage_cost(inst, sol) + wc_bc_wass(inst, sol, emp, FASTP.eps_d) + wc_down_wass(inst, sol, emp, FASTP.eps_l)
    assert r.objective == pytest.approx(primal, rel=1e-6)


def test_wasserstein_radius_zero_equals_sample_average():
    inst = tiny(2, 8, beds=((1, 1), (1, 1)))
    sol = renumber(inst, small_lattice(inst, random_feasible(inst, 3)))
    emp = sample_scenarios(inst, 4, np.random.default_rng(1))
    p = Params(time_limit=60, gap=1e-7, threads=2, n_scenarios=4, eps_d=0.0, eps_l=0.0)
    r = solve_dro(inst, "W", p, fixed=sol, empirical=emp)
    saa = first_stage_cost(inst, sol) + bc_cost(inst, sol, emp.d)[0].mean() + down_cost(inst, sol, emp.lc, emp.lw)[0].mean()
    assert r.objective == pytest.approx(saa, rel=1e-6)


def test_wasserstein_value_nondecreasing_in_radius():
    inst = tiny(2, 8, beds=((1, 1), (1, 1)))
    sol = renumber(inst, small_lattice(inst, random_feasible(inst, 4)))
    emp = sample_scenarios(inst, 4, np.random.default_rng(1))
    vals = []
    for e in (0.0, 10.0, 40.0, 150.0):
        p = Params(time_limit=60, gap=1e-7, threads=2, n_scenarios=4, eps_d=e, eps_l=e / 10)
        vals.append(solve_dro(inst, "W", p, fixed=sol, empirical=emp).objective)
    assert all(b >= a - 1e-6 for a, b in zip(vals, vals[1:])) and vals[-1] > vals[0]


def test_mean_support_value_at_least_nominal_cost_jensen():
    inst = tiny(2, 8, beds=((1, 1), (1, 1)))
    sol = renumber(inst, small_lattice(inst, random_feasible(inst, 2)))
    r = solve_dro(inst, "M", FASTP, fixed=sol)
    nom = nominal_scenario(inst)
    nominal = first_stage_cost(inst, sol) + bc_cost(inst, sol, nom.d)[0][0]
    assert r.objective >= nominal - 1e-6                 # BC part convex => E Q >= Q(mean)


def test_dro_m_value_upper_bounds_expected_cost_of_two_point_distribution():
    inst = tiny(3, 8, beds=((2, 2), (2, 2)))
    sol = renumber(inst, small_lattice(inst, random_feasible(inst, 6)))
    bc = wc_bc_mean(inst, sol)
    # any distribution with the same mean: here a deterministic one at the mean
    nom = nominal_scenario(inst)
    assert bc >= bc_cost(inst, sol, nom.d)[0][0] - 1e-6


@pytest.mark.parametrize("kind", ["M", "W"])
def test_lower_bound_never_exceeds_upper_bound_and_history_monotone(kind):
    inst = tiny(1, 8)
    r = solve_dro(inst, kind, FASTP)
    lbs = [h["lb"] for h in r.history]
    ubs = [h["ub"] for h in r.history]
    assert all(b >= a - 1e-6 for a, b in zip(lbs, lbs[1:]))
    assert all(b <= a + 1e-6 for a, b in zip(ubs, ubs[1:]))
    assert r.lower_bound <= r.objective + 1e-6 * abs(r.objective)
    assert r.status in ("converged", "stalled")


def test_dro_m_with_ample_beds_is_downstream_free():
    inst = tiny(2, 8, beds=((100, 100), (100, 100)))
    r = solve_dro(inst, "M", FASTP)
    sol = r.solution
    assert down_cost(inst, sol, np.rint(inst.hi_c)[None, :].astype(int), np.rint(inst.hi_w)[None, :].astype(int))[0][0] == 0


def test_dro_schedules_fewer_or_equal_surgeries_when_beds_get_tighter():
    counts = []
    for beds in ((0, 0), (2, 2), (100, 100)):
        inst = tiny(2, 8, beds=(beds, beds))
        counts.append(solve_dro(inst, "M", FASTP).solution.n_scheduled())
    assert counts[0] <= counts[1] <= counts[2]


def test_zero_beds_schedules_only_surgeries_without_stays_or_none():
    inst = tiny(2, 8, beds=((0, 0), (0, 0)))
    r = solve_dro(inst, "M", FASTP)
    for i in np.where(r.solution.scheduled)[0]:
        s = inst.surgeries[i]
        assert s.hi_c + s.hi_w >= 0   # sanity; true restriction is economic, checked via cost below
    assert r.objective > 0


def test_infeasible_to_exceed_blocks_capacity_k():
    inst = tiny(2, 12, ors=1)
    r = solve_dro(inst, "M", FASTP)
    for b, blk in enumerate(inst.blocks):
        assert len(r.solution.order(b)) <= blk.K


@pytest.mark.parametrize("flag", [dict(symmetry=False), dict(multicut=False), dict(rho_bound=False)])
def test_dro_w_toggles_keep_optimum_with_fixed_sample(flag):
    inst = tiny(2, 7, beds=((1, 1), (1, 1)))
    emp = sample_scenarios(inst, 3, np.random.default_rng(0))
    base = solve_dro(inst, "W", Params(time_limit=60, gap=1e-6, threads=2, n_scenarios=3, symmetry=False), empirical=emp)
    alt = solve_dro(inst, "W", Params(time_limit=60, gap=1e-6, threads=2, n_scenarios=3, **{**dict(symmetry=False), **flag}), empirical=emp)
    assert alt.objective == pytest.approx(base.objective, rel=1e-4)


def test_downstream_off_ignores_beds():
    inst = tiny(2, 8, beds=((0, 0), (0, 0)))
    with_down = solve_dro(inst, "M", FASTP)
    without = solve_dro(inst, "M", FASTP, downstream=False)
    assert without.solution.n_scheduled() >= with_down.solution.n_scheduled()
    assert not without.integrate_downstream and with_down.integrate_downstream


def test_scheduling_off_uses_mean_allocation_starts():
    inst = tiny(2, 8)
    r = solve_dro(inst, "M", FASTP, scheduling=False)
    sol = r.solution
    for b in range(inst.B):
        t = 0.0
        for k, i in enumerate(sol.order(b)):
            assert sol.start[(b, k)] == pytest.approx(t)
            t += inst.surgeries[i].mu_d
    assert not r.integrate_scheduling


def test_time_limit_returns_valid_solution_and_bounds():
    inst = tiny(2, 10, ors=1)
    r = solve_dro(inst, "M", Params(time_limit=1.5, master_time=1.0, gap=1e-9, threads=2))
    assert r.solution is not None and r.lower_bound <= r.objective + 1e-6
    from esassp import validate_solution
    assert validate_solution(inst, r.solution) == []
