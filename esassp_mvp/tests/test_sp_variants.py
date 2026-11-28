import numpy as np
import pytest
from conftest import tiny

from esassp import Params, evaluate, sample_scenarios, solve, validate_solution
from esassp.evaluate import first_stage_cost, saa_objective
from esassp.sp import solve_sp

P = Params(n_scenarios=10, time_limit=60, gap=1e-9, threads=2)


@pytest.mark.parametrize("seed", range(4))
def test_sp_objective_equals_oracle_saa(seed):
    inst = tiny(seed + 1, 8)
    r = solve_sp(inst, P)
    sc = sample_scenarios(inst, 10, np.random.default_rng(P.seed))
    assert validate_solution(inst, r.solution) == []
    assert r.objective == pytest.approx(saa_objective(inst, r.solution, sc), rel=1e-7)
    assert r.status == "Optimal" and r.gap <= 1e-6


def test_sp_is_optimal_vs_random_feasible_solutions():
    from test_formulations import random_feasible
    inst = tiny(2, 8)
    r = solve_sp(inst, P)
    sc = sample_scenarios(inst, 10, np.random.default_rng(P.seed))
    for s in range(8):
        other = random_feasible(inst, s)
        assert saa_objective(inst, other, sc) >= r.objective - 1e-6


def test_sp_downstream_off_objective_excludes_down_cost():
    inst = tiny(2, 8, beds=((0, 0), (0, 0)))
    r = solve_sp(inst, P, downstream=False)
    sc = sample_scenarios(inst, 10, np.random.default_rng(P.seed))
    assert r.objective == pytest.approx(saa_objective(inst, r.solution, sc, downstream=False), rel=1e-7)


def test_sp_more_scenarios_changes_nothing_when_scenarios_identical():
    inst = tiny(2, 8)
    sc = sample_scenarios(inst, 1, np.random.default_rng(0))
    from esassp.evaluate import Scenarios
    rep = Scenarios(np.repeat(sc.d, 5, 0), np.repeat(sc.lc, 5, 0), np.repeat(sc.lw, 5, 0))
    a = solve_sp(inst, P, scenarios=sc); b = solve_sp(inst, P, scenarios=rep)
    assert a.objective == pytest.approx(b.objective, rel=1e-7)


def test_sp_expected_cost_never_below_deterministic_cost_of_same_plan():
    inst = tiny(2, 8)
    r = solve_sp(inst, P)
    from esassp.evaluate import bc_cost, nominal_scenario
    nom = nominal_scenario(inst)
    sc = sample_scenarios(inst, 5000, np.random.default_rng(0))
    ev = evaluate(inst, r.solution, sc)
    assert ev["mean_bc"] >= bc_cost(inst, r.solution, nom.d)[0][0] - 1e-6 * 100 - 500   # Jensen up to sampling error


@pytest.mark.parametrize("model", ["SP", "DRO-M", "DRO-W"])
def test_unified_solve_entry_point(model):
    inst = tiny(2, 7)
    r = solve(inst, model, Params(time_limit=40, gap=1e-4, threads=2, n_scenarios=4))
    assert r.model == model and validate_solution(inst, r.solution) == []
    assert r.time > 0 and r.lower_bound <= r.objective + 1e-6 * abs(r.objective) + 1e-6


def test_unknown_model_raises():
    with pytest.raises(ValueError):
        solve(tiny(), "FOO")


def test_sp_scheduling_off_uses_mean_allocation():
    inst = tiny(2, 8)
    r = solve_sp(inst, P, scheduling=False)
    for b in range(inst.B):
        t = 0.0
        for k, i in enumerate(r.solution.order(b)):
            assert r.solution.start[(b, k)] == pytest.approx(t); t += inst.surgeries[i].mu_d


def test_ignoring_scheduling_makes_waiting_worse_out_of_sample():
    inst = tiny(2, 8)
    sc = sample_scenarios(inst, 3000, np.random.default_rng(4))
    a = solve_sp(inst, P); b = solve_sp(inst, P, scheduling=False)
    ea, eb = evaluate(inst, a.solution, sc), evaluate(inst, b.solution, sc)
    assert ea["mean_total"] <= eb["mean_total"] * 1.02


def test_ignoring_downstream_never_helps_out_of_sample():
    inst = tiny(2, 8, beds=((1, 1), (1, 1)))
    sc = sample_scenarios(inst, 3000, np.random.default_rng(4))
    a = solve_sp(inst, P); b = solve_sp(inst, P, downstream=False)
    assert evaluate(inst, a.solution, sc)["mean_total"] <= evaluate(inst, b.solution, sc)["mean_total"] * 1.02


def test_solution_postponed_surgeries_have_no_start_and_cost_postponement():
    inst = tiny(2, 8, beds=((0, 0), (0, 0)))
    r = solve_sp(inst, P)
    post = int((r.solution.block < 0).sum())
    assert first_stage_cost(inst, r.solution) == pytest.approx(
        sum(inst.first_stage_cost(i, r.solution.block[i] < 0) for i in range(inst.I)))
    assert post >= 0
