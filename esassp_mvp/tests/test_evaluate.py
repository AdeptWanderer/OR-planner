import numpy as np
import pytest
from conftest import hand_instance, sol_from, tiny

from esassp import sample_scenarios
from esassp.evaluate import (Scenarios, bc_cost, block_costs, down_cost, evaluate, first_stage_cost, nominal_scenario,
                             occupancy, rel_tc, saa_objective)


# ------------------------------------------------------------------ hand-computed Q^BC
def test_bc_no_waiting_no_overtime_idle_only():
    inst = hand_instance()
    sol = sol_from(inst, {0: (0, 0), 1: (0, 1)}, {(0, 0): 0.0, (0, 1): 150.0})
    d = np.array([[100.0, 100.0, 100.0]])
    r = block_costs(inst, sol, d, 0)
    assert r["wait"].sum() == 0 and r["over"][0] == 0
    assert r["idle"][0] == pytest.approx(480 - 200)
    cost, _ = bc_cost(inst, sol, d)
    assert cost[0] == pytest.approx(20 * 280)


def test_bc_waiting_propagates():
    inst = hand_instance()
    sol = sol_from(inst, {0: (0, 0), 1: (0, 1), 2: (0, 2)}, {(0, 0): 0, (0, 1): 100, (0, 2): 200})
    d = np.array([[150.0, 150.0, 100.0]])
    r = block_costs(inst, sol, d, 0)
    # surgery2 starts at 150 (waits 50); ends 300 -> surgery3 scheduled 200, waits 100
    assert r["wait"][0].tolist() == [0.0, 50.0, 100.0]
    assert r["over"][0] == 0
    assert r["idle"][0] == pytest.approx(480 - 400)
    cost, parts = bc_cost(inst, sol, d)
    assert parts["wait"][0] == 150 and cost[0] == pytest.approx(10 * 150 + 20 * 80)


def test_bc_overtime_and_idle_identity():
    inst = hand_instance(L=300.0)
    sol = sol_from(inst, {0: (0, 0), 1: (0, 1), 2: (0, 2)}, {(0, 0): 0, (0, 1): 100, (0, 2): 200})
    d = np.array([[120.0, 120.0, 120.0]])
    r = block_costs(inst, sol, d, 0)
    assert r["over"][0] == pytest.approx(360 - 300)          # last ends at 360 (0 + waits 20,40)
    # idle = L - sum d + over = 300 - 360 + 60 = 0
    assert r["idle"][0] == pytest.approx(0.0)


def test_bc_empty_block_is_all_idle():
    inst = hand_instance()
    sol = sol_from(inst, {}, {})
    cost, parts = bc_cost(inst, sol, np.ones((2, 3)) * 100)
    assert (parts["idle"] == 480).all() and cost[0] == pytest.approx(20 * 480)


def test_bc_vectorised_equals_row_by_row():
    inst = tiny()
    from esassp.sp import solve_sp
    from esassp import Params
    sol = solve_sp(inst, Params(n_scenarios=5, time_limit=30, threads=2)).solution
    sc = sample_scenarios(inst, 30, np.random.default_rng(0))
    all_c, _ = bc_cost(inst, sol, sc.d)
    for r in (0, 7, 29):
        one, _ = bc_cost(inst, sol, sc.d[r:r + 1])
        assert one[0] == pytest.approx(all_c[r])


def test_bc_cost_monotone_in_durations_for_overtime_waiting():
    inst = hand_instance()
    sol = sol_from(inst, {0: (0, 0), 1: (0, 1), 2: (0, 2)}, {(0, 0): 0, (0, 1): 100, (0, 2): 200})
    base = np.array([[100.0, 100.0, 100.0]])
    big = base + 100
    rb, rg = block_costs(inst, sol, base, 0), block_costs(inst, sol, big, 0)
    assert rg["wait"].sum() >= rb["wait"].sum() and rg["over"][0] >= rb["over"][0]


# ------------------------------------------------------------------ hand-computed Q^down
def test_occupancy_hand_case():
    inst = hand_instance(T=7)
    sol = sol_from(inst, {0: (0, 0)}, {(0, 0): 0})
    lc, lw = np.array([[2, 0, 0]]), np.array([[3, 0, 0]])
    icu, ward = occupancy(inst, sol, lc, lw)
    assert icu[0].tolist() == [1, 1, 0, 0, 0, 0, 0]
    assert ward[0].tolist() == [0, 0, 1, 1, 1, 0, 0]


def test_occupancy_direct_to_ward_and_truncation():
    inst = hand_instance(T=4)
    sol = sol_from(inst, {0: (0, 0)}, {(0, 0): 0})
    icu, ward = occupancy(inst, sol, np.array([[0, 0, 0]]), np.array([[10, 0, 0]]))
    assert icu.sum() == 0 and ward[0].tolist() == [1, 1, 1, 1]          # horizon truncation: days >= T ignored


def test_down_cost_excess_beds():
    inst = hand_instance(icu=1, ward=1)
    sol = sol_from(inst, {0: (0, 0), 1: (0, 1), 2: (0, 2)}, {(0, 0): 0, (0, 1): 100, (0, 2): 200})
    lc = np.array([[1, 1, 1]]); lw = np.array([[0, 0, 0]])
    cost, det = down_cost(inst, sol, lc, lw)
    assert det["icu"][0, 0] == 3 and det["icu_excess"][0, 0] == 2
    assert cost[0] == pytest.approx(1000 * 2)
    lw2 = np.array([[1, 1, 1]])
    cost2, det2 = down_cost(inst, sol, lc, lw2)
    assert det2["ward_excess"][0, 1] == 2 and cost2[0] == pytest.approx(1000 * 2 + 500 * 2)


def test_down_cost_zero_when_capacity_ample():
    inst = hand_instance(icu=10, ward=10)
    sol = sol_from(inst, {0: (0, 0), 1: (0, 1), 2: (0, 2)}, {(0, 0): 0, (0, 1): 100, (0, 2): 200})
    cost, _ = down_cost(inst, sol, np.array([[2, 2, 2]]), np.array([[3, 3, 3]]))
    assert cost[0] == 0


def test_down_cost_monotone_in_beds():
    inst = tiny()
    from esassp.sp import solve_sp
    from esassp import Params
    sol = solve_sp(inst, Params(n_scenarios=5, time_limit=30, threads=2)).solution
    sc = sample_scenarios(inst, 100, np.random.default_rng(1))
    c0, _ = down_cost(inst, sol, sc.lc, sc.lw)
    inst.icu_beds += 2; inst.ward_beds += 2
    c1, _ = down_cost(inst, sol, sc.lc, sc.lw)
    assert (c1 <= c0 + 1e-9).all()


def test_unscheduled_patients_use_no_beds():
    inst = hand_instance()
    sol = sol_from(inst, {}, {})
    icu, ward = occupancy(inst, sol, np.full((2, 3), 2), np.full((2, 3), 3))
    assert icu.sum() == 0 and ward.sum() == 0


# ------------------------------------------------------------------ scenarios
def test_sample_shapes_dtypes_and_determinism():
    inst = tiny()
    a = sample_scenarios(inst, 50, np.random.default_rng(3))
    b = sample_scenarios(inst, 50, np.random.default_rng(3))
    assert a.d.shape == (50, inst.I) and a.lc.dtype.kind == "i" and a.n == 50
    assert np.array_equal(a.d, b.d) and np.array_equal(a.lw, b.lw)
    c = a.subset(slice(0, 10)); assert c.n == 10


@pytest.mark.parametrize("family", ["gamma", "lognormal", "uniform"])
def test_families_respect_support(family):
    inst = tiny()
    sc = sample_scenarios(inst, 500, np.random.default_rng(0), family=family)
    assert (sc.d >= inst.lo_d - 1e-9).all() and (sc.d <= inst.hi_d + 1e-9).all()
    assert (sc.lc >= inst.lo_c).all() and (sc.lc <= inst.hi_c).all()


def test_unknown_family_raises():
    with pytest.raises(ValueError):
        sample_scenarios(tiny(), 5, np.random.default_rng(0), family="weibull")


@pytest.mark.parametrize("delta", [0.2, 0.5])
def test_misspecification_widens_duration_support_and_lengthens_stays(delta):
    inst = tiny()
    base = sample_scenarios(inst, 3000, np.random.default_rng(0))
    mis = sample_scenarios(inst, 3000, np.random.default_rng(0), delta=delta)
    assert (mis.d >= (1 - delta) * inst.lo_d - 1e-9).all() and (mis.d <= (1 + delta) * inst.hi_d + 1e-9).all()
    assert mis.d.min() <= base.d.min() + 1e-9 and mis.d.max() >= base.d.max() - 1e-9
    assert mis.lw.mean() >= base.lw.mean() - 1e-9 and (mis.lw <= inst.hi_w).all()


def test_nominal_scenario_is_rounded_mean():
    inst = tiny()
    nom = nominal_scenario(inst)
    assert nom.n == 1 and np.allclose(nom.d[0], inst.mu_d) and np.array_equal(nom.lc[0], np.rint(inst.mu_c))


def test_sample_mean_close_to_mu():
    inst = tiny()
    sc = sample_scenarios(inst, 20000, np.random.default_rng(5))
    assert np.abs(sc.d.mean(0) - inst.mu_d).max() < 3.0
    assert np.abs(sc.lc.mean(0) - inst.mu_c).max() < 0.1


# ------------------------------------------------------------------ evaluate()
def test_evaluate_consistency():
    inst = tiny()
    from esassp.sp import solve_sp
    from esassp import Params
    sol = solve_sp(inst, Params(n_scenarios=8, time_limit=30, threads=2)).solution
    sc = sample_scenarios(inst, 400, np.random.default_rng(2))
    ev = evaluate(inst, sol, sc)
    assert np.allclose(ev["total"], ev["first_stage"] + ev["bc_cost"] + ev["down_cost"])
    assert ev["q75_total"] <= ev["q90_total"] + 1e-9
    for k in ("pe_icu", "pe_ward"):
        assert ((ev[k] >= 0) & (ev[k] <= 1)).all()
    for k in ("me_icu", "me_ward"):
        assert (ev[k] >= 0).all()
    assert ev["n_scheduled"] == sol.n_scheduled()
    assert saa_objective(inst, sol, sc) == pytest.approx(ev["first_stage"] + ev["mean_bc"] + ev["mean_down"])
    assert saa_objective(inst, sol, sc, downstream=False) == pytest.approx(ev["first_stage"] + ev["mean_bc"])


def test_pe_zero_when_beds_ample_and_one_when_zero_beds():
    inst = hand_instance(icu=100, ward=100)
    sol = sol_from(inst, {0: (0, 0)}, {(0, 0): 0})
    sc = sample_scenarios(inst, 50, np.random.default_rng(0))
    assert evaluate(inst, sol, sc)["pe_icu"].max() == 0
    inst.icu_beds[:] = 0
    ev = evaluate(inst, sol, sc)
    assert ev["pe_icu"][0] == 1.0 and ev["me_icu"][0] == 1.0


def test_first_stage_cost_counts_postponed():
    inst = hand_instance()
    sol = sol_from(inst, {0: (0, 0)}, {(0, 0): 0})
    assert first_stage_cost(inst, sol) == pytest.approx(inst.first_stage_cost(0, False) + 2 * inst.first_stage_cost(1, True))


def test_rel_tc_sign():
    assert rel_tc(np.array([2.0, 2.0]), np.array([1.0, 1.0])) == pytest.approx(1.0)
    assert rel_tc(np.array([1.0]), np.array([2.0])) == pytest.approx(-0.5)
