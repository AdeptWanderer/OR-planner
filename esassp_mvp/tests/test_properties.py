"""Randomised invariants of the oracle and the generator (fast, many draws)."""
import numpy as np
import pytest
from conftest import hand_instance, sol_from, tiny

from esassp import generate_instance, sample_scenarios
from esassp.evaluate import bc_cost, block_costs, down_cost, occupancy
from esassp.model import Costs

RNG = np.random.default_rng(2024)


def random_block(m, L=480.0):
    inst = hand_instance(L=L, n=m, K=m)
    starts = np.sort(RNG.uniform(0, L * 0.8, m))
    sol = sol_from(inst, {i: (0, i) for i in range(m)}, {(0, i): float(starts[i]) for i in range(m)})
    return inst, sol


@pytest.mark.parametrize("trial", range(30))
def test_bc_identities(trial):
    m = int(RNG.integers(1, 5))
    inst, sol = random_block(m)
    d = RNG.uniform(10, 220, (16, m))
    D = np.zeros((16, inst.I)); D[:, :m] = d
    r = block_costs(inst, sol, D, 0)
    assert (r["wait"] >= 0).all() and (r["over"] >= 0).all() and (r["idle"] >= 0).all()
    assert (r["wait"][:, 0] == 0).all()                                     # first surgery never waits
    assert np.allclose(r["idle"], np.maximum(0, 480 - d.sum(1) + r["over"]))   # identity (4e)
    # completion time of the last surgery
    end = r["wait"][:, -1] + sol.start[(0, m - 1)] + d[:, -1]
    assert np.allclose(r["over"], np.maximum(0, end - 480))


@pytest.mark.parametrize("trial", range(15))
def test_bc_cost_scales_linearly_with_prices(trial):
    inst, sol = random_block(3)
    D = RNG.uniform(10, 250, (8, inst.I))
    c1, _ = bc_cost(inst, sol, D)
    inst.costs = Costs(c_o=inst.costs.o * 2, c_g=inst.costs.g * 2, c_w=inst.costs.w * 2, c_icu=1, c_ward=1)
    c2, _ = bc_cost(inst, sol, D)
    assert np.allclose(c2, 2 * c1)


@pytest.mark.parametrize("trial", range(15))
def test_late_enough_starts_give_no_waiting(trial):
    m = 3
    inst, sol = random_block(m, L=2000.0)
    d = RNG.uniform(10, 100, (10, inst.I))
    for k in range(m):
        sol.start[(0, k)] = 250.0 * k                                       # more than max duration apart
    assert block_costs(inst, sol, d, 0)["wait"].sum() == 0


@pytest.mark.parametrize("trial", range(15))
def test_bc_cost_monotone_in_last_duration(trial):
    inst, sol = random_block(3)
    D = RNG.uniform(10, 150, (6, inst.I))
    D2 = D.copy(); D2[:, 2] += 40
    r1, r2 = block_costs(inst, sol, D, 0), block_costs(inst, sol, D2, 0)
    assert (r2["over"] >= r1["over"] - 1e-9).all()


@pytest.mark.parametrize("trial", range(20))
def test_occupancy_bed_days_conserved(trial):
    inst = hand_instance(T=int(RNG.integers(4, 10)), n=3, K=3)
    sol = sol_from(inst, {0: (0, 0), 1: (0, 1)}, {(0, 0): 0, (0, 1): 100})
    lc = RNG.integers(0, 4, (12, 3)); lw = RNG.integers(0, 5, (12, 3))
    icu, ward = occupancy(inst, sol, lc, lw)
    for r in range(12):
        exp_icu = sum(min(lc[r, i], inst.T) for i in (0, 1))
        exp_ward = sum(max(0, min(lc[r, i] + lw[r, i], inst.T) - lc[r, i]) for i in (0, 1))
        assert icu[r].sum() == exp_icu and ward[r].sum() == exp_ward
    assert (icu >= 0).all() and (ward >= 0).all()


@pytest.mark.parametrize("trial", range(15))
def test_down_cost_nonnegative_and_zero_without_stays(trial):
    inst = hand_instance(icu=0, ward=0)
    sol = sol_from(inst, {0: (0, 0), 1: (0, 1)}, {(0, 0): 0, (0, 1): 100})
    lc = RNG.integers(0, 3, (5, 3)); lw = RNG.integers(0, 3, (5, 3))
    c, _ = down_cost(inst, sol, lc, lw)
    assert (c >= 0).all()
    c0, _ = down_cost(inst, sol, np.zeros((5, 3), int), np.zeros((5, 3), int))
    assert (c0 == 0).all()


@pytest.mark.parametrize("seed", range(12))
@pytest.mark.parametrize("lib", ["cardiac", "general"])
def test_generator_blocks_hold_type_demand(seed, lib):
    inst = generate_instance(int(RNG.integers(8, 40)), lib, seed, n_ors=int(RNG.integers(1, 4)))
    for b in inst.blocks:
        assert 1 <= b.K <= 6
    assert all(inst.eligible_blocks(i) for i in range(inst.I))   # every surgery can be scheduled somewhere
    assert inst.mu_d.min() > 0


@pytest.mark.parametrize("seed", range(10))
def test_scenarios_are_independent_across_surgeries(seed):
    inst = tiny(seed + 1, 8)
    sc = sample_scenarios(inst, 4000, np.random.default_rng(seed))
    c = np.corrcoef(sc.d[:, 0], sc.d[:, 1])[0, 1]
    assert abs(c) < 0.1


@pytest.mark.parametrize("delta", [0.0, 0.1, 0.2, 0.3, 0.5])
def test_mean_ward_los_nondecreasing_in_delta(delta):
    inst = generate_instance(16, "cardiac", 1)
    a = sample_scenarios(inst, 4000, np.random.default_rng(1), delta=0.0).lw.mean()
    b = sample_scenarios(inst, 4000, np.random.default_rng(1), delta=delta).lw.mean()
    assert b >= a - 0.02


@pytest.mark.parametrize("n", [1, 2, 3, 5])
def test_first_stage_cost_uniform_for_identical_surgeries(n):
    inst = hand_instance(n=n, K=n)
    costs = {inst.first_stage_cost(i, False) for i in range(n)}
    assert len(costs) == 1
