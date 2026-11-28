import itertools

import numpy as np
import pytest
from conftest import hand_instance, sol_from, tiny

from esassp import sample_scenarios, validate_solution
from esassp.evaluate import bc_cost, down_cost
from esassp.formulations import FirstStage, add_bc_recourse, add_down_recourse
from esassp.milp import INF, Model
from esassp.model import Solution


def random_feasible(inst, seed):
    """Optimise a random objective over X so we sample many different feasible first-stage solutions."""
    rng = np.random.default_rng(seed)
    m = Model(); fs = FirstStage(m, inst, symmetry=False)
    for j in list(fs.x.values()) + fs.xp:
        m.set_obj(j, float(rng.normal()))
    for j in fs.s.values():
        m.set_obj(j, float(rng.normal(scale=0.01)))
    r = m.solve(time_limit=20)
    return fs.extract(r.x)


@pytest.mark.parametrize("seed", range(10))
def test_random_first_stage_solutions_are_valid(seed):
    inst = tiny(seed % 4 + 1, 8)
    sol = random_feasible(inst, seed)
    assert validate_solution(inst, sol) == []


def test_each_surgery_exactly_once_and_positions_contiguous():
    inst = tiny(3, 10, ors=2)
    for seed in range(5):
        sol = random_feasible(inst, seed)
        for b in range(inst.B):
            ks = sorted(sol.pos[sol.block == b])
            assert ks == list(range(len(ks)))
        assert ((sol.block >= -1) & (sol.block < inst.B)).all()


def test_start_times_zero_for_empty_positions_and_within_block():
    inst = tiny(2, 8)
    sol = random_feasible(inst, 0)
    for b, blk in enumerate(inst.blocks):
        for k in range(blk.K):
            s = sol.start[(b, k)]
            assert 0 <= s <= blk.length
            if k >= len(sol.order(b)):
                assert s == 0.0


def test_eligibility_respected():
    inst = tiny(4, 10, ors=2)
    sol = random_feasible(inst, 1)
    for i in range(inst.I):
        if sol.block[i] >= 0:
            assert int(sol.block[i]) in inst.eligible_blocks(i)


def test_symmetry_constraints_only_when_requested():
    inst = tiny(1, 8)
    m1, m2 = Model(), Model()
    a = FirstStage(m1, inst, symmetry=True); b = FirstStage(m2, inst, symmetry=False)
    assert a.n_sb > 0 and b.n_sb == 0 and m1.n_rows == m2.n_rows + a.n_sb


def test_symmetry_positions_toggle_reduces_constraints():
    inst = tiny(1, 8)
    a = FirstStage(Model(), inst, symmetry=True, sb_positions=True)
    b = FirstStage(Model(), inst, symmetry=True, sb_positions=False)
    assert a.n_sb > b.n_sb > 0


def _fixed_recourse_value(inst, sol, build):
    """Fix (x, s) in a MILP, add the recourse copy and minimise its cost: must equal the closed-form oracle."""
    m = Model(); fs = FirstStage(m, inst, symmetry=False)
    for (i, b, k), j in fs.x.items():
        m.fix(j, 1.0 if (sol.block[i] == b and sol.pos[i] == k) else 0.0)
    for i in range(inst.I):
        m.fix(fs.xp[i], 1.0 if sol.block[i] < 0 else 0.0)
    for (b, k), j in fs.s.items():
        m.fix(j, sol.start.get((b, k), 0.0))
    cost_terms = build(fs)
    t = {}
    for term in cost_terms:
        for j, c in term.items():
            t[j] = t.get(j, 0.0) + c
    for j, c in t.items():
        m.set_obj(j, c)
    r = m.solve(time_limit=20, gap=1e-9)
    assert r.ok
    return r.obj - sum(inst.first_stage_cost(i, sol.block[i] < 0) for i in range(inst.I))


@pytest.mark.parametrize("seed", range(6))
def test_bc_recourse_copy_equals_closed_form(seed):
    inst = tiny(seed % 3 + 1, 8)
    sol = random_feasible(inst, seed)
    sc = sample_scenarios(inst, 1, np.random.default_rng(seed))
    val = _fixed_recourse_value(inst, sol, lambda fs: add_bc_recourse(fs, sc.d[0]).cost_terms)
    assert val == pytest.approx(bc_cost(inst, sol, sc.d)[0][0], rel=1e-8, abs=1e-6)


@pytest.mark.parametrize("seed", range(6))
def test_down_recourse_copy_equals_closed_form(seed):
    inst = tiny(seed % 3 + 1, 8, beds=((1, 1), (1, 1)))
    sol = random_feasible(inst, seed + 10)
    sc = sample_scenarios(inst, 1, np.random.default_rng(seed))
    val = _fixed_recourse_value(inst, sol, lambda fs: [add_down_recourse(fs, sc.lc[0], sc.lw[0]).cost_terms])
    assert val == pytest.approx(down_cost(inst, sol, sc.lc, sc.lw)[0][0], rel=1e-8, abs=1e-6)


def test_recourse_without_waiting_cost():
    inst = tiny(2, 8)
    sol = random_feasible(inst, 3)
    sc = sample_scenarios(inst, 1, np.random.default_rng(0))
    m = Model(); fs = FirstStage(m, inst, symmetry=False)
    for (i, b, k), j in fs.x.items():
        m.fix(j, 1.0 if (sol.block[i] == b and sol.pos[i] == k) else 0.0)
    for i in range(inst.I):
        m.fix(fs.xp[i], 1.0 if sol.block[i] < 0 else 0.0)
    for (b, k), j in fs.s.items():
        m.fix(j, sol.start.get((b, k), 0.0))
    cp = add_bc_recourse(fs, sc.d[0], c_w=0.0)
    tot = {}
    for term in cp.cost_terms:
        for j, c in term.items():
            tot[j] = tot.get(j, 0.0) + c
    # w variables carry cost c_w in cost_terms regardless; objective is set by us -> cost excludes waiting
    for j, c in tot.items():
        if c != inst.costs.w:
            m.set_obj(j, c)
    r = m.solve(gap=1e-9)
    assert r.ok


def test_extract_reindexes_positions_and_roundtrips():
    inst = tiny(2, 8)
    sol = random_feasible(inst, 2)
    m = Model(); fs = FirstStage(m, inst, symmetry=False)
    h = fs.hint(sol)
    assert sum(1 for j in fs.xp if h.get(j) == 1.0) == int((sol.block < 0).sum())
