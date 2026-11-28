import numpy as np
import pytest
from conftest import hand_instance, sol_from, tiny

from esassp import generate_instance, paper_instance
from esassp.generator import BED_PRESETS, CARDIAC, GENERAL, LIBRARIES, type_summary, _gamma_q
from esassp.model import Block, Costs, Solution, mean_allocation


@pytest.mark.parametrize("lib", ["cardiac", "general"])
@pytest.mark.parametrize("n", [5, 12, 30, 61])
def test_generator_counts_and_ids(lib, n):
    inst = generate_instance(n, lib, 1, n_ors=2)
    assert inst.I == n
    assert [s.id for s in inst.surgeries] == list(range(n))
    assert all(0 <= s.type < len(inst.types) for s in inst.surgeries)


@pytest.mark.parametrize("seed", range(8))
def test_bounds_bracket_mean(seed):
    inst = generate_instance(20, "cardiac", seed)
    for s in inst.surgeries:
        assert s.lo_d <= s.mu_d <= s.hi_d
        assert s.lo_c <= s.mu_c <= s.hi_c
        assert s.lo_w <= s.mu_w <= s.hi_w
        assert s.lo_c >= 0 and s.lo_w >= 0 and s.lo_d > 0


def test_seed_changes_instance_and_same_seed_reproduces():
    a, b, c = generate_instance(20, "cardiac", 1), generate_instance(20, "cardiac", 2), generate_instance(20, "cardiac", 1)
    assert [s.type for s in a.surgeries] == [s.type for s in c.surgeries]
    assert [s.type for s in a.surgeries] != [s.type for s in b.surgeries]


def test_type_weights_fix_the_mix():
    inst = generate_instance(20, "cardiac", 1, type_weights=[1, 0, 0, 0, 0])
    assert {s.type for s in inst.surgeries} == {0}


@pytest.mark.parametrize("ors", [1, 2, 3, 4])
def test_blocks_closed_scheduling_and_layout(ors):
    inst = generate_instance(24, "cardiac", 1, n_ors=ors)
    assert inst.B == 5 * ors
    assert all(len(b.types) == 1 for b in inst.blocks)
    assert all(b.K >= 1 and b.length == 480.0 for b in inst.blocks)
    assert {b.day for b in inst.blocks} <= set(range(5))
    assert len({(b.day, b.room) for b in inst.blocks}) == inst.B
    assert inst.eligible_blocks(0)


def test_every_surgery_has_an_eligible_block():
    for seed in range(6):
        inst = generate_instance(18, "general", seed, n_ors=2)
        for i in range(inst.I):
            assert inst.eligible_blocks(i), (seed, i)


@pytest.mark.parametrize("preset", list(BED_PRESETS))
def test_bed_presets_weekday_weekend(preset):
    inst = generate_instance(10, "cardiac", 1, beds=preset)
    (iw, ie), (ww, we) = BED_PRESETS[preset]
    assert list(inst.icu_beds[:5]) == [iw] * 5 and list(inst.icu_beds[5:]) == [ie] * 2
    assert list(inst.ward_beds[:5]) == [ww] * 5 and list(inst.ward_beds[5:]) == [we] * 2


def test_custom_beds_and_horizon():
    inst = generate_instance(10, "cardiac", 1, beds=((3, 1), (4, 2)), T=14)
    assert inst.T == 14 and len(inst.icu_beds) == 14 and inst.icu_beds[7] == 3 and inst.ward_beds[12] == 2


def test_cost_defaults_follow_paper_ratios():
    c = Costs(c_o=26.0)
    assert c.g == pytest.approx(26 / 1.5) and c.w == pytest.approx(26 / 1.5) and c.o == 26
    c2 = Costs(c_o=100.0, c_g=7.0, c_w=3.0)
    assert (c2.g, c2.w) == (7.0, 3.0)


def test_first_stage_costs_eq30():
    inst = generate_instance(10, "cardiac", 1, costs=Costs(c_o=26.0))
    assert inst.first_stage_cost(0, False) == pytest.approx(0.5 * 26)
    assert inst.first_stage_cost(0, True) == pytest.approx(1.5 * 0.5 * 26)
    assert inst.first_stage_cost(0, True) > inst.first_stage_cost(0, False)


def test_paper_instances_shapes():
    r = paper_instance("R", 28)
    c = paper_instance("Case", 24, beds="(5,2)", cost=2)
    assert r.name == "R-28" and r.I == 28 and c.costs.c_o == 100.0
    assert {t.name for t in r.types} == {t.name for t in GENERAL}


def test_gamma_quantile_helpers():
    assert _gamma_q(0, 1, 0.5) == 0.0
    assert _gamma_q(10, 3, 0.2) < _gamma_q(10, 3, 0.8)
    rng = np.random.default_rng(0)
    s = type_summary(CARDIAC[0], rng)
    assert s["lo_d"] < s["mu_d"] < s["hi_d"]


def test_type_library_provenance_labels():
    assert all(t.source for t in CARDIAC + GENERAL)
    ent = next(t for t in GENERAL if t.name == "ENT")
    assert (ent.d_mean, ent.d_sd) == (74, 37)
    oph = next(t for t in GENERAL if t.name == "OPHTH")
    assert (oph.d_mean, oph.d_sd) == (38, 19)
    assert set(LIBRARIES) == {"cardiac", "general"}


def test_instance_helpers():
    inst = tiny()
    assert inst.T_ext > inst.T
    assert inst.mu_d.shape == (inst.I,) and (inst.lo_d <= inst.hi_d).all()
    assert "I" in inst.summary() and inst.summary()["B"] == inst.B
    assert inst.day_name(0) == "Mon" and inst.day_name(6) == "Sun"
    assert "OR1" in inst.block_label(0)


def test_klass_groups_identical_surgeries():
    inst = generate_instance(20, "cardiac", 1)
    groups = {}
    for s in inst.surgeries:
        groups.setdefault(s.klass, []).append(s.type)
    assert all(len(set(v)) == 1 for v in groups.values())
    assert len(groups) == len({s.type for s in inst.surgeries})


def test_solution_order_copy_and_counts():
    inst = hand_instance()
    sol = sol_from(inst, {0: (0, 1), 1: (0, 0)}, {(0, 0): 0.0, (0, 1): 100.0})
    assert sol.order(0) == [1, 0]
    assert sol.n_scheduled() == 2 and list(sol.scheduled) == [True, True, False]
    c = sol.copy(); c.block[0] = -1
    assert sol.block[0] == 0 and c.n_scheduled() == 1
    assert sol.start_of(0) == 100.0


def test_mean_allocation_cumulative_means():
    inst = hand_instance()
    sol = sol_from(inst, {0: (0, 0), 1: (0, 1), 2: (0, 2)}, {(0, 0): 5.0, (0, 1): 5.0, (0, 2): 5.0})
    ma = mean_allocation(inst, sol)
    assert [ma.start[(0, k)] for k in range(3)] == [0.0, 100.0, 200.0]
    assert sol.start[(0, 0)] == 5.0                     # original untouched
