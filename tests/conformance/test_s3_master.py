"""Section 3.1 master problem (1)-(7) and the reduced cost of Section 3.2."""
import numpy as np
import pytest

from conftest import real_duals
from orsched import generate_instance
from orsched.master import Master
from verification.paper_models import master_lp_from_formulas, reduced_cost_formula


@pytest.mark.parametrize("S,n,seed", [("A", 20, 1), ("B", 20, 2), ("C", 24, 3)])
def test_master_lp_equals_formulas(S, n, seed):
    """The production RMP (HiGHS, with artificials) and an LP written straight
    from equations (1)-(6) (scipy) have the same optimal value."""
    inst = generate_instance(n, S, seed=seed, n_days=3)
    m, y = real_duals(inst, iterations=3)[-1]
    obj, x, art, y = m.solve_lp()
    assert art.sum() < 1e-7, "artificial variables active - formulas LP would be infeasible"
    ref, _, _ = master_lp_from_formulas(inst, m.columns)
    assert obj == pytest.approx(ref, abs=1e-6)


def test_column_coefficients_match_definitions():
    """B_j, a'_ij, t_i a'_ij (5), a_ijt coverage (6) and b_jd (4) of a column."""
    inst = generate_instance(20, "B", seed=4, n_days=3)
    m, y = real_duals(inst, iterations=2)[-1]
    for c in m.columns[:40]:
        idx, val = m.column_entries(c)
        e = dict(zip(idx, val))
        assert e[m.r_rooms[c.day]] == 1                                   # b_jd
        for i, s in c.items:
            sg = inst.surgeries[i]
            assert e[m.r_assign[i]] == 1                                   # a'_ij
            assert e[m.r_surg[(sg.surgeon, c.day)]] == sum(
                inst.surgeries[k].duration for k, _ in c.items if inst.surgeries[k].surgeon == sg.surgeon)
            for t in range(s, s + sg.duration):                            # a_ijt over [t', t'+t_i)
                assert e[m.r_color[(c.day, sg.surgeon, t)]] == 1
        n_color = sum(1 for r in idx if r in set(m.r_color.values()))
        assert n_color == sum(inst.surgeries[i].duration for i, _ in c.items)


def test_reduced_cost_is_the_papers_sigma():
    """value(col) used by pricing == sigma_j of Section 3.2 under pi = -y."""
    inst = generate_instance(24, "C", seed=5, n_days=3)
    hist = real_duals(inst, iterations=3)
    for m, y in hist:
        duals = {}
        for i, r in m.r_assign.items():
            duals[("2", i) if inst.is_mandatory(i) else ("3", i)] = -y[r]
        for d, r in m.r_rooms.items():
            duals[("4", d)] = -y[r]
        for (l, d), r in m.r_surg.items():
            duals[("5", l, d)] = -y[r]
        for (d, l, t), r in m.r_color.items():
            duals[("6", d, l, t)] = -y[r]
        for c in m.columns:
            assert m.column_value(c, y) == pytest.approx(reduced_cost_formula(inst, c, duals), abs=1e-6)


def test_lp_optimality_no_column_in_rmp_has_positive_sigma():
    """At an RMP optimum every column in it has sigma_j <= 0 (LP optimality)."""
    inst = generate_instance(20, "A", seed=6, n_days=3)
    m, y = real_duals(inst, iterations=3)[-1]
    m.solve_lp()
    _, _, _, y = m.solve_lp()
    assert max(m.column_value(c, y) for c in m.columns) <= 1e-6
