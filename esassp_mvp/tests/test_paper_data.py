"""The generator must reproduce the supplement's data tables exactly (Appendix G)."""
import numpy as np
import pytest

from esassp import paper_instance
from esassp.generator import PAPER_CASE, PAPER_M, PAPER_M_SHARE, PAPER_R, _G4

G1 = {"CABG+AVR": (286, 87, 4, 6.5, 7, 5.9), "IsoAVR": (207, 125, 3, 3.2, 6, 4.5), "IsoCABG": (247, 81, 3, 3.8, 5, 4.6),
      "IsoMVR": (250, 77, 3, 2.9, 6, 5.8), "TAVR": (83, 36, 2, 2, 4, 3.8)}
G7 = {"CARD": (99, 53, 2, 2, 2, 2), "GASTRO": (132, 76, 1, 1, 1, 1), "GYN": (78, 52, 2, 2, 2, 2), "MED": (75, 72, .05, .05, 1, 1),
      "ORTHO": (142, 58, 1.5, 1.5, 1.5, 1.5), "URO": (72, 38, .8, .8, 1, 1), "ENT": (74, 37, .1, .1, .1, .1),
      "NEURO": (160, 77, 2, 2, 2, 2), "VASCULAR": (120, 61, 3.5, 3.5, 3.5, 3.5), "OPHTH": (38, 19, .05, .05, .05, .05)}
G5 = {"ENT": (74, 37, .1, .1, .1, .1), "OBGYN": (86, 40, 2, 2, 2, 2), "ORTHO": (107, 44, 1.5, 1.5, 1.5, 1.5), "NEURO": (160, 77, 2, 2, 2, 2),
      "GEN": (93, 49, .05, .05, .05, .05), "OPHTH": (38, 19, .05, .05, .05, .05), "VASCULAR": (120, 61, 3.5, 3.5, 3.5, 3.5),
      "CARDIAC": (240, 103, 2, 2, 3.5, 3.5), "UROLOGY": (64, 52, .8, .8, .8, .8)}


def row(t):
    return (t.d_mean, t.d_sd, t.c_mean, t.c_sd, t.w_mean, t.w_sd)


@pytest.mark.parametrize("lib,table", [(PAPER_CASE, G1), (PAPER_R, G7), (PAPER_M, G5)])
def test_type_statistics_match_tables(lib, table):
    assert {t.name for t in lib} == set(table)
    for t in lib:
        assert row(t) == pytest.approx(table[t.name]), t.name
        assert t.source.startswith("paper")


def test_m_shares_sum_to_100():
    assert sum(PAPER_M_SHARE) == pytest.approx(100.0, abs=0.05) and len(PAPER_M_SHARE) == len(PAPER_M)


@pytest.mark.parametrize("n", [10, 20, 25, 40])
def test_case_instance_structure(n):
    inst = paper_instance("Case", n)
    assert inst.B == 25 and inst.I == n                              # 5 ORs x 5 days (Table G.2)
    counts = np.bincount([s.type for s in inst.surgeries], minlength=5)
    assert counts.max() - counts.min() <= 1 and counts.sum() == n    # I/5 each
    for b in inst.blocks:
        assert b.types == (b.room,)                                  # OR j is dedicated to type j every day
    assert sorted({b.day for b in inst.blocks}) == [0, 1, 2, 3, 4]


@pytest.mark.parametrize("n", [10, 20, 30])
def test_r_instance_structure(n):
    inst = paper_instance("R", n)
    assert inst.B == 50 and len({b.room for b in inst.blocks}) == 10            # Table G.6
    counts = np.bincount([s.type for s in inst.surgeries], minlength=10)
    assert counts.max() - counts.min() <= 1


def test_m_instance_follows_block_schedule_and_shares():
    inst = paper_instance("M", 60)
    assert inst.B == sum(1 for row in _G4.values() for c in row if c)             # Table G.4 cells
    names = [t.name for t in inst.types]
    cells = {(b.room + 1, b.day): names[b.types[0]] for b in inst.blocks}
    for room, rowv in _G4.items():
        for day, nm in enumerate(rowv):
            assert cells.get((room, day)) == nm
    counts = np.bincount([s.type for s in inst.surgeries], minlength=9) / 60 * 100
    assert abs(counts - np.array(PAPER_M_SHARE)).max() < 3.0                        # rounding at I = 60


@pytest.mark.parametrize("kind", ["Case", "R", "M"])
def test_paper_instances_all_surgeries_schedulable_and_bounded(kind):
    inst = paper_instance(kind, 30)
    for i in range(inst.I):
        assert inst.eligible_blocks(i)
    for s in inst.surgeries:
        assert s.lo_d <= s.mu_d <= s.hi_d and s.lo_c <= s.hi_c and s.lo_w <= s.hi_w


def test_cost_and_bed_presets():
    inst = paper_instance("Case", 10, beds="(5,2)", cost=2)
    assert inst.costs.o == 100.0 and inst.costs.g == pytest.approx(100 / 1.5)
    assert list(inst.icu_beds) == [5] * 5 + [2] * 2 and list(inst.ward_beds) == [5] * 5 + [2] * 2


def test_paper_instance_solves_and_validates():
    from esassp import Params, solve, validate_solution
    inst = paper_instance("Case", 10, beds="(5,2)")
    r = solve(inst, "SP", Params(n_scenarios=5, time_limit=60, threads=2))
    assert validate_solution(inst, r.solution) == []
