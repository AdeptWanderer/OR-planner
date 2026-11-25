import itertools
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orsched import BPCParams, Column, generate_instance, solve, validate_solution  # noqa: E402
from orsched.compact import solve_compact  # noqa: E402
from orsched.cuts import lifted_cover_row  # noqa: E402
from orsched.dataio import instance_from_frames, instance_to_frames  # noqa: E402
from orsched.heuristic import constructive_heuristic  # noqa: E402
from orsched.master import Master  # noqa: E402
from orsched.pricing import knapsack_bound, solve_pricing  # noqa: E402
from orsched.validate import validate_columns  # noqa: E402

FAST = BPCParams(cg_time_limit=60, branch_time_limit=60, final_ip_time=20, workers=2)


@pytest.mark.parametrize("args", [
    dict(n_surgeries=12, instance_set="B", seed=1, n_days=2, n_surgeons=4),
    dict(n_surgeries=14, instance_set="A", seed=2, rooms=2, n_days=2, n_surgeons=4),
    dict(n_surgeries=16, instance_set="C", seed=8, n_days=2, n_surgeons=4),
    dict(n_surgeries=15, instance_set="A", seed=6, rooms=2, n_days=3, n_surgeons=4, n_infection_types=2),
])
def test_bpc_matches_exact_monolithic_model(args):
    inst = generate_instance(**args)
    sol = solve(inst, FAST)
    assert validate_solution(inst, sol) == []
    ref = solve_compact(inst, 120, workers=2)
    assert ref["status"] == "OPTIMAL"
    assert sol.lower_bound == pytest.approx(ref["lb"])
    assert sol.upper_bound >= ref["lb"] - 1e-6          # bound is valid


def test_heuristic_is_feasible_and_generator_repairs_deadlines():
    for S in "ABC":
        inst = generate_instance(60, S, seed=3)
        cols, _ = constructive_heuristic(inst, np.random.default_rng(3 + 7919))
        assert validate_columns(inst, cols) == []


def test_reduced_costs_nonpositive_after_lp():
    inst = generate_instance(30, "B", seed=4)
    cols, _ = constructive_heuristic(inst)
    m = Master(inst)
    m.add_columns(cols)
    obj, x, art, y = m.solve_lp()
    for c in m.columns:
        assert m.column_value(c, y) <= 1e-6
    # day_duals reproduce column_value
    for c in m.columns:
        G, pi6, const = m.day_duals(y, c.day)
        v = const + sum(G[i] - pi6[inst.surgeries[i].surgeon, s:s + inst.surgeries[i].duration].sum()
                        for i, s in c.items)
        assert v == pytest.approx(m.column_value(c, y), abs=1e-6)


def test_pricing_columns_have_positive_value_and_respect_knapsack_bound():
    inst = generate_instance(40, "A", seed=5)
    cols, _ = constructive_heuristic(inst)
    m = Master(inst)
    m.add_columns(cols)
    obj, x, art, y = m.solve_lp()
    for d in range(inst.n_days):
        G, pi6, const = m.day_duals(y, d)
        new, st = solve_pricing(inst, d, G, pi6, const, time_limit=5, workers=2)
        for c in new:
            assert m.column_value(c, y) > 0
            assert validate_columns(inst, [c], require_mandatory=False) == []
            assert m.column_value(c, y) <= st.value_bound + 1e-3


def test_lifted_cover_inequality_is_valid_by_enumeration():
    rng = np.random.default_rng(0)
    for trial in range(30):
        inst = generate_instance(10, "C", seed=trial, n_surgeons=1, n_days=1, max_deadline=1)
        cap = int(inst.surgeon_max[0, 0])
        items = list(range(inst.n))
        dur = {i: inst.surgeries[i].duration for i in items}
        order = list(rng.permutation(items))
        C, w = [], 0
        for i in order:
            C.append(i); w += dur[i]
            if w > cap:
                break
        if w <= cap:
            continue
        for i in list(C):
            if w - dur[i] > cap:
                C.remove(i); w -= dur[i]
        row = lifted_cover_row(inst, 0, 0, C)
        for r in range(len(items) + 1):
            for S in itertools.combinations(items, r):
                if sum(dur[i] for i in S) <= cap:
                    assert sum(row.coef.get(i, 0) for i in S) <= row.hi + 1e-9


def test_validator_detects_violations():
    inst = generate_instance(20, "A", seed=1, cleaning_minutes=30, infectious_share=1.0, n_infection_types=2)
    a, b = [s for s in inst.surgeries if s.infection == 1][0], [s for s in inst.surgeries if s.infection == 2][0]
    col = Column(0, ((a.id, 0), (b.id, a.duration)))           # missing cleaning
    assert any("cleaning" in e for e in validate_columns(inst, [col], require_mandatory=False))
    col2 = Column(0, ((a.id, 0), (a.id, 50)))
    assert any("2 times" in e for e in validate_columns(inst, [col2], require_mandatory=False))


def test_csv_roundtrip():
    inst = generate_instance(25, "B", seed=9)
    s, g = instance_to_frames(inst)
    inst2 = instance_from_frames(s, g, inst.rooms, cleaning_minutes=30)
    assert [x.duration for x in inst2.surgeries] == [x.duration for x in inst.surgeries]
    assert (inst2.surgeon_max == inst.surgeon_max).all()
    assert inst2.mandatory == inst.mandatory
