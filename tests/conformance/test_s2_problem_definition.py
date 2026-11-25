"""Section 2 (problem statement, assumptions 1-7, objective) and Section 6.1 (instances)."""
import numpy as np
import pytest

from orsched import Column, generate_instance
from orsched.validate import validate_columns


def test_cleaning_rule_assumption3():
    """OCT is needed when switching to a different infection type or to a
    non-infectious case; never between same-type cases; never before the
    first / after the last case."""
    inst = generate_instance(30, "A", seed=1, infectious_share=0.6, n_infection_types=2)
    by_type = {}
    for s in inst.surgeries:
        by_type.setdefault(s.infection, []).append(s.id)
    a0, a1, a2 = by_type[0][0], by_type[1][0], by_type[2][0]
    b1 = by_type[1][1]
    assert inst.clean(a1, a2) == inst.cleaning          # infection type change
    assert inst.clean(a1, a0) == inst.cleaning          # to non-infectious
    assert inst.clean(a0, a1) == inst.cleaning          # non-infectious -> infectious (Table 1, case 2)
    assert inst.clean(a1, b1) == 0                      # same infection type
    assert inst.clean(a0, by_type[0][1]) == 0           # two non-infectious
    assert inst.clean(None, a1) == 0 and inst.clean(a1, None) == 0


def test_objective_is_total_scheduled_surgery_time():
    """Objective (1): B_j = total duration of the surgeries in schedule j."""
    inst = generate_instance(20, "A", seed=2)
    col = Column(0, ((0, 0), (1, inst.surgeries[0].duration + inst.cleaning)))
    assert col.value(inst) == inst.surgeries[0].duration + inst.surgeries[1].duration


@pytest.mark.parametrize("violation", ["deadline", "surgeon_max", "surgeon_overlap", "rooms", "mandatory",
                                       "room_overlap", "cleaning", "horizon", "twice"])
def test_validator_enforces_every_constraint(violation):
    """The independent checker rejects each violation of constraints 1-4 / assumptions."""
    inst = generate_instance(30, "A", seed=3, infectious_share=0.5)
    S = inst.surgeries
    T = inst.slots_per_day
    mand = inst.mandatory
    if violation == "deadline":
        i = next(s.id for s in S if s.deadline == 0)
        cols = [Column(1, ((i, 0),))]
    elif violation == "surgeon_max":
        l = S[0].surgeon
        mine = [s.id for s in S if s.surgeon == l and s.deadline >= 0]
        cols, st = [], 0
        for k, i in enumerate(mine[:3]):
            cols.append(Column(0, ((i, 0),)))
        inst.surgeon_max[l, 0] = S[mine[0]].duration  # make the cap binding
    elif violation == "surgeon_overlap":
        l = S[0].surgeon
        a, b = [s.id for s in S if s.surgeon == l][:2]
        cols = [Column(0, ((a, 0),)), Column(0, ((b, 1),))]
        inst.surgeon_max[l, 0] = T
    elif violation == "rooms":
        cols = [Column(0, ((i, 0),)) for i in range(inst.rooms[0] + 1)]
    elif violation == "mandatory":
        cols = []
    elif violation == "room_overlap":
        cols = [Column(0, ((0, 0), (1, 1)))]
    elif violation == "cleaning":
        a = next(s.id for s in S if s.infection == 0)
        b = next(s.id for s in S if s.infection == 1)
        cols = [Column(0, ((a, 0), (b, S[a].duration)))]
    elif violation == "horizon":
        cols = [Column(0, ((0, T - S[0].duration + 1),))]
    elif violation == "twice":
        cols = [Column(0, ((0, 0),)), Column(0, ((0, 60),))]
    errs = validate_columns(inst, cols, require_mandatory=(violation == "mandatory"))
    assert errs, f"validator missed a {violation} violation"


def test_generator_matches_section_6_1():
    """5-day horizon, 8 surgeons, 5-min slots, OCT 30 min, deadlines U[1,14]
    (postponed only when infeasible), half infectious, A/B durations U[2h,4h],
    6 rooms in set A, C mostly < 2 h, surgeon limits averaging ~320 min."""
    for S in "ABC":
        inst = generate_instance(100, S, seed=11)
        assert inst.n_days == 5 and inst.n_surgeons == 8 and inst.slot_minutes == 5
        assert inst.cleaning * inst.slot_minutes == 30
        assert inst.slots_per_day * inst.slot_minutes == 480
        dur = np.array([s.duration * 5 for s in inst.surgeries])
        if S in "AB":
            assert dur.min() >= 120 and dur.max() <= 240
        else:
            assert np.median(dur) < 120
        assert all(0 <= s.deadline <= 13 + 5 for s in inst.surgeries)
        share = np.mean([s.infection > 0 for s in inst.surgeries])
        assert share == pytest.approx(0.5, abs=0.01)
        assert 280 <= inst.surgeon_max.mean() * 5 <= 380
        if S == "A":
            assert inst.rooms == [6] * 5
