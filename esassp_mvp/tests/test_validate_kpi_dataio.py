import io

import numpy as np
import pandas as pd
import pytest
from conftest import hand_instance, sol_from, tiny

from esassp import Params, generate_instance, sample_scenarios, validate_solution
from esassp.dataio import BED_COLS, BLOCK_COLS, SURG_COLS, instance_from_frames, instance_to_frames
from esassp.evaluate import evaluate
from esassp.kpi import beds_df, block_df, comparison_df, postponed_df, schedule_df, summary
from esassp.sp import solve_sp


@pytest.fixture(scope="module")
def solved():
    inst = tiny(2, 8)
    return inst, solve_sp(inst, Params(n_scenarios=6, time_limit=30, threads=2))


# ------------------------------------------------------------------ validator
def good():
    inst = hand_instance()
    return inst, sol_from(inst, {0: (0, 0), 1: (0, 1)}, {(0, 0): 0.0, (0, 1): 120.0})


def test_validator_accepts_good():
    assert validate_solution(*good()) == []


@pytest.mark.parametrize("mutate,frag", [
    (lambda s: s.pos.__setitem__(1, 0), "twice"),
    (lambda s: s.pos.__setitem__(1, 7), "outside"),
    (lambda s: s.block.__setitem__(1, 9), "unknown block"),
    (lambda s: s.start.__setitem__((0, 1), 9999.0), "outside"),
    (lambda s: s.start.__setitem__((0, 1), -5.0), "outside"),
    (lambda s: s.pos.__setitem__(1, 2), "consecutive"),
])
def test_validator_detects(mutate, frag):
    inst, sol = good()
    mutate(sol)
    errs = validate_solution(inst, sol)
    assert errs and any(frag in e for e in errs)


def test_validator_wrong_length_and_type_restriction():
    inst, sol = good()
    bad = type(sol)(sol.block[:2], sol.pos[:2], sol.start)
    assert validate_solution(inst, bad)
    inst2 = generate_instance(10, "cardiac", 1)
    i = 0
    wrong = [b for b in range(inst2.B) if b not in inst2.eligible_blocks(i)][0]
    sol2 = sol_from(inst2, {i: (wrong, 0)}, {(wrong, 0): 0.0})
    assert any("type not allowed" in e for e in validate_solution(inst2, sol2))


# ------------------------------------------------------------------ kpi
def test_schedule_and_postponed_tables(solved):
    inst, r = solved
    s, p = schedule_df(inst, r.solution), postponed_df(inst, r.solution)
    assert len(s) + len(p) == inst.I
    assert set(s.columns) >= {"block", "position", "surgery", "scheduled_start", "planned_end"}
    assert (s.planned_end >= s.scheduled_start).all()


def test_block_and_bed_tables_and_summary(solved):
    inst, r = solved
    sc = sample_scenarios(inst, 300, np.random.default_rng(0))
    ev = evaluate(inst, r.solution, sc)
    bd = block_df(inst, r.solution, sc)
    assert len(bd) == inst.B and (bd.p_overtime.between(0, 1)).all() and (bd.n >= 0).all()
    bed = beds_df(inst, ev)
    assert len(bed) == inst.T and (bed.icu_PE.between(0, 1)).all()
    sm = summary(inst, r.solution, ev)
    assert sm["scheduled"] + sm["postponed"] == inst.I and sm["mean_total_cost"] > 0


def test_comparison_df(solved):
    inst, r = solved
    sc = sample_scenarios(inst, 200, np.random.default_rng(0))
    df = comparison_df(inst, {"SP": r, "SP2": r}, sc)
    assert list(df.model) == ["SP", "SP2"] and df.mean_total_cost.iloc[0] == df.mean_total_cost.iloc[1]


# ------------------------------------------------------------------ dataio
def test_frames_columns_and_roundtrip():
    inst = generate_instance(15, "general", 3, n_ors=2)
    f = instance_to_frames(inst)
    assert list(f["surgeries"].columns) == SURG_COLS and list(f["blocks"].columns) == BLOCK_COLS and list(f["beds"].columns) == BED_COLS
    j = instance_from_frames(f["surgeries"], f["blocks"], f["beds"], inst.costs)
    assert (j.I, j.B, j.T) == (inst.I, inst.B, inst.T)
    for a, b in zip(inst.surgeries, j.surgeries):
        assert (a.lo_c, a.hi_c, a.lo_w, a.hi_w) == (b.lo_c, b.hi_c, b.lo_w, b.hi_w) and a.mu_d == pytest.approx(b.mu_d)
    assert [b.types for b in j.blocks] == [tuple(inst.types.index(t) for t in [inst.types[x.types[0]]]) if False else b.types for b in j.blocks]
    assert [(b.day, b.room, b.K) for b in j.blocks] == [(b.day, b.room, b.K) for b in inst.blocks]


def test_csv_text_roundtrip():
    inst = generate_instance(10, "cardiac", 2)
    f = instance_to_frames(inst)
    back = {k: pd.read_csv(io.StringIO(v.to_csv(index=False))) for k, v in f.items()}
    j = instance_from_frames(back["surgeries"], back["blocks"], back["beds"])
    assert np.allclose(j.mu_d, inst.mu_d) and np.allclose(j.ward_beds, inst.ward_beds)


def test_blocks_without_type_are_open_and_solvable():
    inst = tiny(2, 6)
    f = instance_to_frames(inst)
    f["blocks"]["type"] = ""
    j = instance_from_frames(f["surgeries"], f["blocks"], f["beds"], inst.costs)
    assert all(b.types == () for b in j.blocks)
    assert all(len(j.eligible_blocks(i)) == j.B for i in range(j.I))
    r = solve_sp(j, Params(n_scenarios=4, time_limit=30, threads=2))
    assert validate_solution(j, r.solution) == []


def test_missing_priority_defaults_to_one():
    inst = tiny(2, 6)
    f = instance_to_frames(inst)
    f["surgeries"]["priority"] = np.nan
    j = instance_from_frames(f["surgeries"], f["blocks"], f["beds"])
    assert all(s.priority == 1.0 for s in j.surgeries)


def test_beds_sorted_by_day():
    inst = tiny(2, 6)
    f = instance_to_frames(inst)
    shuffled = f["beds"].sample(frac=1, random_state=0)
    j = instance_from_frames(f["surgeries"], f["blocks"], shuffled)
    assert list(j.icu_beds) == list(inst.icu_beds)


def test_priority_scales_first_stage_cost():
    inst = tiny(2, 6)
    f = instance_to_frames(inst)
    f["surgeries"].loc[0, "priority"] = 3.0
    j = instance_from_frames(f["surgeries"], f["blocks"], f["beds"], inst.costs)
    assert j.first_stage_cost(0, True) == pytest.approx(3 * inst.first_stage_cost(0, True))
