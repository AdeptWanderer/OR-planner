"""Regression library: one test per problem found while building and running
the solver.  Each test names the bug it guards against.  Keep adding here."""
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scenarios"))

from orsched import BPCParams, generate_instance, solve, validate_solution  # noqa: E402
from orsched.bounds import day_level_bound  # noqa: E402
from orsched.compact import solve_compact  # noqa: E402
from orsched.heuristic import constructive_heuristic  # noqa: E402
from orsched.master import ExtraRow, Master  # noqa: E402
from orsched.validate import validate_columns  # noqa: E402

QUICK = dict(cg_time_limit=30, branch_time_limit=20, final_ip_time=10, polish_time=0, workers=2)


# ---------------------------------------------------------------- master problem
def test_geq_branching_row_has_artificial_so_master_stays_feasible():
    """Bug: a '#cases >= k' branching row made the restricted master infeasible
    before pricing could add columns, and the node was wrongly declared infeasible."""
    inst = generate_instance(12, "A", seed=1, n_days=2, n_surgeons=3)
    row = ExtraRow(coef={i: 1.0 for i in range(inst.n)}, lo=inst.n, hi=np.inf, kind="branch")
    m = Master(inst, [row])                    # no columns at all
    obj, x, art, y = m.solve_lp()              # must not raise
    assert art.sum() > 0                       # covered by artificials, not infeasible


def test_big_m_penalty_dominates_all_case_minutes():
    """Bug: penalty 10*T let the LP keep a mandatory case partly uncovered (artificial
    0.025) because that was cheaper; nodes were then pruned as 'infeasible'."""
    inst = generate_instance(100, "B", seed=1)
    m = Master(inst)
    assert m.penalty > sum(s.duration for s in inst.surgeries)


def test_artificials_positive_never_prune_a_node_by_themselves():
    """Same bug, end to end: B100 seed 1 used to report UB = LB = 10,695 after 3 nodes
    (invalid).  With the fix the upper bound stays above an independently found plan."""
    inst = generate_instance(30, "B", seed=1, n_days=3)
    sol = solve(inst, BPCParams(**QUICK))
    ref = solve_compact(inst, 60, workers=2)
    assert sol.upper_bound >= (ref["lb"] or 0) - 1e-6


def test_no_upper_bound_on_columns_keeps_reduced_costs_exact():
    """Bug: x_j <= 1 bounds gave columns at their bound a positive reduced cost,
    breaking the LP-duality argument behind the Lagrangian bound."""
    inst = generate_instance(30, "B", seed=4)
    cols, _ = constructive_heuristic(inst)
    m = Master(inst)
    m.add_columns(cols)
    _, _, _, y = m.solve_lp()
    assert max(m.column_value(c, y) for c in m.columns) <= 1e-6


# ---------------------------------------------------------------- pricing
def test_short_case_pricing_proves_optimality_quickly():
    """Bug: set-C pricing could not prove optimality within 30 s; the redundant
    makespan constraint fixed it (C-16 root went from 60 s to under 1 s)."""
    inst = generate_instance(16, "C", seed=8, n_days=2, n_surgeons=4)
    sol = solve(inst, BPCParams(cg_time_limit=60, branch_time_limit=0, final_ip_time=10, polish_time=0, workers=2))
    assert sol.stats["root"]["converged"]
    assert sol.stats["root"]["time"] < 30


@pytest.mark.filterwarnings("error::DeprecationWarning")   # bug: fork() of a threaded process can deadlock
def test_parallel_pricing_gives_the_same_result_as_sequential():
    inst = generate_instance(40, "B", seed=2)
    a = solve(inst, BPCParams(**QUICK, parallel_pricing=1))
    b = solve(inst, BPCParams(**QUICK, parallel_pricing=4))
    assert validate_solution(inst, b) == []
    assert a.stats["root"]["lp"] == pytest.approx(b.stats["root"]["lp"], rel=1e-6, abs=1e-6) or \
        abs(a.lower_bound - b.lower_bound) <= 0.02 * a.lower_bound


# ---------------------------------------------------------------- heuristics
def test_multistart_heuristic_returns_a_valid_plan_when_single_start_fails():
    """Bug: on Riverside the single greedy start missed a mandatory case, so the
    solver had no incumbent at all; 200 random tie-breaks found a valid plan."""
    import riverside as R
    inst, _ = R.build()
    single, _ = constructive_heuristic(inst, np.random.default_rng(0))
    assert validate_columns(inst, single) != []           # the failure we hit
    s = __import__("orsched.bpc", fromlist=["Solver"]).Solver(
        inst, BPCParams(heuristic_starts=60, cg_time_limit=1, branching=False, final_ip_time=1, polish_time=0))
    sol = s.run()
    assert s.heuristic["errors"] == []
    assert validate_columns(inst, s.heuristic["columns"]) == []


def test_dayplan_columns_are_individually_feasible_and_plan_valid_on_small_instance():
    from orsched.dayplan import dayplan_heuristic
    inst = generate_instance(30, "B", seed=3, n_days=3)
    cols, info = dayplan_heuristic(inst, time_per_day=10, workers=2)
    for c in cols:
        assert validate_columns(inst, [c], require_mandatory=False) == []
    assert validate_columns(inst, cols) == []


# ---------------------------------------------------------------- bounds
@pytest.mark.parametrize("S,seed", [("A", 1), ("B", 2), ("C", 3)])
def test_day_level_bound_is_a_valid_upper_bound(S, seed):
    inst = generate_instance(16, S, seed=seed, n_days=2, n_surgeons=4)
    ub, _, proved = day_level_bound(inst, integer=True)
    ref = solve_compact(inst, 60, workers=2)
    assert ref["status"] == "OPTIMAL"
    assert ub * inst.slot_minutes >= ref["lb"] - 1e-6


def test_highs_thread_settings_do_not_clash_in_one_process():
    """Bug: protect_at_risk created a HiGHS instance with default threads, which fixed
    the process-wide pool at 4; every later threads=1 instance (day-level bound,
    master LP) then failed with status kNotset.  Runs in a fresh process because the
    pool is global."""
    code = textwrap.dedent(f"""
        import sys; sys.path.insert(0, {str(ROOT)!r}); sys.path.insert(0, {str(ROOT / 'scenarios')!r})
        import riverside as R
        from orsched.bounds import day_level_bound
        from orsched.master import Master
        inst, _ = R.build(extra_leave_week2=("Ortho B",))
        R.protect_at_risk(inst, "Ortho B")
        b, _, proved = day_level_bound(inst, integer=False)
        base, _ = R.build()
        b2, _, proved2 = day_level_bound(base, integer=False)
        Master(base).solve_lp()
        print("OK", proved2, b2)
    """)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
    assert "OK True" in out.stdout, out.stdout + out.stderr


# ---------------------------------------------------------------- scenario data
def test_what_if_variants_keep_the_same_waiting_list():
    """Bug: applying extra leave before drawing cases changed the random draws, so the
    'what-if' compared different patients (Neuro A suddenly overloaded)."""
    import riverside as R
    key = lambda inst: [(s.duration, s.surgeon, s.deadline, s.infection) for s in inst.surgeries]
    base, _ = R.build()
    for kw in (dict(extra_leave_week2=("Ortho B",)), dict(neuro_extra_sessions=True), dict(friday_rooms=8)):
        v, _ = R.build(**kw)
        assert key(v) == key(base), kw


def test_protect_at_risk_is_feasible_and_partitions_the_at_risk_cases():
    import riverside as R
    inst, _ = R.build(extra_leave_week2=("Ortho B",))
    ob = inst.surgeon_names.index("Ortho B")
    risk = {s.id for s in inst.surgeries if s.surgeon == ob and inst.is_mandatory(s.id) and s.deadline >= 5}
    prot, brch = R.protect_at_risk(inst, "Ortho B")
    assert set(prot) | set(brch) == risk and not set(prot) & set(brch)
    hard = sum(s.duration for s in inst.surgeries if s.surgeon == ob and inst.is_mandatory(s.id) and s.deadline < 5)
    assert hard + sum(inst.surgeries[i].duration for i in prot) <= inst.surgeon_max[ob, :5].sum()


# ---------------------------------------------------------------- robustness
def test_checkpoint_resume_reproduces_a_finished_run(tmp_path):
    inst = generate_instance(30, "A", seed=5, n_days=3)
    p = BPCParams(**QUICK, checkpoint_path=str(tmp_path / "ck.pkl"), checkpoint_every=1)
    a = solve(inst, p)
    b = solve(inst, p)                       # resumes from the 'done' checkpoint
    assert (a.lower_bound, a.upper_bound) == (b.lower_bound, b.upper_bound)
    assert validate_solution(inst, b) == []


def test_lifting_uses_largest_cover_weights():
    """Eq. (39) as printed (smallest weights) cuts off feasible points; ours must not."""
    from orsched.cuts import lifted_cover_row
    from orsched.model import Instance, Surgery
    surg = [Surgery(0, 3, 0, 0), Surgery(1, 3, 0, 0), Surgery(2, 8, 0, 0), Surgery(3, 3, 0, 0)]
    inst = Instance(surg, 1, [1], 96, np.array([[13]]), 0)
    row = lifted_cover_row(inst, 0, 0, [0, 1, 2])
    assert sum(row.coef.get(i, 0) for i in (3, 0, 1)) <= row.hi    # {3,3,3} = 9 <= 13 must stay feasible


def test_dayplan_schedules_every_mandatory_case_on_riverside():
    """Bug: the day packing maximised minutes only, so optional cases pushed mandatory
    cases to their last day, where they no longer fitted (1-2 missed per scenario).
    Fix: force the mandatory cases the relaxation put on each day.  Slow (~5 min)."""
    import riverside as R
    from orsched.dayplan import dayplan_heuristic
    inst, _ = R.build()
    cols, info = dayplan_heuristic(inst, time_per_day=30, workers=4)
    assert validate_columns(inst, cols) == []


def test_day_level_bound_reports_infeasibility_explicitly():
    """Bug: on an infeasible instance (Ortho B leave before triage) the bound came back
    as inf / 0 with 'not proven' instead of saying 'infeasible'."""
    import riverside as R
    inst, _ = R.build(extra_leave_week2=("Ortho B",))
    b, assign, proved = day_level_bound(inst, integer=True)
    assert b == -1.0 and proved
