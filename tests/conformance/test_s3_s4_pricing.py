"""Section 3.2 pricing model and Section 4 enhancements.

The production pricing uses CP-SAT optional intervals; the paper uses a
position-based CP model (8)-(15) with objective (17).  These tests show that
both give the SAME optimal value on realistic and adversarial dual vectors:

* without enhancements: production == literal position model;
* with all enhancements: production == literal position model + DR1.

Dominance Rule 1 is NOT an exact pruning rule for the subproblem: the paper's
proof shows an optimal IORPS solution satisfies it, not that the pricing
optimum does for every dual vector (found by this suite: e.g. an optimal
column starting at slot 2).  It is a restriction of the column set that keeps
an optimal integer solution, so the method's bounds stay valid.  DR2-DR4 and
the knapsack check (FID) are exact on top of it, which is what is asserted.
"""
import numpy as np
import pytest

from conftest import converged_duals, dr4_instance, random_duals, real_duals, small_instances
from orsched.pricing import knapsack_bound, solve_pricing, start_costs
from orsched.validate import validate_columns
from verification.paper_models import paper_fid_bound, position_pricing

STATS = {"dr2": 0, "dr3": 0, "dr4": 0, "cond": 0, "fid": 0, "cases": 0}


def _exact_prod(inst, d, G, pi6, c, dom, fid):
    cols, st = solve_pricing(inst, d, G, pi6, c, time_limit=60, use_dominance=dom, use_fid=fid,
                             max_columns=10_000, workers=4)
    assert st.proven or cols, f"production pricing not proven: {st.cp_status}"
    for col in cols:
        assert validate_columns(inst, [col], require_mandatory=False) == []
    return (max(st.best_value, 0.0) if cols else 0.0), st


def _cases():
    out = []
    rng = np.random.default_rng(2024)
    for inst in small_instances():
        for (m, y) in real_duals(inst, iterations=2):
            for d in range(inst.n_days):
                out.append((inst, d) + m.day_duals(y, d))
        for _ in range(3):
            d = int(rng.integers(inst.n_days))
            out.append((inst, d) + random_duals(inst, rng))
        # converged root duals: most days have no improving column (FID regime)
        m, y = converged_duals(inst)
        for d in range(inst.n_days):
            out.append((inst, d) + m.day_duals(y, d))
        # high room price: many subproblems infeasible w.r.t. (19)
        d = int(rng.integers(inst.n_days))
        G, pi6, c = random_duals(inst, rng)
        out.append((inst, d, G, pi6, c - rng.uniform(40, 120)))
    return out


CASES = _cases()


def _check(ref, status, bound, v, tol, what):
    ref_pos = max(ref if ref is not None else 0.0, 0.0)
    if status == "OPTIMAL":
        assert v == pytest.approx(ref_pos, abs=tol), f"{what}: {v} != reference {ref_pos}"
    else:   # reference not proven within its limit: ours (proven) must lie in [best plan, bound]
        assert ref_pos - tol <= v <= max(bound, 0.0) + tol, f"{what}: {v} outside [{ref_pos}, {bound}]"


@pytest.mark.parametrize("k", range(len(CASES)))
def test_pricing_optimum_equals_position_model(k):
    inst, d, G, pi6, c = CASES[k]
    tol = 0.01 + 1e-3 * inst.n            # integer scaling of CP-SAT objectives
    ref, _, status, bound = position_pricing(inst, d, G, pi6, c, time_limit=180)
    ref1, _, status1, bound1 = position_pricing(inst, d, G, pi6, c, time_limit=180, dr1=True)
    assert status in ("OPTIMAL", "FEASIBLE") and status1 in ("OPTIMAL", "FEASIBLE")
    plain, _ = _exact_prod(inst, d, G, pi6, c, dom=False, fid=False)
    full, st = _exact_prod(inst, d, G, pi6, c, dom=True, fid=True)
    _check(ref, status, bound, plain, tol, "no enhancements vs paper model")
    _check(ref1, status1, bound1, full, tol, "all enhancements vs paper model + DR1")
    assert full <= plain + tol, "a restriction cannot improve the optimum"
    assert st.value_bound >= full - tol, "reported bound below the restricted optimum"
    STATS["dr2"] += st.dr2; STATS["dr3"] += st.dr3; STATS["dr4"] += st.dr4
    STATS["cond"] += st.conditioned; STATS["fid"] += int(st.fid_infeasible); STATS["cases"] += 1
    STATS["dr1_changed"] = STATS.get("dr1_changed", 0) + int(full < plain - tol)


def test_enhancements_actually_fired():
    """Guard against a vacuous equivalence test: the rules must have pruned."""
    if STATS["cases"] < len(CASES):
        pytest.skip("run together with test_pricing_optimum_equals_position_model")
    assert STATS["dr2"] > 0
    assert STATS["dr3"] + STATS["cond"] > 0
    assert STATS["fid"] > 0


def test_dr4_fires_and_keeps_the_optimum():
    """DR3(c) conditions S-long -> S-mid -> S-short (each dominated by the next,
    the dominator's surgeon can fill the room); 40+35+30 > 96, so DR4 removes
    S-long.  The pricing optimum must not change."""
    inst = dr4_instance()
    G = np.array([20.0, 30.0, 40.0, 1.0])            # shorter case is worth more
    pi6 = np.zeros((4, inst.slots_per_day))
    cols, st = solve_pricing(inst, 0, G, pi6, -0.5, time_limit=20, max_columns=10_000)
    assert st.dr4 >= 1, f"DR4 did not fire: {st}"
    plain, _ = _exact_prod(inst, 0, G, pi6, -0.5, dom=False, fid=False)
    ref, _, status, _ = position_pricing(inst, 0, G, pi6, -0.5, time_limit=60, dr1=True)
    assert status == "OPTIMAL"
    assert max(st.best_value, 0) == pytest.approx(plain, abs=0.01) == pytest.approx(max(ref, 0), abs=0.01)


@pytest.mark.parametrize("k", range(0, len(CASES), 2))
def test_knapsack_bound_matches_paper_steps_and_is_valid(k):
    """Our greedy == the paper's step-by-step algorithm (4.2), and it bounds
    the true subproblem optimum (24)-(27) is a relaxation."""
    inst, d, G, pi6, c = CASES[k]
    cand = [i for i in inst.eligible(d) if inst.surgeries[i].duration <= inst.surgeon_max[inst.surgeries[i].surgeon, d]]
    dbest = {}
    T, SD = inst.slots_per_day, min(inst.surgeries[i].duration for i in inst.eligible(d))
    for i in cand:
        cs = start_costs(inst, pi6, i)
        v = np.arange(len(cs)); v = v[(v == 0) | (v >= SD)]
        dbest[i] = G[i] - cs[v].min()
    ours = knapsack_bound(inst, d, cand, dbest)
    paper = paper_fid_bound(inst, d, cand, dbest)
    assert ours == pytest.approx(paper, abs=1e-9)
    ref, _, st_, bd = position_pricing(inst, d, G, pi6, c, time_limit=120, dr1=True)
    best = ref if st_ == "OPTIMAL" else bd
    assert ours + c >= (ref if ref is not None else -np.inf) - 0.02, "knapsack bound below a feasible schedule"


def test_dr1_removes_exactly_1_to_SD_minus_1():
    """Dominance rule 1: V_1 == 0 or V_1 >= SD_d (no other start values removed)."""
    inst = small_instances()[2]
    d = 0
    SD = min(inst.surgeries[i].duration for i in inst.eligible(d))
    G = np.array([s.duration for s in inst.surgeries], float)
    pi6 = np.zeros((inst.n_surgeons, inst.slots_per_day))
    cols, st = solve_pricing(inst, d, G, pi6, -1.0, time_limit=20, max_columns=10_000)
    assert st.dr1_removed == pytest.approx(100 * (SD - 1) / inst.slots_per_day)
    for col in cols:
        assert all(s == 0 or s >= SD for _, s in col.items)


def test_pricing_respects_deadlines_eq8():
    inst = small_instances()[1]
    for (m, y) in real_duals(inst, iterations=2):
        for d in range(inst.n_days):
            cols, _ = solve_pricing(inst, d, *m.day_duals(y, d), time_limit=10)
            for col in cols:
                assert all(inst.surgeries[i].deadline >= d for i, _ in col.items)
