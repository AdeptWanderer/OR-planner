"""Algorithm 1 (price-and-cut) and Algorithm 2 (branch-and-price-and-cut) control flow."""
import math

import pytest

from orsched import BPCParams, generate_instance, solve


@pytest.fixture(scope="module")
def run():
    inst = generate_instance(40, "B", seed=1)
    sol = solve(inst, BPCParams(cg_time_limit=60, branch_time_limit=40, final_ip_time=15, polish_time=0, workers=2))
    return inst, sol


def test_algorithm1_phase_logic(run):
    """Phase 0 (CP time limit) -> phase 1 only after an iteration adding no
    column; phase 1 -> phase 0 after columns are found; the node stops only
    after a phase-1 iteration that added nothing (or on the time limit)."""
    inst, sol = run
    by_node = {}
    for r in sol.stats["iter_log"]:
        by_node.setdefault(r["node"], []).append(r)
    for node, its in by_node.items():
        for prev, cur in zip(its, its[1:]):
            if cur["phase"] == 1:
                assert prev["phase"] == 0 and prev["new_columns"] == 0
            if prev["phase"] == 1 and prev["new_columns"] > 0:
                assert cur["phase"] == 0
    root = by_node[0]
    if sol.stats["root"]["converged"]:
        assert root[-1]["phase"] == 1 and root[-1]["new_columns"] == 0


def test_bounds_are_monotone_and_consistent(run):
    inst, sol = run
    root_its = [r for r in sol.stats["iter_log"] if r["node"] == 0]
    lb = [r["lagrangian_bound"] for r in root_its]
    assert all(b2 <= b1 + 1e-9 for b1, b2 in zip(lb, lb[1:])), "best bound must never increase"
    assert sol.lower_bound <= sol.upper_bound + 1e-6
    assert sol.upper_bound <= sol.stats["root"]["bound"] * inst.slot_minutes + 1e-6
    assert sol.lower_bound >= sol.stats["heuristic_minutes"]


def test_algorithm2_best_first_and_child_bounds(run):
    """Nodes are taken best-first (popped bounds non-increasing) and a child's
    bound never exceeds its parent's."""
    inst, sol = run
    nodes = [n for n in sol.stats["node_log"] if n["id"] != 0]
    if not nodes:
        pytest.skip("solved at the root")
    pb = [n["pbound"] for n in nodes if n.get("pbound") is not None]
    assert all(b2 <= b1 + 1e-6 for b1, b2 in zip(pb, pb[1:]))
    bound = {n["id"]: n["bound"] for n in sol.stats["node_log"]}
    for n in nodes:
        if n["bound"] is not None and bound.get(n["parent"]) is not None:
            assert n["bound"] <= bound[n["parent"]] + 1e-6
