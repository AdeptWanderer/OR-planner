import numpy as np
import pytest

from esassp.milp import INF, Model


def test_simple_lp():
    m = Model(); x = m.var(0, 10, -1); y = m.var(0, 10, -2)
    m.add({x: 1, y: 1}, hi=6); m.add({x: 1, y: -1}, lo=-2)
    r = m.solve()
    assert r.ok and r.obj == pytest.approx(-(2 + 2 * 4)) and r.x[y] == pytest.approx(4.0)


def test_knapsack_milp_optimal():
    w, v, cap = [3, 4, 5, 9, 4], [3, 4, 4, 10, 4], 11
    m = Model(); xs = [m.binary(-vi) for vi in v]
    m.add({x: wi for x, wi in zip(xs, w)}, hi=cap)
    r = m.solve(gap=0)
    best = max(sum(vi for vi, b in zip(v, bits) if b) for bits in __import__("itertools").product([0, 1], repeat=5)
               if sum(wi for wi, b in zip(w, bits) if b) <= cap)
    assert -r.obj == pytest.approx(best) and r.gap == 0 and r.bound == pytest.approx(r.obj)


def test_equality_fix_and_bounds():
    m = Model(); x = m.var(0, 5, 1); y = m.var(-3, 3, 1)
    m.add({x: 1, y: 1}, 2, 2); m.fix(y, -1)
    r = m.solve()
    assert r.x[y] == pytest.approx(-1) and r.x[x] == pytest.approx(3)


def test_infeasible_returns_not_ok():
    m = Model(); x = m.var(0, 1, 1); m.add({x: 1}, lo=2)
    assert not m.solve().ok


def test_duplicate_terms_are_summed_and_zero_dropped():
    m = Model(); x = m.var(0, 10, 1)
    m.add([(x, 1.0), (x, 1.0), (x, -2.0)], hi=1)          # 0*x <= 1 : harmless
    m.add([(x, 1.0), (x, 1.0)], lo=4)
    assert m.solve().x[x] == pytest.approx(2.0)


def test_row_and_var_counts():
    m = Model(); a = m.var(); b = m.binary(); m.add({a: 1, b: 1}, hi=1)
    assert m.n_vars == 2 and m.n_rows == 1


def test_set_obj_and_negative_lower_bounds():
    m = Model(); x = m.var(-INF, INF, 0); m.add({x: 1}, lo=-4, hi=4); m.set_obj(x, 1)
    assert m.solve().x[x] == pytest.approx(-4)


def test_dual_bound_valid_when_time_limited():
    rng = np.random.default_rng(0)
    n = 40
    m = Model(); xs = [m.binary(-float(rng.integers(5, 30))) for _ in range(n)]
    for _ in range(10):
        m.add({x: float(rng.integers(3, 20)) for x in xs}, hi=100)
    r = m.solve(time_limit=1, gap=0)
    assert r.ok and r.bound <= r.obj + 1e-6


def test_set_threads_idempotent():
    from esassp.milp import set_threads
    set_threads(2); set_threads(3)
    m = Model(); x = m.binary(-1)
    assert m.solve().ok
