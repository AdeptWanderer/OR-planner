"""Shared helpers for the paper-conformance suite."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from orsched import generate_instance                       # noqa: E402
from orsched.heuristic import constructive_heuristic       # noqa: E402
from orsched.master import Master                          # noqa: E402
from orsched.pricing import solve_pricing                  # noqa: E402


def real_duals(inst, iterations=3, seed=0):
    """Duals of the RMP after a few CG iterations (realistic pricing inputs).

    Yields (master, y) after each iteration."""
    cols, _ = constructive_heuristic(inst, np.random.default_rng(seed))
    m = Master(inst)
    m.add_columns(cols)
    out = []
    for _ in range(iterations):
        obj, x, art, y = m.solve_lp()
        out.append((m, y.copy()))
        new = []
        for d in range(inst.n_days):
            G, pi6, c = m.day_duals(y, d)
            cs, _ = solve_pricing(inst, d, G, pi6, c, time_limit=5, workers=2)
            new += cs
        if not m.add_columns(new):
            break
    return out


def random_duals(inst, rng):
    """Adversarial random dual vector for one day: arbitrary G, sparse pi6 >= 0."""
    G = np.array([s.duration * rng.uniform(-0.6, 1.6) for s in inst.surgeries])
    pi6 = np.zeros((inst.n_surgeons, inst.slots_per_day))
    mask = rng.random(pi6.shape) < 0.15
    pi6[mask] = rng.uniform(0, 1.2, mask.sum())
    const = -rng.uniform(0, 40)
    return G, pi6, const


def small_instances():
    """A spread of small instances covering sets A/B/C, cleaning, infection types."""
    specs = [
        dict(n_surgeries=12, instance_set="A", seed=1, n_days=2, n_surgeons=4, rooms=2),
        dict(n_surgeries=14, instance_set="B", seed=2, n_days=2, n_surgeons=4),
        dict(n_surgeries=14, instance_set="C", seed=3, n_days=2, n_surgeons=4),
        dict(n_surgeries=12, instance_set="C", seed=4, n_days=2, n_surgeons=3, n_infection_types=3),
        dict(n_surgeries=12, instance_set="A", seed=5, n_days=3, n_surgeons=4, cleaning_minutes=60, rooms=2),
        dict(n_surgeries=14, instance_set="C", seed=6, n_days=2, n_surgeons=4, cleaning_minutes=0),
    ]
    return [generate_instance(**s) for s in specs]
