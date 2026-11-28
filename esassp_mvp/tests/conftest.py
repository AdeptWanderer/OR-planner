import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from esassp import Params, generate_instance  # noqa: E402
from esassp.dro import solve_dro  # noqa: E402
from esassp.model import Block, Costs, Instance, Solution, Surgery, SurgeryType  # noqa: E402

FASTP = Params(time_limit=60, master_time=30, gap=1e-6, threads=2, n_scenarios=4, eps_d=40.0, eps_l=3.0)


def tiny(seed=2, n=8, beds=((1, 1), (2, 1)), library="general", ors=1, **kw):
    return generate_instance(n, library, seed, n_ors=ors, beds=beds, **kw)


def hand_instance(L=480.0, n=3, T=7, icu=2, ward=2, K=3):
    """n identical surgeries, one block on day 0: duration 100 (lo 80, hi 120), ICU 1-2 d (mean 1.5), ward 2-3 d (2.5)."""
    types = [SurgeryType("X", 100, 20, 1.5, 0.5, 2.5, 0.5)]
    surg = [Surgery(i, 0, f"X{i}", 1.0, 100.0, 20.0, 80.0, 120.0, 1.5, 0.5, 1, 2, 2.5, 0.5, 2, 3) for i in range(n)]
    blocks = [Block(0, 0, 0, L, K, (0,))]
    return Instance(types, surg, blocks, T, np.full(T, float(icu)), np.full(T, float(ward)), Costs(c_o=30.0, c_g=20.0, c_w=10.0,
                                                                                                      c_icu=1000, c_ward=500), "hand")


def sol_from(inst, assign, starts):
    """assign: {i: (b, k)}; starts: {(b,k): s}"""
    block = -np.ones(inst.I, int); pos = -np.ones(inst.I, int)
    for i, (b, k) in assign.items():
        block[i], pos[i] = b, k
    return Solution(block, pos, dict(starts))


@pytest.fixture(scope="session")
def inst():
    return tiny()


@pytest.fixture(scope="session")
def dro_m(inst):
    return solve_dro(inst, "M", FASTP)


@pytest.fixture(scope="session")
def small_cardiac():
    return generate_instance(12, "cardiac", 1, n_ors=1, beds="(5,2)")
