"""Closed-form second-stage evaluation (the *oracle*) and out-of-sample simulation.

Q^BC  : waiting recursion  w_k = max(0, w_{k-1} + s_{k-1} + d_{k-1} - s_k), overtime from the last completion,
        idle = L - sum d + overtime (constraints (4b)-(4e) solved in closed form: the LP optimum is the
        component-wise smallest feasible w, o, g).
Q^down: occupancy counts minus beds, only days t < T are costed (stays running past the horizon are ignored).
Everything is vectorised over scenarios, so 10 000 out-of-sample scenarios cost milliseconds.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from .generator import _gamma
from .model import Instance, Solution


# --------------------------------------------------------------------------- scenarios
@dataclass
class Scenarios:
    d: np.ndarray        # (S, I) minutes
    lc: np.ndarray       # (S, I) ICU LOS, integer days
    lw: np.ndarray       # (S, I) ward LOS, integer days

    @property
    def n(self) -> int: return self.d.shape[0]

    def subset(self, idx) -> "Scenarios":
        return Scenarios(self.d[idx], self.lc[idx], self.lw[idx])


def sample_scenarios(inst: Instance, n: int, rng: np.random.Generator, delta: float = 0.0,
                     family: str = "gamma") -> Scenarios:
    """Sample (d, l^c, l^w).

    delta = 0 : the in-sample distribution (gamma(mu_type, sd_type) clipped to [lo, hi]).
    delta > 0 : mis-specified distribution of §6.4/6.5 — durations clipped to [(1-delta) lo, (1+delta) hi];
                LOS gamma with mean (1+delta)*mu and the same variance, clipped to [lo, hi].
    family "lognormal" / "uniform" are available for the paper's additional distribution checks.
    """
    I = inst.I
    d = np.empty((n, I)); lc = np.empty((n, I)); lw = np.empty((n, I))
    for i, s in enumerate(inst.surgeries):
        t = inst.types[s.type]
        d[:, i] = _draw(t.d_mean, t.d_sd, n, rng, family)
        d[:, i] = np.clip(d[:, i], (1 - delta) * s.lo_d, (1 + delta) * s.hi_d)
        lc[:, i] = np.clip(np.rint(_draw(t.c_mean * (1 + delta), t.c_sd, n, rng, family)), s.lo_c, s.hi_c)
        lw[:, i] = np.clip(np.rint(_draw(t.w_mean * (1 + delta), t.w_sd, n, rng, family)), s.lo_w, s.hi_w)
    return Scenarios(d, lc.astype(int), lw.astype(int))


def _draw(mean, sd, n, rng, family):
    if mean <= 0 or sd <= 0:
        return np.full(n, max(mean, 0.0))
    if family == "gamma":
        return _gamma(mean, sd, n, rng)
    if family == "lognormal":
        s2 = np.log(1 + (sd / mean) ** 2)
        return rng.lognormal(np.log(mean) - s2 / 2, np.sqrt(s2), n)
    if family == "uniform":
        h = sd * np.sqrt(3)
        return rng.uniform(max(mean - h, 0), mean + h, n)
    raise ValueError(family)


def nominal_scenario(inst: Instance) -> Scenarios:
    return Scenarios(inst.mu_d[None, :].copy(), np.rint(inst.mu_c)[None, :].astype(int),
                     np.rint(inst.mu_w)[None, :].astype(int))


# --------------------------------------------------------------------------- Q^BC
def block_costs(inst: Instance, sol: Solution, d: np.ndarray, b: int) -> Dict[str, np.ndarray]:
    """Waiting (per-surgery list), overtime and idle of block b for every scenario row of ``d`` (S, I)."""
    S = d.shape[0]
    order = sol.order(b)
    L = inst.blocks[b].length
    wait = np.zeros((S, len(order)))
    w = np.zeros(S)
    prev_end = None
    tot = np.zeros(S)
    for k, i in enumerate(order):
        sk = sol.start[(b, k)]
        if k == 0:
            w = np.zeros(S)
        else:
            w = np.maximum(0.0, prev_end - sk)
        wait[:, k] = w
        prev_end = w + sk + d[:, i]
        tot += d[:, i]
    if order:
        over = np.maximum(0.0, prev_end - L)
        idle = np.maximum(0.0, L - tot + over)
    else:
        over = np.zeros(S); idle = np.full(S, L)
    return dict(wait=wait, over=over, idle=idle)


def bc_cost(inst: Instance, sol: Solution, d: np.ndarray, by_block: bool = False):
    c = inst.costs
    S = d.shape[0]
    tot = np.zeros((S, inst.B))
    parts = dict(wait=np.zeros(S), over=np.zeros(S), idle=np.zeros(S))
    for b in range(inst.B):
        r = block_costs(inst, sol, d, b)
        wsum = r["wait"].sum(axis=1)
        tot[:, b] = c.w * wsum + c.o * r["over"] + c.g * r["idle"]
        parts["wait"] += wsum; parts["over"] += r["over"]; parts["idle"] += r["idle"]
    return (tot if by_block else tot.sum(axis=1)), parts


# --------------------------------------------------------------------------- Q^down
def occupancy(inst: Instance, sol: Solution, lc: np.ndarray, lw: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """ICU / ward beds needed per day, shape (S, T).  ICU days a..a+lc-1, ward days a+lc..a+lc+lw-1."""
    S, T = lc.shape[0], inst.T
    icu = np.zeros((S, T)); ward = np.zeros((S, T))
    ts = np.arange(T)[None, :]
    for i in np.where(sol.scheduled)[0]:
        a = inst.blocks[int(sol.block[i])].day
        c = lc[:, i][:, None]; w = lw[:, i][:, None]
        icu += (ts >= a) & (ts < a + c)
        ward += (ts >= a + c) & (ts < a + c + w)
    return icu, ward


def down_cost(inst: Instance, sol: Solution, lc: np.ndarray, lw: np.ndarray):
    icu, ward = occupancy(inst, sol, lc, lw)
    ue = np.maximum(0.0, icu - inst.icu_beds[None, :])
    ne = np.maximum(0.0, ward - inst.ward_beds[None, :])
    cost = inst.costs.c_icu * ue.sum(axis=1) + inst.costs.c_ward * ne.sum(axis=1)
    return cost, dict(icu=icu, ward=ward, icu_excess=ue, ward_excess=ne)


def first_stage_cost(inst: Instance, sol: Solution) -> float:
    return float(sum(inst.first_stage_cost(i, sol.block[i] < 0) for i in range(inst.I)))


# --------------------------------------------------------------------------- full evaluation
def evaluate(inst: Instance, sol: Solution, sc: Scenarios, downstream: bool = True) -> Dict[str, object]:
    """Out-of-sample metrics of §6.4–6.6 for a fixed first-stage solution."""
    bc, parts = bc_cost(inst, sol, sc.d)
    dc, det = down_cost(inst, sol, sc.lc, sc.lw)
    fs = first_stage_cost(inst, sol)
    total = fs + bc + dc
    icu_need, ward_need = det["icu"], det["ward"]
    pe_icu = (icu_need > inst.icu_beds[None, :]).mean(axis=0)
    pe_ward = (ward_need > inst.ward_beds[None, :]).mean(axis=0)

    def me(need, beds):
        ex = np.maximum(0.0, need - beds[None, :])
        pos = ex > 0
        return np.where(pos.sum(0) > 0, ex.sum(0) / np.maximum(pos.sum(0), 1), 0.0)

    return dict(
        first_stage=fs, bc_cost=bc, down_cost=dc, total=total,
        mean_total=float(total.mean()), q75_total=float(np.quantile(total, 0.75)), q90_total=float(np.quantile(total, 0.9)),
        mean_bc=float(bc.mean()), mean_down=float(dc.mean()),
        wait=parts["wait"], over=parts["over"], idle=parts["idle"],
        pe_icu=pe_icu, pe_ward=pe_ward, me_icu=me(icu_need, inst.icu_beds), me_ward=me(ward_need, inst.ward_beds),
        icu_need=icu_need, ward_need=ward_need, n_scheduled=sol.n_scheduled(),
    )


def saa_objective(inst: Instance, sol: Solution, sc: Scenarios, downstream: bool = True) -> float:
    ev = evaluate(inst, sol, sc)
    return float(ev["first_stage"] + ev["mean_bc"] + (ev["mean_down"] if downstream else 0.0))


def rel_tc(tc_a: np.ndarray, tc_b: np.ndarray) -> float:
    """(TC_a - TC_b) / TC_b on the mean, as the paper's RelTC (a = SP, b = DRO)."""
    return float((tc_a.mean() - tc_b.mean()) / tc_b.mean())
