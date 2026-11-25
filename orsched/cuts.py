"""Lifted minimal cover inequalities (Section 5.3) on the reformulated
surgeon-capacity knapsacks (36):  sum_{i in I_l} t_i z_id <= A_ld,
with z_id = sum_{j in J_d, i in j} x_j.

Lifting: the paper uses the fast sequence-independent lifting procedure of
Atamtürk (2005), eq. (39).  We use the classical Balas (1975) form of that
procedure in which theta_r is the sum of the r *largest* cover weights; this
is guaranteed valid (with the r smallest weights the inequality can cut off
feasible points, e.g. C={3,3,8}, b=13, t_i=3).
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

from .master import ExtraRow
from .model import Column, Instance


def z_values(inst: Instance, columns: Sequence[Column], x: np.ndarray) -> Dict[tuple, float]:
    z: Dict[tuple, float] = {}
    for c, v in zip(columns, x):
        if v <= 1e-9:
            continue
        for i, _ in c.items:
            z[(i, c.day)] = z.get((i, c.day), 0.0) + v
    return z


def lifted_cover_row(inst: Instance, d: int, l: int, C: List[int]) -> ExtraRow:
    w = sorted((inst.surgeries[i].duration for i in C), reverse=True)
    theta = np.cumsum(w)                       # theta_r, r = 1..|C|
    coef = {i: 1.0 for i in C}
    for i in inst.by_surgeon(l):
        if i in coef or inst.surgeries[i].deadline < d:
            continue
        t = inst.surgeries[i].duration
        r = int(np.searchsorted(theta, t, side="right"))   # #theta_r <= t
        if r > 0:
            coef[i] = float(min(r, len(C)))
    return ExtraRow(coef=coef, lo=-np.inf, hi=float(len(C) - 1), day=d, kind="cut",
                    tag=("lci", d, l, tuple(sorted(C))))


def separate(inst: Instance, columns: Sequence[Column], x: np.ndarray, existing: set,
             max_cuts: int = 50, eps: float = 1e-4) -> List[ExtraRow]:
    z = z_values(inst, columns, x)
    cuts: List[ExtraRow] = []
    for d in range(inst.n_days):
        for l in range(inst.n_surgeons):
            items = [i for i in inst.by_surgeon(l) if z.get((i, d), 0.0) > 1e-6]
            if len(items) < 2:
                continue
            cap = inst.surgeon_max[l, d]
            dur = {i: inst.surgeries[i].duration for i in items}
            if sum(dur.values()) <= cap:
                continue
            # greedy cover: prefer z close to 1, heavy items
            order = sorted(items, key=lambda i: ((1.0 - z[(i, d)]) / dur[i], -dur[i]))
            C, w = [], 0
            for i in order:
                C.append(i); w += dur[i]
                if w > cap:
                    break
            if w <= cap:
                continue
            # make minimal: drop items with smallest z while still a cover
            for i in sorted(C, key=lambda i: z[(i, d)]):
                if w - dur[i] > cap:
                    C.remove(i); w -= dur[i]
            row = lifted_cover_row(inst, d, l, C)
            if row.tag in existing:
                continue
            lhs = sum(cf * z.get((i, d), 0.0) for i, cf in row.coef.items())
            if lhs > row.hi + eps:
                cuts.append(row)
                existing.add(row.tag)
                if len(cuts) >= max_cuts:
                    return cuts
    return cuts
