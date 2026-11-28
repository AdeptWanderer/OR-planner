"""Instance generator (synthetic reconstruction of the paper's §6.1 recipe).

The paper's hospital data is private and Appendices G/H of the supplement are not available, so type parameters
are *reconstructed*: only ENT (mu 74, sd 37), OPHTH (mu 38, sd 19) and VASCULAR (mean LOS 3.5 d) are quoted in the
text; every other number is invented but plausible and is labelled ``source="synthetic"``.
Bounds are the 20 % / 80 % quantiles of the type distributions, as in the paper.
"""
from __future__ import annotations

import math
from dataclasses import replace
from typing import Dict, List, Optional, Sequence

import numpy as np
from scipy import stats

from .model import Block, Costs, Instance, Surgery, SurgeryType

# ------------------------------------------------------------------ type libraries
CARDIAC = [  # inspired by the collaborating hospital (5 cardiac types) — all numbers synthetic
    SurgeryType("CABG", 270, 60, 2.2, 1.2, 5.0, 2.0),
    SurgeryType("AVR", 230, 50, 2.0, 1.0, 5.0, 2.0),
    SurgeryType("MVR", 260, 70, 2.8, 1.5, 6.0, 2.5),
    SurgeryType("TAVR", 85, 30, 0.9, 0.8, 3.0, 1.5),
    SurgeryType("OTHER", 150, 45, 1.2, 0.9, 4.0, 2.0),
]
GENERAL = [  # Mannino et al.-style specialties; only ENT / OPHTH durations and VASCULAR LOS are quoted in the paper
    SurgeryType("ENT", 74, 37, 0.0, 0.0, 0.6, 0.7, "paper text (duration); LOS synthetic"),
    SurgeryType("OPHTH", 38, 19, 0.0, 0.0, 0.1, 0.3, "paper text (duration); LOS synthetic"),
    SurgeryType("MED", 95, 40, 0.3, 0.5, 1.8, 1.2),
    SurgeryType("VASCULAR", 150, 55, 1.0, 0.9, 2.5, 1.5, "paper text (avg LOS 3.5 d); rest synthetic"),
    SurgeryType("ORTHO", 120, 45, 0.4, 0.6, 3.2, 1.6),
    SurgeryType("GENSURG", 110, 50, 0.6, 0.8, 2.8, 1.5),
]
LIBRARIES: Dict[str, List[SurgeryType]] = {"cardiac": CARDIAC, "general": GENERAL}

BED_PRESETS = {"(10,7)": ((10, 7), (10, 7)), "(5,2)": ((5, 2), (5, 2))}   # (icu(week, weekend), ward(week, weekend))


def _gamma(mean: float, sd: float, size, rng: np.random.Generator) -> np.ndarray:
    if mean <= 0 or sd <= 0:
        return np.full(size, max(mean, 0.0))
    k = (mean / sd) ** 2
    return rng.gamma(k, sd ** 2 / mean, size)


def _gamma_q(mean: float, sd: float, q: float) -> float:
    if mean <= 0 or sd <= 0:
        return max(mean, 0.0)
    k = (mean / sd) ** 2
    return float(stats.gamma.ppf(q, k, scale=sd ** 2 / mean))


def type_summary(t: SurgeryType, rng: np.random.Generator, n_mc: int = 20000) -> dict:
    """Bounds (20/80 % quantiles) and the mean of the *clipped* distribution, which is what the DRO-M set uses."""
    lo_d, hi_d = round(_gamma_q(t.d_mean, t.d_sd, 0.2), 1), round(_gamma_q(t.d_mean, t.d_sd, 0.8), 1)
    lo_c, hi_c = int(round(_gamma_q(t.c_mean, t.c_sd, 0.2))), int(round(_gamma_q(t.c_mean, t.c_sd, 0.8)))
    lo_w, hi_w = int(round(_gamma_q(t.w_mean, t.w_sd, 0.2))), int(round(_gamma_q(t.w_mean, t.w_sd, 0.8)))
    hi_c, hi_w = max(hi_c, lo_c), max(hi_w, lo_w)
    d = np.clip(_gamma(t.d_mean, t.d_sd, n_mc, rng), lo_d, hi_d)
    c = np.clip(np.rint(_gamma(t.c_mean, t.c_sd, n_mc, rng)), lo_c, hi_c)
    w = np.clip(np.rint(_gamma(t.w_mean, t.w_sd, n_mc, rng)), lo_w, hi_w)
    return dict(mu_d=float(d.mean()), sd_d=float(d.std()), lo_d=lo_d, hi_d=hi_d,
                mu_c=float(c.mean()), sd_c=float(c.std()), lo_c=lo_c, hi_c=hi_c,
                mu_w=float(w.mean()), sd_w=float(w.std()), lo_w=lo_w, hi_w=hi_w)


def _beds(T: int, weekday: int, weekend: int, surg_days: int = 5) -> np.ndarray:
    return np.array([weekday if (t % 7) < surg_days else weekend for t in range(T)], dtype=float)


def generate_instance(n_surgeries: int = 24, library: str = "cardiac", seed: int = 1, n_ors: int = 2,
                      T: int = 7, surgery_days: int = 5, block_minutes: float = 480.0,
                      beds: str | tuple = "(5,2)", costs: Optional[Costs] = None,
                      type_weights: Optional[Sequence[float]] = None, max_positions: int = 6,
                      name: Optional[str] = None) -> Instance:
    """Closed block scheduling: every block is dedicated to one surgery type (as at the collaborating hospital).

    ``beds`` is a preset name ("(10,7)" / "(5,2)") or ((icu_week, icu_weekend), (ward_week, ward_weekend)).
    """
    rng = np.random.default_rng(seed)
    types = LIBRARIES[library]
    nt = len(types)
    # random type mix (Dirichlet) so that every seed is a different instance; pass type_weights for a fixed mix
    w = rng.dirichlet(np.full(nt, 6.0)) if type_weights is None else np.asarray(type_weights, float)
    w = w / w.sum()
    # number of surgeries per type (largest remainder)
    raw = w * n_surgeries
    cnt = np.floor(raw).astype(int)
    for j in np.argsort(-(raw - cnt))[: n_surgeries - cnt.sum()]:
        cnt[j] += 1
    summ = [type_summary(t, np.random.default_rng(1000 + j)) for j, t in enumerate(types)]
    surgeries: List[Surgery] = []
    for j in range(nt):
        for r in range(cnt[j]):
            surgeries.append(Surgery(len(surgeries), j, f"{types[j].name}-{r + 1:02d}", 1.0, **summ[j]))
    # blocks: n_ors * surgery_days OR-days assigned to types proportionally to expected duration demand
    nblocks = n_ors * surgery_days
    demand = np.array([cnt[j] * summ[j]["mu_d"] for j in range(nt)])
    share = demand / demand.sum() * nblocks
    nb = np.maximum(np.where(cnt > 0, 1, 0), np.floor(share).astype(int))
    while nb.sum() > nblocks:
        nb[np.argmax(nb - share)] -= 1
    while nb.sum() < nblocks:
        nb[np.argmax(share - nb)] += 1
    # interleave types so that a type's blocks fall on different days where possible
    slots = [(d, r) for d in range(surgery_days) for r in range(n_ors)]
    order = [int(j) for j in np.argsort(-demand)]
    by_type = {j: [j] * int(nb[j]) for j in order}
    seq: List[int] = []
    while any(by_type.values()):
        for j in order:
            if by_type[j]:
                seq.append(by_type[j].pop())
    slot_types: List[List[int]] = [[j] for j in seq]
    present = [j for j in range(nt) if cnt[j] > 0]
    for j in present:                       # fewer blocks than types: leftover types share the least loaded block (open block)
        if not any(j in t for t in slot_types):
            load = [sum(cnt[x] * summ[x]["mu_d"] for x in t) for t in slot_types]
            slot_types[int(np.argmin(load))].append(j)
    blocks: List[Block] = []
    for (d, r), ts in zip(slots, slot_types):
        lo = min(summ[j]["lo_d"] for j in ts)
        ncls = sum(cnt[j] for j in ts)
        K = int(max(1, min(max_positions, ncls, math.floor(block_minutes / max(lo, 1.0)))))
        blocks.append(Block(len(blocks), d, r, block_minutes, K, tuple(sorted(ts))))
    if isinstance(beds, str):
        (iw, ie), (ww, we) = BED_PRESETS[beds]
    else:
        (iw, ie), (ww, we) = beds
    inst = Instance(types=list(types), surgeries=surgeries, blocks=blocks, T=T,
                    icu_beds=_beds(T, iw, ie, surgery_days), ward_beds=_beds(T, ww, we, surgery_days),
                    costs=costs or Costs(), name=name or f"{library}-{n_surgeries}-s{seed}", seed=seed,
                    notes="synthetic reconstruction; see README")
    return inst


def paper_instance(kind: str, n: int, beds: str = "(10,7)", cost: int = 1, seed: int = 1) -> Instance:
    """Instances shaped like the paper's R-n (6 general specialties, 10 ORs) and Case-n (cardiac, 5 ORs).
    Our sizes are far smaller because HiGHS replaces CPLEX; the *recipe* is the paper's."""
    c = Costs(c_o=26.0, c_icu=1000, c_ward=1000) if cost == 1 else Costs(c_o=100.0, c_icu=1000, c_ward=1000)
    if kind.lower().startswith("r"):
        return generate_instance(n, "general", seed, n_ors=max(1, round(n / 14)), beds=beds, costs=c, name=f"R-{n}")
    return generate_instance(n, "cardiac", seed, n_ors=max(1, round(n / 12)), beds=beds, costs=c, name=f"Case-{n}")
