"""Random instance generator following Section 6.1 of the paper.

Instance sets
-------------
A : durations ~ U[2h, 4h], 6 rooms per day (rooms not the bottleneck).
B : as A but the number of rooms is set by the room-sizing heuristic
    (online Appendix 6) -> rooms are restrictive.
C : short durations from a Pearson type III distribution (Fei et al. 2009),
    rooms set by the room-sizing heuristic.

Common: 5-day horizon, 8 surgeons, surgeon daily limits averaging ~320 min
(capped at the 8 h room day), deadlines ~ U{1..14} (deadline > 5 -> optional),
half of the cases infectious, cleaning (OCT) = 30 min, 5-minute slots.
The constructive heuristic is then used to postpone deadlines that cannot be
met, guaranteeing feasibility (as in the paper).
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

from .model import Instance, Surgery


def room_sizing_heuristic(durations_min, surgeons, surgeon_max_min, n_days, day_minutes):
    """Appendix 6: average daily working hours of each surgeon assuming the
    workload is spread evenly over the horizon -> rooms needed per day."""
    L = surgeon_max_min.shape[0]
    daily_need = 0.0
    for l in range(L):
        work = sum(d for d, s in zip(durations_min, surgeons) if s == l) / n_days
        daily_need += min(work, float(np.mean(surgeon_max_min[l])))
    return max(1, int(math.ceil(daily_need / day_minutes)))


def generate_instance(
    n_surgeries: int = 40,
    instance_set: str = "A",
    n_days: int = 5,
    n_surgeons: int = 8,
    rooms: Optional[int] = None,
    day_minutes: int = 480,
    slot_minutes: int = 5,
    cleaning_minutes: int = 30,
    infectious_share: float = 0.5,
    n_infection_types: int = 1,
    max_deadline: int = 14,
    surgeon_mean_minutes: float = 320.0,
    seed: int = 0,
    make_feasible: bool = True,
) -> Instance:
    rng = np.random.default_rng(seed)
    instance_set = instance_set.upper()

    # ---- durations (minutes, multiple of slot) ----------------------------
    if instance_set in ("A", "B"):
        dur = rng.integers(120 // slot_minutes, 240 // slot_minutes + 1, size=n_surgeries) * slot_minutes
    elif instance_set == "C":
        # Pearson III == shifted gamma; parameters chosen so that cases are
        # mostly < 2 h (mean ~ 75 min) as described in the paper.
        raw = 30 + rng.gamma(shape=2.0, scale=22.0, size=n_surgeries)
        dur = np.clip(np.round(raw / slot_minutes) * slot_minutes, 30, 240).astype(int)
    else:
        raise ValueError("instance_set must be A, B or C")

    surgeons = rng.integers(0, n_surgeons, size=n_surgeries)

    # ---- surgeon daily limits (minutes) -------------------------------------
    # mean ~320 min, sd ~ sqrt(20)*... -> capped to the room day; multiples of 30.
    A = rng.normal(surgeon_mean_minutes, 60.0, size=(n_surgeons, n_days))
    A = np.clip(np.round(A / 30.0) * 30.0, 120, day_minutes).astype(int)
    # make sure every surgeon can at least perform his/her longest case
    for l in range(n_surgeons):
        longest = max([d for d, s in zip(dur, surgeons) if s == l], default=0)
        A[l] = np.maximum(A[l], int(math.ceil(longest / 30.0) * 30))
    A = np.minimum(A, day_minutes)

    deadlines = rng.integers(1, max_deadline + 1, size=n_surgeries) - 1  # 0-based
    infect = np.zeros(n_surgeries, dtype=int)
    k = int(round(infectious_share * n_surgeries))
    idx = rng.choice(n_surgeries, size=k, replace=False)
    infect[idx] = rng.integers(1, n_infection_types + 1, size=k)

    if rooms is None:
        if instance_set == "A":
            rooms = 6
        else:
            rooms = room_sizing_heuristic(dur, surgeons, A, n_days, day_minutes)

    surg = [
        Surgery(id=i, duration=int(dur[i]) // slot_minutes, surgeon=int(surgeons[i]),
                deadline=int(deadlines[i]), infection=int(infect[i]),
                label=f"S{i + 1:03d}")
        for i in range(n_surgeries)
    ]
    inst = Instance(
        surgeries=surg, n_days=n_days, rooms=[int(rooms)] * n_days,
        slots_per_day=day_minutes // slot_minutes,
        surgeon_max=(A // slot_minutes).astype(int),
        cleaning=cleaning_minutes // slot_minutes, slot_minutes=slot_minutes,
        name=f"{instance_set}-{n_surgeries}-seed{seed}",
    )
    if make_feasible:
        from .heuristic import constructive_heuristic
        _, postponed = constructive_heuristic(inst, rng=np.random.default_rng(seed + 7919),
                                              postpone=True)
        if postponed:
            surg = [Surgery(s.id, s.duration, s.surgeon, postponed.get(s.id, s.deadline),
                            s.infection, s.label) for s in inst.surgeries]
            inst.surgeries = surg
    return inst
