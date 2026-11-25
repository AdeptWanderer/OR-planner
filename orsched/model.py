"""Core data structures.

Time is discretised into slots of ``slot_minutes`` minutes (paper: 5 min).
All durations / limits inside an :class:`Instance` are expressed in slots.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class Surgery:
    id: int                 # 0..n-1 (index into Instance.surgeries)
    duration: int           # t_i, slots
    surgeon: int            # s_i, 0..L-1
    deadline: int           # dd_i, 0-based day index; >= n_days means optional
    infection: int = 0      # 0 = non-infectious, k>0 = infection type k
    label: str = ""         # free text (patient / procedure reference)


@dataclass
class Instance:
    surgeries: List[Surgery]
    n_days: int
    rooms: List[int]                    # |K_d| per day
    slots_per_day: int                  # |T_d| (same for every day)
    surgeon_max: np.ndarray             # A_ld, shape (L, D), slots
    cleaning: int                       # OCT, slots
    slot_minutes: int = 5
    surgeon_names: Optional[List[str]] = None
    day_names: Optional[List[str]] = None
    name: str = "instance"

    # ------------------------------------------------------------------ sets
    @property
    def n(self) -> int:
        return len(self.surgeries)

    @property
    def n_surgeons(self) -> int:
        return int(self.surgeon_max.shape[0])

    def is_mandatory(self, i: int) -> bool:
        return self.surgeries[i].deadline < self.n_days

    @property
    def mandatory(self) -> List[int]:
        return [s.id for s in self.surgeries if s.deadline < self.n_days]

    @property
    def optional(self) -> List[int]:
        return [s.id for s in self.surgeries if s.deadline >= self.n_days]

    def eligible(self, d: int) -> List[int]:
        """Surgeries that may be performed on day d (dd_i >= d)."""
        return [s.id for s in self.surgeries if s.deadline >= d]

    def by_surgeon(self, l: int) -> List[int]:
        return [s.id for s in self.surgeries if s.surgeon == l]

    def clean(self, i: Optional[int], j: Optional[int]) -> int:
        """CL_{ij}: cleaning needed between consecutive cases i -> j.

        Cleaning is required when the infection type changes (switching to a
        different infection type, or from infectious to non-infectious and
        vice-versa).  No cleaning before the first / after the last case.
        """
        if i is None or j is None:
            return 0
        return self.cleaning if self.surgeries[i].infection != self.surgeries[j].infection else 0

    def surgeon_name(self, l: int) -> str:
        if self.surgeon_names and l < len(self.surgeon_names):
            return self.surgeon_names[l]
        return f"Surgeon {l + 1}"

    def day_name(self, d: int) -> str:
        if self.day_names and d < len(self.day_names):
            return self.day_names[d]
        return ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][d % 7] if self.n_days <= 7 else f"Day {d + 1}"

    def minutes(self, slots: float) -> float:
        return slots * self.slot_minutes


@dataclass(frozen=True)
class Column:
    """A feasible schedule for one operating room on one day.

    ``items`` is a tuple of (surgery, start_slot) sorted by start.
    """
    day: int
    items: Tuple[Tuple[int, int], ...]

    def surgeries(self) -> List[int]:
        return [i for i, _ in self.items]

    def value(self, inst: Instance) -> int:
        """B_j: total scheduled surgery time (slots)."""
        return sum(inst.surgeries[i].duration for i, _ in self.items)

    def key(self):
        return (self.day, self.items)


@dataclass
class Solution:
    """An integer solution: list of room-day schedules (one per used room)."""
    columns: List[Column]
    objective: float                    # total scheduled minutes
    upper_bound: float                  # minutes
    lower_bound: float                  # minutes (== objective of best integer)
    stats: Dict = field(default_factory=dict)

    @property
    def gap(self) -> float:
        if self.upper_bound <= 0:
            return 0.0
        return 100.0 * (self.upper_bound - self.lower_bound) / self.upper_bound

    def assignment(self) -> Dict[int, Tuple[int, int, int]]:
        """surgery -> (day, room_index_within_day, start_slot)."""
        out = {}
        room_counter: Dict[int, int] = {}
        for col in sorted(self.columns, key=lambda c: (c.day, c.items[0][1] if c.items else 0, c.items)):
            k = room_counter.get(col.day, 0)
            room_counter[col.day] = k + 1
            for i, s in col.items:
                out[i] = (col.day, k, s)
        return out
