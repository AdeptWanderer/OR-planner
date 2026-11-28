"""Core data structures for the ESASSP (OR-to-downstream elective surgery planning).

Notation follows Shehadeh, Tsang, Padman & Kilic (EJOR 329, 2026):
  I surgeries, B blocks (an OR on a given day, closed block scheduling), K_b positions per block,
  T planning days, Bed^icu_t / Bed^ward_t downstream capacities,
  d_i surgery duration (minutes), l^c_i / l^w_i post-operative LOS in ICU / ward (days).
Durations are continuous (minutes), LOS values are integers (days).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


@dataclass(frozen=True)
class SurgeryType:
    name: str
    d_mean: float; d_sd: float            # duration (min)
    c_mean: float; c_sd: float            # ICU LOS (days)
    w_mean: float; w_sd: float            # ward LOS (days)
    source: str = "synthetic"             # where the numbers come from (README honesty)


@dataclass(frozen=True)
class Surgery:
    id: int
    type: int                             # index into Instance.types
    label: str = ""
    priority: float = 1.0                 # multiplies first-stage penalties (higher = more urgent)
    # distribution summaries (filled by the generator / data import)
    mu_d: float = 0.0; sd_d: float = 0.0; lo_d: float = 0.0; hi_d: float = 0.0
    mu_c: float = 0.0; sd_c: float = 0.0; lo_c: int = 0; hi_c: int = 0
    mu_w: float = 0.0; sd_w: float = 0.0; lo_w: int = 0; hi_w: int = 0

    @property
    def klass(self) -> tuple:
        """Surgeries with the same class are interchangeable (symmetry breaking)."""
        return (self.type, self.priority, round(self.mu_d, 6), round(self.mu_c, 6), round(self.mu_w, 6),
                round(self.lo_d, 6), round(self.hi_d, 6), self.lo_c, self.hi_c, self.lo_w, self.hi_w)


@dataclass(frozen=True)
class Block:
    id: int
    day: int                              # t_b, 0-based day index in [0, T)
    room: int                             # OR index
    length: float                         # L_b, minutes
    K: int                                # max number of surgeries in the block
    types: Tuple[int, ...] = ()           # surgery types allowed (empty = any)


@dataclass(frozen=True)
class Costs:
    c_o: float = 26.0                     # overtime per minute (Cost 1)
    c_g: Optional[float] = None           # idle per minute (default c_o / 1.5)
    c_w: Optional[float] = None           # waiting per minute (default c_o / 1.5)
    c_icu: float = 1000.0                 # per ICU bed-day above capacity
    c_ward: float = 1000.0                # per ward bed-day above capacity
    alpha: float = 0.5                    # first-stage penalty scale  c_{i,b} = alpha * c_o
    postpone_factor: float = 1.5          # c_{i,b'} = postpone_factor * alpha * c_o

    @property
    def g(self) -> float: return self.c_o / 1.5 if self.c_g is None else self.c_g
    @property
    def w(self) -> float: return self.c_o / 1.5 if self.c_w is None else self.c_w
    @property
    def o(self) -> float: return self.c_o


@dataclass
class Instance:
    types: List[SurgeryType]
    surgeries: List[Surgery]
    blocks: List[Block]
    T: int
    icu_beds: np.ndarray                  # length T
    ward_beds: np.ndarray                 # length T
    costs: Costs = field(default_factory=Costs)
    name: str = "instance"
    seed: int = 0
    notes: str = ""

    # ---------------------------------------------------------------- sizes
    @property
    def I(self) -> int: return len(self.surgeries)
    @property
    def B(self) -> int: return len(self.blocks)

    # per-surgery vectors -----------------------------------------------------
    def _vec(self, attr: str, dtype=float) -> np.ndarray:
        return np.array([getattr(s, attr) for s in self.surgeries], dtype=dtype)

    @property
    def mu_d(self): return self._vec("mu_d")
    @property
    def lo_d(self): return self._vec("lo_d")
    @property
    def hi_d(self): return self._vec("hi_d")
    @property
    def mu_c(self): return self._vec("mu_c")
    @property
    def mu_w(self): return self._vec("mu_w")
    @property
    def lo_c(self): return self._vec("lo_c", int)
    @property
    def hi_c(self): return self._vec("hi_c", int)
    @property
    def lo_w(self): return self._vec("lo_w", int)
    @property
    def hi_w(self): return self._vec("hi_w", int)

    def eligible_blocks(self, i: int) -> List[int]:
        t = self.surgeries[i].type
        return [b.id for b in self.blocks if (not b.types) or t in b.types]

    def first_stage_cost(self, i: int, postponed: bool) -> float:
        s = self.surgeries[i]
        base = self.costs.alpha * self.costs.o * s.priority
        return base * (self.costs.postpone_factor if postponed else 1.0)

    @property
    def T_ext(self) -> int:
        """Timeline long enough for every stay that starts inside the horizon to finish."""
        return self.T + int(max((s.hi_c + s.hi_w for s in self.surgeries), default=0)) + 1

    def day_name(self, t: int) -> str:
        return DAY_NAMES[t % 7] if self.T <= 7 else f"D{t + 1}"

    def block_label(self, b: int) -> str:
        blk = self.blocks[b]
        return f"OR{blk.room + 1}·{self.day_name(blk.day)}"

    def summary(self) -> Dict[str, float]:
        return dict(I=self.I, B=self.B, T=self.T,
                    mean_total_duration=float(self.mu_d.sum()),
                    block_minutes=float(sum(b.length for b in self.blocks)),
                    icu_beds=float(self.icu_beds.sum()), ward_beds=float(self.ward_beds.sum()))


@dataclass
class Solution:
    """First-stage decisions.  ``block[i]`` is -1 for a postponed surgery."""
    block: np.ndarray                     # (I,) int
    pos: np.ndarray                       # (I,) int, 0-based position inside block (-1 if postponed)
    start: Dict[Tuple[int, int], float]   # s[b, k]

    def order(self, b: int) -> List[int]:
        idx = np.where(self.block == b)[0]
        return [int(i) for i in idx[np.argsort(self.pos[idx])]]

    @property
    def scheduled(self) -> np.ndarray:
        return self.block >= 0

    def start_of(self, i: int) -> float:
        return self.start[(int(self.block[i]), int(self.pos[i]))]

    def copy(self) -> "Solution":
        return Solution(self.block.copy(), self.pos.copy(), dict(self.start))

    def n_scheduled(self) -> int:
        return int(self.scheduled.sum())


def mean_allocation(inst: Instance, sol: Solution) -> Solution:
    """Start times = cumulative *mean* durations in the sequence (what hospitals do without a scheduling model)."""
    out = sol.copy()
    for b in range(inst.B):
        t = 0.0
        for k, i in enumerate(sol.order(b)):
            out.start[(b, k)] = t
            t += inst.surgeries[i].mu_d
    return out
