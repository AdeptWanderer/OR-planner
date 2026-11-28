from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .model import Solution


@dataclass
class Params:
    n_scenarios: int = 50                 # N (SAA sample / Wasserstein empirical size)
    time_limit: float = 120.0             # total wall-clock budget (s)
    master_time: float = 60.0             # per master solve (s)
    gap: float = 1e-3                     # C&CG / MIP relative gap
    threads: int = 4
    seed: int = 1
    symmetry: bool = True                 # SB constraints (27)-(29)
    multicut: bool = True                 # per-block cuts (24)/(25)
    rho_bound: bool = True                # valid inequality (26)
    eps_d: float = 50.0                   # Wasserstein radius for durations
    eps_l: float = 5.0                    # Wasserstein radius for LOS
    sub_method: str = "auto"              # "enum" | "milp" | "auto" for the BC sub-problem
    max_iter: int = 200
    log: Any = None                       # callable(str)


@dataclass
class Result:
    model: str                            # "SP" | "DRO-M" | "DRO-W"
    solution: Solution
    objective: float                      # model objective at the returned solution (UB for DRO)
    lower_bound: float
    gap: float
    time: float
    status: str
    integrate_downstream: bool = True
    integrate_scheduling: bool = True
    history: List[Dict[str, float]] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)
