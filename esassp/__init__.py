"""ESASSP — OR-to-downstream elective surgery planning under uncertainty (SP / DRO-M / DRO-W).

Implementation of Shehadeh, Tsang, Padman & Kilic, EJOR 329 (2026) 288-307.
"""
from .dro import solve_dro
from .evaluate import (Scenarios, evaluate, nominal_scenario, rel_tc, sample_scenarios)
from .generator import generate_instance, paper_instance
from .model import Block, Costs, Instance, Solution, Surgery, SurgeryType, mean_allocation
from .result import Params, Result
from .sp import solve_sp
from .validate import validate_solution


def solve(inst: Instance, model: str = "DRO-M", params: Params | None = None, downstream: bool = True,
          scheduling: bool = True) -> Result:
    """model: "SP" | "DRO-M" | "DRO-W".  ``downstream=False`` ignores ICU / ward capacity,
    ``scheduling=False`` ignores waiting times / start times (the paper's non-integrated baselines, §6.7)."""
    params = params or Params()
    if model == "SP":
        return solve_sp(inst, params, downstream=downstream, scheduling=scheduling)
    if model in ("DRO-M", "M"):
        return solve_dro(inst, "M", params, downstream, scheduling)
    if model in ("DRO-W", "W"):
        return solve_dro(inst, "W", params, downstream, scheduling)
    raise ValueError(model)
