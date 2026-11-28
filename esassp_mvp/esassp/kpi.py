"""Tables and KPIs used by the dashboard, README benchmarks and the static web export."""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .evaluate import Scenarios, bc_cost, block_costs, evaluate, nominal_scenario
from .model import Instance, Solution


def schedule_df(inst: Instance, sol: Solution) -> pd.DataFrame:
    rows = []
    for b, blk in enumerate(inst.blocks):
        t_end = 0.0
        for k, i in enumerate(sol.order(b)):
            s = inst.surgeries[i]
            start = sol.start[(b, k)]
            rows.append(dict(block=inst.block_label(b), day=inst.day_name(blk.day), room=f"OR{blk.room + 1}", position=k + 1,
                             surgery=s.label or f"S{i}", type=inst.types[s.type].name,
                             scheduled_start=round(start, 1), expected_duration=round(s.mu_d, 1),
                             planned_end=round(start + s.mu_d, 1),
                             icu_days=round(s.mu_c, 1), ward_days=round(s.mu_w, 1)))
    return pd.DataFrame(rows)


def postponed_df(inst: Instance, sol: Solution) -> pd.DataFrame:
    rows = [dict(surgery=inst.surgeries[i].label or f"S{i}", type=inst.types[inst.surgeries[i].type].name,
                 mu_duration=round(inst.surgeries[i].mu_d, 1), mu_icu=round(inst.surgeries[i].mu_c, 1),
                 mu_ward=round(inst.surgeries[i].mu_w, 1)) for i in range(inst.I) if sol.block[i] < 0]
    return pd.DataFrame(rows)


def block_df(inst: Instance, sol: Solution, sc: Scenarios) -> pd.DataFrame:
    """Per block expected wait / overtime / idle under ``sc``."""
    rows = []
    for b, blk in enumerate(inst.blocks):
        r = block_costs(inst, sol, sc.d, b)
        rows.append(dict(block=inst.block_label(b), n=len(sol.order(b)),
                         planned_min=round(sum(inst.surgeries[i].mu_d for i in sol.order(b)), 1),
                         exp_wait=round(float(r["wait"].sum(1).mean()), 1), exp_overtime=round(float(r["over"].mean()), 1),
                         exp_idle=round(float(r["idle"].mean()), 1),
                         p_overtime=round(float((r["over"] > 1e-9).mean()), 3)))
    return pd.DataFrame(rows)


def beds_df(inst: Instance, ev: Dict[str, object]) -> pd.DataFrame:
    T = inst.T
    return pd.DataFrame(dict(day=[inst.day_name(t) for t in range(T)],
                             icu_beds=inst.icu_beds, ward_beds=inst.ward_beds,
                             icu_mean_need=np.asarray(ev["icu_need"]).mean(0).round(2),
                             ward_mean_need=np.asarray(ev["ward_need"]).mean(0).round(2),
                             icu_PE=np.asarray(ev["pe_icu"]).round(3), ward_PE=np.asarray(ev["pe_ward"]).round(3),
                             icu_ME=np.asarray(ev["me_icu"]).round(2), ward_ME=np.asarray(ev["me_ward"]).round(2)))


def summary(inst: Instance, sol: Solution, ev: Dict[str, object]) -> Dict[str, float]:
    over = np.asarray(ev["over"]); idle = np.asarray(ev["idle"]); wait = np.asarray(ev["wait"])
    return dict(scheduled=int(sol.n_scheduled()), postponed=int(inst.I - sol.n_scheduled()),
                mean_total_cost=float(ev["mean_total"]), q90_total_cost=float(ev["q90_total"]),
                mean_bc_cost=float(ev["mean_bc"]), mean_down_cost=float(ev["mean_down"]),
                first_stage_cost=float(ev["first_stage"]),
                mean_wait_min=float(wait.mean()), mean_overtime_min=float(over.mean()), mean_idle_min=float(idle.mean()),
                max_pe_icu=float(np.max(ev["pe_icu"])), max_pe_ward=float(np.max(ev["pe_ward"])),
                mean_pe_icu=float(np.mean(ev["pe_icu"])), mean_pe_ward=float(np.mean(ev["pe_ward"])),
                max_me_icu=float(np.max(ev["me_icu"])), max_me_ward=float(np.max(ev["me_ward"])))


def comparison_df(inst: Instance, runs: Dict[str, object], sc: Scenarios) -> pd.DataFrame:
    """runs: label -> Result.  One row per model, all evaluated on the same out-of-sample scenarios."""
    rows = []
    for name, r in runs.items():
        ev = evaluate(inst, r.solution, sc)
        s = summary(inst, r.solution, ev)
        rows.append(dict(model=name, **{k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items()},
                         solve_s=round(r.time, 1), gap_pct=round(100 * r.gap, 2)))
    return pd.DataFrame(rows)
