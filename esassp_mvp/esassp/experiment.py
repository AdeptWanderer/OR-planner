"""One-instance experiment in the spirit of the paper's §6: solve SP / DRO-M / DRO-W (and the non-integrated
variants), then evaluate every first-stage solution out of sample under the in-sample and mis-specified
distributions.  Produces a JSON-serialisable record used by the benchmarks, the dashboard and the web artifact."""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import numpy as np

from . import solve
from .evaluate import evaluate, rel_tc, sample_scenarios
from .kpi import summary
from .model import Instance
from .result import Params, Result

DELTAS = (0.0, 0.2, 0.5)


def instance_record(inst: Instance) -> Dict[str, Any]:
    return dict(
        name=inst.name, T=inst.T, icu_beds=inst.icu_beds.tolist(), ward_beds=inst.ward_beds.tolist(),
        costs=dict(o=inst.costs.o, g=inst.costs.g, w=inst.costs.w, icu=inst.costs.c_icu, ward=inst.costs.c_ward),
        types=[t.name for t in inst.types],
        surgeries=[dict(id=s.id, label=s.label, type=s.type, mu_d=round(s.mu_d, 1), lo_d=s.lo_d, hi_d=s.hi_d,
                        mu_c=round(s.mu_c, 2), lo_c=s.lo_c, hi_c=s.hi_c, mu_w=round(s.mu_w, 2), lo_w=s.lo_w, hi_w=s.hi_w)
                   for s in inst.surgeries],
        blocks=[dict(id=b.id, day=b.day, room=b.room, L=b.length, K=b.K, types=list(b.types)) for b in inst.blocks])


def solution_record(inst: Instance, r: Result) -> Dict[str, Any]:
    sol = r.solution
    return dict(model=r.model, downstream=r.integrate_downstream, scheduling=r.integrate_scheduling,
                objective=r.objective, lb=r.lower_bound, gap=r.gap, time=r.time, status=r.status,
                scheduled=int(sol.n_scheduled()),
                plan=[[int(i), int(sol.block[i]), int(sol.pos[i]), round(float(sol.start_of(i)), 1)] for i in range(inst.I)
                      if sol.block[i] >= 0],
                postponed=[int(i) for i in range(inst.I) if sol.block[i] < 0],
                history=[{k: (round(v, 4) if isinstance(v, float) else v) for k, v in h.items()} for h in r.history][-60:],
                stats={k: v for k, v in r.stats.items() if isinstance(v, (int, float, str))})


def eval_record(inst: Instance, r: Result, scen: Dict[float, Any], sp_tc: Optional[Dict[float, np.ndarray]] = None) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for dl, sc in scen.items():
        ev = evaluate(inst, r.solution, sc)
        s = summary(inst, r.solution, ev)
        rec = dict(s)
        rec.update(pe_icu=np.asarray(ev["pe_icu"]).round(4).tolist(), pe_ward=np.asarray(ev["pe_ward"]).round(4).tolist(),
                   me_icu=np.asarray(ev["me_icu"]).round(3).tolist(), me_ward=np.asarray(ev["me_ward"]).round(3).tolist(),
                   icu_need=np.asarray(ev["icu_need"]).mean(0).round(3).tolist(),
                   icu_need_q20=np.quantile(ev["icu_need"], 0.2, axis=0).round(2).tolist(),
                   icu_need_q80=np.quantile(ev["icu_need"], 0.8, axis=0).round(2).tolist(),
                   ward_need=np.asarray(ev["ward_need"]).mean(0).round(3).tolist(),
                   ward_need_q20=np.quantile(ev["ward_need"], 0.2, axis=0).round(2).tolist(),
                   ward_need_q80=np.quantile(ev["ward_need"], 0.8, axis=0).round(2).tolist(),
                   q75_total=float(ev["q75_total"]))
        wait = np.asarray(ev["wait"]); idle = np.asarray(ev["idle"])
        h, edges = np.histogram(wait, bins=24)
        rec["wait_hist"] = dict(counts=h.tolist(), edges=edges.round(1).tolist())
        if sp_tc is not None and dl in sp_tc:
            rec["rel_tc_vs_sp"] = rel_tc(sp_tc[dl], ev["total"])
        out[str(dl)] = rec
        out[str(dl)]["_tc"] = ev["total"]          # stripped before JSON dump by the caller
    return out


def run_experiment(inst: Instance, params: Params, models=("SP", "DRO-M", "DRO-W"), baselines: bool = True,
                   n_oos: int = 5000, seed: int = 123, log=print, deltas=DELTAS,
                   baseline_time_limit: Optional[float] = None) -> Dict[str, Any]:
    t0 = time.time()
    scen = {dl: sample_scenarios(inst, n_oos, np.random.default_rng(seed + int(dl * 100)), delta=dl) for dl in deltas}
    runs: Dict[str, Result] = {}
    for m in models:
        log(f"  solving {m} ...")
        runs[m] = solve(inst, m, params)
        log(f"    {m}: obj {runs[m].objective:.1f} LB {runs[m].lower_bound:.1f} gap {100 * runs[m].gap:.2f}% "
            f"{runs[m].time:.1f}s scheduled {runs[m].solution.n_scheduled()}")
    if baselines:
        import dataclasses
        bp = dataclasses.replace(params, time_limit=min(params.time_limit, baseline_time_limit or params.time_limit),
                                 master_time=min(params.master_time, baseline_time_limit or params.master_time))
        for m in [x for x in ("SP", "DRO-M") if x in models]:
            for tag, kw in (("w/o downstream", dict(downstream=False)), ("w/o scheduling", dict(scheduling=False))):
                log(f"  solving {m} {tag} ...")
                runs[f"{m} {tag}"] = solve(inst, m, bp, **kw)
    ev: Dict[str, Any] = {}
    tc_sp = None
    if "SP" in runs:
        tc_sp = {dl: evaluate(inst, runs["SP"].solution, scen[dl])["total"] for dl in scen}
    for name, r in runs.items():
        ev[name] = eval_record(inst, r, scen, tc_sp)
    tcs = {name: {k: v.pop("_tc") for k, v in d.items()} for name, d in ev.items()}
    # RelTC of the non-integrated variants vs their integrated counterpart (§6.7)
    for name in list(ev):
        base = name.split(" w/o")[0]
        if " w/o" in name and base in tcs:
            for dl in tcs[name]:
                ev[name][dl]["rel_tc_vs_integrated"] = rel_tc(tcs[name][dl], tcs[base][dl])
    return dict(instance=instance_record(inst), runs={k: solution_record(inst, r) for k, r in runs.items()}, eval=ev,
                params=dict(N=params.n_scenarios, eps_d=params.eps_d, eps_l=params.eps_l, time_limit=params.time_limit,
                            gap=params.gap, n_oos=n_oos), wall=time.time() - t0)
