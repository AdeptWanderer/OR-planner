"""Distributionally robust ESASSP: column-and-constraint generation (Algorithm 1) for DRO-M (mean-support ambiguity,
eq. (17)-(18)) and DRO-W (1-Wasserstein ambiguity, eq. (20)-(22)).

Master (relaxation, scenario subset O):
  min  c.x + mu^d.psi + mu^c.phi_ + mu^w.varpi + dBC + ddown
  s.t. (x,s) in X,  McCormick links psi = rho x, phi_ = phi sigma, varpi = zeta sigma  (18c)-(18e)
       for every scenario r:  recourse copy (4b)-(4e)/(5b)-(5e) and the cuts
         dBC_b   >= Q_b(x,s,d^r)   - sum_i d^r_i psi_ib          (24, multi-cut)
         ddown   >= Q^down(x,l^r)  - sum_i (l^c_i phi_i + l^w_i varpi_i)
Separation: exact worst case for the current (x, s, rho, phi, zeta) — see ``subproblems``.
Termination: (UB - LB)/UB <= gap, with LB = dual bound of the master and UB from the true sub-problem values.

DRO-W (derived here: Appendix D with the algorithm is not part of the paper PDF) uses one multiplier nu for the
duration ball and one multiplier phi_W for the LOS ball, and for every empirical sample n and scenario r the cut
         dBC_{n,b} >= Q_b(d^r) - nu * sum_{i in b} |d^r_i - d^n_i|,
         ddown_n   >= Q^down(l^r) - phi_W * sum_{i sched} (|l^c,r_i - l^c,n_i| + |l^w,r_i - l^w,n_i|).
Recourse copies are shared by all n, so the master grows with the number of scenarios, not with N * scenarios.
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from .evaluate import Scenarios, nominal_scenario, sample_scenarios, first_stage_cost
from .formulations import FirstStage, add_bc_recourse, add_down_recourse
from .milp import INF, Model
from .model import Instance, Solution, mean_allocation
from .result import Params, Result
from .subproblems import bc_sub, down_sub

Expr = Tuple[float, Dict[int, float]]


def _mccormick(m: Model, P: int, base: int, lo: float, hi: float, A: Expr):
    """P = base * A  with A in [0,1] a linear expression and base in [lo, hi]."""
    c0, terms = A
    def row(coefs, const_lo=None, **kw):
        m.add(coefs, **kw)
    t = {P: 1.0}
    for j, v in terms.items(): t[j] = t.get(j, 0.0) - lo * v
    m.add(t, lo=lo * c0)                                       # P >= lo A
    t = {P: 1.0}
    for j, v in terms.items(): t[j] = t.get(j, 0.0) - hi * v
    m.add(t, hi=hi * c0)                                       # P <= hi A
    t = {P: 1.0, base: -1.0}
    for j, v in terms.items(): t[j] = t.get(j, 0.0) - hi * v
    m.add(t, lo=-hi + hi * c0)                                 # P >= base - hi (1 - A)
    t = {P: 1.0, base: -1.0}
    for j, v in terms.items(): t[j] = t.get(j, 0.0) - lo * v
    m.add(t, hi=-lo + lo * c0)                                 # P <= base - lo (1 - A)


class DROMaster:
    def __init__(self, inst: Instance, params: Params, kind: str, downstream: bool = True, scheduling: bool = True,
                 fixed: Optional[Solution] = None, empirical: Optional[Scenarios] = None):
        assert kind in ("M", "W")
        self.inst, self.p, self.kind = inst, params, kind
        self.downstream, self.scheduling = downstream, scheduling
        c = inst.costs
        self.cw = c.w if scheduling else 0.0
        Kmax = max(b.K for b in inst.blocks)
        self.rho_lo = -c.g
        self.rho_hi = 2 * c.o + c.g + self.cw * Kmax
        self.phi_lo, self.phi_hi = -c.c_ward, c.c_icu + c.c_ward
        self.zeta_lo, self.zeta_hi = 0.0, c.c_ward
        m = self.m = Model()
        self.fs = fs = FirstStage(m, inst, symmetry=params.symmetry and fixed is None)
        if fixed is not None:
            for i in range(inst.I):
                bb = int(fixed.block[i])
                for (ii, b, k), j in fs.x.items():
                    if ii == i:
                        m.fix(j, 1.0 if (b == bb and k == int(fixed.pos[i])) else 0.0)
                m.fix(fs.xp[i], 1.0 if bb < 0 else 0.0)
            for (b, k), j in fs.s.items():
                m.fix(j, fixed.start.get((b, k), 0.0))
        self.sc_d: List[np.ndarray] = []; self.sc_l: List[Tuple[np.ndarray, np.ndarray]] = []
        sig = {i: (1.0, {j: -1.0 for j in [fs.xp[i]]}) for i in range(inst.I)}     # sigma_i = 1 - xp_i
        self.sigma = sig
        big_d = float(inst.hi_d.sum()) * max(abs(self.rho_lo), self.rho_hi) + 1.0
        big_l = float((inst.hi_c + inst.hi_w).sum()) * max(abs(self.phi_lo), self.phi_hi) + 1.0
        if kind == "M":
            self.rho = [m.var(self.rho_lo, self.rho_hi) for _ in range(inst.I)]
            self.psi: Dict[Tuple[int, int], int] = {}
            for i in range(inst.I):
                for b in fs.elig[i]:
                    P = m.var(min(self.rho_lo, 0.0), max(self.rho_hi, 0.0), inst.surgeries[i].mu_d)
                    self.psi[(i, b)] = P
                    _mccormick(m, P, self.rho[i], self.rho_lo, self.rho_hi, (0.0, fs.a(i, b)))
                if params.rho_bound:                                               # valid inequality (26)
                    t = {self.rho[i]: 1.0}
                    for b in fs.elig[i]:
                        for k in range(inst.blocks[b].K):
                            t[fs.x[(i, b, k)]] = -(c.g + 2 * c.o + (inst.blocks[b].K - k) * self.cw)
                    m.add(t, hi=0.0)
            if downstream:
                self.phi = [m.var(self.phi_lo, self.phi_hi) for _ in range(inst.I)]
                self.zeta = [m.var(self.zeta_lo, self.zeta_hi) for _ in range(inst.I)]
                self.phis, self.zetas = [], []
                for i in range(inst.I):
                    P1 = m.var(min(self.phi_lo, 0.0), max(self.phi_hi, 0.0), inst.surgeries[i].mu_c)
                    P2 = m.var(0.0, self.zeta_hi, inst.surgeries[i].mu_w)
                    _mccormick(m, P1, self.phi[i], self.phi_lo, self.phi_hi, sig[i])
                    _mccormick(m, P2, self.zeta[i], self.zeta_lo, self.zeta_hi, sig[i])
                    self.phis.append(P1); self.zetas.append(P2)
            self.dbc = [m.var(-big_d, INF, 1.0) for _ in range(inst.B)] if params.multicut else [m.var(-big_d * inst.B, INF, 1.0)]
            self.ddown = m.var(-big_l, INF, 1.0) if downstream else None
        else:
            assert empirical is not None
            self.emp = empirical
            N = empirical.n
            self.nu_hi = self.rho_hi
            self.phiw_hi = c.c_icu + c.c_ward
            self.nu = m.var(0.0, self.nu_hi, params.eps_d)
            self.kappa: Dict[Tuple[int, int], int] = {}
            for i in range(inst.I):
                for b in fs.elig[i]:
                    P = m.var(0.0, self.nu_hi)
                    self.kappa[(i, b)] = P
                    _mccormick(m, P, self.nu, 0.0, self.nu_hi, (0.0, fs.a(i, b)))
            if downstream:
                self.phiw = m.var(0.0, self.phiw_hi, params.eps_l)
                self.pi = []
                for i in range(inst.I):
                    P = m.var(0.0, self.phiw_hi)
                    _mccormick(m, P, self.phiw, 0.0, self.phiw_hi, sig[i])
                    self.pi.append(P)
            lbd = -float(inst.hi_d.sum()) * self.nu_hi - 1.0
            self.dbc_n = [[m.var(lbd, INF, 1.0 / N) for _ in range(inst.B)] for _ in range(N)]
            self.ddown_n = [m.var(-float((inst.hi_c + inst.hi_w).sum()) * self.phiw_hi - 1.0, INF, 1.0 / N) for _ in range(N)] \
                if downstream else None

    # ------------------------------------------------------------------ scenarios
    def add_bc_scenario(self, d: np.ndarray):
        inst, m, fs = self.inst, self.m, self.fs
        self.sc_d.append(d.copy())
        cp = add_bc_recourse(fs, d, 0.0, c_w=self.cw)
        if self.kind == "M":
            cuts = []
            for b in range(inst.B):
                t = dict(cp.cost_terms[b])
                for i in range(inst.I):
                    if (i, b) in self.psi:
                        t[self.psi[(i, b)]] = t.get(self.psi[(i, b)], 0.0) - float(d[i])
                cuts.append(t)
            if self.p.multicut:
                for b in range(inst.B):
                    t = cuts[b]; t[self.dbc[b]] = t.get(self.dbc[b], 0.0) - 1.0
                    m.add(t, hi=0.0)
            else:
                t: Dict[int, float] = {}
                for cb in cuts:
                    for j, v in cb.items(): t[j] = t.get(j, 0.0) + v
                t[self.dbc[0]] = -1.0
                m.add(t, hi=0.0)
        else:
            N = self.emp.n
            for n in range(N):
                for b in range(inst.B):
                    t = dict(cp.cost_terms[b])
                    for i in range(inst.I):
                        if (i, b) in self.kappa:
                            dev = abs(float(d[i]) - float(self.emp.d[n, i]))
                            if dev > 0:
                                t[self.kappa[(i, b)]] = t.get(self.kappa[(i, b)], 0.0) - dev
                    t[self.dbc_n[n][b]] = t.get(self.dbc_n[n][b], 0.0) - 1.0
                    m.add(t, hi=0.0)

    def add_down_scenario(self, lc: np.ndarray, lw: np.ndarray):
        inst, m, fs = self.inst, self.m, self.fs
        self.sc_l.append((lc.copy(), lw.copy()))
        cp = add_down_recourse(fs, lc, lw, 0.0)
        if self.kind == "M":
            t = dict(cp.cost_terms)
            for i in range(inst.I):
                t[self.phis[i]] = t.get(self.phis[i], 0.0) - float(lc[i])
                t[self.zetas[i]] = t.get(self.zetas[i], 0.0) - float(lw[i])
            t[self.ddown] = -1.0
            m.add(t, hi=0.0)
        else:
            for n in range(self.emp.n):
                t = dict(cp.cost_terms)
                for i in range(inst.I):
                    dev = abs(int(lc[i]) - int(self.emp.lc[n, i])) + abs(int(lw[i]) - int(self.emp.lw[n, i]))
                    if dev:
                        t[self.pi[i]] = t.get(self.pi[i], 0.0) - dev
                t[self.ddown_n[n]] = t.get(self.ddown_n[n], 0.0) - 1.0
                m.add(t, hi=0.0)

    # ------------------------------------------------------------------ master solve
    def solve(self, time_limit: float):
        res = self.m.solve(time_limit=time_limit, gap=(self.p.master_gap if self.p.master_gap is not None else min(self.p.gap / 4, 1e-3)), threads=self.p.threads)   # no warm start: a partial hint was accepted by HiGHS as an (invalid) incumbent
        if not res.ok:
            raise RuntimeError(f"master failed: {res.status}")
        sol = self.fs.extract(res.x)
        return res, sol


def solve_dro(inst: Instance, kind: str = "M", params: Params = Params(), downstream: bool = True,
              scheduling: bool = True, fixed: Optional[Solution] = None, empirical: Optional[Scenarios] = None) -> Result:
    """Run C&CG.  kind = "M" (mean-support) or "W" (Wasserstein)."""
    t0 = time.time()
    log = params.log or (lambda s: None)
    if kind == "W" and empirical is None:
        empirical = sample_scenarios(inst, params.n_scenarios, np.random.default_rng(params.seed))
    M = DROMaster(inst, params, kind, downstream, scheduling, fixed, empirical)
    nom = nominal_scenario(inst)
    M.add_bc_scenario(nom.d[0])
    if downstream:
        M.add_down_scenario(nom.lc[0], nom.lw[0])
    LB, UB = -np.inf, np.inf
    best: Optional[Solution] = None
    hist: List[Dict[str, float]] = []
    status = "max_iter"
    c = inst.costs
    mu_c, mu_w = inst.mu_c, inst.mu_w
    method = params.sub_method
    t_sub_total = t_master_total = 0.0
    for it in range(1, params.max_iter + 1):
        remaining = params.time_limit - (time.time() - t0)
        if remaining <= 0.5:
            status = "time_limit"; break
        tm = time.time()
        res, sol = M.solve(min(params.master_time, remaining))
        t_master = time.time() - tm
        t_master_total += t_master
        LB = max(LB, res.bound)
        x = res.x
        ts = time.time()
        # ---------------- separation
        new_d: List[Tuple[float, np.ndarray]] = []       # (violation, scenario)
        new_l: List[Tuple[float, np.ndarray, np.ndarray]] = []
        if kind == "M":
            rho = np.array([x[j] for j in M.rho])
            sumV = 0.0; sumdelta = 0.0
            dstar = inst.mu_d.copy()
            viol_bc = 0.0
            for b in range(inst.B):
                v, d = bc_sub(inst, sol, b, rho=rho, method=method)
                dl = x[M.dbc[b]] if params.multicut else None
                sumV += v
                if params.multicut:
                    sumdelta += dl; viol_bc += max(0.0, v - dl)
                msk = ~np.isnan(d)
                dstar[msk] = d[msk]
            if not params.multicut:
                sumdelta = x[M.dbc[0]]; viol_bc = max(0.0, sumV - sumdelta)
            V_down = 0.0; d_down = 0.0; viol_dn = 0.0; lcs = lws = None
            if downstream:
                phi = np.array([x[j] for j in M.phi]); zeta = np.array([x[j] for j in M.zeta])
                V_down, lcs, lws = down_sub(inst, sol, phi=phi, zeta=zeta)
                d_down = x[M.ddown]; viol_dn = max(0.0, V_down - d_down)
            ub_true = res.obj - sumdelta - d_down + sumV + V_down
            nv = (viol_bc, dstar)
            if viol_bc > 0:
                new_d.append((viol_bc, dstar))
            if viol_dn > 0:
                new_l.append((viol_dn, lcs, lws))
        else:
            N = empirical.n
            nu = float(x[M.nu]); phiw = float(x[M.phiw]) if downstream else 0.0
            tot_extra = 0.0; tot_delta = 0.0
            for n in range(N):
                dstar = empirical.d[n].copy(); sumV = 0.0; sumdelta = 0.0
                for b in range(inst.B):
                    v, d = bc_sub(inst, sol, b, nu=nu, dn=empirical.d[n], method=method)
                    sumV += v; sumdelta += x[M.dbc_n[n][b]]
                    msk = ~np.isnan(d); dstar[msk] = d[msk]
                tot_extra += sumV / N; tot_delta += sumdelta / N
                if sumV > sumdelta + 1e-6:
                    new_d.append((sumV - sumdelta, dstar))
                if downstream:
                    v, lcs, lws = down_sub(inst, sol, phiW=phiw, lcn=empirical.lc[n], lwn=empirical.lw[n])
                    dn_ = x[M.ddown_n[n]]
                    tot_extra += v / N; tot_delta += dn_ / N
                    if v > dn_ + 1e-6:
                        new_l.append((v - dn_, lcs, lws))
            ub_true = res.obj - tot_delta + tot_extra
        t_sub_total += time.time() - ts
        if ub_true < UB:
            UB, best = ub_true, sol.copy()
        gap = (UB - LB) / max(abs(UB), 1e-9)
        hist.append(dict(iter=it, lb=float(LB), ub=float(UB), gap=float(gap), master_obj=float(res.obj),
                         n_bc=len(M.sc_d), n_down=len(M.sc_l), t_master=t_master, t_sub=time.time() - ts,
                         t_total=time.time() - t0, rows=M.m.n_rows, cols=M.m.n_vars))
        log(f"[{kind}] it {it:3d}  LB {LB:12.2f}  UB {UB:12.2f}  gap {100 * gap:6.3f}%  "
            f"|O|={len(M.sc_d)}/{len(M.sc_l)}  master {t_master:5.1f}s sub {time.time() - ts:5.1f}s")
        if gap <= params.gap:
            status = "converged"; break
        # ---------------- add scenarios
        def dedup_d(d):
            return any(np.allclose(d, e) for e in M.sc_d)
        def dedup_l(lc, lw):
            return any(np.array_equal(lc, a) and np.array_equal(lw, b) for a, b in M.sc_l)
        added = 0
        cap = 1 if kind == "M" else max(2, (empirical.n // 3))
        for v, d in sorted(new_d, key=lambda z: -z[0])[:cap]:
            if not dedup_d(d):
                M.add_bc_scenario(d); added += 1
        for v, lc, lw in sorted(new_l, key=lambda z: -z[0])[:cap]:
            if not dedup_l(lc, lw):
                M.add_down_scenario(lc, lw); added += 1
        if added == 0:
            status = "stalled"       # separation returns known scenarios: numerically converged
            LB = max(LB, UB - 1e-6 * abs(UB)) if (UB - LB) / max(abs(UB), 1e-9) < 5 * params.gap else LB
            break
    if best is None:
        raise RuntimeError("no solution found")
    final = mean_allocation(inst, best) if not scheduling else best
    gap = max(0.0, (UB - LB) / max(abs(UB), 1e-9))
    return Result(f"DRO-{kind}", final, float(UB), float(LB), float(gap), time.time() - t0, status, downstream, scheduling,
                  history=hist, stats=dict(iters=len(hist), n_bc=len(M.sc_d), n_down=len(M.sc_l), n_sb=M.fs.n_sb,
                                           t_master=t_master_total, t_sub=t_sub_total, rows=M.m.n_rows, cols=M.m.n_vars))
