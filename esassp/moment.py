"""Brute-force *primal* worst-case expectations (independent oracle for the dual / C&CG reformulations).

For a fixed first-stage solution (x, s):
  mean-support :  sup_{P: E d = mu, supp in box} E Q^BC   -> moment LP over the 2^m box vertices of every block
                  sup_{P: E l = mu, supp in lattice} E Q^down -> moment LP over the integer LOS lattice
  Wasserstein  :  sup_{W1(P, P_N) <= eps} E Q            -> transport LP over {lo, d^n, hi}^m candidate points
Only usable for tiny instances (that is the point: it shares no code with the dual machinery except the closed-form Q).
"""
from __future__ import annotations

import itertools
from typing import Optional

import numpy as np
from scipy.optimize import linprog

from .evaluate import Scenarios, down_cost
from .model import Instance, Solution
from .subproblems import _block_cost_rows


def wc_bc_mean(inst: Instance, sol: Solution) -> float:
    total = 0.0
    for b in range(inst.B):
        order = sol.order(b)
        m = len(order)
        if m == 0:
            total += inst.costs.g * inst.blocks[b].length
            continue
        lo, hi, mu = inst.lo_d[order], inst.hi_d[order], inst.mu_d[order]
        V = np.array(list(itertools.product(*[(lo[k], hi[k]) for k in range(m)])))
        q = _block_cost_rows(inst, sol, b, order, V)
        A = np.vstack([V.T, np.ones(len(V))]); rhs = np.append(mu, 1.0)
        r = linprog(-q, A_eq=A, b_eq=rhs, bounds=(0, None), method="highs")
        if r.status != 0:
            raise RuntimeError("moment LP infeasible (mean outside the support box?)")
        total += -r.fun
    return float(total)


def _lattice(inst: Instance, sol: Solution):
    P = [int(i) for i in np.where(sol.scheduled)[0]]
    rng = [[(c, w) for c in range(inst.surgeries[i].lo_c, inst.surgeries[i].hi_c + 1)
            for w in range(inst.surgeries[i].lo_w, inst.surgeries[i].hi_w + 1)] for i in P]
    return P, list(itertools.product(*rng))


def wc_down_mean(inst: Instance, sol: Solution, max_points: int = 60000) -> float:
    P, pts = _lattice(inst, sol)
    if len(pts) > max_points:
        raise ValueError("lattice too large for brute force")
    nc = np.rint(inst.mu_c).astype(int); nw = np.rint(inst.mu_w).astype(int)
    lcs = np.tile(nc, (len(pts), 1)); lws = np.tile(nw, (len(pts), 1))
    for r, combo in enumerate(pts):
        for i, (c, w) in zip(P, combo):
            lcs[r, i], lws[r, i] = c, w
    q, _ = down_cost(inst, sol, lcs, lws)
    rows = [lcs[:, i].astype(float) for i in P] + [lws[:, i].astype(float) for i in P] + [np.ones(len(pts))]
    rhs = [inst.mu_c[i] for i in P] + [inst.mu_w[i] for i in P] + [1.0]
    r = linprog(-q, A_eq=np.vstack(rows), b_eq=np.array(rhs), bounds=(0, None), method="highs")
    if r.status != 0:
        raise RuntimeError("down moment LP infeasible")
    return float(-r.fun)


def _transport_lp(points_cost, dist, eps, N):
    """max sum_n sum_j pi_nj q_j  s.t. sum_j pi_nj = 1/N, sum pi_nj dist_nj <= eps."""
    nj = points_cost.shape[0]
    c = -np.tile(points_cost, N)
    A_eq = np.zeros((N, N * nj))
    for n in range(N):
        A_eq[n, n * nj:(n + 1) * nj] = 1.0
    r = linprog(c, A_ub=dist.reshape(1, -1), b_ub=[eps], A_eq=A_eq, b_eq=np.full(N, 1.0 / N), bounds=(0, None), method="highs")
    if r.status != 0:
        raise RuntimeError("transport LP failed")
    return float(-r.fun)


def wc_bc_wass(inst: Instance, sol: Solution, emp: Scenarios, eps: float) -> float:
    """Blocks are coupled through the single radius eps, so the transport LP is solved jointly (small instances)."""
    N = emp.n
    # all blocks jointly: candidate = product of per-block candidate sets would explode; use a Lagrangian-free route:
    # exact LP over per-block candidate points, with the budget shared -> concave in budget split; solve one LP
    # with variables pi_{n,b,j} where each block's transport mass is independent but the *budget* is shared.
    blocks = [b for b in range(inst.B) if sol.order(b)]
    cols_cost, cols_dist, idx = [], [], []
    base = sum(inst.costs.g * inst.blocks[b].length for b in range(inst.B) if not sol.order(b))
    for b in blocks:
        order = sol.order(b)
        lo, hi = inst.lo_d[order], inst.hi_d[order]
        for n in range(N):
            mid = np.clip(emp.d[n, order], lo, hi)
            C = np.array(list(itertools.product(*[sorted({lo[k], mid[k], hi[k]}) for k in range(len(order))])))
            cols_cost.append(_block_cost_rows(inst, sol, b, order, C))
            cols_dist.append(np.abs(C - mid[None, :]).sum(axis=1))
            idx.append((b, n))
    nvar = sum(len(c) for c in cols_cost)
    cvec = -np.concatenate(cols_cost) ; dist = np.concatenate(cols_dist)
    A_eq = np.zeros((len(idx), nvar)); off = 0
    for r, c in enumerate(cols_cost):
        A_eq[r, off:off + len(c)] = 1.0; off += len(c)
    res = linprog(cvec, A_ub=dist.reshape(1, -1), b_ub=[eps], A_eq=A_eq, b_eq=np.full(len(idx), 1.0 / N),
                  bounds=(0, None), method="highs")
    if res.status != 0:
        raise RuntimeError("wass bc LP failed")
    return float(-res.fun + base)


def wc_down_wass(inst: Instance, sol: Solution, emp: Scenarios, eps: float, max_points: int = 6000) -> float:
    P, pts = _lattice(inst, sol)
    if len(pts) > max_points:
        raise ValueError("lattice too large for brute force")
    N = emp.n
    nc = np.rint(inst.mu_c).astype(int); nw = np.rint(inst.mu_w).astype(int)
    lcs = np.tile(nc, (len(pts), 1)); lws = np.tile(nw, (len(pts), 1))
    for r, combo in enumerate(pts):
        for i, (c, w) in zip(P, combo):
            lcs[r, i], lws[r, i] = c, w
    q, _ = down_cost(inst, sol, lcs, lws)
    dist = np.zeros((N, len(pts)))
    for n in range(N):
        dist[n] = sum(np.abs(lcs[:, i] - emp.lc[n, i]) + np.abs(lws[:, i] - emp.lw[n, i]) for i in P)
    nj = len(pts)
    c = -np.tile(q, N)
    A_eq = np.zeros((N, N * nj))
    for n in range(N):
        A_eq[n, n * nj:(n + 1) * nj] = 1.0
    res = linprog(c, A_ub=dist.reshape(1, -1), b_ub=[eps], A_eq=A_eq, b_eq=np.full(N, 1.0 / N), bounds=(0, None), method="highs")
    if res.status != 0:
        raise RuntimeError("wass down LP failed")
    return float(-res.fun)
