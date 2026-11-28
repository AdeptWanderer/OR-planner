"""Worst-case (separation) problems of the C&CG algorithm.

BC  : for a block with ordered surgeries i_1..i_m and fixed (x, s),
        mean-support:  V = max_{d in box}  Q_b(d) - sum_i rho_i d_i                (eq. (10), per block)
        Wasserstein :  V = max_{d in S}    Q_b(d) - nu * sum_i |d_i - d^n_i|       (eq. (21), per block)
      Q_b is convex piecewise linear in d, so the optimum lies on a vertex (box) / in {lo, d^n, hi}^m (Wasserstein).
      * ``enum`` : exact enumeration of those candidates with the closed-form oracle (fast for m <= ~11).
      * ``milp`` : the dual of (4b)-(4e) with binary vertex selectors and linearised bilinear terms (the paper's route).
        Dual feasible set (derived here, matches (10b)-(10c)):  a_k - a_{k+1} - b_k <= c^w,  sum_k b_k - g <= c^o,
        g <= c^g,  all >= 0, which gives the bounds  g <= c^g,  b_k <= c^o + c^g,  a_k <= c^w (m-k) + c^o + c^g.
Down: max over integer LOS vectors of  Q^down(x, l) - sum_i (phi_i l^c_i + zeta_i l^w_i)  [mean-support]
      or  Q^down(x, l) - phi * sum_i (|l^c_i - l^{c,n}_i| + |l^w_i - l^{w,n}_i|)        [Wasserstein].
      Solved as a MILP over the monotone-binary LOS characterisation of Prop. 3 (ICU exit r, ward exit h).
"""
from __future__ import annotations

import itertools
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .milp import INF, Model
from .model import Instance, Solution


# ============================================================================= BC
def _block_cost_rows(inst: Instance, sol: Solution, b: int, order: Sequence[int], dm: np.ndarray) -> np.ndarray:
    """Q_b for many duration rows dm (S, m) in sequence order (closed form of (4b)-(4e))."""
    c = inst.costs
    L = inst.blocks[b].length
    S, m = dm.shape
    if m == 0:
        return np.full(S, c.g * L)
    wsum = np.zeros(S)
    end = None
    for k in range(m):
        sk = sol.start[(b, k)]
        w = np.zeros(S) if k == 0 else np.maximum(0.0, end - sk)
        wsum += w
        end = w + sk + dm[:, k]
    over = np.maximum(0.0, end - L)
    idle = np.maximum(0.0, L - dm.sum(axis=1) + over)
    return c.w * wsum + c.o * over + c.g * idle


def bc_sub(inst: Instance, sol: Solution, b: int, rho: Optional[np.ndarray] = None, nu: Optional[float] = None,
           dn: Optional[np.ndarray] = None, method: str = "auto", enum_limit: int = 11):
    """Return (value, d_worst (I,)) of the block-b worst-case problem.

    Mean-support when ``rho`` is given, Wasserstein when ``nu`` and the empirical sample ``dn`` (I,) are given.
    ``d_worst`` is defined for the surgeries of block b only (others NaN)."""
    order = sol.order(b)
    m = len(order)
    d_out = np.full(inst.I, np.nan)
    if m == 0:
        return float(inst.costs.g * inst.blocks[b].length), d_out
    lo = inst.lo_d[order]; hi = inst.hi_d[order]
    if method == "auto":
        method = "enum" if m <= enum_limit else "milp"
    if method == "enum":
        if nu is None:
            cand = np.array(list(itertools.product(*[(lo[k], hi[k]) for k in range(m)])))
            pen = cand @ rho[order]
        else:
            mid = np.clip(dn[order], lo, hi)
            cand = np.array(list(itertools.product(*[sorted({lo[k], mid[k], hi[k]}) for k in range(m)])))
            pen = nu * np.abs(cand - mid[None, :]).sum(axis=1)
        val = _block_cost_rows(inst, sol, b, order, cand) - pen
        j = int(np.argmax(val))
        d_out[order] = cand[j]
        return float(val[j]), d_out
    return _bc_sub_milp(inst, sol, b, order, rho, nu, dn)


def _bc_sub_milp(inst, sol, b, order, rho, nu, dn):
    c = inst.costs
    m_ = len(order)
    L = inst.blocks[b].length
    cw, co, cg = c.w, c.o, c.g
    lo = inst.lo_d[order]; hi = inst.hi_d[order]
    s = [sol.start[(b, k)] for k in range(m_)]
    M = Model()
    # dual variables (maximise -> minimise the negative)
    al = [None] + [M.var(0, cw * (m_ - k) + co + cg) for k in range(1, m_)]
    be = [M.var(0, co + cg) for _ in range(m_)]
    ga = M.var(0, cg)
    for k in range(1, m_):
        t = {al[k]: 1.0, be[k]: -1.0}
        if k + 1 < m_:
            t[al[k + 1]] = -1.0
        M.add(t, hi=cw)
    M.add({**{be[k]: 1.0 for k in range(m_)}, ga: -1.0}, hi=co)
    wass = nu is not None
    base = np.clip(dn[order], lo, hi) if wass else lo
    up = hi - base
    dwn = base - lo if wass else np.zeros(m_)
    cobj: Dict[int, float] = {}                                                     # objective coefficients (to maximise)

    def addc(j, v):
        cobj[j] = cobj.get(j, 0.0) + v
    const = 0.0
    # constant dual terms: sum_{k>=1} a_k (s_{k-1}-s_k) + sum_k b_k (s_k - L) + g L
    for k in range(1, m_):
        addc(al[k], s[k - 1] - s[k])
    for k in range(m_):
        addc(be[k], s[k] - L)
    addc(ga, L)
    for k in range(m_):
        i = order[k]
        Ebounds = (-cg, (cw * (m_ - k - 1) + co + cg if k + 1 < m_ else 0.0) + co + cg)
        E = {be[k]: 1.0, ga: -1.0}
        if k + 1 < m_:
            E[al[k + 1]] = 1.0
        # base * E  (- rho_i * base for mean-support)
        for j, cf in E.items():
            addc(j, base[k] * cf)
        if not wass:
            const -= rho[i] * base[k]
        pen_up = rho[i] if not wass else nu
        if up[k] > 1e-12:
            tp = M.binary()
            zp = M.var(-INF, INF)
            lo_f, hi_f = Ebounds[0] - pen_up, Ebounds[1] - pen_up
            # zp <= hi_f*tp ; zp <= (E - pen_up) - lo_f (1 - tp)
            M.add({zp: 1.0, tp: -hi_f}, hi=0.0)
            t = {zp: 1.0}
            for j, cf in E.items():
                t[j] = t.get(j, 0.0) - cf
            t[tp] = t.get(tp, 0.0) - lo_f
            M.add(t, hi=-pen_up - lo_f)
            addc(zp, up[k])
            M._tp = getattr(M, "_tp", {}); M._tp[k] = tp
        if wass and dwn[k] > 1e-12:
            tm = M.binary()
            zm = M.var(-INF, INF)
            lo_f, hi_f = Ebounds[0] + nu, Ebounds[1] + nu
            # zm >= (E + nu) - hi_f (1 - tm) ; zm >= lo_f * tm ; objective term -dwn * zm
            t = {zm: 1.0, tm: -hi_f}
            for j, cf in E.items():
                t[j] = t.get(j, 0.0) - cf
            M.add(t, lo=nu - hi_f)
            M.add({zm: 1.0, tm: -lo_f}, lo=0.0)
            addc(zm, -dwn[k])
            M._tm = getattr(M, "_tm", {}); M._tm[k] = tm
            if k in getattr(M, "_tp", {}):
                M.add({M._tp[k]: 1.0, tm: 1.0}, hi=1.0)
    for j, v in cobj.items():
        M.set_obj(j, -v)
    r = M.solve(time_limit=30, gap=1e-9, threads=1)
    if not r.ok:
        raise RuntimeError(f"BC sub-problem MILP failed: {r.status}")
    d = base.copy()
    for k in range(m_):
        if k in getattr(M, "_tp", {}) and r.x[M._tp[k]] > 0.5:
            d[k] += up[k]
        if k in getattr(M, "_tm", {}) and r.x[M._tm[k]] > 0.5:
            d[k] -= dwn[k]
    out = np.full(inst.I, np.nan)
    out[order] = d
    return float(-r.obj + const), out


# ============================================================================= downstream
def _stays(inst: Instance, sol: Solution):
    """Scheduled patients with their surgery day."""
    return [(int(i), inst.blocks[int(sol.block[i])].day) for i in np.where(sol.scheduled)[0]]


def down_sub(inst: Instance, sol: Solution, phi: Optional[np.ndarray] = None, zeta: Optional[np.ndarray] = None,
             phiW: Optional[float] = None, lcn: Optional[np.ndarray] = None, lwn: Optional[np.ndarray] = None,
             time_limit: float = 30.0):
    """Worst-case LOS scenario.  Returns (value, lc (I,), lw (I,)); unscheduled patients get their nominal LOS."""
    c = inst.costs
    T, Te = inst.T, inst.T_ext
    wass = phiW is not None
    M = Model()
    obj: Dict[int, float] = {}

    def addo(j, v):
        obj[j] = obj.get(j, 0.0) + v
    pats = _stays(inst, sol)
    e = [M.var(0, 1) for _ in range(T)]; mm = [M.var(0, 1) for _ in range(T)]
    for t in range(T):
        addo(e[t], -c.c_icu * inst.icu_beds[t]); addo(mm[t], -c.c_ward * inst.ward_beds[t])
    lc_expr: Dict[int, Tuple[float, Dict[int, float]]] = {}
    lw_expr: Dict[int, Tuple[float, Dict[int, float]]] = {}
    for i, a in pats:
        s_ = inst.surgeries[i]
        loC, hiC, loW, hiW = s_.lo_c, s_.hi_c, s_.lo_w, s_.hi_w
        R: Dict[int, int] = {}; H: Dict[int, int] = {}
        for t in range(a + loC, min(a + hiC, Te)):
            R[t] = M.binary()
        for t in range(a + loC + loW, min(a + hiC + hiW, Te)):
            H[t] = M.binary()

        def Rv(t):          # (const, var) of R_t
            if t < a + loC: return (0.0, None)
            if t >= a + hiC: return (1.0, None)
            return (0.0, R[t])

        def Hv(t):
            if t < a + loC + loW: return (0.0, None)
            if t >= a + hiC + hiW: return (1.0, None)
            return (0.0, H[t])
        # monotone, H <= R
        for t in range(a + loC, min(a + hiC, Te) - 1):
            M.add({R[t]: 1.0, R[t + 1]: -1.0}, hi=0.0)
        for t in range(a + loC + loW, min(a + hiC + hiW, Te) - 1):
            M.add({H[t]: 1.0, H[t + 1]: -1.0}, hi=0.0)
        for t in range(a + loC + loW, min(a + hiC + hiW, Te)):
            rc, rv = Rv(t)
            if rv is None:
                if rc == 0.0:
                    M.fix(H[t], 0.0)
            else:
                M.add({H[t]: 1.0, rv: -1.0}, hi=0.0)
        # LOS expressions: lc = loC + sum_{t in free R}(1 - R_t)   (days a..a+loC-1 always in ICU)
        lc_c = float(loC) + sum(1.0 for t in R)
        lc_t = {R[t]: -1.0 for t in R}
        lc_expr[i] = (lc_c, lc_t)
        # lw = sum_{t>=a} (R_t - H_t) over the whole timeline
        lw_c = 0.0; lw_t: Dict[int, float] = {}
        for t in range(a, Te):
            rc, rv = Rv(t); hc, hv = Hv(t)
            lw_c += rc - hc
            if rv is not None: lw_t[rv] = lw_t.get(rv, 0.0) + 1.0
            if hv is not None: lw_t[hv] = lw_t.get(hv, 0.0) - 1.0
        lw_expr[i] = (lw_c, lw_t)
        M.add(dict(lw_t), loW - lw_c, hiW - lw_c)
        # occupancy products
        for t in range(a, T):
            rc, rv = Rv(t)
            y_c, y_t = 1.0 - rc, ({rv: -1.0} if rv is not None else {})
            if y_c != 0.0 or y_t:
                p = M.var(0, 1); addo(p, c.c_icu)
                M.add({p: 1.0, e[t]: -1.0}, hi=0.0)
                tt = {p: 1.0}
                for j, cf in y_t.items(): tt[j] = tt.get(j, 0.0) - cf
                M.add(tt, hi=y_c)
            hc, hv = Hv(t)
            z_c = rc - hc
            z_t: Dict[int, float] = {}
            if rv is not None: z_t[rv] = 1.0
            if hv is not None: z_t[hv] = z_t.get(hv, 0.0) - 1.0
            if z_c != 0.0 or z_t:
                q = M.var(0, 1); addo(q, c.c_ward)
                M.add({q: 1.0, mm[t]: -1.0}, hi=0.0)
                tt = {q: 1.0}
                for j, cf in z_t.items(): tt[j] = tt.get(j, 0.0) - cf
                M.add(tt, hi=z_c)
        # penalties
        if not wass:
            for expr, mult in ((lc_expr[i], phi[i]), (lw_expr[i], zeta[i])):
                cst, tm = expr
                for j, cf in tm.items(): addo(j, -mult * cf)
                obj[("const")] = obj.get("const", 0.0) - mult * cst
        else:
            for expr, nom in ((lc_expr[i], lcn[i]), (lw_expr[i], lwn[i])):
                cst, tm = expr
                dev = M.var(0, INF); addo(dev, -phiW)
                t1 = {dev: 1.0}
                for j, cf in tm.items(): t1[j] = t1.get(j, 0.0) - cf
                M.add(t1, lo=cst - nom)                  # dev >= l - nom
                t2 = {dev: 1.0}
                for j, cf in tm.items(): t2[j] = t2.get(j, 0.0) + cf
                M.add(t2, lo=nom - cst)                  # dev >= nom - l
    const = obj.pop("const", 0.0)
    for j, v in obj.items():
        M.set_obj(j, -v)
    r = M.solve(time_limit=time_limit, gap=1e-7, threads=1)
    lc = np.rint(inst.mu_c).astype(int); lw = np.rint(inst.mu_w).astype(int)
    for i, a in pats:
        cst, tm = lc_expr[i]; lc[i] = int(round(cst + sum(cf * r.x[j] for j, cf in tm.items())))
        cst, tm = lw_expr[i]; lw[i] = int(round(cst + sum(cf * r.x[j] for j, cf in tm.items())))
    return float(-r.obj + const), lc, lw
