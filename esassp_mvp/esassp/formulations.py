"""First-stage feasible set X (eq. (2)), symmetry breaking (27)-(29) and scenario copies of the recourse
problems Q^BC (4a)-(4e) and Q^down (5a)-(5e) written directly in (x, s)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .milp import INF, Model
from .model import Instance, Solution


class FirstStage:
    """Variables x[i,b,k] (binary), xp[i] (postponed, the dummy block b'), s[b,k] (continuous)."""

    def __init__(self, m: Model, inst: Instance, symmetry: bool = True, sb_positions: bool = True):
        self.m, self.inst = m, inst
        I, B = inst.I, inst.B
        self.x: Dict[Tuple[int, int, int], int] = {}
        self.xp: List[int] = []
        self.s: Dict[Tuple[int, int], int] = {}
        self.elig: List[List[int]] = [inst.eligible_blocks(i) for i in range(I)]
        for i in range(I):
            for b in self.elig[i]:
                for k in range(inst.blocks[b].K):
                    self.x[(i, b, k)] = m.binary(inst.first_stage_cost(i, False), f"x_{i}_{b}_{k}")
            self.xp.append(m.binary(inst.first_stage_cost(i, True), f"xp_{i}"))
        for b, blk in enumerate(inst.blocks):
            for k in range(blk.K):
                self.s[(b, k)] = m.var(0.0, blk.length, 0.0, False, f"s_{b}_{k}")
        self.by_block: List[List[List[int]]] = [[[i for i in range(I) if (i, b, k) in self.x]
                                                 for k in range(blk.K)] for b, blk in enumerate(inst.blocks)]
        self._constraints()
        self.n_sb = 0
        if symmetry:
            self._symmetry(sb_positions)

    # -------------------------------------------------------------- helpers
    def sigma(self, i: int) -> Dict[int, float]:
        """sum_{b in B, k} x[i,b,k]   (1 - xp[i])"""
        return {self.x[(i, b, k)]: 1.0 for b in self.elig[i] for k in range(self.inst.blocks[b].K)}

    def a(self, i: int, b: int) -> Dict[int, float]:
        """sum_k x[i,b,k]  (i in block b)"""
        return {self.x[(i, b, k)]: 1.0 for k in range(self.inst.blocks[b].K) if (i, b, k) in self.x}

    def occ(self, b: int, k: int) -> Dict[int, float]:
        return {self.x[(i, b, k)]: 1.0 for i in self.by_block[b][k]}

    def dur(self, b: int, k: int, d: np.ndarray, scale: float = 1.0) -> Dict[int, float]:
        return {self.x[(i, b, k)]: scale * float(d[i]) for i in self.by_block[b][k]}

    # -------------------------------------------------------------- constraints (2)
    def _constraints(self):
        m, inst = self.m, self.inst
        for i in range(inst.I):
            t = dict(self.sigma(i)); t[self.xp[i]] = 1.0
            m.add(t, 1.0, 1.0)
        for b, blk in enumerate(inst.blocks):
            for k in range(blk.K):
                occ = self.occ(b, k)
                if occ:
                    m.add(occ, hi=1.0)
                if k >= 1:
                    t = dict(occ)
                    for j, c in self.occ(b, k - 1).items():
                        t[j] = t.get(j, 0.0) - c
                    if t:
                        m.add(t, hi=0.0)
                t = {self.s[(b, k)]: 1.0}
                for j, c in occ.items():
                    t[j] = -blk.length * c
                m.add(t, hi=0.0)                                     # s_bk <= L_b * sum_i x_ibk

    # -------------------------------------------------------------- symmetry breaking (27)-(29)
    def _symmetry(self, sb_positions: bool):
        inst, m = self.inst, self.m
        classes: Dict[tuple, List[int]] = {}
        for s in inst.surgeries:
            classes.setdefault((s.klass, tuple(self.elig[s.id])), []).append(s.id)
        # a block is "class-exclusive" when only one class may use it (closed block scheduling)
        users: Dict[int, set] = {}
        for key, members in classes.items():
            for b in self.elig[members[0]]:
                users.setdefault(b, set()).add(key)
        for key, members in classes.items():
            blocks = self.elig[members[0]]
            excl = all(len(users[b]) == 1 for b in blocks)
            for i, j in zip(members[:-1], members[1:]):
                # (27) if i is in block b then j is in a block >= b (or postponed)
                for pos, b in enumerate(blocks):
                    t = dict(self.a(i, b))
                    for b2 in blocks[pos:]:
                        for jj, c in self.a(j, b2).items():
                            t[jj] = t.get(jj, 0.0) - c
                    t[self.xp[j]] = t.get(self.xp[j], 0.0) - 1.0
                    m.add(t, hi=0.0); self.n_sb += 1
                # (28) if i is postponed, j is postponed
                m.add({self.xp[i]: 1.0, self.xp[j]: -1.0}, hi=0.0); self.n_sb += 1
                # (29) if i is at position k of b, j is at k+1 of b, later block, or postponed
                if sb_positions and excl:
                    for pos, b in enumerate(blocks):
                        K = inst.blocks[b].K
                        for k in range(K):
                            t = {self.x[(i, b, k)]: 1.0}
                            if k + 1 < K:
                                t[self.x[(j, b, k + 1)]] = -1.0
                            for b2 in blocks[pos + 1:]:
                                for jj, c in self.a(j, b2).items():
                                    t[jj] = t.get(jj, 0.0) - c
                            t[self.xp[j]] = t.get(self.xp[j], 0.0) - 1.0
                            m.add(t, hi=0.0); self.n_sb += 1

    # -------------------------------------------------------------- extract
    def extract(self, xv: np.ndarray) -> Solution:
        inst = self.inst
        block = -np.ones(inst.I, int); pos = -np.ones(inst.I, int)
        for (i, b, k), j in self.x.items():
            if xv[j] > 0.5:
                block[i], pos[i] = b, k
        start = {}
        for b in range(inst.B):
            order = [i for i in np.where(block == b)[0]]
            order.sort(key=lambda i: pos[i])
            for newk, i in enumerate(order):
                pos[i] = newk
            for k in range(inst.blocks[b].K):
                start[(b, k)] = float(np.clip(xv[self.s[(b, k)]], 0, inst.blocks[b].length)) if k < len(order) else 0.0
        return Solution(block, pos, start)

    def hint(self, sol: Solution) -> Dict[int, float]:
        h: Dict[int, float] = {}
        for i in range(self.inst.I):
            if sol.block[i] >= 0 and (i, int(sol.block[i]), int(sol.pos[i])) in self.x:
                h[self.x[(i, int(sol.block[i]), int(sol.pos[i]))]] = 1.0
            else:
                h[self.xp[i]] = 1.0
        for (b, k), j in self.s.items():
            h[j] = sol.start.get((b, k), 0.0)
        return h


# ===================================================================== recourse copies
@dataclass
class BCCopy:
    cost_terms: List[Dict[int, float]]      # per block: {var: coef} of c^w sum w + c^g g + c^o o
    vars: Dict[str, int] = field(default_factory=dict)


@dataclass
class DownCopy:
    cost_terms: Dict[int, float]
    u: List[int]
    n: List[int]


def add_bc_recourse(fs: FirstStage, d: np.ndarray, prob_obj: float = 0.0, c_w: Optional[float] = None) -> BCCopy:
    """Constraints (4b)-(4e) for one duration scenario d (length I).  If ``prob_obj`` > 0 the cost is also put into
    the objective with that weight (SP / SAA).  Returns per-block cost expressions."""
    m, inst = fs.m, fs.inst
    cw = inst.costs.w if c_w is None else c_w
    cg, co = inst.costs.g, inst.costs.o
    out: List[Dict[int, float]] = []
    for b, blk in enumerate(inst.blocks):
        K, L = blk.K, blk.length
        dmax = max((inst.surgeries[i].hi_d for i in range(inst.I) if (i, b, 0) in fs.x), default=0.0)
        o = m.var(0, INF, prob_obj * co); g = m.var(0, INF, prob_obj * cg)
        cost = {o: co, g: cg}
        w = [None] + [m.var(0, INF, prob_obj * cw) for _ in range(1, K)]          # (4b): w_{b,1} = 0
        for k in range(1, K):
            cost[w[k]] = cw
            M = L + k * dmax
            # (4c)  w_k - w_{k-1} - s_{k-1} + s_k - sum d x_{k-1} - M * occ_k >= -M
            t = {w[k]: 1.0, fs.s[(b, k - 1)]: -1.0, fs.s[(b, k)]: 1.0}
            if k - 1 >= 1:
                t[w[k - 1]] = -1.0
            for j, c in fs.dur(b, k - 1, d, -1.0).items():
                t[j] = t.get(j, 0.0) + c
            for j, c in fs.occ(b, k).items():
                t[j] = t.get(j, 0.0) - M * c
            m.add(t, lo=-M)
        for k in range(K):                                                        # (4d)
            t = {o: 1.0, fs.s[(b, k)]: -1.0}
            if k >= 1:
                t[w[k]] = -1.0
            for j, c in fs.dur(b, k, d, -1.0).items():
                t[j] = t.get(j, 0.0) + c
            m.add(t, lo=-L)
        t = {g: 1.0, o: -1.0}                                                     # (4e)
        for k in range(K):
            for j, c in fs.dur(b, k, d, 1.0).items():
                t[j] = t.get(j, 0.0) + c
        m.add(t, lo=L)
        out.append(cost)
    return BCCopy(out)


def add_down_recourse(fs: FirstStage, lc: np.ndarray, lw: np.ndarray, prob_obj: float = 0.0) -> DownCopy:
    """Constraints (5b)-(5e) for one LOS scenario, eliminating y, z:  u_t >= ICU_t - Bed, n_t >= ward_t - Bed."""
    m, inst = fs.m, fs.inst
    T = inst.T
    u = [m.var(0, INF, prob_obj * inst.costs.c_icu) for _ in range(T)]
    n = [m.var(0, INF, prob_obj * inst.costs.c_ward) for _ in range(T)]
    ticu: List[Dict[int, float]] = [{u[t]: 1.0} for t in range(T)]
    tward: List[Dict[int, float]] = [{n[t]: 1.0} for t in range(T)]
    for i in range(inst.I):
        for b in fs.elig[i]:
            a = inst.blocks[b].day
            ci, wi = int(lc[i]), int(lw[i])
            for t in range(T):
                inicu = a <= t < a + ci
                inward = a + ci <= t < a + ci + wi
                if inicu or inward:
                    for j in fs.a(i, b):
                        if inicu:
                            ticu[t][j] = ticu[t].get(j, 0.0) - 1.0
                        if inward:
                            tward[t][j] = tward[t].get(j, 0.0) - 1.0
    for t in range(T):
        m.add(ticu[t], lo=-float(inst.icu_beds[t]))
        m.add(tward[t], lo=-float(inst.ward_beds[t]))
    cost = {}
    for t in range(T):
        cost[u[t]] = inst.costs.c_icu
        cost[n[t]] = inst.costs.c_ward
    return DownCopy(cost, u, n)
