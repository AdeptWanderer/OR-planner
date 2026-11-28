"""Independent feasibility checker for first-stage solutions (constraints (2))."""
from __future__ import annotations

from typing import List

import numpy as np

from .model import Instance, Solution


def validate_solution(inst: Instance, sol: Solution) -> List[str]:
    errs: List[str] = []
    if len(sol.block) != inst.I:
        return ["solution length != number of surgeries"]
    seen = {}
    for i in range(inst.I):
        b, k = int(sol.block[i]), int(sol.pos[i])
        if b < 0:
            continue
        if b >= inst.B:
            errs.append(f"surgery {i}: unknown block {b}"); continue
        blk = inst.blocks[b]
        if blk.types and inst.surgeries[i].type not in blk.types:
            errs.append(f"surgery {i}: type not allowed in block {b}")
        if not (0 <= k < blk.K):
            errs.append(f"surgery {i}: position {k} outside 0..{blk.K - 1} of block {b}")
        if (b, k) in seen:
            errs.append(f"block {b} position {k} used twice (surgeries {seen[(b, k)]}, {i})")
        seen[(b, k)] = i
    for b, blk in enumerate(inst.blocks):
        ks = sorted(k for (bb, k) in seen if bb == b)
        if ks != list(range(len(ks))):
            errs.append(f"block {b}: positions {ks} are not consecutive from 0")
        for k in ks:
            s = sol.start.get((b, k), 0.0)
            if s < -1e-6 or s > blk.length + 1e-6:
                errs.append(f"block {b} position {k}: start {s:.1f} outside [0, {blk.length}]")
    return errs
