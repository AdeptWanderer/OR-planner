"""Price-and-cut (Algorithm 1) and branch-and-price-and-cut (Algorithm 2)."""
from __future__ import annotations

import heapq
import itertools
import math
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .cuts import separate
from .heuristic import constructive_heuristic
from .master import ExtraRow, Master
from .model import Column, Instance, Solution
from .pricing import NodeRestrictions, PricingStats, column_value_on_day, solve_pricing
from .validate import validate_columns


@dataclass
class BPCParams:
    cp_time_limit: float = 2.0          # CPtimelimit for the fast phase (paper: 10 s)
    cp_exact_time_limit: float = 30.0   # time limit in CheckOptimalityPhase (paper: inf)
    use_dominance: bool = True          # Section 4.1
    use_fid: bool = True                # Section 4.2
    use_cuts: bool = True               # Section 5.3
    copy_columns: bool = True           # Algorithm 1, line 12
    branching: bool = True              # Algorithm 2 (False -> price-and-cut only)
    cg_time_limit: float = 120.0        # root CG time limit (paper: 2 h)
    branch_time_limit: float = 120.0    # time for the branching tree (paper: 2 h)
    prove_fraction: float = 0.125       # share of branch time proving node optimality (paper: 15 min / 2 h)
    final_ip_time: float = 60.0         # IP on all generated columns (paper: 1 h)
    polish_time: float = 30.0           # CP-SAT polishing of the incumbent (0 = off; not in paper)
    max_columns_per_sub: int = 15
    max_cut_rounds: int = 5
    max_cuts_per_round: int = 40
    max_nodes: int = 500
    workers: int = 4
    seed: int = 0
    # --- long-run robustness (not in the paper) ---
    checkpoint_path: Optional[str] = None   # pickle file; resume from it if it exists
    checkpoint_every: float = 300.0         # seconds between checkpoints
    log_cap: int = 20000                    # max records kept in pricing / iteration / node logs


@dataclass
class IterLog:
    node: int
    iteration: int
    time: float
    rmp_obj: float
    lagrangian_bound: float
    n_columns: int
    n_cuts: int
    new_columns: int
    phase: int
    artificial: float


class Solver:
    def __init__(self, inst: Instance, params: BPCParams = BPCParams(),
                 log: Optional[Callable[[str], None]] = None,
                 progress: Optional[Callable[[dict], None]] = None):
        self.inst, self.p = inst, params
        self.logf = log or (lambda s: None)
        self.progress = progress or (lambda d: None)
        self.pool: Dict[tuple, Column] = {}
        self.cuts: List[ExtraRow] = []
        self.cut_tags: set = set()
        self.iter_log: List[IterLog] = []
        self.pricing_log: List[dict] = []
        self.node_log: List[dict] = []
        self.best_cols: Optional[List[Column]] = None
        self.best_obj = -math.inf
        self.t0 = time.time()
        self._node_ids = itertools.count()
        self.n_nodes = 0
        self.pricing_agg: Dict[str, float] = {}
        self.trajectory: List[tuple] = []        # (elapsed s, best plan, best bound) in slots
        self.phase = "root"
        self.used = {"root": 0.0, "branch": 0.0}
        self._phase_t0 = time.time()
        self._last_ckpt = time.time()
        self._root_bound_sofar = math.inf
        self.heap: list = []
        self._cnt = 0
        self.unproven: List[float] = []
        self.root_bound = math.inf
        self.polish = None
        self.lb_before_polish = -math.inf

    # ------------------------------------------------------------- helpers
    def elapsed(self):
        return time.time() - self.t0

    def add_pool(self, cols: Sequence[Column]):
        for c in cols:
            self.pool.setdefault(c.key(), c)

    def update_incumbent(self, cols, obj, source):
        if cols is None or obj is None or obj <= self.best_obj + 1e-9:
            return False
        if validate_columns(self.inst, cols):
            return False
        self.best_cols, self.best_obj = list(cols), float(obj)
        self.trajectory.append((round(self.elapsed(), 1), self.best_obj, self.current_ub()))
        self.logf(f"[{self.elapsed():7.1f}s] new incumbent {obj * self.inst.slot_minutes:.0f} min ({source})")
        return True

    def build_master(self, restr: NodeRestrictions, branch_rows: Sequence[ExtraRow]) -> Master:
        m = Master(self.inst, list(self.cuts) + list(branch_rows))
        m.add_columns([c for c in self.pool.values() if restr.compatible(c)])
        return m

    # ----------------------------------------------------- Algorithm 1
    def price_and_cut(self, node_id: int, restr: NodeRestrictions, branch_rows: Sequence[ExtraRow],
                      deadline: float, prove: bool = True):
        inst, p = self.inst, self.p
        master = self.build_master(restr, branch_rows)
        phase = 0
        it = 0
        best_bound = math.inf
        converged = False
        n_cut_start = len(self.cuts)
        while True:
            it += 1
            obj, x, art, y = master.solve_lp()
            # ---- cutting planes (lines 4-8)
            if p.use_cuts:
                for _ in range(p.max_cut_rounds):
                    new = separate(inst, master.columns, x, self.cut_tags, p.max_cuts_per_round)
                    if not new:
                        break
                    for r in new:
                        master.add_row(r)
                        self.cuts.append(r)
                    obj, x, art, y = master.solve_lp()
            # ---- pricing (lines 9-11)
            new_cols: List[Column] = []
            day_bounds = []
            tl = p.cp_time_limit if phase == 0 else p.cp_exact_time_limit
            all_proven = True
            duals = {}
            for d in range(inst.n_days):
                G, pi6, const = master.day_duals(y, d)
                duals[d] = (G, pi6, const)
                remaining = max(0.5, deadline - time.time())
                cols, st = solve_pricing(inst, d, G, pi6, const, restr, min(tl, remaining),
                                         p.use_dominance, p.use_fid, p.max_columns_per_sub,
                                         p.workers, p.seed + it)
                rec = asdict(st); rec.update(node=node_id, iteration=it, day=d, phase=phase)
                if len(self.pricing_log) < p.log_cap:
                    self.pricing_log.append(rec)
                for k_, v_ in rec.items():
                    if isinstance(v_, (bool, int, float)) and k_ not in ("node", "iteration", "day", "phase"):
                        self.pricing_agg[k_] = self.pricing_agg.get(k_, 0.0) + float(v_)
                self.pricing_agg["calls"] = self.pricing_agg.get("calls", 0.0) + 1
                new_cols += cols
                all_proven &= st.proven
                day_bounds.append(max(0.0, st.value_bound))
            # ---- copy columns to other days (line 12)
            if p.copy_columns and new_cols:
                extra = []
                for c in new_cols:
                    for d2 in range(inst.n_days):
                        if d2 == c.day:
                            continue
                        G, pi6, const = duals[d2]
                        v = column_value_on_day(inst, c, d2, G, pi6, const, restr)
                        if v is not None and v > 1e-6:
                            extra.append(Column(d2, c.items))
                new_cols += extra
            lag = obj + sum(inst.rooms[d] * day_bounds[d] for d in range(inst.n_days))
            best_bound = min(best_bound, lag)
            added = master.add_columns(new_cols)
            self.add_pool(new_cols)
            if node_id == 0 or len(self.iter_log) < p.log_cap:
                self.iter_log.append(IterLog(node_id, it, self.elapsed(), obj, best_bound,
                                             len(master.columns), len(self.cuts), added, phase, float(art.sum())))
            if node_id == 0:
                self._root_bound_sofar = min(self._root_bound_sofar, best_bound)
                self.maybe_checkpoint()
            self.progress(dict(node=node_id, iteration=it, rmp=obj, bound=best_bound,
                               columns=len(master.columns), elapsed=self.elapsed()))
            self.logf(f"[{self.elapsed():7.1f}s] node {node_id} it {it:3d} phase {phase} "
                      f"RMP {obj * inst.slot_minutes:9.1f}  bound {best_bound * inst.slot_minutes:9.1f}  "
                      f"+cols {added:3d}  cols {len(master.columns)}  cuts {len(self.cuts)}")
            # ---- phase logic (lines 14-22)
            if added == 0:
                if phase == 0 and prove:
                    phase = 1
                elif phase == 0 and not prove:
                    break
                else:
                    converged = all_proven
                    if converged:
                        best_bound = min(best_bound, obj)
                    break
            elif phase == 1:
                phase = 0
            if time.time() > deadline:
                break
        obj, x, art, y = master.solve_lp()
        return dict(master=master, obj=obj, x=x, art=art, bound=best_bound, converged=converged,
                    iterations=it)

    # ----------------------------------------------------- branching
    def choose_branch(self, master: Master, x: np.ndarray):
        inst = self.inst
        cols = master.columns
        frac = lambda v: abs(v - round(v)) > 1e-5
        # (1) total number of scheduled surgeries
        m = float(sum(len(c.items) * v for c, v in zip(cols, x)))
        if frac(m):
            return ("count", None, m)
        # (2) per surgeon
        best, bl = None, None
        for l in range(inst.n_surgeons):
            ml = float(sum(sum(1 for i, _ in c.items if inst.surgeries[i].surgeon == l) * v
                           for c, v in zip(cols, x)))
            if frac(ml):
                score = abs((ml - math.floor(ml)) - 0.5)
                if best is None or score < best:
                    best, bl = score, (l, ml)
        if bl is not None:
            return ("surgeon", bl[0], bl[1])
        # (3) Ryan-Foster pair
        w: Dict[Tuple[int, int], float] = {}
        for c, v in zip(cols, x):
            if v < 1e-6 or v > 1 - 1e-6:
                continue
            S = sorted(c.surgeries())
            for a_i in range(len(S)):
                for b_i in range(a_i + 1, len(S)):
                    key = (S[a_i], S[b_i])
                    w[key] = w.get(key, 0.0) + v
        bestp, bs = None, None
        for key, v in w.items():
            if frac(v):
                score = abs((v - math.floor(v)) - 0.5)
                if bs is None or score < bs:
                    bs, bestp = score, key
        if bestp is not None:
            return ("pair", bestp, w[bestp])
        return None

    def children(self, branch, restr: NodeRestrictions, rows: List[ExtraRow]):
        inst = self.inst
        kind, what, val = branch
        if kind == "count":
            coef = {i: 1.0 for i in range(inst.n)}
            return [(restr, rows + [ExtraRow(coef, math.ceil(val), np.inf, kind="branch", tag=("cnt>=",))], f"#cases ≥ {math.ceil(val)}"),
                    (restr, rows + [ExtraRow(coef, -np.inf, math.floor(val), kind="branch", tag=("cnt<=",))], f"#cases ≤ {math.floor(val)}")]
        if kind == "surgeon":
            coef = {i: 1.0 for i in inst.by_surgeon(what)}
            nm = inst.surgeon_name(what)
            return [(restr, rows + [ExtraRow(coef, math.ceil(val), np.inf, kind="branch")], f"{nm}: #cases ≥ {math.ceil(val)}"),
                    (restr, rows + [ExtraRow(coef, -np.inf, math.floor(val), kind="branch")], f"{nm}: #cases ≤ {math.floor(val)}")]
        a, b = what
        la, lb = inst.surgeries[a].label, inst.surgeries[b].label
        r1 = NodeRestrictions(restr.together + [(a, b)], list(restr.apart))
        r2 = NodeRestrictions(list(restr.together), restr.apart + [(a, b)])
        return [(r1, rows, f"{la} & {lb} same room"), (r2, rows, f"{la} & {lb} different rooms")]

    # ----------------------------------------------------- Algorithm 2
    # ------------------------------------------------ checkpoint / resume
    STATE = ("phase", "used", "pool", "cuts", "cut_tags", "best_cols", "best_obj", "heuristic", "root",
             "trivial_ub", "heap", "_cnt", "unproven", "n_nodes", "root_bound", "trajectory", "node_log",
             "iter_log", "pricing_log", "pricing_agg", "polish", "lb_before_polish", "_root_bound_sofar",
             "_next_node", "global_ub")

    def current_ub(self) -> float:
        if self.phase == "root":
            return min(self._root_bound_sofar, getattr(self, "trivial_ub", math.inf))
        if self.phase == "branch":
            ob = [-h[0] for h in self.heap] + self.unproven
            return min(self.root_bound, max(ob + [self.best_obj])) if ob else max(self.best_obj, 0)
        return getattr(self, "global_ub", self.root_bound)

    def _account(self):
        now = time.time()
        if self.phase in self.used:
            self.used[self.phase] += now - self._phase_t0
        self._phase_t0 = now

    def set_phase(self, ph):
        self._account()
        self.phase = ph
        self.checkpoint()

    def checkpoint(self):
        if not self.p.checkpoint_path:
            return
        import os, pickle
        self._account()
        self._next_node = next(self._node_ids)
        self._node_ids = itertools.count(self._next_node)
        self.trajectory.append((round(self.elapsed(), 1), self.best_obj, self.current_ub()))
        st = {k: getattr(self, k, None) for k in self.STATE}
        st["elapsed"] = self.elapsed()
        tmp = self.p.checkpoint_path + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump(st, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, self.p.checkpoint_path)       # atomic
        self._last_ckpt = time.time()
        try:                                          # give freed memory back to the OS
            import ctypes, gc
            gc.collect(); ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            pass

    def maybe_checkpoint(self):
        if self.p.checkpoint_path and time.time() - self._last_ckpt >= self.p.checkpoint_every:
            self.checkpoint()

    def try_resume(self) -> bool:
        import os, pickle
        path = self.p.checkpoint_path
        if not path or not os.path.exists(path):
            return False
        with open(path, "rb") as f:
            st = pickle.load(f)
        for k in self.STATE:
            if k in st and st[k] is not None:
                setattr(self, k, st[k])
        self._node_ids = itertools.count(st.get("_next_node") or 0)
        self.t0 = time.time() - st["elapsed"]
        self._phase_t0 = time.time()
        self.logf(f"[{self.elapsed():7.1f}s] resumed from checkpoint in phase '{self.phase}' "
                  f"(root used {self.used['root']:.0f}s, branching used {self.used['branch']:.0f}s)")
        return True

    # ----------------------------------------------------- main driver
    def run(self) -> Solution:
        inst, p = self.inst, self.p
        resumed = self.try_resume()
        if not resumed:
            rng = np.random.default_rng(p.seed)
            init_cols, _ = constructive_heuristic(inst, rng)
            self.add_pool(init_cols)
            heur_obj = sum(c.value(inst) for c in init_cols)
            heur_errs = validate_columns(inst, init_cols)
            self.heuristic = dict(objective=heur_obj, columns=init_cols, errors=heur_errs)
            self.trivial_ub = min(sum(inst.rooms[d] * inst.slots_per_day for d in range(inst.n_days)),
                                  int(inst.surgeon_max.sum()), sum(s.duration for s in inst.surgeries))
            if not heur_errs:
                self.update_incumbent(init_cols, heur_obj, "constructive heuristic")
        heur_obj, heur_errs = self.heuristic["objective"], self.heuristic["errors"]
        trivial_ub = self.trivial_ub

        # ---------------- root node (resumable: the master is rebuilt from the pool)
        if self.phase == "root":
            remaining = max(5.0, p.cg_time_limit - self.used["root"])
            res = self.price_and_cut(0, NodeRestrictions(), [], time.time() + remaining, prove=True)
            self.root_bound = min(trivial_ub, res["bound"], self._root_bound_sofar)
            self.root = dict(lp=res["obj"], bound=self.root_bound, converged=res["converged"],
                             iterations=res["iterations"], time=self.elapsed(),
                             columns=len(self.pool), cuts=len(self.cuts))
            if not res["converged"]:
                self.logf(f"root CG not converged in time limit; bound = {self.root_bound * inst.slot_minutes:.0f} min")
            cols, obj = res["master"].solve_ip(min(p.final_ip_time, 30.0), start=self.best_cols)
            self.update_incumbent(cols, obj, "root RMP-IP")
            self.node_log.append(dict(id=0, parent=None, label="root", bound=self.root_bound, lp=res["obj"],
                                      converged=res["converged"], status="processed", depth=0))
            self.n_nodes = 1
            self.heap = []
            if p.branching and self.root_bound > self.best_obj + 1 - 1e-6:
                br = self.choose_branch(res["master"], res["x"])
                if br is not None:
                    for r, rows, lbl in self.children(br, NodeRestrictions(), []):
                        self._cnt += 1
                        heapq.heappush(self.heap, (-self.root_bound, self._cnt, r, rows, lbl, 0, 1, True))
            del res
            self.set_phase("branch")

        if self.phase == "branch":
            self.branch_and_bound()
            ob = [-h[0] for h in self.heap] + self.unproven
            self.global_ub = min(self.root_bound, max(ob + [self.best_obj])) if ob else self.best_obj
            self.global_ub = max(self.global_ub, self.best_obj)
            self.heap = []
            self.set_phase("final")
        self.global_ub = self.current_ub() if not hasattr(self, "global_ub") else self.global_ub

        # ---------------- final IP on all generated columns (line 16)
        if self.phase == "final":
            fm = Master(inst)
            fm.add_columns(list(self.pool.values()))
            cols, obj = fm.solve_ip(max(5.0, p.final_ip_time), start=self.best_cols)
            del fm
            self.update_incumbent(cols, obj, "final IP over all columns")
            self.lb_before_polish = self.best_obj
            self.set_phase("polish")

        # ---------------- polishing (extension)
        if self.phase == "polish":
            global_ub = self.global_ub
            if p.polish_time > 0 and self.best_cols is not None and self.best_obj < math.floor(global_ub + 1e-6):
                from .compact import solve_compact
                r = solve_compact(inst, p.polish_time, p.workers, hint=self.best_cols,
                                  lower_bound=int(round(self.best_obj)))
                self.polish = dict(status=r["status"], minutes=r["lb"], time=r["time"])
                if r["columns"]:
                    cols = r["columns"]
                    if self.update_incumbent(cols, sum(c.value(inst) for c in cols), "CP-SAT polishing"):
                        self.add_pool(cols)
                if r["status"] == "OPTIMAL":
                    self.global_ub = min(self.global_ub, self.best_obj)
            self.set_phase("done")

        global_ub = max(self.global_ub, self.best_obj)
        if global_ub < math.inf:
            global_ub = math.floor(global_ub + 1e-6)
        self.global_ub = global_ub
        self.trajectory.append((round(self.elapsed(), 1), self.best_obj, global_ub))
        sm = inst.slot_minutes
        if self.best_cols is None:
            sol = Solution([], 0.0, global_ub * sm, 0.0)
        else:
            sol = Solution(self.best_cols, self.best_obj * sm, global_ub * sm, self.best_obj * sm)
        agg = dict(self.pricing_agg)
        sol.stats = dict(
            heuristic_minutes=heur_obj * sm, heuristic_errors=heur_errs,
            root=self.root, n_columns=len(self.pool), n_cuts=len(self.cuts),
            n_nodes=self.n_nodes, time=self.elapsed(),
            iter_log=[asdict(r) for r in self.iter_log], pricing_log=self.pricing_log, pricing_agg=agg,
            node_log=self.node_log, trivial_ub=trivial_ub * sm, params=asdict(p), polish=self.polish,
            lb_before_polish=(self.lb_before_polish * sm if self.lb_before_polish > -math.inf else None),
            trajectory=[(t, b * sm, u * sm) for t, b, u in self.trajectory],
            time_used=dict(self.used),
        )
        self.logf(f"[{self.elapsed():7.1f}s] done: LB {sol.lower_bound:.0f}  UB {sol.upper_bound:.0f}  gap {sol.gap:.2f}%")
        return sol

    def _log_node(self, **kw):
        kw.setdefault("pbound", getattr(self, "_cur_pbound", None))   # bound when popped (best-first audit)
        if len(self.node_log) < self.p.log_cap:
            self.node_log.append(kw)

    def branch_and_bound(self):
        """Best-first B&P&C over self.heap; resumable (state lives on self)."""
        p, inst = self.p, self.inst
        remaining = max(0.0, p.branch_time_limit - self.used["branch"])
        prove_left = max(0.0, p.branch_time_limit * p.prove_fraction - self.used["branch"])
        end = time.time() + remaining
        prove_end = time.time() + prove_left
        while self.heap and time.time() < end and self.n_nodes < p.max_nodes:
            self.maybe_checkpoint()
            negb, _, restr, rows, lbl, parent, depth, parent_proven = heapq.heappop(self.heap)
            pbound = -negb
            if math.floor(pbound + 1e-6) <= self.best_obj + 1e-6:
                continue   # pruned by bound
            self._cur_pbound = pbound
            nid = next(self._node_ids) + 1
            self.n_nodes += 1
            prove = time.time() < prove_end
            node_deadline = min(end, time.time() + max(10.0, p.cg_time_limit / 2))
            try:
                res = self.price_and_cut(nid, restr, rows, node_deadline, prove=prove)
            except RuntimeError as e:      # numerical trouble: keep the parent bound (conservative)
                self.unproven.append(pbound)
                self._log_node(id=nid, parent=parent, label=lbl, bound=pbound, lp=None,
                               converged=False, status=f"error: {e}", depth=depth)
                continue
            # every bound computed in price_and_cut (Lagrangian / LP) is valid
            bound = min(pbound, res["bound"])
            # NB: artificials > 0 at convergence do NOT prove infeasibility (finite big-M);
            # the penalised LP value is still a valid bound, so we only prune by bound.
            if bound < 0:
                self._log_node(id=nid, parent=parent, label=lbl, bound=None, lp=res["obj"],
                               converged=res["converged"], status="infeasible", depth=depth)
                continue
            status = "processed"
            x = res["x"]
            master = res["master"]
            integral = np.all(np.minimum(np.abs(x), np.abs(1 - x)) < 1e-6) and res["art"].sum() < 1e-6
            if integral:
                cols = [c for c, v in zip(master.columns, x) if v > 0.5]
                self.update_incumbent(cols, sum(c.value(inst) for c in cols), f"integral LP at node {nid}")
                status = "integral"
                if res["converged"]:
                    bound = min(bound, res["obj"])
            if math.floor(bound + 1e-6) <= self.best_obj + 1e-6:
                status = "pruned" if status != "integral" else status
                self._log_node(id=nid, parent=parent, label=lbl, bound=bound, lp=res["obj"],
                               converged=res["converged"], status=status, depth=depth)
                continue
            brc = self.choose_branch(master, x) if not integral else None
            if brc is None:
                cols, obj = master.solve_ip(min(30.0, max(1.0, end - time.time())), start=self.best_cols)
                self.update_incumbent(cols, obj, f"RMP-IP at node {nid}")
                if not integral:
                    self.unproven.append(bound)
                self._log_node(id=nid, parent=parent, label=lbl, bound=bound, lp=res["obj"],
                               converged=res["converged"], status="leaf-IP", depth=depth)
                continue
            self._log_node(id=nid, parent=parent, label=lbl, bound=bound, lp=res["obj"],
                           converged=res["converged"], status="branched", depth=depth)
            for r, rws, l2 in self.children(brc, restr, rows):
                self._cnt += 1
                heapq.heappush(self.heap, (-bound, self._cnt, r, rws, l2, nid, depth + 1, res["converged"]))
            del res, master


def solve(inst: Instance, params: BPCParams = BPCParams(), log=None, progress=None) -> Solution:
    s = Solver(inst, params, log=log, progress=progress)
    sol = s.run()
    sol.stats["heuristic_columns"] = s.heuristic["columns"]
    return sol
