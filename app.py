"""Streamlit dashboard for the CP-based branch-and-price-and-cut OR scheduler.

run:  streamlit run app.py
"""
from __future__ import annotations

import io
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from orsched import BPCParams, generate_instance, solve
from orsched.compact import solve_compact
from orsched.dataio import instance_from_frames, instance_to_frames
from orsched.heuristic import constructive_heuristic
from orsched.kpi import (cleaning_df, room_utilization, schedule_df, summary,
                         surgeon_utilization, waitlist_df)
from orsched.validate import validate_columns

st.set_page_config(page_title="OR Planner · B&P&C", page_icon="🏥", layout="wide")

# ----------------------------------------------------------------- palette
SERIES = ["#257de0", "#f47230", "#00b894", "#f2a600", "#ee7fac", "#008a17", "#6a4ce2", "#ec4647"]
GRAY, INK, INK2, GRID = "#9c9c9c", "#191919", "#666666", "#e6e6e6"
SEQ = [[0, "#cce3fd"], [0.5, "#519ced"], [1, "#0e3d7e"]]


def surgeon_color(l: int) -> str:
    return SERIES[l % len(SERIES)]


def style(fig, h=360, legend=True):
    has_title = bool(fig.layout.title.text)
    fig.update_layout(height=h, margin=dict(l=10, r=10, t=(78 if has_title and legend else 44), b=10),
                      title=dict(y=0.98, yanchor="top", x=0, xanchor="left") if has_title else None, plot_bgcolor="rgba(0,0,0,0)",
                      paper_bgcolor="rgba(0,0,0,0)", font=dict(size=13),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0) if legend else None,
                      showlegend=legend, hoverlabel=dict(bgcolor="white"))
    fig.update_xaxes(gridcolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


def hhmm(minutes: float, start_hour: int = 8) -> str:
    t = int(start_hour * 60 + minutes)
    return f"{t // 60:02d}:{t % 60:02d}"


# ----------------------------------------------------------------- sidebar
st.sidebar.title("🏥 OR Planner")
st.sidebar.caption("Integrated operating-room planning & scheduling — CP-based branch-and-price-and-cut "
                   "(Hashemi Doulabi, Rousseau & Pesant, IJOC 2016)")

with st.sidebar.expander("1 · Instance", expanded=True):
    source = st.radio("Data source", ["Generate (paper §6.1)", "Upload CSV"], horizontal=False)
    if source.startswith("Generate"):
        iset = st.selectbox("Instance set", ["A", "B", "C"],
                            help="A: 2–4 h cases, 6 rooms · B: 2–4 h cases, rooms sized by heuristic (tight) · "
                                 "C: short cases (Pearson III), rooms sized by heuristic")
        n = st.slider("Number of surgeries", 10, 120, 40, 5)
        days = st.slider("Days in horizon", 1, 7, 5)
        n_surg = st.slider("Surgeons", 2, 12, 8)
        rooms_override = st.number_input("Rooms per day (0 = set default)", 0, 20, 0)
        infect = st.slider("Share of infectious cases", 0.0, 1.0, 0.5, 0.05)
        ntypes = st.slider("Infection types", 1, 4, 1)
        oct_ = st.slider("Cleaning time OCT (min)", 0, 60, 30, 5)
        maxdl = st.slider("Deadlines drawn from day 1 … ", 1, 21, 14,
                          help="Deadlines beyond the horizon make the case optional (it may be postponed).")
        smean = st.slider("Mean surgeon max time / day (min)", 120, 480, 320, 10)
        seed = st.number_input("Random seed", 0, 10_000, 1)
    else:
        up_s = st.file_uploader("surgeries.csv", type="csv")
        up_g = st.file_uploader("surgeons.csv", type="csv")
        rooms_txt = st.text_input("Rooms per day (comma separated, or one number)", "4")
        oct_ = st.number_input("Cleaning time OCT (min)", 0, 120, 30, 5)
    day_minutes = st.number_input("OR day length (min)", 120, 900, 480, 30)
    slot = st.select_slider("Time slot (min)", [1, 5, 10, 15], 5)

with st.sidebar.expander("2 · Algorithm", expanded=True):
    use_dr = st.checkbox("Dominance rules (§4.1)", True)
    use_fid = st.checkbox("Fast infeasibility detection (§4.2)", True)
    use_cuts = st.checkbox("Lifted minimal cover cuts (§5.3)", True)
    copy_cols = st.checkbox("Copy columns to other days", True)
    branching = st.checkbox("Branching (B&P&C) — off = price-and-cut", True)
    cp_tl = st.slider("CP time limit, fast phase (s)", 0.5, 10.0, 2.0, 0.5)
    cp_ex = st.slider("CP time limit, optimality phase (s)", 2.0, 120.0, 20.0, 1.0)
    cg_tl = st.slider("Root column generation limit (s)", 10, 1800, 60, 10)
    br_tl = st.slider("Branching limit (s)", 0, 1800, 60, 10)
    ip_tl = st.slider("Final IP limit (s)", 5, 600, 30, 5)
    pol_tl = st.slider("CP-SAT polishing (s, 0 = off)", 0, 600, 30, 5,
                       help="Extension beyond the paper: a monolithic CP-SAT model warm-started with the "
                            "B&P&C incumbent. Helps on short-case instances where columns are hard to combine.")
    workers = st.slider("CP-SAT workers", 1, 16, 4)

# ------------------------------------------------------------ build instance
@st.cache_data(show_spinner=False)
def _gen(iset, n, days, n_surg, rooms, infect, ntypes, oct_, maxdl, smean, seed, day_minutes, slot):
    inst = generate_instance(n, iset, n_days=days, n_surgeons=n_surg, rooms=rooms or None,
                             day_minutes=day_minutes, slot_minutes=slot, cleaning_minutes=oct_,
                             infectious_share=infect, n_infection_types=ntypes, max_deadline=maxdl,
                             surgeon_mean_minutes=smean, seed=int(seed))
    s, g = instance_to_frames(inst)
    return s, g, inst.rooms, inst.name


if source.startswith("Generate"):
    base_s, base_g, base_rooms, inst_name = _gen(iset, n, days, n_surg, int(rooms_override), infect, ntypes,
                                                 oct_, maxdl, smean, seed, day_minutes, slot)
else:
    if up_s is None or up_g is None:
        st.info("Upload **surgeries.csv** and **surgeons.csv** (templates in the *Instance* tab), "
                "or switch to *Generate*.")
        s0, g0, _, _ = _gen("A", 20, 5, 4, 3, 0.5, 1, 30, 14, 320, 1, 480, 5)
        st.download_button("surgeries.csv template", s0.to_csv(index=False), "surgeries.csv")
        st.download_button("surgeons.csv template", g0.to_csv(index=False), "surgeons.csv")
        st.stop()
    base_s, base_g = pd.read_csv(up_s), pd.read_csv(up_g)
    base_rooms = [int(x) for x in rooms_txt.replace(" ", "").split(",") if x]
    inst_name = "uploaded"

key = (inst_name, len(base_s), source)
if st.session_state.get("data_key") != key:
    st.session_state.data_key = key
    st.session_state.surg_df = base_s.copy()
    st.session_state.surgeon_df = base_g.copy()
    st.session_state.rooms = list(base_rooms)
    st.session_state.pop("sol", None)

try:
    inst = instance_from_frames(st.session_state.surg_df, st.session_state.surgeon_df,
                                st.session_state.rooms, day_minutes=int(day_minutes), slot_minutes=int(slot),
                                cleaning_minutes=int(oct_), name=inst_name)
except Exception as e:  # noqa
    st.error(f"Could not build the instance: {e}")
    st.stop()

M = inst.slot_minutes

# ------------------------------------------------------------------- header
st.title("Operating-room planning & scheduling")
c = st.columns(5)
c[0].metric("Surgeries", inst.n)
c[1].metric("Mandatory (deadline in horizon)", len(inst.mandatory))
c[2].metric("Surgeons", inst.n_surgeons)
c[3].metric("OR-days available", sum(inst.rooms))
c[4].metric("Demand / OR capacity",
            f"{100 * sum(s.duration for s in inst.surgeries) / (sum(inst.rooms) * inst.slots_per_day):.0f}%")

run = st.button("▶  Solve with branch-and-price-and-cut", type="primary", use_container_width=True)

tabs = st.tabs(["① How it works", "② Instance & data", "③ Solver run", "④ Schedule",
                "⑤ Hospital KPIs", "⑥ Compare methods"])

# =================================================================== RUN
if run:
    params = BPCParams(cp_time_limit=cp_tl, cp_exact_time_limit=cp_ex, use_dominance=use_dr, use_fid=use_fid,
                       use_cuts=use_cuts, copy_columns=copy_cols, branching=branching, cg_time_limit=cg_tl,
                       branch_time_limit=br_tl, final_ip_time=ip_tl, polish_time=pol_tl,
                       workers=workers)
    log_lines = []
    with st.status("Solving…", expanded=True) as status_box:
        ph_txt = st.empty()
        ph_chart = st.empty()
        pts = []
        last = [0.0]

        def on_log(s):
            log_lines.append(s)
            ph_txt.code("\n".join(log_lines[-8:]))

        def on_progress(d):
            pts.append(d)
            if time.time() - last[0] > 0.7:
                last[0] = time.time()
                df = pd.DataFrame(pts)
                fig = go.Figure()
                fig.add_scatter(y=df.rmp * M, mode="lines", name="RMP objective (LP)", line=dict(color=SERIES[0], width=2))
                fig.add_scatter(y=np.minimum(df.bound * M, df.rmp.max() * M * 1.5), mode="lines",
                                name="Upper bound (Lagrangian)", line=dict(color=SERIES[1], width=2, dash="dot"))
                ph_chart.plotly_chart(style(fig, 260), use_container_width=True)

        t0 = time.time()
        sol = solve(inst, params, log=on_log, progress=on_progress)
        status_box.update(label=f"Done in {time.time() - t0:.1f}s — gap {sol.gap:.2f}%", state="complete",
                          expanded=False)
    st.session_state.sol = sol
    st.session_state.sol_inst_sig = (inst.n, tuple(inst.rooms), int(inst.surgeon_max.sum()))
    st.session_state.log = log_lines

sol = st.session_state.get("sol")
if sol is not None and st.session_state.get("sol_inst_sig") != (inst.n, tuple(inst.rooms), int(inst.surgeon_max.sum())):
    sol = None

# =================================================================== TAB 1
with tabs[0]:
    st.subheader("The idea in one picture")
    st.graphviz_chart("""
digraph G {
  rankdir=LR; bgcolor="transparent"; node [shape=box, style="rounded,filled", fillcolor="#eef5fe", color="#257de0", fontname="Helvetica", fontsize=11];
  edge [color="#666666", fontname="Helvetica", fontsize=10];
  H  [label="Constructive heuristic\\n(initial room-day schedules)"];
  RMP [label="Restricted master LP\\nchoose room-day schedules\\nmax Σ B_j x_j  (1)–(7)", fillcolor="#fff1e9", color="#f47230"];
  CUT [label="Lifted minimal cover cuts\\non surgeon daily limits (38)"];
  SUB [label="Pricing: one CP model per day\\nbest single-room schedule\\nwith positive reduced cost (18)–(19)", fillcolor="#e6f8f4", color="#00b894"];
  DR [label="Dominance rules 1–4\\n(shrink domains)"];
  FID [label="Knapsack bound\\n(fast infeasibility detection)"];
  BR [label="Branching\\n(1) #cases  (2) #cases/surgeon\\n(3) pair same room?"];
  IP [label="IP over all columns\\n→ final schedule + gap", fillcolor="#fff6dd", color="#f2a600"];
  H -> RMP; RMP -> CUT [label="fractional x"]; CUT -> RMP;
  RMP -> SUB [label="dual prices π"]; DR -> SUB; FID -> SUB; SUB -> RMP [label="new columns"];
  RMP -> BR [label="no column improves\\n& x fractional"]; BR -> RMP [label="child nodes"];
  BR -> IP;
}""")
    a, b = st.columns(2)
    with a:
        st.markdown("""
**Problem (IORPS).** Over a planning horizon (typically one week) decide *which* surgeries are done, *on which day*,
*in which room* and *at what time*, so that the **total scheduled surgery time is maximised** while
- every **mandatory** case (deadline inside the horizon) is done before its deadline; optional cases may be postponed,
- each **surgeon** stays below his/her maximum operating minutes per day,
- a surgeon is **never in two rooms** at the same time (open scheduling: rooms are shared by surgeons),
- an **obligatory cleaning** (OCT, e.g. 30 min) is inserted when a room switches infection type,
- rooms open at most `|K_d|` per day, each for a fixed day length.

**Why decomposition?** Compact models with `x[i,k,d,t]` variables explode (the paper's IP has a 47% average gap).
Instead, a *column* is a complete **schedule of one room for one day**. The master problem just picks
the best combination of columns; generating good columns is delegated to a constraint-programming
model that is very good at sequencing and timing.
""")
    with b:
        st.markdown("""
**Column generation loop (Algorithm 1).**
1. Solve the LP relaxation of the master with the current columns → dual prices for: each case (2)–(3),
   rooms per day (4), surgeon-hours (5) and surgeon *time-slot* occupation (6).
2. Separate **lifted cover cuts** on surgeon daily capacity and re-solve.
3. For every day solve the **CP pricing problem**: find one room schedule whose
   value `Σ G_di − Σ π*(start) − π⁴_d` is positive (a column that improves the LP).
   - *Dominance rules* fix the first start to 0 or ≥ the shortest case, drop cases that can only hurt,
     and drop/condition cases dominated by a better case of the same (or a free) surgeon.
   - The *knapsack bound* proves many subproblems useless before any search.
4. Phase 0 runs CP with a short time limit; when nothing is found it switches to the
   *optimality phase* to prove the LP bound.
5. **Branch-and-price-and-cut (Algorithm 2)** branches on (1) total #cases, (2) #cases of a surgeon,
   (3) Ryan–Foster *same room / different room* pairs, best-first.
6. A final IP over all generated columns gives the schedule; LP/Lagrangian bounds give the **optimality gap**.
7. *(Extension)* Optional CP-SAT polishing of the incumbent.
""")
    with st.expander("Key formulas"):
        st.latex(r"\max \sum_{j\in J} B_j x_j \quad\text{s.t.}\quad \sum_{j\in J_i} x_j = 1\ (i\in I_1),\ "
                 r"\sum_{j\in J_i} x_j\le 1\ (i\in I_2),\ \sum_{j\in J'_d} x_j\le |K_d|")
        st.latex(r"\sum_{j\in J'_d}\sum_{i\in I'_l} t_i a'_{ij}x_j \le A_{ld},\qquad "
                 r"\sum_{j\in J'_d}\sum_{i\in I'_l}\sum_{t'\le t<t'+t_i} a_{ijt'}x_j\le 1")
        st.latex(r"\text{pricing: }\max\Big\{\sum_p G_{d[W_p]}-\sum_p \pi^*_{d[W_p][V_p]}-\pi^{(4)}_d\Big\}"
                 r",\quad G_{di}=t_i-\pi^{(2,3)}_i-t_i\pi^{(5)}_{s_i d}")
        st.latex(r"\text{Lagrangian bound: } z_{LP}\le z_{RMP}+\sum_d |K_d|\cdot \max(0,\bar\sigma_d)")
    with st.expander("Implementation notes (what differs from the paper)"):
        st.markdown("""
- **Solvers**: HiGHS (LP/MIP) and OR-Tools CP-SAT instead of CPLEX / CP Optimizer. The pricing CP uses optional
  interval variables (equivalent to the position model (8)–(15)); cleaning is enforced between every pair of
  different-infection cases, which is equivalent to enforcing it between consecutive cases.
- A redundant **makespan cut** (total duration + one cleaning per extra infection group ≤ day length) makes the CP
  pricing much faster on short-case instances.
- **Cover lifting** uses Balas' sequence-independent coefficients with the *largest* cover weights (the
  "nondecreasing" ordering printed in eq. (39) can produce invalid cuts).
- Dominance rules 2–4 and the knapsack bound are applied at the root of each CP search (p* = 1).
- Valid **upper bounds** at every iteration from the Lagrangian bound, so time-limited runs still report a true gap.
- **Artificial variables** keep the master feasible for mandatory cases and ≥-branching rows.
- **CP-SAT polishing** (optional, extension): after the final IP, a monolithic CP-SAT model warm-started with the
  incumbent improves the schedule; the B&P&C bound is still used for the gap.
""")
    if sol is not None:
        st.subheader("Trace of your last run")
        it = pd.DataFrame(sol.stats["iter_log"])
        if len(it):
            r0 = it[it.node == 0]
            st.markdown(f"Root node: **{len(r0)} CG iterations**, LP value **{sol.stats['root']['lp'] * M:,.0f} min**, "
                        f"{'converged ✅' if sol.stats['root']['converged'] else 'stopped by time limit ⏱'}; "
                        f"{sol.stats['n_columns']} columns and {sol.stats['n_cuts']} cuts generated, "
                        f"{sol.stats['n_nodes']} branch-and-bound nodes.")

# =================================================================== TAB 2
with tabs[1]:
    st.subheader("Cases (editable)")
    st.caption("Edit durations, deadlines (day number; > horizon = optional), surgeons or infection types, "
               "then hit **Solve**. Infection 0 = non-infectious, k = infection type k.")
    edited = st.data_editor(st.session_state.surg_df, num_rows="dynamic", use_container_width=True, height=300,
                            key=f"ed_s_{st.session_state.data_key}")
    a, b = st.columns([2, 1])
    with a:
        st.subheader("Surgeon max operating minutes per day")
        edited_g = st.data_editor(st.session_state.surgeon_df, use_container_width=True,
                                  key=f"ed_g_{st.session_state.data_key}")
    with b:
        st.subheader("Rooms per day")
        rdf = pd.DataFrame({"day": [inst.day_name(d) for d in range(inst.n_days)],
                            "rooms": (st.session_state.rooms * inst.n_days)[:inst.n_days]
                            if len(st.session_state.rooms) == 1 else st.session_state.rooms})
        edited_r = st.data_editor(rdf, use_container_width=True, disabled=["day"],
                                  key=f"ed_r_{st.session_state.data_key}")
    if st.button("Apply edits"):
        st.session_state.surg_df = edited
        st.session_state.surgeon_df = edited_g
        st.session_state.rooms = [int(x) for x in edited_r.rooms]
        st.session_state.pop("sol", None)
        st.rerun()
    s_csv, g_csv = instance_to_frames(inst)
    d1, d2 = st.columns(2)
    d1.download_button("Download surgeries.csv", s_csv.to_csv(index=False), "surgeries.csv")
    d2.download_button("Download surgeons.csv", g_csv.to_csv(index=False), "surgeons.csv")

    st.subheader("Demand vs. capacity")
    dem = pd.DataFrame([dict(day=min(s.deadline, inst.n_days), mins=s.duration * M) for s in inst.surgeries])
    by = dem.groupby("day").mins.sum().reindex(range(inst.n_days + 1), fill_value=0)
    lab = [f"due {inst.day_name(d)}" for d in range(inst.n_days)] + ["optional"]
    cap = [inst.rooms[d] * inst.slots_per_day * M for d in range(inst.n_days)]
    fig = go.Figure()
    fig.add_bar(x=lab, y=by.values, name="case minutes by deadline", marker_color=SERIES[0],
                hovertemplate="%{x}: %{y:,.0f} min<extra></extra>")
    fig.add_scatter(x=lab[:-1], y=np.cumsum(cap), name="cumulative OR capacity", mode="lines+markers",
                    line=dict(color=SERIES[1], width=2), marker=dict(size=8),
                    hovertemplate="by %{x}: %{y:,.0f} min<extra></extra>")
    fig.add_scatter(x=lab[:-1], y=np.cumsum(by.values[:-1]), name="cumulative mandatory demand",
                    mode="lines+markers", line=dict(color=SERIES[2], width=2), marker=dict(size=8),
                    hovertemplate="by %{x}: %{y:,.0f} min<extra></extra>")
    fig.update_yaxes(title="minutes")
    st.plotly_chart(style(fig, 330), use_container_width=True)

# =================================================================== TAB 3
with tabs[2]:
    if sol is None:
        st.info("Press **Solve** to run the algorithm.")
    else:
        k = st.columns(6)
        k[0].metric("Scheduled (LB)", f"{sol.lower_bound:,.0f} min")
        k[1].metric("Upper bound", f"{sol.upper_bound:,.0f} min")
        k[2].metric("Optimality gap", f"{sol.gap:.2f}%")
        k[3].metric("vs. heuristic", f"+{sol.lower_bound - sol.stats['heuristic_minutes']:,.0f} min")
        k[4].metric("Columns / cuts", f"{sol.stats['n_columns']} / {sol.stats['n_cuts']}")
        k[5].metric("Time", f"{sol.stats['time']:.1f}s")
        pol = sol.stats.get("polish")
        if pol:
            st.caption(f"CP-SAT polishing: {pol['status']} in {pol['time']:.1f}s"
                       + (f" → {pol['minutes']:,.0f} min" if pol['minutes'] else ""))
        errs = validate_columns(inst, sol.columns)
        if errs:
            st.error("Schedule violates constraints:\n" + "\n".join(errs[:10]))
        else:
            st.success("Independent validator: every constraint satisfied (deadlines, rooms, surgeon limits, "
                       "no surgeon overlap, cleaning).")

        it_all = pd.DataFrame(sol.stats["iter_log"])
        st.subheader("Convergence")
        nodes = sorted(it_all.node.unique())
        nsel = st.selectbox("Node", nodes, format_func=lambda v: "root (node 0)" if v == 0 else f"node {v}",
                            help="Column-generation iterations of one branch-and-bound node. Iterations where "
                                 "artificial variables are still positive (master not yet feasible) are hidden.")
        it = it_all[it_all.node == nsel].copy()
        feas = it.artificial < 1e-6
        fig = go.Figure()
        x = it.iteration
        fig.add_scatter(x=x, y=np.where(feas, it.rmp_obj * M, np.nan), name="Master LP objective",
                        line=dict(color=SERIES[0], width=2), marker=dict(size=8),
                        customdata=np.stack([it.n_columns, it.new_columns, it.phase], -1),
                        hovertemplate="it %{x}: LP %{y:,.0f} min<br>%{customdata[0]} columns "
                                      "(+%{customdata[1]}) · phase %{customdata[2]}<extra></extra>")
        fig.add_scatter(x=x, y=np.minimum(it.lagrangian_bound * M, sol.stats["trivial_ub"]),
                        name="Best valid bound (Lagrangian)", line=dict(color=SERIES[1], width=2, dash="dot"),
                        marker=dict(size=8), hovertemplate="it %{x}: bound %{y:,.0f} min<extra></extra>")
        fig.add_hline(y=sol.lower_bound, line=dict(color=SERIES[2], width=2),
                      annotation_text=f"best schedule {sol.lower_bound:,.0f}", annotation_position="bottom right")
        fig.add_hline(y=sol.stats["heuristic_minutes"], line=dict(color=GRAY, width=1, dash="dash"),
                      annotation_text="heuristic", annotation_position="bottom left")
        lo = min(sol.stats["heuristic_minutes"], float(np.nanmin(np.where(feas, it.rmp_obj * M, np.inf)))
                 if feas.any() else sol.stats["heuristic_minutes"]) * 0.97
        hi = float(np.minimum(it.lagrangian_bound * M, sol.stats["trivial_ub"]).iloc[min(2, len(it) - 1)])
        fig.update_yaxes(title="scheduled minutes", range=[lo, max(hi, sol.upper_bound) * 1.03 + 1])
        fig.update_xaxes(title="column-generation iteration", dtick=max(1, len(it) // 12))
        st.caption("Phase 0 = CP with short time limit; phase 1 = optimality-proof phase. The bound is valid at "
                   "every iteration, so the gap is honest even when a time limit stops the run.")
        st.plotly_chart(style(fig, 380), use_container_width=True)

        pr = pd.DataFrame(sol.stats["pricing_log"])
        st.subheader("Pricing-subproblem enhancements (cf. paper Table 2)")
        if len(pr):
            cand = pr.candidates.replace(0, np.nan)
            t2 = pd.DataFrame([{
                "DR1 – start values removed (%)": pr.dr1_removed.mean(),
                "DR2 – cases removed (%)": 100 * (pr.dr2 / cand).mean(),
                "DR3 – cases removed (%)": 100 * (pr.dr3 / cand).mean(),
                "DR4 – cases removed (%)": 100 * (pr.dr4 / cand).mean(),
                "Conditioned cases (%)": 100 * (pr.conditioned / cand).mean(),
                "FID1 – infeasible by knapsack (%)": 100 * pr.fid_infeasible.mean(),
                "FID2 – emptied by dominance (%)": 100 * pr.fid2_infeasible.mean(),
                "Stopped at knapsack bound (%)": 100 * pr.hit_knapsack_bound.mean(),
                "CP calls": len(pr), "CP time (s)": pr.cp_time.sum()}]).T.rename(columns={0: "value"})
            a, b = st.columns([1, 1])
            a.dataframe(t2.style.format("{:.2f}"), use_container_width=True)
            byday = pr.groupby("day").agg(calls=("cp_time", "size"), cp_time=("cp_time", "sum"),
                                          n_cols=("n_columns", "sum")).reset_index()
            byday["day"] = [inst.day_name(d) for d in byday.day]
            fig = go.Figure(go.Bar(x=byday.day, y=byday.n_cols, marker_color=SERIES[0],
                                   hovertemplate="%{x}: %{y} columns<extra></extra>"))
            fig.update_layout(title="Columns generated per day subproblem")
            b.plotly_chart(style(fig, 300, legend=False), use_container_width=True)

        nl = pd.DataFrame(sol.stats["node_log"])
        st.subheader(f"Branch-and-bound tree ({len(nl)} nodes)")
        if len(nl) > 30:
            vc = nl.status.astype(str).value_counts()
            st.markdown(" · ".join(f"**{v}** {k}" for k, v in vc.items()))
            show_nl = nl.assign(bound_min=nl.bound * M, lp_min=nl.lp * M)[
                ["id", "parent", "depth", "label", "status", "lp_min", "bound_min", "converged"]]
            st.dataframe(show_nl, use_container_width=True, hide_index=True, height=300)
        elif len(nl) > 1:
            colors = {"branched": "#eef5fe", "pruned": "#efefef", "integral": "#e6f8f4", "leaf-IP": "#fff6dd",
                      "infeasible": "#fde8e8", "processed": "#eef5fe"}
            lines = ['digraph T { bgcolor="transparent"; node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=10];']
            for _, r in nl.head(120).iterrows():
                bd = "—" if pd.isna(r.bound) else f"{r.bound * M:,.0f}"
                lines.append(f'n{r.id} [label="#{r.id} {r.label}\\nbound {bd}\\n{r.status}", '
                             f'fillcolor="{colors.get(str(r.status), "#ffffff")}"];')
                if r.parent is not None and not pd.isna(r.parent):
                    lines.append(f"n{int(r.parent)} -> n{r.id};")
            lines.append("}")
            st.graphviz_chart("\n".join(lines))
        else:
            st.caption("Solved at the root node — no branching needed.")
        with st.expander("Solver log"):
            st.code("\n".join(st.session_state.get("log", [])))

# =================================================================== TAB 4
def gantt(inst, cols, d, title=""):
    df = schedule_df(inst, cols)
    df = df[df.day == d + 1]
    cl = cleaning_df(inst, cols)
    cl = cl[cl.day == d + 1]
    rooms = [f"OR {k + 1}" for k in range(inst.rooms[d])]
    fig = go.Figure()
    for l in range(inst.n_surgeons):
        sub = df[df.surgeon_idx == l]
        if not len(sub):
            continue
        fig.add_bar(y=sub.room, x=sub.duration_min, base=sub.start_min, orientation="h",
                    name=inst.surgeon_name(l), marker=dict(color=surgeon_color(l), line=dict(color="white", width=2),
                                                         pattern=dict(shape=["/" if v else "" for v in sub.infectious],
                                                                      fgcolor="rgba(255,255,255,.8)", size=7,
                                                                      fillmode="overlay")),
                    text=sub.case, textposition="inside", insidetextanchor="middle",
                    customdata=np.stack([sub.case, sub.surgeon, [hhmm(v) for v in sub.start_min],
                                         [hhmm(v) for v in sub.end_min], sub.infection,
                                         sub.deadline_day.fillna(0).astype(int)], -1),
                    hovertemplate="<b>%{customdata[0]}</b> · %{customdata[1]}<br>%{customdata[2]}–%{customdata[3]}"
                                  "<br>infection type %{customdata[4]} · deadline day %{customdata[5]}<extra></extra>")
    if len(cl):
        fig.add_bar(y=cl.room, x=cl.end_min - cl.start_min, base=cl.start_min, orientation="h", name="Cleaning",
                    marker=dict(color="#d9d9d9", line=dict(color="white", width=2)),
                    hovertemplate="cleaning %{base}–%{x}<extra></extra>")
    T = inst.slots_per_day * M
    ticks = list(range(0, T + 1, 60))
    fig.update_xaxes(range=[0, T], tickvals=ticks, ticktext=[hhmm(t) for t in ticks])
    fig.update_yaxes(categoryorder="array", categoryarray=rooms[::-1])
    fig.update_layout(barmode="overlay", title=title, bargap=0.25)
    return style(fig, 110 + 46 * len(rooms))


with tabs[3]:
    if sol is None:
        st.info("Press **Solve** to compute a schedule.")
    else:
        d = st.radio("Day", list(range(inst.n_days)), format_func=inst.day_name, horizontal=True)
        st.plotly_chart(gantt(inst, sol.columns, d, f"{inst.day_name(d)} — rooms (hatched = infectious case)"),
                        use_container_width=True)
        # surgeon view
        df = schedule_df(inst, sol.columns)
        sd = df[df.day == d + 1]
        if len(sd):
            fig = go.Figure()
            for l in sorted(sd.surgeon_idx.unique()):
                sub = sd[sd.surgeon_idx == l]
                fig.add_bar(y=sub.surgeon, x=sub.duration_min, base=sub.start_min, orientation="h",
                            marker=dict(color=surgeon_color(l), line=dict(color="white", width=2)), showlegend=False,
                            text=sub.room, textposition="inside",
                            customdata=np.stack([sub.case, sub.room], -1),
                            hovertemplate="%{customdata[0]} in %{customdata[1]}<extra></extra>")
            T = inst.slots_per_day * M
            ticks = list(range(0, T + 1, 60))
            fig.update_xaxes(range=[0, T], tickvals=ticks, ticktext=[hhmm(t) for t in ticks])
            fig.update_layout(barmode="overlay", title="Surgeon timelines (room label inside) — never two places at once")
            st.plotly_chart(style(fig, 120 + 34 * sd.surgeon.nunique(), legend=False), use_container_width=True)
        st.subheader("Full schedule")
        show = df.assign(start=[hhmm(v) for v in df.start_min], end=[hhmm(v) for v in df.end_min])[
            ["day_name", "room", "start", "end", "case", "surgeon", "duration_min", "infection", "mandatory",
             "deadline_day", "slack_days"]]
        st.dataframe(show, use_container_width=True, height=320)
        buf = io.StringIO(); show.to_csv(buf, index=False)
        st.download_button("Download schedule.csv", buf.getvalue(), "schedule.csv")

# =================================================================== TAB 5
with tabs[4]:
    if sol is None:
        st.info("Press **Solve** to see KPIs.")
    else:
        S = summary(inst, sol.columns)
        H = summary(inst, sol.stats["heuristic_columns"])
        k = st.columns(6)
        k[0].metric("OR utilisation",  f"{S['utilization_opened_rooms']:.1f}%",
                    f"{S['utilization_opened_rooms'] - H['utilization_opened_rooms']:+.1f} pt vs heuristic")
        k[1].metric("Surgery minutes", f"{S['scheduled_minutes']:,.0f}",
                    f"{S['scheduled_minutes'] - H['scheduled_minutes']:+,.0f} vs heuristic")
        k[2].metric("Mandatory cases done", f"{S['mandatory_scheduled']}/{S['mandatory_total']}")
        k[3].metric("Optional cases done", f"{S['optional_scheduled']}/{S['optional_total']}",
                    f"{S['optional_scheduled'] - H['optional_scheduled']:+d} vs heuristic")
        k[4].metric("Rooms opened", f"{S['rooms_opened']}/{S['rooms_available']}")
        k[5].metric("Cleanings", f"{S['cleanings']}", help=f"{S['cleaning_minutes']:.0f} min of cleaning",
                    delta=f"{S['cleanings'] - H['cleanings']:+d} vs heuristic", delta_color="inverse")

        a, b = st.columns(2)
        ru = room_utilization(inst, sol.columns)
        with a:
            piv = ru.pivot(index="room", columns="day_name", values="utilization")
            piv = piv[[inst.day_name(d) for d in range(inst.n_days) if inst.day_name(d) in piv.columns]]
            fig = go.Figure(go.Heatmap(z=piv.values, x=piv.columns, y=piv.index, colorscale=SEQ, zmin=0, zmax=100,
                                       text=np.round(piv.values).astype("float"), texttemplate="%{text:.0f}%",
                                       hovertemplate="%{y} · %{x}: %{z:.1f}% surgery time<extra></extra>",
                                       colorbar=dict(title="%")))
            fig.update_yaxes(autorange="reversed")
            fig.update_layout(title="Room utilisation (surgery time / day length)")
            st.plotly_chart(style(fig, 330, legend=False), use_container_width=True)
        with b:
            agg = ru.groupby("day_name", sort=False)[["surgery_min", "cleaning_min", "idle_min"]].sum().reset_index()
            fig = go.Figure()
            for col, nm, c_ in [("surgery_min", "Surgery", SERIES[0]), ("cleaning_min", "Cleaning", GRAY),
                                ("idle_min", "Idle (incl. closed rooms)", "#e6e6e6")]:
                fig.add_bar(x=agg.day_name, y=agg[col], name=nm, marker=dict(color=c_, line=dict(color="white", width=2)),
                            hovertemplate=nm + " %{y:,.0f} min<extra></extra>")
            fig.update_layout(barmode="stack", title="How OR time is spent per day")
            fig.update_yaxes(title="minutes")
            st.plotly_chart(style(fig, 330), use_container_width=True)

        su = surgeon_utilization(inst, sol.columns)
        a, b = st.columns(2)
        with a:
            g = su.groupby(["surgeon", "surgeon_idx"])[["scheduled_min", "max_min"]].sum().reset_index()
            g = g.sort_values("surgeon_idx")
            fig = go.Figure()
            fig.add_bar(y=g.surgeon, x=g.max_min, orientation="h", name="Available (max hours)",
                        marker=dict(color="#e6e6e6"), hovertemplate="available %{x:,.0f} min<extra></extra>")
            fig.add_bar(y=g.surgeon, x=g.scheduled_min, orientation="h", name="Scheduled",
                        marker=dict(color=SERIES[0]), width=0.45, hovertemplate="scheduled %{x:,.0f} min<extra></extra>")
            fig.update_layout(barmode="overlay", title="Surgeon workload over the horizon")
            fig.update_yaxes(autorange="reversed")
            st.plotly_chart(style(fig, 330), use_container_width=True)
        with b:
            df = schedule_df(inst, sol.columns)
            mand = df[df.mandatory]
            if len(mand):
                vc = mand.slack_days.value_counts().sort_index()
                fig = go.Figure(go.Bar(x=[f"{int(v)} d" for v in vc.index], y=vc.values, marker_color=SERIES[2],
                                       hovertemplate="%{y} cases done %{x} before deadline<extra></extra>"))
                fig.update_layout(title="Mandatory cases: days between surgery and deadline")
                fig.update_yaxes(title="cases")
                st.plotly_chart(style(fig, 330, legend=False), use_container_width=True)
        st.subheader("Waiting list after this plan")
        wl = waitlist_df(inst, sol.columns)
        if len(wl):
            st.caption("Cases not scheduled in the horizon (optional, postponed to the next planning period).")
            st.dataframe(wl, use_container_width=True, hide_index=True)
        else:
            st.success("Every case is scheduled within the horizon.")

# =================================================================== TAB 6
with tabs[5]:
    st.markdown("Compare the B&P&C schedule with the **constructive heuristic** and a **monolithic CP-SAT model** "
                "(all decisions in one model, in the spirit of the paper's pure CP / IP baselines). Paper results "
                "(Table 6 & 7) are shown for reference.")
    tl = st.slider("Time limit for the monolithic model (s)", 10, 600, 60, 10)
    if st.button("Run monolithic CP-SAT baseline"):
        with st.spinner("Solving the compact model…"):
            st.session_state.compact = solve_compact(inst, tl, workers)
    rows = []
    hc, _ = constructive_heuristic(inst, np.random.default_rng(0))
    rows.append(dict(method="Constructive heuristic (§6.1)", scheduled_min=sum(c.value(inst) for c in hc) * M,
                     upper_bound=None, gap=None, time_s=None))
    if sol is not None:
        rows.append(dict(method="Branch-and-price-and-cut", scheduled_min=sol.lower_bound,
                         upper_bound=sol.upper_bound, gap=sol.gap, time_s=sol.stats["time"]))
    cp = st.session_state.get("compact")
    if cp:
        g = None if cp["lb"] is None else 100 * (cp["ub"] - cp["lb"]) / cp["ub"]
        rows.append(dict(method=f"Monolithic CP-SAT ({cp['status']})", scheduled_min=cp["lb"], upper_bound=cp["ub"],
                         gap=g, time_s=cp["time"]))
    st.dataframe(pd.DataFrame(rows), use_container_width=True)
    st.subheader("Paper reference — average optimality gaps (%)")
    st.dataframe(pd.DataFrame({
        "method": ["IP (compact, CPLEX)", "Pure CP", "CP→IP", "Original CG", "Enhanced CG", "Price-and-cut",
                   "Branch-and-price", "Branch-and-price-and-cut"],
        "avg gap %": [46.94, 17.68, 11.84, 9.90, 7.82, 4.76, 5.36, 2.81]}), use_container_width=True, hide_index=True)
