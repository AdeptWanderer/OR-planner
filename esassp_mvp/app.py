"""Streamlit dashboard — OR-to-downstream elective surgery planning under uncertainty (SP / DRO-M / DRO-W).

run:  streamlit run app_esassp.py
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from esassp import Costs, Params, generate_instance
from esassp.dataio import instance_from_frames, instance_to_frames
from esassp.experiment import DELTAS, run_experiment
from esassp.model import DAY_NAMES

st.set_page_config(page_title="ESASSP · OR-to-downstream planner", page_icon="🏥", layout="wide")

SERIES = ["#257de0", "#f47230", "#00b894", "#f2a600", "#ee7fac", "#008a17", "#6a4ce2", "#ec4647"]
MODEL_COLOR = {"SP": "#f47230", "DRO-M": "#257de0", "DRO-W": "#00b894"}
GRID, INK2 = "#e6e6e6", "#666666"


def style(fig, h=340, legend=True):
    fig.update_layout(height=h, margin=dict(l=10, r=10, t=40 if legend else 20, b=10), plot_bgcolor="rgba(0,0,0,0)",
                      paper_bgcolor="rgba(0,0,0,0)", font=dict(size=13), showlegend=legend,
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0), hoverlabel=dict(bgcolor="white"))
    fig.update_xaxes(gridcolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


def mcolor(name: str) -> str:
    return MODEL_COLOR.get(name.split(" ")[0], "#9c9c9c")


def hhmm(minutes: float, start_hour: int = 8) -> str:
    t = int(start_hour * 60 + minutes)
    return f"{t // 60:02d}:{t % 60:02d}"


# ----------------------------------------------------------------- sidebar
st.sidebar.title("🏥 ESASSP planner")
st.sidebar.caption("Integrated surgery assignment, sequencing & scheduling with ICU / ward capacity — "
                   "SP and distributionally robust models (Shehadeh, Tsang, Padman & Kilic, EJOR 2026)")

with st.sidebar.expander("1 · Instance", expanded=True):
    source = st.radio("Data source", ["Generate (synthetic, paper §6.1 recipe)", "Upload CSV"])
    if source.startswith("Generate"):
        library = st.selectbox("Surgery types", ["cardiac", "general"],
                               help="cardiac: 5 cardiac types (collaborating hospital style) · general: 6 specialties "
                                    "(Mannino-style). All numbers except ENT/OPHTH durations and VASCULAR LOS are synthetic.")
        n = st.slider("Number of surgeries I", 8, 60, 20, 1)
        ors = st.slider("Operating rooms (Mon–Fri blocks)", 1, 4, 2)
        beds = st.selectbox("Beds (weekday, weekend)", ["(10,7)", "(5,2)"], index=1,
                            help="ICU and ward capacity on weekdays / weekends. Tight capacities show the effect best.")
        cost = st.radio("Cost structure", ["Cost 1 (overtime $26/min)", "Cost 2 (overtime $100/min)"])
        seed = st.number_input("Random seed", 0, 10_000, 1)
    else:
        up_s = st.file_uploader("surgeries.csv", type="csv")
        up_b = st.file_uploader("blocks.csv", type="csv")
        up_d = st.file_uploader("beds.csv", type="csv")
        cost = st.radio("Cost structure", ["Cost 1 (overtime $26/min)", "Cost 2 (overtime $100/min)"])

with st.sidebar.expander("2 · Models & algorithm", expanded=True):
    models = st.multiselect("Models", ["SP", "DRO-M", "DRO-W"], default=["SP", "DRO-M"],
                            help="SP: sample average approximation. DRO-M: mean–support ambiguity. DRO-W: 1-Wasserstein ball.")
    baselines = st.checkbox("Also solve non-integrated baselines (w/o downstream, w/o scheduling)", True)
    N = st.slider("Scenarios N (SP sample / DRO-W empirical)", 5, 100, 20, 5)
    eps_d = st.slider("Wasserstein radius ε^d (minutes)", 5.0, 200.0, 50.0, 5.0)
    eps_l = st.slider("Wasserstein radius ε^l (days)", 0.5, 20.0, 5.0, 0.5)
    tl = st.slider("Time limit per model (s)", 10, 900, 90, 10)
    gap = st.select_slider("Target gap", [0.0001, 0.001, 0.005, 0.01, 0.02, 0.05], 0.001)
    threads = st.slider("HiGHS threads", 1, 8, 4)
    sb = st.checkbox("Symmetry-breaking constraints (27)–(29)", True)
    mc = st.checkbox("Per-block cuts (24)/(25)", True)
    rb = st.checkbox("Valid inequality (26) on ρ", True)

with st.sidebar.expander("3 · Out-of-sample evaluation", expanded=False):
    n_oos = st.slider("Simulation scenarios", 500, 20000, 5000, 500)
    delta_view = st.select_slider("Mis-specification Δ (shown in tabs ⑤–⑥)", list(DELTAS), 0.0,
                                  help="Δ>0: longer ICU/ward stays than assumed in the optimisation, wider duration range.")

costs = Costs(c_o=26.0) if cost.startswith("Cost 1") else Costs(c_o=100.0)


# ----------------------------------------------------------------- instance
@st.cache_data(show_spinner=False)
def _gen(library, n, ors, beds, c_o, seed):
    inst = generate_instance(n, library, int(seed), n_ors=ors, beds=beds, costs=Costs(c_o=c_o))
    return instance_to_frames(inst), inst.name


if source.startswith("Generate"):
    frames, inst_name = _gen(library, n, ors, beds, costs.c_o, seed)
else:
    if not (up_s and up_b and up_d):
        st.info("Upload **surgeries.csv**, **blocks.csv** and **beds.csv** (templates below), or switch to *Generate*.")
        f0, _ = _gen("cardiac", 12, 1, "(5,2)", 26.0, 1)
        for k, v in f0.items():
            st.download_button(f"{k}.csv template", v.to_csv(index=False), f"{k}.csv", key=f"tpl{k}")
        st.stop()
    frames, inst_name = dict(surgeries=pd.read_csv(up_s), blocks=pd.read_csv(up_b), beds=pd.read_csv(up_d)), "uploaded"

sig = (inst_name, len(frames["surgeries"]), len(frames["blocks"]), costs.c_o,
       None if not source.startswith("Generate") else (library, n, ors, beds, seed))
if st.session_state.get("sig") != sig:
    st.session_state.sig = sig
    st.session_state.frames = {k: v.copy() for k, v in frames.items()}
    st.session_state.pop("exp", None)
try:
    fr = st.session_state.frames
    inst = instance_from_frames(fr["surgeries"], fr["blocks"], fr["beds"], costs, inst_name)
except Exception as e:  # noqa
    st.error(f"Could not build the instance: {e}")
    st.stop()

# ----------------------------------------------------------------- header
st.title("Operating-room-to-downstream elective surgery planning")
c = st.columns(5)
c[0].metric("Surgeries", inst.I)
c[1].metric("Surgical blocks (OR-days)", inst.B)
c[2].metric("Expected demand / OR time", f"{100 * inst.mu_d.sum() / sum(b.length for b in inst.blocks):.0f}%")
c[3].metric("Expected ICU bed-days / capacity", f"{inst.mu_c.sum():.0f} / {inst.icu_beds.sum():.0f}")
c[4].metric("Expected ward bed-days / capacity", f"{inst.mu_w.sum():.0f} / {inst.ward_beds.sum():.0f}")

run = st.button("▶  Solve and evaluate", type="primary", width="stretch", disabled=not models)
tabs = st.tabs(["① How it works", "② Instance & data", "③ Solver run", "④ Schedule", "⑤ Robustness", "⑥ Compare models"])

# =================================================================== RUN
if run:
    params = Params(n_scenarios=N, time_limit=float(tl), master_time=float(max(10, tl / 2)), gap=float(gap), threads=threads,
                    symmetry=sb, multicut=mc, rho_bound=rb, eps_d=eps_d, eps_l=eps_l)
    lines = []
    with st.status("Solving…", expanded=True) as box:
        ph = st.empty()

        def on_log(s):
            lines.append(s)
            ph.code("\n".join(lines[-10:]))
        params.log = on_log
        t0 = time.time()
        st.session_state.exp = run_experiment(inst, params, models=tuple(models), baselines=baselines, n_oos=n_oos, log=on_log)
        st.session_state.log = lines
        box.update(label=f"Done in {time.time() - t0:.1f}s", state="complete", expanded=False)

exp = st.session_state.get("exp")

# =================================================================== TAB 1
with tabs[0]:
    st.subheader("The problem")
    st.markdown(
        "A hospital must decide, for a week, **which elective surgeries to perform and which to postpone**, in **which OR block** "
        "and in **what order and start time** — while surgery durations and the post-operative ICU / ward stays are uncertain. "
        "Too many surgeries → OR overtime, patient waiting and **no recovery bed** (the real bottleneck); too few → idle OR time and "
        "long waiting lists. Solving assignment, sequencing and scheduling **together with the downstream beds** is the paper's point.")
    st.graphviz_chart("""
digraph G { rankdir=LR; bgcolor="transparent"; node [shape=box, style="rounded,filled", fillcolor="#eef5fe", color="#257de0", fontname="Helvetica", fontsize=11];
 edge [color="#666666", fontname="Helvetica", fontsize=10];
 M [label="Master (MILP, HiGHS)\\nx[i,b,k], s[b,k] + multipliers ρ, φ, ζ\\nrecourse copies for scenarios in O", fillcolor="#fff1e9", color="#f47230"];
 S1 [label="Separation BC (per block)\\nworst durations d*:\\nvertex enumeration / dual MILP", fillcolor="#e6f8f4", color="#00b894"];
 S2 [label="Separation downstream\\nworst LOS l*:\\nMILP on monotone ICU/ward-exit binaries", fillcolor="#e6f8f4", color="#00b894"];
 U [label="UB = master − δ + true values\\nLB = master dual bound", shape=note, fillcolor="#fff"];
 M -> S1 [label="x, s, ρ"]; M -> S2 [label="x, φ, ζ"]; S1 -> M [label="new scenario d*"]; S2 -> M [label="new scenario l*"]; M -> U; }
""")
    l, r = st.columns(2)
    with l:
        st.markdown("**Second-stage costs** (given plan x, s and random d, l)")
        st.latex(r"Q^{BC}=\min \sum_{b,k} c^w w_{bk}+\sum_b (c^g g_b + c^o o_b)\quad \text{(4b)–(4e)}")
        st.latex(r"Q^{down}=\min \sum_t c^{icu}_t u_t + c^{ward}_t n_t,\ \ u_t\ge \text{ICU}_t-\text{Bed}^{icu}_t\ \text{(5b)–(5e)}")
        st.markdown("**DRO-M dual** (Props. 1–2): the worst case over all distributions with the given mean and support")
        st.latex(r"\sup_{P:\,E d=\mu}E\,Q^{BC}=\min_{\rho}\ \sum_i \mu_i\rho_i x_i+\max_{d\in[\underline d,\overline d]}\big[Q^{BC}-\textstyle\sum_i d_i\rho_i x_i\big]")
    with r:
        st.markdown("**DRO-W**: Wasserstein ball of radius ε around N historical samples")
        st.latex(r"\sup_{W_1(P,\hat P_N)\le\varepsilon}E\,Q=\min_{\nu\ge 0}\ \varepsilon\nu+\tfrac1N\sum_n\max_{d}\big[Q(d)-\nu\|d-d^n\|_1\big]")
        st.markdown("""
**Why the separation problems are small**: `Q^BC` is convex piecewise-linear in the durations, so the worst case sits on the
vertices of the duration box (Wasserstein: `{lo, d^n, hi}` per surgery) — enumerated for blocks up to 11 surgeries, otherwise a MILP
built from the dual of (4b)–(4e). The downstream worst case is an integer program over ICU/ward exit days.
""")
    with st.expander("Differences from the paper (honest list)"):
        st.markdown("""
* **Solver**: HiGHS replaces CPLEX; instances are therefore much smaller (I ≈ 10–40 interactive) than the paper's I = 50–100.
* **Data**: the hospital data are private → *synthetic reconstruction* of the recipe (20/80 % quantile bounds, gamma durations/LOS, Cost 1/Cost 2,
  (10,7)/(5,2) beds). Only ENT/OPHTH durations and VASCULAR LOS are quoted in the paper text.
* **DRO-W algorithm**: the paper's Appendix D is not in the PDF; the C&CG for DRO-W is **derived here** from (20)–(22) and verified against a primal transport LP.
* **Stays beyond day T** are not costed (the paper mixes [T] and [T+2] in different places).
* **Multiplier products** are linearised per (surgery, block) for ρ and per surgery for the LOS multipliers (equivalent, smaller).
* **No warm start** of the master (HiGHS accepted an invalid partial incumbent in tests).
* **Symmetry breaking** assumes interchangeable surgeries — exact for DRO-M, a modelling choice for SAA-based SP / DRO-W.
""")
    if st.session_state.get("log"):
        with st.expander("Trace of the last run"):
            st.code("\n".join(st.session_state.log[-60:]))

# =================================================================== TAB 2
with tabs[1]:
    st.subheader("Surgeries")
    st.caption("Editable. Durations in minutes, ICU/ward stays in days. *lo/hi* = 20 % / 80 % quantiles (the support); *mean* feeds the DRO-M ambiguity set.")
    st.session_state.frames["surgeries"] = st.data_editor(st.session_state.frames["surgeries"], num_rows="fixed",
                                                          width="stretch", height=330, key="ed_s")
    l, r = st.columns(2)
    with l:
        st.subheader("Blocks (OR-days)")
        st.session_state.frames["blocks"] = st.data_editor(st.session_state.frames["blocks"], width="stretch", height=300, key="ed_b")
    with r:
        st.subheader("Bed capacity per day")
        st.session_state.frames["beds"] = st.data_editor(st.session_state.frames["beds"], width="stretch", height=300, key="ed_d")
    d1, d2, d3 = st.columns(3)
    for col, (k, v) in zip((d1, d2, d3), instance_to_frames(inst).items()):
        col.download_button(f"⬇ {k}.csv", v.to_csv(index=False), f"{k}.csv")
    st.subheader("Expected demand vs capacity")
    tn = [t.name for t in inst.types]
    dem = pd.DataFrame(dict(type=tn, surgeries=[sum(1 for s in inst.surgeries if s.type == j) for j in range(len(tn))],
                            OR_minutes=[sum(s.mu_d for s in inst.surgeries if s.type == j) for j in range(len(tn))],
                            block_minutes=[sum(b.length for b in inst.blocks if j in b.types) for j in range(len(tn))],
                            ICU_days=[sum(s.mu_c for s in inst.surgeries if s.type == j) for j in range(len(tn))],
                            ward_days=[sum(s.mu_w for s in inst.surgeries if s.type == j) for j in range(len(tn))]))
    fig = go.Figure()
    fig.add_bar(x=dem.type, y=dem.OR_minutes, name="expected surgery minutes", marker_color=SERIES[0])
    fig.add_bar(x=dem.type, y=dem.block_minutes, name="block minutes available", marker_color="#cce3fd")
    st.plotly_chart(style(fig, 280), width="stretch")
    fig = go.Figure()
    fig.add_bar(x=[inst.day_name(t) for t in range(inst.T)], y=inst.icu_beds, name="ICU beds", marker_color=SERIES[1])
    fig.add_bar(x=[inst.day_name(t) for t in range(inst.T)], y=inst.ward_beds, name="ward beds", marker_color=SERIES[2])
    st.plotly_chart(style(fig, 240), width="stretch")

# helpers for the solved tabs -------------------------------------------------
def runs_available():
    return list(exp["runs"].keys()) if exp else []


def need_run():
    st.info("Press **▶ Solve and evaluate** first.")


# =================================================================== TAB 3
with tabs[2]:
    if not exp:
        need_run()
    else:
        rows = []
        for k, r in exp["runs"].items():
            rows.append(dict(model=k, objective=round(r["objective"], 1), lower_bound=round(r["lb"], 1), gap_pct=round(100 * r["gap"], 3),
                             status=r["status"], seconds=round(r["time"], 1), scheduled=r["scheduled"],
                             iterations=r["stats"].get("iters", ""), scenarios_BC=r["stats"].get("n_bc", ""),
                             scenarios_down=r["stats"].get("n_down", "")))
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        pick = st.selectbox("Convergence of", [k for k, r in exp["runs"].items() if r["history"]] or ["-"])
        if pick in exp["runs"] and exp["runs"][pick]["history"]:
            h = pd.DataFrame(exp["runs"][pick]["history"])
            fig = go.Figure()
            fig.add_scatter(x=h.t_total, y=h.ub, name="upper bound (best true value)", mode="lines+markers", line=dict(color=SERIES[1]))
            fig.add_scatter(x=h.t_total, y=h.lb, name="lower bound (master)", mode="lines+markers", line=dict(color=SERIES[0]))
            fig.update_xaxes(title="seconds"); fig.update_yaxes(title="objective")
            st.plotly_chart(style(fig, 320), width="stretch")
            fig = go.Figure()
            fig.add_scatter(x=h.iter, y=100 * h.gap, mode="lines+markers", line=dict(color=SERIES[2]))
            fig.update_xaxes(title="C&CG iteration"); fig.update_yaxes(title="gap %", type="log")
            st.plotly_chart(style(fig, 240, False), width="stretch")
            st.dataframe(h[["iter", "lb", "ub", "gap", "n_bc", "n_down", "rows", "cols", "t_master", "t_sub"]], width="stretch", hide_index=True, height=240)

# =================================================================== TAB 4
with tabs[3]:
    if not exp:
        need_run()
    else:
        k = st.selectbox("Plan of", runs_available(), key="plan_pick")
        run_rec = exp["runs"][k]
        plan = {p[0]: p for p in run_rec["plan"]}
        st.caption("Each bar is the **time allocated** to a surgery (difference of consecutive scheduled starts); the darker part is its expected duration. "
                   "Robust models typically allocate more time to early surgeries and less to the last one (paper Fig. 2).")
        fig = go.Figure()
        ylab = []
        for b in inst.blocks:
            y = f"{inst.block_label(b.id)}"
            ylab.append(y)
            items = sorted([p for p in run_rec["plan"] if p[1] == b.id], key=lambda p: p[2])
            for idx, (i, _, pos, start) in enumerate(items):
                nxt = items[idx + 1][3] if idx + 1 < len(items) else b.length
                s = inst.surgeries[i]
                col = SERIES[s.type % len(SERIES)]
                fig.add_bar(y=[y], x=[max(nxt - start, 1)], base=[start], orientation="h", marker=dict(color=col, opacity=0.35, line=dict(color=col, width=1)),
                            hovertemplate=f"{s.label}<br>start {hhmm(start)}<br>allocated {nxt - start:.0f} min<br>expected {s.mu_d:.0f} min<extra></extra>", showlegend=False)
                fig.add_bar(y=[y], x=[min(s.mu_d, max(nxt - start, 1))], base=[start], orientation="h", marker=dict(color=col),
                            text=s.label, textposition="inside", insidetextanchor="start", hoverinfo="skip", showlegend=False)
        fig.update_layout(barmode="overlay")
        fig.update_xaxes(title="minutes from block start", range=[0, max(b.length for b in inst.blocks)])
        fig.update_yaxes(categoryorder="array", categoryarray=ylab[::-1])
        st.plotly_chart(style(fig, 70 + 30 * inst.B, False), width="stretch")
        l, r = st.columns([3, 2])
        with l:
            ev = exp["eval"][k][str(delta_view)]
            days = [inst.day_name(t) for t in range(inst.T)]
            fig = go.Figure()
            fig.add_scatter(x=days, y=ev["icu_need"], name="ICU beds needed (mean)", line=dict(color=SERIES[1], width=3))
            fig.add_scatter(x=days, y=ev["ward_need"], name="ward beds needed (mean)", line=dict(color=SERIES[2], width=3))
            fig.add_scatter(x=days, y=inst.icu_beds, name="ICU capacity", line=dict(color=SERIES[1], dash="dot"))
            fig.add_scatter(x=days, y=inst.ward_beds, name="ward capacity", line=dict(color=SERIES[2], dash="dot"))
            fig.add_scatter(x=days + days[::-1], y=ev["icu_need_q80"] + ev["icu_need_q20"][::-1], fill="toself", fillcolor="rgba(244,114,48,.12)",
                            line=dict(width=0), name="ICU 20–80 %", hoverinfo="skip")
            st.plotly_chart(style(fig, 300), width="stretch")
        with r:
            st.markdown(f"**Scheduled {run_rec['scheduled']} of {inst.I}**")
            post = pd.DataFrame([dict(surgery=inst.surgeries[i].label, type=tn[inst.surgeries[i].type], mu_min=inst.surgeries[i].mu_d,
                                      mu_icu=inst.surgeries[i].mu_c, mu_ward=inst.surgeries[i].mu_w) for i in run_rec["postponed"]])
            st.markdown("Postponed")
            st.dataframe(post, hide_index=True, width="stretch", height=230)
        sched = pd.DataFrame([dict(block=inst.block_label(b), position=pos + 1, surgery=inst.surgeries[i].label, scheduled_start=hhmm(st_),
                                   expected_min=round(inst.surgeries[i].mu_d)) for i, b, pos, st_ in run_rec["plan"]])
        st.download_button("⬇ schedule.csv", sched.to_csv(index=False), "schedule.csv")

# =================================================================== TAB 5
with tabs[4]:
    if not exp:
        need_run()
    else:
        st.caption(f"Out-of-sample simulation with {exp['params']['n_oos']:,} scenarios. Δ = {delta_view}: "
                   + ("same distribution as used for optimisation." if delta_view == 0 else "mis-specified — longer stays and wider durations than assumed."))
        sel = st.multiselect("Plans to compare", runs_available(), default=[m for m in runs_available() if " " not in m])
        days = [inst.day_name(t) for t in range(inst.T)]
        l, r = st.columns(2)
        for col, unit, key in ((l, "ICU", "icu"), (r, "Ward", "ward")):
            fig = go.Figure()
            for m in sel:
                ev = exp["eval"][m][str(delta_view)]
                fig.add_bar(x=days, y=ev[f"pe_{key}"], name=m, marker_color=mcolor(m))
            fig.update_yaxes(title="probability of exceedance (PE)", range=[0, 1])
            col.markdown(f"**{unit} beds — probability that the beds needed exceed capacity**")
            col.plotly_chart(style(fig, 280), width="stretch")
            fig = go.Figure()
            for m in sel:
                ev = exp["eval"][m][str(delta_view)]
                fig.add_bar(x=days, y=ev[f"me_{key}"], name=m, marker_color=mcolor(m))
            fig.update_yaxes(title="mean exceedance (ME), beds")
            col.markdown(f"**{unit} beds — mean number of beds above capacity when exceeded**")
            col.plotly_chart(style(fig, 240), width="stretch")
        fig = go.Figure()
        for m in sel:
            wh = exp["eval"][m][str(delta_view)]["wait_hist"]
            ed = np.array(wh["edges"]); fig.add_scatter(x=(ed[:-1] + ed[1:]) / 2, y=np.array(wh["counts"]) / sum(wh["counts"]), name=m,
                                                         mode="lines", line=dict(color=mcolor(m), shape="spline"))
        fig.update_xaxes(title="total patient waiting in the OR per week (min)"); fig.update_yaxes(title="probability")
        st.markdown("**Waiting time in the ORs** (paper Fig. 5/6)")
        st.plotly_chart(style(fig, 280), width="stretch")

# =================================================================== TAB 6
with tabs[5]:
    if not exp:
        need_run()
    else:
        dl = str(delta_view)
        rows = []
        for m in runs_available():
            ev = exp["eval"][m][dl]; r = exp["runs"][m]
            rows.append({"model": m, "scheduled": r["scheduled"], "mean total cost": round(ev["mean_total_cost"]),
                         "90th pct cost": round(ev["q90_total_cost"]), "first-stage": round(ev["first_stage_cost"]),
                         "mean OR cost": round(ev["mean_bc_cost"]), "mean downstream cost": round(ev["mean_down_cost"]),
                         "avg wait (min)": round(ev["mean_wait_min"]), "avg overtime (min)": round(ev["mean_overtime_min"], 1),
                         "avg idle (h)": round(ev["mean_idle_min"] / 60, 1), "mean PE ICU": round(ev["mean_pe_icu"], 3),
                         "mean PE ward": round(ev["mean_pe_ward"], 3), "RelTC vs SP": round(ev.get("rel_tc_vs_sp", float("nan")), 3),
                         "RelTC w/o vs integrated": round(ev.get("rel_tc_vs_integrated", float("nan")), 3), "solve (s)": round(r["time"], 1),
                         "gap %": round(100 * r["gap"], 2)})
        df = pd.DataFrame(rows)
        st.dataframe(df, width="stretch", hide_index=True)
        st.download_button("⬇ comparison.csv", df.to_csv(index=False), "comparison.csv")
        fig = go.Figure()
        for m in runs_available():
            xs = [str(d) for d in DELTAS]
            fig.add_scatter(x=xs, y=[exp["eval"][m][x]["mean_total_cost"] for x in xs], name=m, mode="lines+markers",
                            line=dict(color=mcolor(m), dash="dot" if " " in m else "solid"))
        fig.update_xaxes(title="mis-specification Δ"); fig.update_yaxes(title="mean total cost (out of sample)")
        st.markdown("**Robustness to distribution shift** — how the cost of each plan grows when the true stays are longer than assumed")
        st.plotly_chart(style(fig, 320), width="stretch")
        st.markdown("""
**Reading guide (paper §6)**: SP schedules the most surgeries (better access) but exceeds recovery-bed capacity far more often and degrades fastest
under Δ; DRO plans schedule fewer surgeries, protect the ICU/ward and keep OR waiting short; **ignoring the downstream units** schedules more surgeries but the
beds needed exceed capacity (large RelTC), and **ignoring scheduling** (start times at mean-duration allocation) lengthens OR waiting.
""")
