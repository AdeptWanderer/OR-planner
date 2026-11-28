import json
from pathlib import Path

import numpy as np
import pytest
from conftest import tiny

from esassp import Params
from esassp.experiment import DELTAS, eval_record, instance_record, run_experiment, solution_record

APP = Path(__file__).resolve().parents[1] / "app.py"


@pytest.fixture(scope="module")
def exp():
    inst = tiny(2, 8, beds=((1, 1), (1, 1)))
    p = Params(time_limit=30, master_time=15, gap=1e-3, threads=2, n_scenarios=5)
    return inst, run_experiment(inst, p, models=("SP", "DRO-M", "DRO-W"), baselines=True, n_oos=300, log=lambda s: None)


def test_experiment_structure(exp):
    inst, e = exp
    assert set(e["runs"]) == {"SP", "DRO-M", "DRO-W", "SP w/o downstream", "SP w/o scheduling", "DRO-M w/o downstream", "DRO-M w/o scheduling"}
    assert set(e["eval"]["SP"]) == {str(d) for d in DELTAS}
    assert e["instance"]["name"] == inst.name and len(e["instance"]["surgeries"]) == inst.I


def test_experiment_is_json_serialisable_and_has_no_arrays(exp):
    _, e = exp
    s = json.dumps(e)
    assert "\"_tc\"" not in s and len(s) > 1000


def test_plans_reference_valid_blocks_and_surgeries(exp):
    inst, e = exp
    for r in e["runs"].values():
        ids = [p[0] for p in r["plan"]] + r["postponed"]
        assert sorted(ids) == list(range(inst.I))
        assert all(0 <= p[1] < inst.B for p in r["plan"])
        assert r["scheduled"] == len(r["plan"])


def test_rel_tc_fields(exp):
    _, e = exp
    assert e["eval"]["SP"]["0.0"]["rel_tc_vs_sp"] == pytest.approx(0.0, abs=1e-12)
    assert "rel_tc_vs_integrated" in e["eval"]["SP w/o downstream"]["0.0"]
    assert "rel_tc_vs_integrated" not in e["eval"]["SP"]["0.0"]


def test_eval_arrays_have_horizon_length_and_valid_ranges(exp):
    inst, e = exp
    ev = e["eval"]["DRO-M"]["0.5"]
    for k in ("pe_icu", "pe_ward", "me_icu", "me_ward", "icu_need", "ward_need"):
        assert len(ev[k]) == inst.T
    assert all(0 <= x <= 1 for x in ev["pe_icu"] + ev["pe_ward"])
    assert sum(ev["wait_hist"]["counts"]) == 300


def test_cost_grows_with_misspecification(exp):
    _, e = exp
    for name in ("SP", "DRO-M"):
        c = [e["eval"][name][str(d)]["mean_total_cost"] for d in DELTAS]
        assert c[2] >= c[0] * 0.95


def test_instance_record_fields():
    inst = tiny(2, 6)
    rec = instance_record(inst)
    assert rec["T"] == inst.T and len(rec["blocks"]) == inst.B and len(rec["icu_beds"]) == inst.T


def test_history_recorded_for_dro_only(exp):
    _, e = exp
    assert e["runs"]["SP"]["history"] == [] and e["runs"]["DRO-M"]["history"]


# ------------------------------------------------------------------ Streamlit app (headless)
def test_app_renders_without_exception():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(APP), default_timeout=90).run()
    assert not at.exception and len(at.tabs) == 6 and len(at.metric) == 5


def test_app_solves_and_fills_tabs():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(APP), default_timeout=180).run()
    at.sidebar.slider[0].set_value(8)            # surgeries
    at.sidebar.slider[1].set_value(1)            # ORs
    at.sidebar.checkbox[0].uncheck()             # no baselines
    at.sidebar.multiselect[0].set_value(["SP", "DRO-M"])
    at.run()
    at.button[0].click().run()
    assert not at.exception
    assert any("Done" in str(s.label) for s in at.status) or at.session_state["exp"] is not None
    assert set(at.session_state["exp"]["runs"]) == {"SP", "DRO-M"}
