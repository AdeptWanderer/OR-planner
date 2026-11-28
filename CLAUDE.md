# OR-planner

Paper → MVP workspace for operating-room planning. **Read `docs/LEARNINGS.md` first** (workflow, paper notes,
pitfalls) and append to it when you learn something.

* `orsched/` + `app.py` + `web/` — MVP #1 (CP-based branch-and-price-and-cut, Hashemi Doulabi et al. 2016).
* `esassp/` + `app_esassp.py` + `web_esassp/` — MVP #2 (OR-to-downstream planning under uncertainty, DRO,
  Shehadeh et al. 2026).
* Every MVP: package, independent validator/oracle, exact cross-check tests (`pytest -q tests`), Streamlit app,
  static web artifact from precomputed runs, benchmarks with checkpointing, README with
  "Differences from the paper".
* Develop on branch `claude/pensive-newton-y4w11b`; commit + push at each milestone.
