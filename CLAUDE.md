# OR-planner

Paper -> MVP workspace for operating-room planning. **Read `docs/LEARNINGS.md` first** (workflow, pitfalls) and append to it when you learn something.

* `orsched/` + `app.py` + `web/` — MVP #1 (CP-based branch-and-price-and-cut, Hashemi Doulabi et al. 2016).
* `esassp_mvp/` — MVP #2, an **independent project** (OR-to-downstream planning under uncertainty, SP / DRO-M / DRO-W, Shehadeh et al. 2026):
  own README, requirements, `docs/NOTES.md`, 418 tests (`cd esassp_mvp && python -m pytest -q tests`).
* Every MVP: package, independent oracle/validator, exact cross-check tests, Streamlit app, static web artifact, benchmarks with checkpointing,
  README with "Differences from the paper". Write lots of tests.
* Develop on branch `claude/pensive-newton-y4w11b`; commit + push at each milestone.
