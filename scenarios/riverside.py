"""Riverside Regional Hospital: a two-week integrated OR planning problem.

A realistic-scale, synthetic instance of the IORPS (Hashemi Doulabi et al. 2016):
  * 10 operating days (two weeks Mon-Fri), 480-minute room days, 5-minute slots
  * 8 operating rooms Mon-Thu, 6 on Fridays (reduced staffing)
  * 14 surgeons in 7 specialties with weekly OR-day calendars (full/half days,
    clinic days = 0 minutes), one surgeon on planned leave in week 2
  * 220 waiting-list cases with procedure-specific lognormal durations
  * clinical urgency: cat 1 (deadline day 1-3), cat 2 (deadline day 4-10),
    cat 3 (may wait: no deadline in the horizon)
  * infection control: MRSA / VRE / C. diff cases; 45-min terminal clean
    whenever a room switches organism or goes back to clean cases
Everything is fictional; the numbers are realistic orders of magnitude.
"""
from __future__ import annotations

import numpy as np

import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))
from orsched.model import Instance, Surgery

SLOT = 5
DAY_MIN = 480
DAYS = ["Mon 1", "Tue 1", "Wed 1", "Thu 1", "Fri 1", "Mon 2", "Tue 2", "Wed 2", "Thu 2", "Fri 2"]

# specialty: [(procedure, mean min, sd min, weight)]
PROCEDURES = {
    "Orthopaedics": [("Total hip replacement", 120, 25, 3), ("Total knee replacement", 115, 25, 3),
                     ("Spinal fusion", 210, 50, 1), ("ACL reconstruction", 90, 20, 2)],
    "General surgery": [("Laparoscopic cholecystectomy", 75, 20, 4), ("Inguinal hernia repair", 60, 15, 4),
                        ("Colectomy", 180, 45, 2), ("Thyroidectomy", 105, 25, 1)],
    "Vascular": [("EVAR", 150, 40, 2), ("Carotid endarterectomy", 120, 30, 2), ("Fem-pop bypass", 200, 45, 1)],
    "Gynaecology": [("Laparoscopic hysterectomy", 125, 30, 3), ("Myomectomy", 110, 30, 1), ("Ovarian cystectomy", 70, 20, 2)],
    "Urology": [("TURP", 70, 20, 3), ("Radical prostatectomy", 190, 40, 1), ("Nephrectomy", 175, 40, 1)],
    "Neurosurgery": [("Craniotomy", 240, 60, 2), ("Lumbar discectomy", 100, 25, 3)],
    "ENT": [("Tonsillectomy", 45, 15, 3), ("Septoplasty", 60, 15, 2), ("Sinus surgery (FESS)", 90, 25, 2)],
}

# surgeon: (specialty, weekly calendar Mon..Fri in OR minutes)
SURGEONS = [
    ("Ortho A", "Orthopaedics", [480, 0, 480, 480, 0]),
    ("Ortho B", "Orthopaedics", [0, 480, 480, 0, 480]),
    ("Ortho C", "Orthopaedics", [480, 480, 0, 240, 0]),
    ("General A", "General surgery", [480, 0, 480, 0, 480]),
    ("General B", "General surgery", [0, 480, 0, 480, 240]),
    ("General C", "General surgery", [480, 480, 240, 0, 0]),
    ("Vascular A", "Vascular", [0, 480, 0, 480, 0]),
    ("Gynae A", "Gynaecology", [480, 0, 480, 0, 0]),
    ("Gynae B", "Gynaecology", [0, 480, 0, 480, 240]),
    ("Urology A", "Urology", [240, 480, 0, 480, 0]),
    ("Urology B", "Urology", [480, 0, 480, 0, 240]),
    ("Neuro A", "Neurosurgery", [480, 480, 0, 480, 0]),
    ("ENT A", "ENT", [0, 0, 480, 240, 480]),
    ("ENT B", "ENT", [480, 240, 0, 0, 480]),
]
LEAVE_WEEK2 = {"ENT B"}           # planned leave in week 2

INFECTIONS = {1: "MRSA", 2: "VRE", 3: "C. diff"}


def build(n_cases=300, seed=7, friday_rooms=6, extra_leave_week2=(), cleaning_min=45):
    rng = np.random.default_rng(seed)
    L, D = len(SURGEONS), len(DAYS)
    A = np.zeros((L, D), dtype=int)          # planned calendars: the waiting list is built on these
    for l, (name, _, cal) in enumerate(SURGEONS):
        for d in range(D):
            mins = cal[d % 5]
            if d >= 5 and name in LEAVE_WEEK2:
                mins = 0
            A[l, d] = mins // SLOT
    # case load proportional to each surgeon's OR time, inflated so demand exceeds supply
    weight = A.sum(axis=1) / A.sum()
    owners = rng.choice(L, size=n_cases, p=weight)
    cases, meta = [], []
    for i, l in enumerate(owners):
        spec = SURGEONS[l][1]
        procs = PROCEDURES[spec]
        w = np.array([p[3] for p in procs], float)
        name, mu, sd, _ = procs[rng.choice(len(procs), p=w / w.sum())]
        s2 = np.log(1 + (sd / mu) ** 2)
        dur = rng.lognormal(np.log(mu) - s2 / 2, np.sqrt(s2))
        dur = int(np.clip(round(dur / SLOT) * SLOT, 30, 420))
        u = rng.random()
        if u < 0.08:
            cat, dl = 1, int(rng.integers(1, 4))          # within 3 days
        elif u < 0.45:
            cat, dl = 2, int(rng.integers(4, 11))         # within the fortnight
        else:
            cat, dl = 3, int(rng.integers(11, 30))        # may wait
        r = rng.random()
        inf = 1 if r < 0.06 else 2 if r < 0.09 else 3 if r < 0.12 else 0
        # a surgeon needs an OR day (with enough minutes) on or before the deadline
        ok_days = [d for d in range(min(dl, D)) if A[l, d] * SLOT >= dur]
        if cat < 3 and not ok_days:
            later = [d for d in range(D) if A[l, d] * SLOT >= dur]
            dl = (later[0] + 1) if later else 30
            cat = 2 if later else 3
        cases.append(Surgery(id=i, duration=dur // SLOT, surgeon=int(l), deadline=dl - 1,
                             infection=inf, label=f"RV{i + 1:03d}"))
        meta.append(dict(case=f"RV{i + 1:03d}", procedure=name, specialty=spec, category=cat,
                         infection=INFECTIONS.get(inf, "none")))
    # what-if: unplanned leave applied AFTER the waiting list is fixed (same patients, less capacity)
    for l, (name, _, _) in enumerate(SURGEONS):
        if name in extra_leave_week2:
            A[l, 5:] = 0
    rooms = [8, 8, 8, 8, friday_rooms] * 2
    inst = Instance(surgeries=cases, n_days=D, rooms=rooms, slots_per_day=DAY_MIN // SLOT,
                    surgeon_max=A, cleaning=cleaning_min // SLOT, slot_minutes=SLOT,
                    surgeon_names=[s[0] for s in SURGEONS], day_names=DAYS, name="riverside")
    return inst, meta


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from orsched.heuristic import constructive_heuristic
    from orsched.validate import validate_columns
    inst, meta = build()
    dem = sum(s.duration for s in inst.surgeries) * SLOT
    print("cases", inst.n, "mandatory", len(inst.mandatory), "cat1", sum(m["category"] == 1 for m in meta),
          "infectious", sum(s.infection > 0 for s in inst.surgeries))
    print("demand min", dem, "| surgeon capacity", inst.surgeon_max.sum() * SLOT,
          "| room capacity", sum(inst.rooms) * DAY_MIN)
    mand = sum(inst.surgeries[i].duration for i in inst.mandatory) * SLOT
    print("mandatory demand", mand)
    cols, post = constructive_heuristic(inst, np.random.default_rng(0))
    errs = validate_columns(inst, cols)
    print("heuristic minutes", sum(c.value(inst) for c in cols) * SLOT, "errors", len(errs), errs[:5])
