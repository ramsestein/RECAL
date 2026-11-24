# Amendments to the RECAL validation protocol

Each entry: number, date, whether it was made before or after any result of the protocol was seen, what changed, why, and which runs it affects.

| # | Date | Before/after results | Change | Rationale | Runs affected |
|---|---|---|---|---|---|
| — | 2026-11-24 | — | Protocol v1.0 registered (`PREREGISTRATION.md`) | Initial registration | — |
| A1 | 2026-11-24 | **Before** (no source model trained, no RECAL or baseline run on these databases) | Data and software freeze, detailed below | Mandatory under §10 | All |

---

## A1 — Data and software freeze

**Status:** DRAFT. A1 takes effect when committed together with the `v0.3.0` tag; the tag's
commit hash is then added below. No source model is trained before that.

### A1.1 Databases (protocol §2)

| Code | Version | `SHA256SUMS.txt` (first 16 hex of its own SHA-256) |
|---|---|---|
| MIMIC | MIMIC-IV 3.1 | `b1f75ad5aba53104` |
| NWICU | NWICU 0.1.0 | `5df101e4039e3c56` |
| INSPIRE | INSPIRE 1.4.2 | `bc72e1f1a8049376` |

eICU-CRD is not used (unchanged from §2).

### A1.2 Operational definitions (protocol §3)

Implemented in `validation/harmonisation/spec.yaml` (spec v1.0) and `extract.py`.

- **ICU stay.** In MIMIC-IV and NWICU, `icustays` also contains intermediate, step-down and
  ward units. Only stays whose `first_careunit` is an adult ICU are eligible (include regex
  `intensive care|icu|ccu`; exclude `intermediate|stepdown|picu|nicu|pacu`), and the index
  stay (first per patient) is chosen after this filter. Excluded stays: 7,493 (MIMIC-IV) and
  1,110 (NWICU). INSPIRE ICU episodes are the post-operative ICU admissions (`icuin_time`);
  several operations sharing one episode count once.
- **Outcome.** ICU mortality as registered: death during the index ICU stay. A death
  charted up to 6 h after ICU discharge is counted (charting delay at the time of death).
  MIMIC-IV/NWICU: `admissions.deathtime`; INSPIRE: `inhosp_death_time`.
- **Windows.** Vital signs: 0–24 h after ICU admission. Laboratory: −6 h to 24 h.
  Weight: −24 h to 24 h (INSPIRE: pre-operative weight).
- **Aggregations.** Vital signs: minimum, maximum, mean. Laboratory: minimum, maximum.
- **Mean arterial pressure** is derived identically in all databases as (SBP + 2·DBP)/3
  from an SBP and a DBP of the same modality charted at the same time.
- **Variables not harmonised** because NWICU 0.1.0 does not record them: GCS, FiO2,
  arterial PO2/PCO2, mechanical ventilation, vasopressors, urine output.
- **Known measurement differences, not corrected** (acquisition drift, §3.4): bicarbonate is
  serum chemistry in MIMIC-IV/NWICU and arterial blood-gas HCO3 in INSPIRE; INSPIRE sodium
  and potassium include blood-gas measurements; 29.5 % of NWICU stays have no charted vital
  sign in the first 24 h.

### A1.3 Cohort flow

| Step | MIMIC-IV 3.1 | NWICU 0.1.0 | INSPIRE 1.4.2 |
|---|---|---|---|
| ICU stays in source table | 94,458 | 28,612 | 19,195 |
| Adult-ICU-unit stays | 86,965 | 27,502 | — |
| First ICU stay per patient | 60,207 | 22,448 | 14,279 |
| Age ≥ 18 | 60,207 | 22,448 | 14,279 |
| ICU stay ≥ 24 h | 47,559 | 16,847 | 7,145 |
| Alive at 24 h (final cohort) | 47,385 | 16,812 | 7,141 |
| ICU deaths | 3,500 (7.4 %) | 1,242 (7.4 %) | 128 (1.8 %) |

### A1.4 Final predictor list

44 predictors, frozen in `validation/harmonisation/final_features.json`, selected by the
registered rule (missing rate ≤ 50 % in every database): age, male sex, weight; minimum,
maximum and mean of heart rate, systolic, diastolic and mean arterial pressure, respiratory
rate, SpO2 and temperature; minimum and maximum of creatinine, urea nitrogen, sodium,
potassium, glucose, haemoglobin, white cells, platelets, total bilirubin and bicarbonate.
Dropped by the rule: albumin, lactate, pH, INR (min and max).

### A1.5 Target cohorts larger than the available events (clarifies §4.3)

A target cohort of E events is drawn only if the Target-Pool contains at least E events;
otherwise that (transfer, E) cell is not run and is reported as not feasible. With ICU
mortality as registered, INSPIRE as target has about 64 events in the Target-Pool, so only
E ∈ {10, 25, 50} are run, and its Target-Holdout (about 64 events) is below the 200-event
minimum of §4.2. Transfers into INSPIRE are therefore reported but flagged as low-precision
ground truth and excluded from the acceptance criterion (§8.3), as §4.2 already specifies.
INSPIRE as source trains the source model on about 100 events; this is reported as a
limitation. The outcome is **not** changed.

### A1.6 Software

- RECAL **v0.3.0** (branch `release/v0.3.0`, commit `61d90ed`; tag pending), implementing
  §4.5 items 1–4: OOF-based gaps and recovery ratio, percentile-bootstrap rr CI,
  `recal_cli/verdict.py` (rules V1/V2 of §5), working `recal` entry point.
- One performance change made before A1: the MMD² permutation test reuses the pooled
  kernel (identical statistic and random stream).
- Environment lock: `requirements-lock.txt` (Python 3.13, Linux x86_64).
- Harmonisation code: branch `validation/harmonisation` (commit at A1).
- **Baseline B7** uses **SKADA 0.6.0** (installable in the locked environment; ADAPT not
  needed).
