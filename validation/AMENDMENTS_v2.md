# Amendments to protocol v2.0 (eICU-CRD evaluation of RECAL 0.4)

| # | Date | Before/after results | Change |
|---|---|---|---|
| — | 2026-11-25 | — | Protocol v2.0 registered (`PREREGISTRATION_v2.md`) |
| B1 | 2026-11-25 | **Before** (no source model applied to eICU data; no RECAL run on eICU) | Data and software freeze, below |

## B1 — Data and software freeze

**Status:** in effect from tag `v0.4.0`.

### B1.1 Harmonisation (eICU-CRD 2.0)

`validation/harmonisation/spec.yaml` (`eicu` block) and `extract.py` (`cohort_eicu`,
`events_eicu`), same cohort rules, windows, aggregations and bounds as A1, and the 44
predictors frozen in A1.

- Vital signs: `vitalPeriodic` (heart rate, respiration, SpO2 `sao2`, invasive systolic and
  diastolic pressure, temperature) and `vitalAperiodic` (non-invasive systolic and diastolic
  pressure). MAP derived as in A1. Temperature also from `nurseCharting`
  ("Temperature (C)" / "Temperature (F)"), because `vitalPeriodic.temperature` is missing in
  89 % of stays; with nurse charting the missing rate is 3 %.
- Laboratory: `lab.labname` creatinine, BUN, sodium, potassium, glucose, Hgb,
  WBC x 1000, platelets x 1000, total bilirubin, bicarbonate.
- Weight: `patient.admissionweight`. Age `> 89` coded 90.
- First ICU stay: first health-system stay (discharge year, then id), first unit visit.
- Outcome: `unitdischargestatus = 'Expired'` (death in the ICU).

### B1.2 Cohort flow

ICU stays 200,859 → first ICU stay per patient 139,367 → age ≥ 18 138,852 → ICU stay ≥ 24 h
96,473 → known discharge status 96,468 (ICU deaths 5,260; 5.5 %).

### B1.3 Predictors

All 44 A1 predictors are available. Missing rates are ≤ 9.3 % except total bilirubin
(42.8 %). The predictor list is not re-selected.

### B1.4 Target hospitals

Only one hospital has ≥ 200 ICU deaths, so the pre-specified fallback (≥ 100 deaths) applies.
This gives **11 hospitals**:

| Hospital | Region | Deaths |
|---|---|---|
| 420 | Northeast | 230 |
| 458 | South | 197 |
| 252 | Midwest | 163 |
| 443 | South | 163 |
| 264 | Midwest | 153 |
| 188 | South | 140 |
| 199 | Northeast | 139 |
| 167 | West | 128 |
| 338 | Midwest | 126 |
| 122 | South | 124 |
| 300 | Midwest | 115 |

There are four S2 source models (left-out regions: Midwest, Northeast, South, West). With a
50 % Target-Pool, E = 100 is feasible only where the Pool has ≥ 100 events. Other cells are
reported as not feasible.

### B1.5 Software

- RECAL **v0.4.0** (tag `v0.4.0`).
- Experiment: `validation/run_validation_v2.py`. Its integration and timing were checked
  only on randomly generated data of the same shape.
