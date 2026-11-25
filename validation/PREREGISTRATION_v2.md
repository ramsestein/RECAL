# RECAL 0.4 — Pre-registered independent evaluation on eICU-CRD (protocol v2.0)

**Date of registration:** 2026-11-25
**Registered by:** Ramsés Marrero García (Hospital Clínic de Barcelona / Universitat de Barcelona)
**Context:** Revision of SoftwareX manuscript SOFTX-D-26-01085.
**Relation to protocol v1.0.** Protocol v1.0 (`PREREGISTRATION.md`, Amendment A1, tag
`v0.3.0`) evaluated RECAL 0.3 on MIMIC-IV, NWICU and INSPIRE. Those databases are now the
**development databases**: their results informed the redesign of RECAL 0.4 (selection of
the adaptation by cross-validation; no deploy-or-retrain verdict). This protocol evaluates
RECAL 0.4, frozen before any analysis, on a database that was not used at any point of
development: **eICU-CRD 2.0**.

**Prior data access, disclosed.** The investigators have credentialed access to eICU-CRD and
had drafted a variable mapping for it in an unrelated project. No model, RECAL run or
outcome analysis on eICU-CRD has been performed for RECAL. Amendment B1 (data and software
freeze, §9) is committed before any source model is applied to eICU data.

---

## 1. Question and claims under test

RECAL 0.4 takes a frozen binary classifier and a small labelled cohort from a new site,
chooses the adaptation (input transformation × recalibration) with the lowest
cross-validated log-loss, and reports the expected performance of the adapted model at that
site. This protocol tests three claims:

- **C1 — Improvement without harm.** On new patients of the target site, the RECAL-adapted
  model has lower log-loss than the frozen model, and rarely worse.
- **C2 — Selection quality.** RECAL is not worse than the best fixed simple alternative
  (logistic recalibration, Platt), and its regret with respect to the candidate that turns
  out best on new patients is small.
- **C3 — Honest estimate.** RECAL's nested-CV estimate of performance agrees with the
  performance later observed on new patients of the same site, and its 95% CIs have
  near-nominal coverage.

RECAL makes no deploy-or-retrain recommendation, and none is evaluated.

## 2. Data

- eICU Collaborative Research Database 2.0 (multi-centre, >200 US hospitals).
- MIMIC-IV 3.1 (development database; used only as the source of one source model, §4.1).
- Harmonisation: `validation/harmonisation/spec.yaml` extended with an `eicu` block using the
  same cohort rules, windows, aggregations, bounds and the **same 44 predictors frozen in
  A1** (`final_features.json`). The predictor list is not re-selected on eICU. A predictor
  that eICU cannot provide is set to missing and reported.
- Cohort (as in v1.0 §3): adults; first ICU stay per patient (first `unitvisitnumber` of the
  first health-system stay); ICU stay ≥ 24 h; alive at 24 h. Outcome: ICU mortality
  (`unitdischargestatus = 'Expired'`). eICU times are offsets from ICU admission.

## 3. Target sites

- **Eligible target hospitals:** hospitals with ≥ 200 ICU deaths in the harmonised cohort.
  If fewer than 8 hospitals qualify, the threshold becomes ≥ 100 deaths (fixed now). The list
  is recorded in Amendment B1.
- Each target hospital is split once, stratified by outcome, into **Target-Pool** (50 %)
  and **Target-Holdout** (50 %). Cohorts are drawn from the Pool, and evaluation uses the
  Holdout only.

## 4. Design

### 4.1 Source models (frozen; XGBoost with the hyperparameters of v1.0 §4.4)

- **S1 — External:** trained on the whole MIMIC-IV cohort.
- **S2 — Leave-region-out:** for a target hospital in eICU region R, trained on all eICU
  hospitals outside R. There is one S2 model per region (and a separate one for hospitals
  with unknown region).

The MLP is not used: in v1.0 it failed to train reliably with few events.

### 4.2 Target cohorts

- E ∈ {25, 50, 100} events, with non-events in the Target-Pool proportion. A cell is not run
  if the Pool has fewer than E events.
- R = 10 replicates per (target hospital, source, E).
- Seeds: `SeedSequence(20261125, spawn_key=key)`, written to `<out>/seeds.json`.

### 4.3 RECAL configuration

RECAL **v0.4.0** (tag recorded in B1) with defaults:

- `selection: cv`, 5 folds, one-standard-error rule;
- candidates: identity / designed / designed_drift × none / intercept / logistic / isotonic;
- nested 5-fold estimate;
- no oracle, verdict or counterfactuals;
- the source cohort passed to RECAL is the source model's training set; for S1 and S2 it is
  subsampled at random to at most 20,000 patients (fixed seed), to bound running time.

### 4.4 Comparators (fitted on the same cohort, evaluated on the same Holdout)

| ID | Comparator |
|---|---|
| F | Frozen source model |
| P | Logistic recalibration (Platt) |
| I | Recalibration-in-the-large |
| T | Isotonic recalibration |
| V | Closed testing procedure (Vergouwe et al., 2017) as in v1.0 |
| K | Every candidate of RECAL's set, fitted on the whole cohort (needed for regret) |

## 5. Endpoints (Target-Holdout)

### 5.1 Primary endpoints

1. **C1: Δ_F = log-loss(RECAL) − log-loss(F).**
   - Superiority is claimed if the upper bound of the 95 % CI of the mean Δ_F is < 0.
   - The **harm rate** is P(Δ_F > +0.005) (0.005 nats per patient).
2. **C2: Δ_P = log-loss(RECAL) − log-loss(P).**
   - Non-inferiority margin: +0.002 nats per patient.
   - Non-inferiority is claimed if the upper bound of the 95 % CI of the mean Δ_P is < +0.002.

### 5.2 Secondary endpoints

- **Regret** = log-loss(RECAL) − min over K of the Holdout log-loss. Reported as median and 90th percentile.
- **Δ against the other comparators:** the same Δ for Brier, AUROC, calibration slope,
  calibration intercept, ECE (10 equal-frequency bins) and ICI, against F, P, I, T and V.
- **C3, accuracy of the estimate:** nested estimate − Holdout value, for AUROC, Brier,
  calibration slope and calibration intercept. Reported as median error and median
  absolute error.
- **C3, coverage:** the share of runs whose 95 % CI contains the Holdout value (AUROC, Brier).
- **Choice frequencies:** how often each candidate is chosen, and how often the identity is
  chosen with no recalibration.
- **Reference model:** the Holdout AUROC of an XGBoost model trained on the whole
  Target-Pool, reported for context only.

### 5.3 Results by subgroup

Every endpoint is reported by source (S1, S2) and by E.

## 6. Analysis

- Runs are clustered within target hospitals. Means and CIs come from a two-stage bootstrap
  (B = 2000): hospitals are resampled, then runs within each resampled hospital.
- The primary endpoints are evaluated separately for S1 and S2, at each E. No correction for
  multiplicity is applied; all cells are reported.
- **Negative results.** If C1 or C2 fails for a source/E cell, this is reported as such.
- Every run is reported. Execution failures are counted and reported.

## 7. Development-database results (not confirmatory)

RECAL 0.4 is also run on the v1.0 design (MIMIC-IV, NWICU, INSPIRE) with the same endpoints.
Those results are labelled **development**: the databases informed the redesign, so they
cannot confirm it.

## 8. Reproducibility

The code (harmonisation, experiment, analysis) and the seeds are in this repository.
Reproduction requires PhysioNet credentials. Data, derived data and trained models are
not distributed.

## 9. Amendments

Amendments are recorded in `AMENDMENTS_v2.md`. **B1** (mandatory, before any source model is
applied to eICU data) records:

- the eICU harmonisation;
- the cohort flow;
- the eligible hospitals and the S2 regions;
- the missing predictors;
- the RECAL v0.4.0 tag and commit hash.

Changes made after results are reported as deviations.

## References

Vergouwe, Y., Nieboer, D., Oostenbrink, R., Debray, T. P. A., Murray, G. D., Kattan, M. W., Koffijberg, H., Moons, K. G. M., & Steyerberg, E. W. (2017). A closed testing procedure to select an appropriate method for updating prediction models. *Statistics in Medicine, 36*(28), 4529–4539. https://doi.org/10.1002/sim.7179

Pollard, T. J., Johnson, A. E. W., Raffa, J. D., Celi, L. A., Mark, R. G., & Badawi, O. (2018). The eICU Collaborative Research Database, a freely available multi-center database for critical care research. *Scientific Data, 5*, 180178. https://doi.org/10.1038/sdata.2018.178
