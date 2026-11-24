# RECAL — Pre-registered validation protocol

**Protocol version:** 1.0
**Date of registration:** 2026-11-24
**Registered by:** Ramsés Marrero García (Hospital Clínic de Barcelona / Universitat de Barcelona)
**Context:** Revision of SoftwareX manuscript SOFTX-D-26-01085 ("RECAL: audited recalibration and alignment of frozen clinical predictive models, with a deploy-or-retrain verdict").
**Repository state at registration:** parent commit `5ea6a2c` (RECAL v0.2.2).

This document fixes, before any RECAL run on the databases listed below, the design, the decision rules, the ground-truth definition, the baselines, the metrics and the analysis plan used to validate RECAL's deploy-or-retrain verdict. Any later change is recorded as a dated amendment in [`AMENDMENTS.md`](AMENDMENTS.md). Analyses not described here are reported as exploratory.

**Prior data access, disclosed.** The investigators already had credentialed access to MIMIC-IV, NWICU and INSPIRE and have used them in other studies (descriptive statistics and score revalidation). No RECAL run, source model or target model for this protocol has been fitted on these databases at the time of registration.

---

## 1. Objectives

**Primary objective.** To estimate how often RECAL's verdict recommends deploying the wrapper when, in held-out data from the target population, the wrapper is not in fact deployable (*false-deploy rate*). The estimate is made across real cross-database transfers and target-cohort sizes.

**Secondary objectives.**

1. To estimate overall verdict accuracy, the rate of indeterminate verdicts and the rate of unnecessary "retrain" recommendations.
2. To estimate the smallest number of target events at which the verdict meets the acceptance criterion in Section 8.3. This answers Reviewer 2, comment 5.
3. To compare the held-out performance of the RECAL wrapper with recalibration-only, alignment-only, domain-adaptation-library and retraining baselines (Section 6). For decision baselines, the comparison also covers their decisions.
4. To quantify the effect of the reviewers' M1 correction by comparing verdict rule V1 (point estimates, out-of-fold) with V2 (interval-aware).

**Out of scope.** The deployment thresholds (0.02/0.05 for the optimism gap, 0.3/0.7 for the recovery ratio) are **not** re-estimated in the primary analysis. They are treated as provisional heuristics and tested as they are. Any data-driven threshold analysis is exploratory (Section 9).

---

## 2. Data sources

All three databases are distributed by PhysioNet under credentialed access. Neither the data, nor derived datasets, nor trained models will be redistributed. Reproduction requires the reader's own credentials. The repository will provide the extraction, harmonisation, training and analysis code with fixed seeds, so that every result can be regenerated from the raw databases.

| Code | Database | Setting |
|---|---|---|
| MIMIC | MIMIC-IV | Beth Israel Deaconess Medical Center, Boston, USA |
| NWICU | Northwestern ICU database (NWICU) | Northwestern Medicine, Chicago, USA |
| INSPIRE | INSPIRE | Seoul National University Hospital, Republic of Korea (surgical population) |

Exact database versions are recorded in Amendment A1. eICU-CRD is **not** part of this protocol. Adding it requires an amendment registered before any result is seen, and the added analyses are then labelled as an extension.

---

## 3. Cohort, outcome and predictors

### 3.1 Cohort

- Adults (age ≥ 18 years).
- First ICU stay per patient. All sampling and splitting is done at patient level, so the same patient never appears in two partitions.
- ICU length of stay ≥ 24 h, because prediction is made at 24 h.
- Known outcome.

### 3.2 Outcome

ICU mortality: death before ICU discharge. Patients who die in the first 24 h are excluded by the length-of-stay criterion. The operational definition for each database, including how INSPIRE's perioperative ICU episodes are identified, is fixed in Amendment A1.

### 3.3 Predictors

- Demographic, physiological and laboratory variables recorded in the first 24 h of the ICU stay.
- Inclusion rule:
  - The variable can be mapped in all three databases.
  - Its missing rate is ≤ 50 % in each database. This matches RECAL's `max_missing_rate = 0.5`.
- Aggregations (for example minimum, maximum and mean in the window) and units are defined once, in the harmonisation code.

### 3.4 Harmonisation freeze

The harmonisation code, the final variable list, the aggregations and the units are frozen in Amendment A1. A1 is committed **before** any source model is trained.

- After A1, harmonisation is not changed in response to RECAL or baseline results.
- Harmonisation problems found later (for example unit mismatches revealed by RECAL's drift audit) are reported as findings. They are not corrected and re-run. They count as acquisition drift and are part of what the experiment measures.
- The `unit_corrections` field of the RECAL configuration is left empty in every run.

---

## 4. Experimental design

### 4.1 Transfers

**Primary (real cross-database transfers).** All six directed pairs S → T with S, T ∈ {MIMIC, NWICU, INSPIRE} and S ≠ T.

**No-shift control (primary).** MIMIC-A → MIMIC-B: MIMIC patients are split at random, 50/50 at patient level, into a source half A and a target half B.

**Supplementary.**

1. *Temporal transfer.* MIMIC, earliest `anchor_year_group` → latest `anchor_year_group`.
2. *Semi-synthetic shifts.* The shifts in Section 4.6 are applied to the MIMIC-A → MIMIC-B control.
3. *Permuted-label control.* See Section 4.7.

### 4.2 Partitions within each transfer

**Source database**

- Random split, stratified by outcome: 80 % development and 20 % source test.
- The source model is trained on the development set only.
- Source performance is reported on the source test set.

**Target database**

- One random split, stratified by outcome: 50 % **Target-Pool** and 50 % **Target-Holdout**.
- RECAL target cohorts and every baseline are fitted only on subsamples of the Target-Pool.
- Ground truth is evaluated only on the Target-Holdout.
- The Target-Holdout is never used to fit, select or tune anything.
- If a Target-Holdout contains fewer than 200 events, that transfer is reported, but it is flagged as having a low-precision ground truth and is excluded from the acceptance criterion.

### 4.3 Target-cohort sizes

Target cohorts are defined by their **number of events E**, so that databases with different mortality can be compared. Results are also reported by total cohort size n.

- E ∈ {10, 25, 50, 100, 200}.
- Each cohort contains exactly E events and round(E · (1 − π_T) / π_T) non-events, where π_T is the outcome prevalence in the Target-Pool. Both are drawn without replacement from the Target-Pool.
- **R = 20 independent subsamples** (seeds) per (transfer, E).

### 4.4 Models

**Primary model family: XGBoost.** Fixed hyperparameters, not tuned on any target data:

| Hyperparameter | Value |
|---|---|
| `n_estimators` | 300 |
| `max_depth` | 4 |
| `learning_rate` | 0.05 |
| `subsample` | 0.8 |
| `colsample_bytree` | 0.8 |
| `min_child_weight` | 5 |

Missing values are handled natively.

**Secondary model family: multilayer perceptron (MLP).**

- Architecture: 64–32 hidden units, ReLU, Adam, early stopping on 10 % of the development set.
- Inputs are standardised; missing values are imputed with the median. Both the scaler and the imputer are fitted on the source development set only.
- Run for E ∈ {25, 100} only.

**Seeds.**

- Master seed: `20261124`.
- Per-run seeds are derived with `numpy.random.SeedSequence(20261124)`, spawned in a fixed, documented order over (transfer, model, E, replicate).
- The seed list is written to disk with the results.

The trained source models are **not** published. The training code and seeds are.

### 4.5 RECAL configuration (identical for every run)

**Software.** RECAL **v0.3.0**, tagged before Amendment A1. Relative to v0.2.2 it must contain:

1. Gaps and recovery ratio computed from out-of-fold (OOF) predictions.
2. A percentile-bootstrap CI for the recovery ratio (Section 5.2).
3. A `verdict` module that implements rules V1 and V2 exactly as written in Section 5.
4. A working `recal` console entry point.

After A1, only bug fixes that do not touch the decision rules or thresholds are allowed. Each one is logged as an amendment, together with the runs it affects.

**Configuration.**

- `recal_core`: package defaults (`pca_k = 5`, `max_n_sweep = 30`, `apply_qt = true`, `apply_woe = "auto"`, `max_missing_rate = 0.5`).
- `regularization`: package defaults.
- `overfitting_check`: `kfold`, `n_splits = 5`, `random_state` = the run seed.
- `evaluation`:
  - `oracle_eval = true`, `oracle_cv = 5`
  - `feature_attribution = false`, `counterfactual_alternatives = 0`
  - `brier_decompose = true`
- No manual overrides of Designer decisions: masking, alignment method and calibration are left to the Designer. Whether Platt calibration was applied is recorded for every run.
- When a stratified 5-fold split cannot be formed (fewer than 5 events in a fold), RECAL must return `INDETERMINATE (insufficient events)`. That verdict counts as "not deploy".

### 4.6 Semi-synthetic shifts (supplementary)

The shifts are applied to MIMIC-B, the target half. Each shift is applied identically to the Target-Pool and to the Target-Holdout, because the shifted population is the target population. They are run at E ∈ {50, 100} with R = 20.

"Top features" means the source model's three most important features, ranked by mean |SHAP| on the source development set.

| ID | Shift | Construction | Expected class |
|---|---|---|---|
| S1 | Prevalence | Resample MIMIC-B to 0.5× and 2× its prevalence | Recoverable |
| S2 | Covariate | Biased selection with weight ∝ exp(β·z) over the standardised top features. β is chosen so that the mean shift is 0.5 SD. | Recoverable / mixed |
| S3 | Acquisition / units | Known rescaling of three continuous laboratory variables (×10, ×0.1, +offset of 1 SD) | Recoverable |
| S4 | Missingness (MNAR) | Values above the 75th percentile of two top features set to missing in 50 % of rows | Mixed |
| S5 | Calibration only | Labels regenerated as y′ ~ Bernoulli(σ(a + b·logit p_S)), with a = −1, b = 0.6; ranking preserved | Recoverable |
| S6 | Concept | Logistic model fitted on MIMIC-B; signs of the three top coefficients flipped; labels regenerated; intercept adjusted to keep prevalence | Not recoverable |

The "expected class" is used only as a secondary check that the ground-truth rule (Section 5.3) behaves as intended. The primary evaluation always uses the ground-truth rule.

### 4.7 Permuted-label control (supplementary)

- Run on MIMIC-A → MIMIC-B and on one real transfer (MIMIC → INSPIRE).
- E = 50, R = 20.
- Outcome labels are permuted at random within both the target cohort and the Target-Holdout.
- Expected:
  - OOF AUROC ≈ 0.5.
  - Green verdicts in ≈ 0 % of runs.
  - Clearly non-zero optimism gaps if there is leakage between the selection steps and evaluation.

---

## 5. Decision rules and ground truth

### 5.1 Quantities computed by RECAL on the target cohort

| Quantity | Definition |
|---|---|
| AUROC_raw | AUROC of the frozen source model on the target cohort |
| AUROC_in | AUROC of the fitted wrapper on the cohort it was fitted on (in-sample) |
| AUROC_oof | AUROC of the wrapper from the k-fold OOF predictions, with profiling, Designer sweep, fitting and calibration repeated inside each fold |
| AUROC_oracle | OOF AUROC of RECAL's target-trained oracle |
| optimism_gap | AUROC_in − AUROC_oof |
| total_gap | AUROC_oracle − AUROC_raw |
| rr | (AUROC_oof − AUROC_raw) / total_gap |
| slope_oof | Calibration slope of the OOF wrapper probabilities |

### 5.2 Confidence interval for rr

rr_CI is a 95 % percentile bootstrap CI (B = 1000). Target-cohort patients are resampled with replacement, stratified by outcome, together with their raw, OOF and oracle OOF scores. The three AUROCs are recomputed in every resample, which keeps their correlation.

This CI does not include the variability of refitting. That limitation is stated in the paper.

### 5.3 Verdict rule V2 (interval-aware, primary)

Conditions are evaluated in this order:

1. **INDETERMINATE (insufficient events)** if E < 10 or a stratified 5-fold split cannot be formed.
2. **NO DISCRIMINATION GAP** if total_gap ≤ 0. Then:
   - **GREEN** if optimism_gap ≤ 0.02 and slope_oof ∈ [0.7, 1.3];
   - otherwise **AMBER**.
3. **RED (retrain)** if optimism_gap > 0.05 or the upper bound of rr_CI < 0.3.
4. **GREEN (deploy wrapper)** if optimism_gap ≤ 0.02, the lower bound of rr_CI ≥ 0.7 and slope_oof ∈ [0.7, 1.3].
5. **AMBER (mixed drift / enlarge cohort)** if rr_CI lies entirely within [0.3, 0.7].
6. **INDETERMINATE (interval crosses a threshold)** otherwise.

### 5.4 Verdict rule V1 (point estimates, OOF, secondary)

V1 follows the rule as described in the submitted manuscript, but uses OOF estimates:

- **RED** if optimism_gap > 0.05 or rr < 0.3.
- **GREEN** if optimism_gap ≤ 0.02 and rr ≥ 0.7.
- **AMBER** otherwise.

Case 2 of V2 (total_gap ≤ 0) applies to V1 as well.

### 5.5 Binary reading of the verdict

- **Deploy** = GREEN.
- **Not deploy** = RED, AMBER or INDETERMINATE.

### 5.6 Ground truth (Target-Holdout)

For every run, a **Reference model** of the same family and with the same fixed hyperparameters as the source model is trained on the **whole Target-Pool**. All metrics are computed on the Target-Holdout.

The wrapper fitted on the target cohort is **deployable** (GT = 1) if and only if all three conditions hold:

1. AUROC_wrapper ≥ AUROC_Reference − δ, with **δ = 0.02**;
2. calibration slope ∈ **[0.7, 1.3]**;
3. |calibration-in-the-large intercept| ≤ **0.2** on the logit scale, estimated with the slope fixed at 1.

Otherwise GT = 0.

The Reference model is used only to define the ground truth. It is not offered as a deployable model.

**Ground-truth sensitivity analyses** (pre-specified):

- δ ∈ {0.01, 0.03, 0.05};
- slope interval [0.8, 1.2];
- intercept bound 0.1.

### 5.7 Error types

| | GT = 1 (deployable) | GT = 0 (not deployable) |
|---|---|---|
| **Deploy** (GREEN) | Correct deploy | **False deploy** (primary endpoint) |
| **Not deploy** | Missed deploy (unnecessary data collection) | Correct no-deploy |

RED verdicts with GT = 1 are reported separately as *unnecessary retrain*.

---

## 6. Baselines

Every baseline is fitted on the same target cohort as RECAL in each run and evaluated on the same Target-Holdout.

The linear predictor (LP) of the frozen model is logit(p_S). For the MLP it is the pre-sigmoid output.

| ID | Method | Fitted on target cohort |
|---|---|---|
| B0 | Frozen source model, no update | — |
| B1 | Recalibration-in-the-large | Intercept only (LP as offset) |
| B2 | Logistic recalibration (Platt) | Intercept and slope on LP |
| B3 | Isotonic recalibration | Monotone mapping of p_S |
| B4 | Closed testing procedure (Vergouwe et al., 2017), adapted to black-box models | See below |
| B5 | CORAL only (RECAL's aligner, no QT/WOE/masking, no calibration) | Alignment |
| B6 | Quantile transform only | Alignment |
| B7 | CORAL from SKADA, with Platt on top | Alignment and calibration |
| B8 | Retraining on the target cohort | Same family and hyperparameters as the source model |
| — | RECAL (full) | As in Section 4.5 |

**B4 in detail.** Four nested logistic models are fitted:

- M0: original model;
- M1: intercept update (recalibration-in-the-large);
- M2: intercept and slope on the LP (logistic recalibration);
- M3: M2 plus main effects of all harmonised predictors.

**Black-box adaptation.** Model revision proper (re-estimating every coefficient of the original model) is not possible for a black-box model. M3, a model extension, takes its place as the most extensive update.

**Closed testing order** (Vergouwe et al., 2017). Each simpler model is compared with the most extensive model, M3, with a likelihood-ratio test at α = 0.05:

1. If M0 vs M3 is not rejected, M0 is chosen.
2. Otherwise, if M1 vs M3 is not rejected, M1 is chosen.
3. Otherwise, if M2 vs M3 is not rejected, M2 is chosen.
4. Otherwise, M3 is chosen.

**Small cohorts.** M3 is used only when E / (number of parameters added) ≥ 10. Otherwise M2 is used as the most extensive model, and this is reported.

**Decision mapping for B4.**

- Selected level M0, M1 or M2 → "deploy (updated)".
- M3 → "revise/retrain".

B4's decisions are scored against the same ground truth as RECAL's verdict. For this, B4's selected model takes the place of the wrapper in the ground-truth rule of Section 5.6.

**Software and versions.** ADAPT is used if SKADA cannot be installed in the frozen environment. The choice is recorded in A1. Library versions are pinned in the lock file of v0.3.0.

---

## 7. Metrics on the Target-Holdout

Reported for RECAL, every baseline, the frozen model and the Reference model:

- AUROC, with DeLong 95 % CI;
- AUPRC;
- Brier score;
- calibration intercept (calibration-in-the-large, slope fixed at 1);
- calibration slope;
- ECE with **10 equal-frequency bins**;
- integrated calibration index (ICI);
- sensitivity, specificity, PPV and NPV at a probability threshold equal to the outcome prevalence of the target cohort;
- net benefit (decision curve analysis) over thresholds 0.05–0.30 in steps of 0.01. Net benefit at 0.10 and 0.20 is tabulated.

---

## 8. Statistical analysis

### 8.1 Primary endpoint

**False-deploy rate** = P(verdict = GREEN | GT = 0), for verdict rule V2 and the XGBoost model, at each E.

- Estimated per transfer, with a Wilson 95 % CI.
- Pooled over the six real transfers with a cluster bootstrap over transfers (B = 2000).

The unconditional proportion P(GREEN ∧ GT = 0) is reported alongside.

### 8.2 Secondary endpoints

- Accuracy of the binary verdict.
- Sensitivity of the verdict for deployable wrappers.
- Indeterminate rate, AMBER rate and unnecessary-retrain rate.
- The same quantities for V1, for the MLP, and for the B4 decisions.
- Distribution of rr_oof and of optimism_gap.
- Bias of RECAL's internal oracle: AUROC_oracle on the target cohort minus AUROC_Reference on the Target-Holdout.
- For each baseline: the mean paired difference from RECAL in every Section 7 metric across replicates, with bootstrap 95 % CI.

### 8.3 Acceptance criterion and minimum cohort size

**Acceptance criterion.** RECAL's V2 verdict is considered acceptable at a given E if the **upper bound of the 95 % CI of the false-deploy rate is ≤ 0.10 in every evaluable real transfer**. A transfer is evaluable if its Target-Holdout has at least 200 events (Section 4.2) and the denominator GT = 0 includes at least 10 runs.

**Minimum cohort size.** E_min is the smallest E that meets the acceptance criterion.

**Negative results.** If no E meets the criterion, this is reported as a negative result. The manuscript will then state that the verdict is not validated at the sizes tested.

### 8.4 Reporting

- Every run is reported (results tables are generated by script from the run logs). No run is excluded except for documented execution failures, which are counted and reported.
- The validation is reported following TRIPOD+AI where applicable.
- **Main text:** primary and secondary endpoints for the six real transfers and the no-shift control, with the baseline comparison at E = 50 and trends across E.
- **Supplementary material:** all remaining E values, the MLP, V1, ground-truth sensitivity analyses, the temporal transfer, the semi-synthetic shifts and the permuted-label control.

---

## 9. Exploratory analyses (not confirmatory)

- ROC-type analysis of rr_oof and optimism_gap against GT, to suggest data-driven thresholds. These would require external validation before use.
- Relationship between the drift categories assigned by RECAL's profiler and the error types.

---

## 10. Amendments and deviations

- Changes are recorded as numbered, dated entries in [`AMENDMENTS.md`](AMENDMENTS.md), committed to this repository.
- Every entry states whether it was made **before** or **after** any result of this protocol was seen.
- Changes made after results are reported as deviations in the manuscript.

**Amendment A1 (data and software freeze)** is mandatory and must precede the training of any source model. It records:

- database versions;
- cohort flow and counts per database;
- the final predictor list, aggregations and units;
- operational outcome definitions;
- the RECAL v0.3.0 tag and commit hash;
- the environment lock file;
- the choice between SKADA and ADAPT for B7.

**Timestamping.** A git commit date alone does not prove when a document was written. This protocol (commit and tag) will therefore also be archived on Zenodo, and the archive DOI recorded in `AMENDMENTS.md`.

---

## 11. Planned code layout

```
validation/
  PREREGISTRATION.md      this document
  AMENDMENTS.md           dated amendment log
  harmonisation/          extraction and harmonisation code (frozen in A1)
  configs/                RECAL YAML per transfer (generated, identical except for paths/seeds)
  run_validation.py       runs all transfers × E × replicates × methods
  analyse.py              generates every table and figure from the run logs
  seeds.json              derived seed list
```

Paths to the local PhysioNet copies are passed as arguments or environment variables. No data path or data file is committed.

---

## References

Collins, G. S., Moons, K. G. M., Dhiman, P., Riley, R. D., Beam, A. L., Van Calster, B., Ghassemi, M., Liu, X., Reitsma, J. B., van Smeden, M., Boulesteix, A.-L., Camaradou, J. C., Celi, L. A., Denaxas, S., Denniston, A. K., Glocker, B., Golub, R. M., Harvey, H., Heinze, G., … Logullo, P. (2024). TRIPOD+AI statement: Updated guidance for reporting clinical prediction models that use regression or machine learning methods. *BMJ, 385*, e078378. https://doi.org/10.1136/bmj-2023-078378

DeLong, E. R., DeLong, D. M., & Clarke-Pearson, D. L. (1988). Comparing the areas under two or more correlated receiver operating characteristic curves: A nonparametric approach. *Biometrics, 44*(3), 837–845. https://doi.org/10.2307/2531595

Steyerberg, E. W. (2019). *Clinical prediction models: A practical approach to development, validation, and updating* (2nd ed.). Springer. https://doi.org/10.1007/978-3-030-16399-0

Sun, B., Feng, J., & Saenko, K. (2016). Return of frustratingly easy domain adaptation. *Proceedings of the AAAI Conference on Artificial Intelligence, 30*(1), 2058–2065. https://doi.org/10.1609/aaai.v30i1.10306

Van Calster, B., McLernon, D. J., van Smeden, M., Wynants, L., & Steyerberg, E. W. (2019). Calibration: The Achilles heel of predictive analytics. *BMC Medicine, 17*, 230. https://doi.org/10.1186/s12916-019-1466-7

Vergouwe, Y., Nieboer, D., Oostenbrink, R., Debray, T. P. A., Murray, G. D., Kattan, M. W., Koffijberg, H., Moons, K. G. M., & Steyerberg, E. W. (2017). A closed testing procedure to select an appropriate method for updating prediction models. *Statistics in Medicine, 36*(28), 4529–4539. https://doi.org/10.1002/sim.7179
