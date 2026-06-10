# OPEN QUESTIONS — RECAL

Open questions and known issues for future releases.

---

## OQ-1: Calibration heterogeneity (strata)

**File:** `recal_core/designer/rules.py::_test_calibration_heterogeneity()`  
**Status:** Conservative (always returns p=1.0)

**Description:**  
The `_test_calibration_heterogeneity()` function currently always returns 1.0,
meaning `select_calibration_method()` never chooses `stratified_platt`.
This is equivalent to assuming "no heterogeneity" and always using `platt_loo`.

For SNUH→Clínic this is correct (n_events=29 → no statistical power to detect
stratified heterogeneity). But in larger cohorts the gain from stratified
calibration might be underestimated.

**Plan v0.3:**  
Implement a Hosmer–Lemeshow test by stratum using `StratifiedPlattRecalibrator._strata_fn`
in `recal/calibration/stratified_platt.py`. Use chi2 with k-1 degrees of freedom.

---

## OQ-2: PCACoralAligner double fit in `_get_aligned_scores()`

**File:** `recal_core/pipeline/auto_adapter.py::_get_aligned_scores()`  
**Status:** Minor inefficiency, not a bug (deterministic result)

**Description:**  
`_get_aligned_scores()` is called both in `fit()` (for calibration data) and in
`predict()`. The `PCACoralAligner` is re-fit on both calls with the same data,
so the result is identical but the computation is duplicated.

**Plan v0.3:**  
Add a `self._aligner_fitted: bool` flag and skip `.fit()` if already fitted.

```python
if not self._aligner_fitted:
    self._fitted_aligner.fit(X_s_corr, X_t_corr)
    self._aligner_fitted = True
X_t_aligned = self._fitted_aligner.transform(X_t_corr, nan_mask=nan_mask_corr)
```

---

## OQ-3: Computed vs pre-computed L_base — a methodological divergence

**File:** `recal_core/profiler/feature_profiler.py::_compute_lbase_scores()`  
**Status:** **Methodological divergence** (not a bug — two different quantities)

**Description:**  
There are two valid ways to estimate the baseline predictive capacity of a
feature, and RECAL currently mixes them without always making the distinction
explicit to the user.

| Variant | Implementation | What it measures |
|---|---|---|
| **Reference (CSV)** | `results/v/v_drift_decomposition.csv` — computed offline with a univariate XGBoost trained on source and evaluated on source. | Non-linear predictive capacity (AUROC). |
| **Fallback (code)** | `_compute_lbase_scores()` — logistic LASSO on source. | Linear correlation with the outcome. |

These are **different quantities**. The CSV values were used to produce the
published SNUH→Clínic benchmark numbers. When the CSV is absent, the LASSO
fallback is silently substituted. Any manuscript reporting benchmark results must
state which variant was used.

The opaque path `results/v/v_drift_decomposition.csv` is not shipped with the
repository; it is an offline artefact. For reproducibility, the CSV should be
renamed to something descriptive (e.g. `feature_baseline_scores.csv`) and
accompanied by a provenance note.

**Plan v0.3:**  
- Rename the fallback field to `lbase_score_approx` to signal it is not identical
to the CSV reference.
- Add an explicit `lbase_method: Literal["xgboost_auroc", "lasso_approx"]` flag
so callers know which quantity they are looking at.
- Move the pre-computed CSV to a documented location (e.g.
`inputs/feature_baselines.csv`) with a README explaining its provenance.

---

## OQ-4: `drift_type_v` parameter in `UnivariateConceptShiftDiagnoser`

**File:** `recal_core/profiler/feature_profiler.py`  
**Status:** To verify in integration tests

**Description:**  
`feature_profiler.py` passes `drift_type_v=[drift_type]` as a kwarg to
`UnivariateConceptShiftDiagnoser.fit()`. If this class does not accept that
parameter, the fallback handles the error (beta3=0, qbh=1.0, flip=False) but
does not report a warning.

**Plan v0.3:**  
Check the constructor signature in `recal/drift/concept_shift_univariate.py`.
If the parameter does not exist, remove it from the call and compute drift_type_v internally.

---

## OQ-5: Multi-target extension

**File:** `recal_core/pipeline/auto_adapter.py`  
**Status:** Not implemented

**Description:**  
The current pipeline assumes a single target hospital. For multi-target transfer
(SNUH → [Clínic, HUGTiP, HJD]), each target needs its own `Profiler` and `AutoAdapter`.
There is no grouping or hierarchical transfer mechanism.

**Plan v0.3:**  
Add `MultiTargetAutoAdapter(targets: dict[str, CohortPair])` that runs RECAL in
parallel and combines reports into a single comparative HTML.

---

## OQ-6: MMD² computation robustness in `global_profiler.py`

**File:** `recal_core/profiler/global_profiler.py`  
**Status:** Sub-optimal in high dimensionality

**Description:**  
The MMD² computation uses an RBF kernel with default bandwidth (median heuristic).
When p >> n_target the computation becomes unstable. It should project onto the
first k PCs before computing MMD².

**Plan v0.3:**  
Pre-project onto the first `min(k_pca, n_target//2)` PCA components before the
kernel computation. Add `pca_before_mmd: bool = True` as a constants parameter.

---

## OQ-7: Reporter without Jinja2 (maintainability)

**File:** `recal_core/reporter/html_report.py`  
**Status:** Functional, but fragile for complex HTML

**Description:**  
The HTML is generated by string concatenation in Python. It works but is hard to
maintain if more sections need to be added.

**Plan v0.3:**  
Evaluate using `jinja2` as an optional dependency. If not installed, fall back to
the current generator.
