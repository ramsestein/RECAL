# Changelog — RECAL

All notable changes to the `recal_core/` package.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [0.1.0] — 2025-06

### Added

#### Block A: Profiler
- `recal_core/profiler/constants.py` — 18 thresholds with empirical justifications.
- `recal_core/profiler/base.py` — Dataclasses `FeatureProfile` (17 fields) and `DriftProfile`
  with helper methods (`features_by_quadrant`, `poisonous_features`, etc.).
- `recal_core/profiler/quadrant.py` — `assign_quadrants(shap, lbase)` → quadrants A/B/C/D.
- `recal_core/profiler/global_profiler.py` — `profile_global()`: MMD², Fisher exact,
  AUROC CI bootstrap (n=500), calibration slope, ECE, CITL.
- `recal_core/profiler/feature_profiler.py` — `profile_features()`: L_base (logistic LASSO),
  SHAP importance, concept shift (beta3, qbh, flip), quadrant assignment, combined score.
- `recal_core/profiler/profiler.py` — `Profiler` class combining global + feature profiles.

#### Block B: Designer
- `recal_core/designer/base.py` — `AdapterConfig` dataclass with all decision fields
  plus `rationale: dict` with justifications.
- `recal_core/designer/rules.py` — 5 deterministic rules:
  - `should_mask_features()` — Activates if n_events ≥ 20.
  - `select_mask_n()` — Elbow of second derivative on combined_scores; capped at 20%.
  - `should_apply_quantile_transform_per_feature()` — NONLINEAR/PARTIAL + cv≥5% + var_ratio outside (0.5, 2.0).
  - `should_apply_woe_per_feature()` — STABLE/LINEAR + n_target_events≥30 + n_source_events≥100 + SHAP≥0.005.
  - `should_apply_pca_coral()` — Always True.
  - `select_pca_coral_k()` — Var≥80% on source PCA; capped at sqrt(n_target).
  - `should_recalibrate()` — |slope-1| > 0.5 AND n_events≥20.
  - `select_calibration_method()` — Isotonic if n≥500; Platt LOO otherwise.
- `recal_core/designer/selector.py` — `ComponentSelector` class: profile → AdapterConfig.

#### Block C: AutoAdapter
- `recal_core/pipeline/auto_adapter.py` — `AutoAdapter` class:
  - `.profile(pair)` → DriftProfile
  - `.design()` → AdapterConfig
  - `.fit(pair)` → self
  - `.predict(pair)` → np.ndarray
  - `.auto_adapt(pair)` → np.ndarray (full pipeline with filter_target)
  - `.profile_from_arrays()` and `._predict_from_arrays()` (test helpers)
- `recal_core/pipeline/__init__.py`

#### Block D: Reporter
- `recal_core/reporter/tables.py` — 4 Markdown table generators.
- `recal_core/reporter/figures.py` — 4 matplotlib figures (quadrant map, calibration curve,
  combined score bar, missing rates).
- `recal_core/reporter/html_report.py` — `generate_html_report()`: self-contained HTML
  with base64-embedded figures.
- `recal_core/reporter/__init__.py`

#### Block E: SNUH→Clínic Validation Test
- `recal_core/tests/test_validation_snuh_clinic.py` — E2E regression test:
  - `TestDataSanity` — Checks n_obs, n_events, baseline AUROC.
  - `TestProfilerOutputs` — Checks slope>2, ≥3 distinct quadrants.
  - `TestDesignerDecisions` — Checks 7 known deterministic decisions.
  - `TestPipelinePerformance` (SLOW) — Checks AUROC∈[0.65, 0.75], slope∈[0.5, 1.5].
  - `test_html_report_generates` (SLOW) — Checks HTML>5000 chars.

#### Block F: Additional Tests
- `recal_core/tests/test_profiler.py` — Unit tests for Profiler with synthetic data.
- `recal_core/tests/test_designer_rules.py` — Unit tests for each Designer rule.
- `recal_core/tests/test_auto_adapter.py` — Integration tests with synthetic data.
- `recal_core/tests/test_dataset_shift_invariance.py` — Invariance (determinism, scaling, monotonicity).

### Notes
- `recal/` was not modified in any commit of this version.
- The Designer does not use target metrics to select hyperparameters
  (honest evaluation guaranteed).

---

## [Unreleased]

See [OPEN_QUESTIONS.md](OPEN_QUESTIONS.md) for planned future work.
