# Changelog — ADAPT

Todos los cambios notables en el paquete `adapt/`.

El formato sigue [Keep a Changelog](https://keepachangelog.com/es/1.0.0/).

---

## [0.1.0] — 2025-06

### Añadido

#### Bloque A: Profiler
- `adapt/profiler/constants.py` — 18 thresholds con justificaciones empíricas.
- `adapt/profiler/base.py` — Dataclasses `FeatureProfile` (17 campos) y `DriftProfile`
  con métodos helpers (`features_by_quadrant`, `ponzonous_features`, etc.).
- `adapt/profiler/quadrant.py` — `assign_quadrants(shap, lbase)` → cuadrantes A/B/C/D.
- `adapt/profiler/global_profiler.py` — `profile_global()`: MMD², Fisher exact,
  AUROC CI bootstrap (n=500), calibration slope, ECE, CITL.
- `adapt/profiler/feature_profiler.py` — `profile_features()`: L_base (LASSO logístico),
  SHAP importance, concept shift (beta3, qbh, flip), quadrant assignment, combined score.
- `adapt/profiler/profiler.py` — Clase `Profiler` que combina global + features.

#### Bloque B: Designer
- `adapt/designer/base.py` — Dataclass `AdapterConfig` con todos los campos de decisión
  más `rationale: dict` con justificaciones.
- `adapt/designer/rules.py` — 5 reglas determinísticas:
  - `should_mask_features()` — Activa si n_events ≥ 20.
  - `select_mask_n()` — Elbow de second derivative en combined_scores; cap 20%.
  - `should_apply_quantile_transform_per_feature()` — NONLINEAR/PARTIAL + cv≥5% + var_ratio fuera de (0.5, 2.0).
  - `should_apply_woe_per_feature()` — STABLE/LINEAR + n_target_events≥30 + n_source_events≥100 + SHAP≥0.005.
  - `should_apply_pca_coral()` — Siempre True.
  - `select_pca_coral_k()` — Var≥80% en source PCA; cap sqrt(n_target).
  - `should_recalibrate()` — |slope-1| > 0.5 AND n_events≥20.
  - `select_calibration_method()` — Isotónica si n≥500; Platt LOO en otro caso.
- `adapt/designer/selector.py` — Clase `ComponentSelector`: profile → AdapterConfig.

#### Bloque C: AutoAdapter
- `adapt/pipeline/auto_adapter.py` — Clase `AutoAdapter`:
  - `.profile(pair)` → DriftProfile
  - `.design()` → AdapterConfig
  - `.fit(pair)` → self
  - `.predict(pair)` → np.ndarray
  - `.auto_adapt(pair)` → np.ndarray (pipeline completa con filter_target)
  - `.profile_from_arrays()` y `._predict_from_arrays()` (helpers para tests)
- `adapt/pipeline/__init__.py`

#### Bloque D: Reporter
- `adapt/reporter/tables.py` — 4 generadores de tablas Markdown.
- `adapt/reporter/figures.py` — 4 figuras matplotlib (cuadrant map, calibration curve,
  combined score bar, missing rates).
- `adapt/reporter/html_report.py` — `generate_html_report()`: HTML autocontenido
  con figuras en base64.
- `adapt/reporter/__init__.py`

#### Bloque E: Test Validación SNUH→Clínic
- `adapt/tests/test_validation_snuh_clinic.py` — Test E2E de regresión:
  - `TestDataSanity` — Verifica n_obs, n_events, AUROC baseline.
  - `TestProfilerOutputs` — Verifica slope>2, ≥3 cuadrantes distintos.
  - `TestDesignerDecisions` — Verifica 7 decisiones determinísticas conocidas.
  - `TestPipelinePerformance` (SLOW) — Verifica AUROC∈[0.65, 0.75], slope∈[0.5, 1.5].
  - `test_html_report_generates` (SLOW) — Verifica HTML>5000 chars.

#### Bloque F: Tests adicionales
- `adapt/tests/test_profiler.py` — Tests unitarios del Profiler con datos sintéticos.
- `adapt/tests/test_designer_rules.py` — Tests unitarios de cada regla del Designer.
- `adapt/tests/test_auto_adapter.py` — Tests de integración con datos sintéticos.
- `adapt/tests/test_dataset_shift_invariance.py` — Invarianza (determinismo, escalado, monotonicidad).

### Notas
- `domain_transfer/` no fue modificado en ningún commit de esta versión.
- El Designer no usa métricas de target para seleccionar hiperparámetros
  (evaluación honesta garantizada).

---

## [Unreleased]

Ver [OPEN_QUESTIONS.md](OPEN_QUESTIONS.md) para trabajo futuro planificado.
