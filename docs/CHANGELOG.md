# Changelog — RECAL

All notable changes to the RECAL project.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [0.2.0] — 2026-06-10

### Renamed / Rebranded
- Project renamed from **ADAPT** to **RECAL** (Recalibration & Alignment Wrapper)
  to differentiate from classical domain-adaptation methods.
- Packages renamed: `domain_transfer` → `recal`, `adapt` → `recal_core`,
  `adapt_cli` → `recal_cli`.
- Output paths updated: `outputs/adapted_models/` → `outputs/recal_models/`,
  `outputs/reports/adapt_report.html` → `outputs/reports/recal_report.html`.

### Added
- `recal/model/xgboost_wrapper.py` — Schema-aware XGBoost loader (requires real
  `model_path`; no hidden dummy training).
- `recal_core/tests/test_e2e_synthetic.py` — End-to-end test that generates
  synthetic source+target, trains a real XGBoost, and runs the full RECAL pipeline.
- `recal_core/tests/data/` — Bundled synthetic CSVs, schema JSON and trained
  XGBoost model for CI (no real clinical data required).
- `docs/TESTING.md` — Test-suite overview, running instructions, data philosophy.

### Changed
- `recal_core/tests/test_validation_snuh_clinic.py` — Now uses synthetic data as
  fallback when real CSVs are absent; assertions branch on `is_synthetic`.
- `tests/test_reproduces_legacy.py` — Same fallback to synthetic data;
  exact-number assertions only run with real data.
- `docs/CHANGELOG.md` and `docs/OPEN_QUESTIONS.md` — Translated to English.

### Fixed
- `pyproject.toml` — Proper `pip install` metadata (readme, license, authors,
  classifiers, project URLs).
- CI badge in `README.md` now points to the new `RECAL` repository.
- `ruff check .` passes cleanly (no unused imports, no dead code).

---

## [0.1.0] — 2026-06 (initial implementation)

### Added
- **Profiler** (`recal_core/profiler/`): global drift profile (MMD², AUROC CI,
  calibration slope, ECE) + per-feature profiles (L_base, SHAP, concept shift,
  quadrant assignment).
- **Designer** (`recal_core/designer/`): deterministic rule engine that selects
  masking, quantile transform, WOE, PCA-CORAL k, and calibration method from the
  drift profile.
- **AutoAdapter** (`recal_core/pipeline/auto_adapter.py`): orchestrates
  profile → design → fit → predict in one class.
- **Reporter** (`recal_core/reporter/`): self-contained HTML report with
  base64-embedded figures and Markdown tables.
- **CLI** (`recal_cli/run.py`): end-to-end orchestrator with YAML config,
  cross-validation, drift attribution, and counterfactual sweep.
- **Alignment algorithms** (`recal/align/`): PCA-CORAL, Quantile Transform,
  WOE encoder, selective alignment, AdaBN, Optimal Transport.
  *Note: AdaBN and Optimal Transport are present in the codebase but are not
  exposed in the documented Designer workflow or the default CLI pipeline.*
- **Calibration** (`recal/calibration/`): Stratified Platt recalibrator with
  isotonic / Platt LOO selection.
- **Validation tests** (`recal_core/tests/test_validation_snuh_clinic.py`):
  E2E regression against SNUH→Clínic benchmark.

---

## [0.2.2] — 2026-06-11

### Added
- **Alignment strategy sweep** (`recal_core/designer/rules.py`): new function
  `select_alignment_strategy()` that compares PCA-CORAL (with k around the
  heuristic optimum) against CORAL pure via mini-sweep on target, selecting
  whichever achieves the highest AUROC. This replaces the previous
  variance-explained heuristic that never considered CORAL pure.
- **`use_coral_pure` flag** (`recal_core/designer/base.py`): new boolean on
  `AdapterConfig` that signals the pipeline to use `CoralAligner` instead of
  `PCACoralAligner` when selected by the alignment sweep.
- **CoralAligner support** (`recal_core/pipeline/auto_adapter.py`): `fit()` and
  `_get_aligned_scores()` now handle both `PCACoralAligner` and `CoralAligner`
  transparently based on `config.use_coral_pure`.

### Changed
- `recal_core/designer/selector.py` — The PCA-CORAL block now calls
  `select_alignment_strategy()` when `pair` and `model` are available, enabling
  the CORAL pure vs PCA-CORAL comparison. Falls back to variance-explained k
  heuristic when no target data is available.
- `recal_core/profiler/feature_profiler.py` — LASSO logistic regression updated
  to use `l1_ratio=1` instead of the deprecated `penalty='l1'` parameter
  (sklearn 1.8+ API).

### Fixed
- `synthetic_data/run_recala.py` — Removed all manual overrides that forced
  masking, WOE, quantile transform, and calibration to `False`. The pipeline
  now delegates fully to the Designer's internal heuristics. Removed manual
  alignment sweep and handcrafted audit trail.
- `recal_core/designer/rules.py` — Added missing `import warnings` needed by
  the new `select_alignment_strategy()`.

---
