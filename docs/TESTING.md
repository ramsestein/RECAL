# Testing Guide

This document explains how the RECAL test suite is organised, what data it uses,
how to run it, and how to interpret the results.

---

## 1. Philosophy

RECAL ships **two categories of tests**:

| Category | Data source | Purpose | Speed |
|---|---|---|---|
| **Fast / CI** | Synthetic CSVs bundled in `recal_core/tests/data/` | Verify the pipeline executes correctly on any machine | ~4 min |
| **Slow / Benchmark** | Real clinical CSVs (not in repo) | Verify exact benchmark numbers (AUROC, slope, etc.) | ~5-10 min |

The fast suite runs in GitHub Actions. The slow suite is run locally when the
real SNUH/Clínic datasets are available.

---

## 2. Test Data

### 2.1 Synthetic data (bundled)

Located in `recal_core/tests/data/`:

| File | Description |
|---|---|
| `synthetic_source.csv` | 800 rows, 114 features, prevalence ~0.48 |
| `synthetic_target.csv` | 150 rows, 114 features, prevalence ~0.19 |
| `synthetic_schema.json` | Ordered list of 114 feature names |
| `synthetic_model.json` | XGBoost model trained on `synthetic_source.csv` |

The target is deliberately injected with covariate shift (mean/variance changes)
and missing values so that `filter_target(0.5)` leaves **105 rows** with **29
events** — matching the real Clínic benchmark dimensions.

### 2.2 Real data (not in repository)

When the following files exist, tests automatically switch to **real data** and
use **exact assertions**:

```
datasets/SNUH_AKI(SNUH_AKI).csv
datasets/Clínic_AKI.csv
inputs/feature_schema.json
model/aki_external.json
```

---

## 3. Running the Tests

### 3.1 Fast suite (CI default)

```bash
pytest -m "not slow"
```

This skips the 6 benchmark-performance tests that require real drift to produce
reliable AUROC/slope improvements.

### 3.2 Slow suite (requires real data)

```bash
pytest -m slow
```

### 3.3 Everything

```bash
pytest
```

### 3.4 Specific file

```bash
pytest recal_core/tests/test_validation_snuh_clinic.py -v
pytest tests/test_reproduces_legacy.py -v
```

---

## 4. Current Results

### Fast suite (`pytest -m "not slow"`)

```
145 passed, 6 deselected, 19 warnings in ~240 s
```

Breakdown:

- `recal_core/tests/test_auto_adapter.py` — 3 tests (profiler, design, arrays)
- `recal_core/tests/test_e2e_synthetic.py` — 3 tests (train XGB, full pipeline, HTML report)
- `recal_core/tests/test_validation_snuh_clinic.py` — 17 tests (data sanity, profiler outputs, designer decisions)
- `tests/test_reproduces_legacy.py` — 4 tests (filter count, positives, AUROC raw, AUROC PCA-CORAL)
- Other unit tests across `recal/` and `recal_core/` — ~118 tests

### Deselected / slow tests

6 tests in `test_validation_snuh_clinic.py`:

- `TestPipelinePerformance::test_auroc_improves`
- `TestPipelinePerformance::test_auroc_in_expected_range`
- `TestPipelinePerformance::test_calibration_slope_improves`
- `TestPipelinePerformance::test_calibration_slope_in_range`
- `TestPipelinePerformance::test_ece_improves`
- `test_html_report_generates`

These are marked `@pytest.mark.slow` because they assert that the pipeline
**improves** AUROC, calibration slope, and ECE. With random synthetic data the
alignment may not improve the model, so these assertions are only reliable with
real clinical drift.

---

## 5. Linting (Ruff)

All Python code is linted with **Ruff**.

```bash
ruff check .
```

Current status: `All checks passed!`

To auto-fix import sorting and trivial issues:

```bash
ruff check . --fix
```

Configuration lives in `pyproject.toml`:

```toml
[tool.ruff.lint]
select = ["E", "F", "I", "W"]
ignore = ["E501"]

[tool.ruff.lint.isort]
known-first-party = ["recal", "recal_core", "recal_cli"]
```

---

## 6. Test Architecture

```
recal_core/tests/
├── test_auto_adapter.py          # Unit tests for AutoAdapter profiler
├── test_e2e_synthetic.py         # Full pipeline with random data
├── test_validation_snuh_clinic.py # SNUH→Clínic regression (synthetic fallback)
└── data/
    ├── synthetic_source.csv
    ├── synthetic_target.csv
    ├── synthetic_schema.json
    └── synthetic_model.json

tests/
└── test_reproduces_legacy.py     # Exact-number regression (synthetic fallback)
```

### Key design patterns

1. **Fallback fixtures** — `test_validation_snuh_clinic.py` detects whether real
   CSVs exist; if not, it loads synthetic data and adjusts assertions.
2. **`is_synthetic` flag** — Returned by fixtures so that assertions can branch
   between exact numbers (real) and permissive ranges (synthetic).
3. **`@pytest.mark.slow`** — Only applied to tests that assert *improvement*
   (AUROC, slope, ECE). Pure functionality tests run fast.
4. **`XGBoostWrapper` requires `model_path`** — No hidden dummy training inside
   the wrapper. Tests train a real XGBoost, save it to disk, and load it.

---

## 7. Adding New Tests

1. If the test needs data, use the synthetic fixture or generate arrays inline.
2. If the test asserts exact benchmark numbers, mark it `@pytest.mark.slow` and
   branch on `is_synthetic` for permissive ranges.
3. Run `ruff check .` before committing.
4. Verify both fast and (if you have real data) slow suites pass.
