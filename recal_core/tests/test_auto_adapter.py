"""
recal_core/tests/test_auto_adapter.py
=================================
Tests de integración del AutoAdapter con datos sintéticos.

No usa datos reales (SNUH/Clínic) para ser rápido.
Los tests E2E con datos reales están en test_validation_snuh_clinic.py.
"""

from __future__ import annotations

import numpy as np
import pytest

# ── Helpers ───────────────────────────────────────────────────────────────────

class FakeModel:
    """Modelo fake con predict_proba() y shap_values()."""

    def __init__(self, n_features: int = 10, seed: int = 42):
        self._rng = np.random.default_rng(seed)
        self._n_features = n_features

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        # Predicción basada en la primera feature (correlacionada con y)
        raw = 1 / (1 + np.exp(-X[:, 0]))
        return np.clip(raw, 0.05, 0.95)

    def shap_values(self, X: np.ndarray) -> np.ndarray:
        shap = np.zeros_like(X)
        shap[:, 0] = X[:, 0] * 0.5  # Solo la feature 0 tiene SHAP no nulo
        return shap


def _make_synthetic_cohort_arrays(n_s=300, n_t=60, p=10, seed=0):
    rng = np.random.default_rng(seed)
    X_s = rng.normal(0, 1, (n_s, p))
    y_s = (X_s[:, 0] + rng.normal(0, 0.3, n_s) > 0).astype(int)
    X_t = rng.normal(1, 1, (n_t, p))  # drift: media desplazada 1
    y_t = (X_t[:, 0] + rng.normal(0, 0.3, n_t) > 1).astype(int)
    y_t[0] = 1
    y_t[1] = 0  # garantizar ambas clases
    schema = [f"feat_{j:02d}" for j in range(p)]
    return X_s, y_s, X_t, y_t, schema


# ── Tests del Profiler integrado en AutoAdapter ───────────────────────────────

class TestAutoAdapterProfile:

    def test_profile_returns_drift_profile(self):
        from recal_core.pipeline.auto_adapter import AutoAdapter
        from recal_core.profiler.base import DriftProfile

        X_s, y_s, X_t, y_t, schema = _make_synthetic_cohort_arrays()
        model = FakeModel(n_features=len(schema))

        # Simular CohortPair-like con solo arrays
        class FakePair:
            X_s = X_s
            y_s = y_s
            X_t = X_t
            y_t = y_t
            X_s_imp = X_s
            X_t_imp = X_t
            mu_s = X_s.mean(axis=0)
            nan_mask_t = np.zeros_like(X_t, dtype=bool)
            idx_corr = list(range(X_t.shape[1]))
            schema_list = schema

        aa = AutoAdapter(model=model, schema=schema)
        profile = aa.profile_from_arrays(X_s, y_s, X_t, y_t)
        assert isinstance(profile, DriftProfile)
        assert len(profile.features) == len(schema)

    def test_profile_from_arrays_helper(self):
        """profile_from_arrays() es un helper conveniente."""
        from recal_core.pipeline.auto_adapter import AutoAdapter

        X_s, y_s, X_t, y_t, schema = _make_synthetic_cohort_arrays()
        model = FakeModel(n_features=len(schema))
        aa = AutoAdapter(model=model, schema=schema)
        profile = aa.profile_from_arrays(X_s, y_s, X_t, y_t)
        assert profile.n_source_obs == len(y_s)
        assert profile.n_target_obs == len(y_t)


# ── Tests del Designer integrado en AutoAdapter ───────────────────────────────

class TestAutoAdapterDesign:

    def test_design_after_profile(self):
        from recal_core.designer.base import AdapterConfig
        from recal_core.pipeline.auto_adapter import AutoAdapter

        X_s, y_s, X_t, y_t, schema = _make_synthetic_cohort_arrays()
        model = FakeModel(n_features=len(schema))
        aa = AutoAdapter(model=model, schema=schema)
        aa.profile_from_arrays(X_s, y_s, X_t, y_t)
        config = aa.design()
        assert isinstance(config, AdapterConfig)

    def test_design_without_profile_raises(self):
        from recal_core.pipeline.auto_adapter import AutoAdapter
        aa = AutoAdapter(model=FakeModel(), schema=["f0", "f1"])
        with pytest.raises(RuntimeError):
            aa.design()


# ── Tests de métricas del pipeline ───────────────────────────────────────────

class TestAutoAdapterPredictSanity:

    def test_predict_returns_probabilities(self):
        """Las probabilidades deben estar en [0, 1]."""
        from recal_core.pipeline.auto_adapter import AutoAdapter

        X_s, y_s, X_t, y_t, schema = _make_synthetic_cohort_arrays()
        model = FakeModel(n_features=len(schema))
        aa = AutoAdapter(model=model, schema=schema)
        profile = aa.profile_from_arrays(X_s, y_s, X_t, y_t)
        aa._profile = profile
        aa.design()

        proba = aa._predict_from_arrays(X_s, y_s, X_t, y_t)
        assert proba.shape == (len(y_t),)
        assert np.all(proba >= 0) and np.all(proba <= 1)

    def test_predict_no_nans(self):
        from recal_core.pipeline.auto_adapter import AutoAdapter

        X_s, y_s, X_t, y_t, schema = _make_synthetic_cohort_arrays()
        model = FakeModel(n_features=len(schema))
        aa = AutoAdapter(model=model, schema=schema)
        aa.profile_from_arrays(X_s, y_s, X_t, y_t)
        aa.design()
        proba = aa._predict_from_arrays(X_s, y_s, X_t, y_t)
        assert not np.any(np.isnan(proba))
