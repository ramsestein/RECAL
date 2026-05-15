"""
adapt/tests/test_profiler.py
============================
Tests unitarios del Profiler.

Usa datos sintéticos para probar cada subcomponente de forma aislada.
"""

from __future__ import annotations

import numpy as np
import pytest


# ── Fixtures sintéticas ───────────────────────────────────────────────────────

def _make_synthetic_pair(
    n_s: int = 200, n_t: int = 50, p: int = 10, seed: int = 42
):
    """Par sintético con drift controlado."""
    rng = np.random.default_rng(seed)

    # Source: features normales
    X_s = rng.normal(0, 1, (n_s, p))
    y_s = (X_s[:, 0] + X_s[:, 1] + rng.normal(0, 0.5, n_s) > 0.5).astype(int)

    # Target: drift en las primeras 3 features (media desplazada)
    X_t = rng.normal(0, 1, (n_t, p))
    X_t[:, :3] += 2.0   # desplazamiento de media
    y_t = (X_t[:, 0] + X_t[:, 1] + rng.normal(0, 0.5, n_t) > 0.5).astype(int)

    # Garantizar al menos un positivo y un negativo en target
    y_t[0] = 1
    y_t[1] = 0

    schema = [f"feat_{j:02d}" for j in range(p)]
    return X_s, y_s, X_t, y_t, schema


class TestGlobalProfiler:
    """Tests de profile_global()."""

    def test_output_keys(self):
        from adapt.profiler.global_profiler import profile_global

        class FakeModel:
            def predict_proba(self, X):
                return np.random.default_rng(0).random(X.shape[0])

        X_s, y_s, X_t, y_t, schema = _make_synthetic_pair()
        result = profile_global(X_s, y_s, X_t, y_t, FakeModel())

        required = [
            "n_source_obs", "n_target_obs", "n_source_events", "n_target_events",
            "prevalence_source", "prevalence_target", "prevalence_shift_pvalue",
            "p_n_ratio_target", "mmd2_source_target", "mmd2_pvalue",
            "pca_variance_explained", "baseline_auroc", "baseline_auroc_ci_low",
            "baseline_auroc_ci_high", "baseline_calibration_slope",
            "baseline_ece", "baseline_citl",
        ]
        for key in required:
            assert key in result, f"Falta la clave: {key}"

    def test_mmd2_nonnegative(self):
        from adapt.profiler.global_profiler import profile_global

        class FakeModel:
            def predict_proba(self, X):
                return np.full(X.shape[0], 0.3)

        X_s, y_s, X_t, y_t, schema = _make_synthetic_pair()
        result = profile_global(X_s, y_s, X_t, y_t, FakeModel())
        assert result["mmd2_source_target"] >= 0

    def test_prevalences_in_unit_interval(self):
        from adapt.profiler.global_profiler import profile_global

        class FakeModel:
            def predict_proba(self, X):
                return np.full(X.shape[0], 0.5)

        X_s, y_s, X_t, y_t, schema = _make_synthetic_pair()
        result = profile_global(X_s, y_s, X_t, y_t, FakeModel())
        assert 0 <= result["prevalence_source"] <= 1
        assert 0 <= result["prevalence_target"] <= 1

    def test_auroc_in_unit_interval(self):
        from adapt.profiler.global_profiler import profile_global

        class FakeModel:
            def predict_proba(self, X):
                rng = np.random.default_rng(99)
                return rng.random(X.shape[0])

        X_s, y_s, X_t, y_t, schema = _make_synthetic_pair()
        result = profile_global(X_s, y_s, X_t, y_t, FakeModel())
        assert 0 <= result["baseline_auroc"] <= 1
        assert result["baseline_auroc_ci_low"] <= result["baseline_auroc"]
        assert result["baseline_auroc"] <= result["baseline_auroc_ci_high"]


class TestQuadrantAssignment:
    """Tests de assign_quadrants()."""

    def test_returns_correct_length(self):
        from adapt.profiler.quadrant import assign_quadrants
        n = 20
        shap = np.random.default_rng(0).random(n)
        lbase = np.random.default_rng(1).random(n)
        result = assign_quadrants(shap, lbase)
        assert len(result) == n

    def test_valid_quadrant_names(self):
        from adapt.profiler.quadrant import assign_quadrants
        valid = {"A_core", "B_noisy_important", "C_redundant", "D_ponzonous"}
        shap = np.array([1.0, 1.0, 0.0, 0.0])
        lbase = np.array([1.0, 0.0, 1.0, 0.0])
        result = assign_quadrants(shap, lbase)
        for q in result:
            assert q in valid, f"Cuadrante inválido: {q}"

    def test_high_high_is_a_core(self):
        """Shap alto + L_base alto → A_core."""
        from adapt.profiler.quadrant import assign_quadrants
        # Extremos claros
        shap = np.array([0.0, 0.0, 0.0, 10.0])
        lbase = np.array([0.0, 0.0, 0.0, 10.0])
        result = assign_quadrants(shap, lbase)
        assert result[-1] == "A_core", f"Expected A_core, got {result[-1]}"

    def test_low_low_is_d_ponzonous(self):
        """Shap bajo + L_base bajo → D_ponzonous."""
        from adapt.profiler.quadrant import assign_quadrants
        shap = np.array([10.0, 10.0, 10.0, 0.0])
        lbase = np.array([10.0, 10.0, 10.0, 0.0])
        result = assign_quadrants(shap, lbase)
        assert result[-1] == "D_ponzonous", f"Expected D_ponzonous, got {result[-1]}"


class TestDriftProfileDataclass:
    """Tests de DriftProfile."""

    def _make_feature(self, name="f", drift_type="STABLE", shap=0.1, lbase=0.5,
                      combined=0.3, quadrant="A_core", cv_t=0.5):
        from adapt.profiler.base import FeatureProfile
        return FeatureProfile(
            name=name, domain="preop", drift_type_v=drift_type,
            lbase_score=lbase, shap_importance=shap, combined_score=combined,
            quadrant=quadrant, univariate_concept_shift_beta3=0.1,
            univariate_concept_shift_qbh=0.5, flip_of_sign=False,
            cv_target=cv_t, cv_source=0.4, var_ratio=1.0,
            near_constant_target=False, missing_rate_source=0.1,
            missing_rate_target=0.2,
        )

    def test_features_by_quadrant(self):
        from adapt.profiler.base import DriftProfile, FeatureProfile
        f1 = self._make_feature("a", quadrant="A_core")
        f2 = self._make_feature("b", quadrant="D_ponzonous")
        f3 = self._make_feature("c", quadrant="A_core")
        profile = DriftProfile(
            n_source_obs=100, n_target_obs=50, n_source_events=30, n_target_events=15,
            prevalence_source=0.3, prevalence_target=0.3, prevalence_shift_pvalue=0.5,
            p_n_ratio_target=0.1, mmd2_source_target=0.1, mmd2_pvalue=0.1,
            pca_variance_explained=[0.4, 0.65, 0.78, 0.88, 0.95],
            baseline_auroc=0.7, baseline_auroc_ci_low=0.6, baseline_auroc_ci_high=0.8,
            baseline_calibration_slope=1.0, baseline_ece=0.05, baseline_citl=0.01,
            features=[f1, f2, f3],
        )
        assert len(profile.features_by_quadrant("A_core")) == 2
        assert len(profile.features_by_quadrant("D_ponzonous")) == 1

    def test_ponzonous_features(self):
        from adapt.profiler.base import DriftProfile
        f1 = self._make_feature("a", quadrant="D_ponzonous")
        f2 = self._make_feature("b", quadrant="A_core")
        profile = DriftProfile(
            n_source_obs=100, n_target_obs=50, n_source_events=30, n_target_events=15,
            prevalence_source=0.3, prevalence_target=0.3, prevalence_shift_pvalue=0.5,
            p_n_ratio_target=0.1, mmd2_source_target=0.1, mmd2_pvalue=0.1,
            pca_variance_explained=[0.4, 0.65, 0.78, 0.88, 0.95],
            baseline_auroc=0.7, baseline_auroc_ci_low=0.6, baseline_auroc_ci_high=0.8,
            baseline_calibration_slope=1.0, baseline_ece=0.05, baseline_citl=0.01,
            features=[f1, f2],
        )
        pz = profile.ponzonous_features()
        assert len(pz) == 1
        assert pz[0].name == "a"
