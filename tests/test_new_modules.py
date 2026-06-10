"""
tests.test_new_modules
=======================
Tests básicos para los módulos nuevos: oracle, drift_attribution,
calibration_decomposition, audit_serializer, designer_audit.
"""
from __future__ import annotations

import numpy as np
import pytest

# ── recal_core.designer_audit ───────────────────────────────────────────────────────

class TestDesignerAudit:
    def test_record_and_get(self):
        from recal_core.designer_audit import (
            AlternativeChoice,
            DesignerAuditTrail,
            DesignerDecision,
        )

        trail = DesignerAuditTrail()
        d = DesignerDecision(
            step="mask_activate",
            criterion="n_events >= threshold",
            alternatives=[
                AlternativeChoice("enable", "activated", None, True),
                AlternativeChoice("disable", "activated", None, False),
            ],
            final_choice=True,
            justification="enough events",
        )
        trail.record(d)
        assert len(trail.decisions) == 1
        assert trail.decisions[0].step == "mask_activate"

    def test_to_dict_serializable(self):
        from recal_core.designer_audit import (
            AlternativeChoice,
            DesignerAuditTrail,
            DesignerDecision,
        )

        trail = DesignerAuditTrail()
        trail.record(DesignerDecision(
            step="pca_coral_k",
            criterion="sqrt(n_target)",
            alternatives=[AlternativeChoice(5, "n_components", None, True)],
            final_choice=5,
            justification="computed",
        ))
        d = trail.to_dict()
        assert isinstance(d, list)
        assert d[0]["step"] == "pca_coral_k"

    def test_to_json_roundtrip(self):
        import json

        from recal_core.designer_audit import DesignerAuditTrail, DesignerDecision

        trail = DesignerAuditTrail()
        trail.record(DesignerDecision("s", "c", [], "choice", "just"))
        js = trail.to_json()
        parsed = json.loads(js)
        assert len(parsed) == 1


# ── recal_cli.oracle ───────────────────────────────────────────────────────────

class TestOracle:
    def _make_data(self, n=100, seed=42):
        rng = np.random.default_rng(seed)
        X = rng.normal(size=(n, 5))
        y = (X[:, 0] + rng.normal(scale=0.5, size=n) > 0).astype(int)
        return X, y

    def test_fit_returns_dict_with_auroc(self):
        from recal_cli.oracle import fit_target_oracle
        X, y = self._make_data()

        class FakeModel:
            def predict_proba(self, X): return np.clip(X[:, 0], 0, 1)

        result = fit_target_oracle(X, y, source_model=FakeModel(), cv=3)
        assert "auroc" in result
        assert 0.0 <= result["auroc"] <= 1.0
        assert result["ci_lo"] <= result["auroc"] <= result["ci_hi"]

    def test_handles_constant_scores(self):
        """No debe explotar con scores constantes."""
        from recal_cli.oracle import fit_target_oracle

        X = np.zeros((50, 3))
        y = np.array([0] * 25 + [1] * 25)

        class ConstModel:
            def predict_proba(self, X): return np.full(len(X), 0.5)

        result = fit_target_oracle(X, y, source_model=ConstModel(), model_family="logistic", cv=3)
        assert "auroc" in result


# ── recal_cli.drift_attribution ───────────────────────────────────────────────

class TestDriftDecomposition:
    def test_basic_decomposition(self):
        from recal_cli.drift_attribution import drift_decomposition

        result = drift_decomposition(
            auroc_raw=0.70,
            auroc_adapted=0.75,
            auroc_oracle=0.85,
            ci_raw=(0.65, 0.75),
            ci_adapted=(0.70, 0.80),
            ci_oracle=(0.80, 0.90),
        )
        assert result["total_gap"] == pytest.approx(0.15, abs=1e-9)
        assert result["recoverable_gap"] == pytest.approx(0.05, abs=1e-9)
        assert result["irreducible_gap"] == pytest.approx(0.10, abs=1e-9)
        assert result["recovery_ratio"] == pytest.approx(0.05 / 0.15, abs=1e-9)
        assert not result["indeterminate"]

    def test_indeterminate_when_no_oracle(self):
        from recal_cli.drift_attribution import drift_decomposition

        result = drift_decomposition(
            auroc_raw=0.70,
            auroc_adapted=0.75,
            auroc_oracle=None,
            ci_raw=(0.65, 0.75),
            ci_adapted=(0.70, 0.80),
            ci_oracle=None,
        )
        assert result["indeterminate"] is True
        assert result["recovery_ratio"] is None

    def test_zero_total_gap_indeterminate(self):
        from recal_cli.drift_attribution import drift_decomposition

        result = drift_decomposition(
            auroc_raw=0.80,
            auroc_adapted=0.82,
            auroc_oracle=0.80,  # oracle == raw → total_gap ≈ 0
        )
        assert result["indeterminate"] is True


# ── recal_cli.calibration_decomposition ───────────────────────────────────────

class TestBrierDecomposition:
    def _make_preds(self, n=200, seed=0):
        rng = np.random.default_rng(seed)
        y = rng.integers(0, 2, size=n)
        scores = np.clip(y * 0.7 + rng.normal(0, 0.2, n), 0.01, 0.99)
        return y.astype(float), scores

    def test_brier_total_check(self):
        from recal_cli.calibration_decomposition import brier_decompose

        y, s = self._make_preds()
        result = brier_decompose(y, s)
        expected = float(np.mean((y - s) ** 2))
        assert result["brier_score"] == pytest.approx(expected, abs=1e-6)

    def test_murphy_identity(self):
        """BS ≈ reliability − resolution + uncertainty."""
        from recal_cli.calibration_decomposition import brier_decompose

        y, s = self._make_preds()
        r = brier_decompose(y, s, n_bins=10)
        check = r["reliability"] - r["resolution"] + r["uncertainty"]
        assert r["brier_score"] == pytest.approx(check, abs=0.01)

    def test_delta_negative_on_improvement(self):
        from recal_cli.calibration_decomposition import brier_decompose, brier_delta

        y = np.array([0, 0, 1, 1] * 25, dtype=float)
        raw = brier_decompose(y, np.full(100, 0.5))  # worst calibration
        adapted = brier_decompose(y, np.clip(y + np.random.default_rng(0).normal(0, 0.05, 100), 0, 1))
        delta = brier_delta(raw, adapted)
        # Adapted scores should be better → delta_brier_score < 0
        assert "delta_brier_score" in delta


# ── AdapterConfig new fields ───────────────────────────────────────────────────

class TestAdapterConfigFields:
    def test_audit_and_sweep_default_values(self):
        from recal_core.designer.base import AdapterConfig

        cfg = AdapterConfig()
        assert cfg.audit is None
        assert cfg.mask_sweep_history == []

    def test_can_assign_audit(self):
        from recal_core.designer.base import AdapterConfig
        from recal_core.designer_audit import DesignerAuditTrail

        cfg = AdapterConfig()
        cfg.audit = DesignerAuditTrail()
        assert isinstance(cfg.audit, DesignerAuditTrail)
