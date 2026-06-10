"""
tests.test_joint_drift
=======================
Tests for recal_cli.joint_drift using synthetic Gaussian data with
known covariance structure.
"""

from __future__ import annotations

import numpy as np
import pytest

from recal_cli.joint_drift import (
    compute_condition_number,
    compute_effective_rank,
    compute_vif,
    joint_drift_report,
    mi_matrix_delta,
)

RNG = np.random.default_rng(0)

# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_correlated(n: int, corr: float, p: int = 3, rng=RNG) -> np.ndarray:
    """Generate n × p Gaussian matrix with equicorrelation `corr`."""
    cov = np.full((p, p), corr)
    np.fill_diagonal(cov, 1.0)
    return rng.multivariate_normal(np.zeros(p), cov, size=n)


def _make_independent(n: int, p: int = 3, rng=RNG) -> np.ndarray:
    """Generate n × p Gaussian matrix with identity covariance."""
    return rng.standard_normal((n, p))


# ── compute_vif ───────────────────────────────────────────────────────────────


class TestComputeVIF:
    def test_independent_features_vif_near_one(self):
        """Independent features → VIF ≈ 1."""
        X = _make_independent(500)
        df = compute_vif(X, ["a", "b", "c"])
        assert list(df.columns) == ["feature", "VIF"]
        assert len(df) == 3
        assert (df["VIF"].values < 3.0).all(), f"Expected VIF ≈ 1, got {df['VIF'].values}"

    def test_highly_correlated_vif_high(self):
        """Highly correlated features → VIF >> 1."""
        X = _make_correlated(500, corr=0.95)
        df = compute_vif(X, ["a", "b", "c"])
        assert df["VIF"].max() > 5.0, f"Expected high VIF, got {df['VIF'].values}"

    def test_feature_names_preserved(self):
        X = _make_independent(100, p=2)
        df = compute_vif(X, ["x1", "x2"])
        assert list(df["feature"]) == ["x1", "x2"]

    def test_wrong_feature_length_raises(self):
        X = _make_independent(50, p=3)
        with pytest.raises(ValueError, match="feature names"):
            compute_vif(X, ["only_one"])

    def test_constant_feature_no_error(self):
        """Constant column → VIF = 1 (no division by zero)."""
        X = _make_independent(100, p=3)
        X[:, 0] = 5.0  # constant
        df = compute_vif(X, ["const", "b", "c"])
        assert np.isfinite(df["VIF"].values).all()

    def test_vif_capped_at_1000(self):
        """Near-perfect collinearity → VIF capped at 1000."""
        rng = np.random.default_rng(42)
        base = rng.standard_normal(300)
        X = np.column_stack([base, base + rng.standard_normal(300) * 1e-6, rng.standard_normal(300)])
        df = compute_vif(X, ["a", "b", "c"])
        assert df["VIF"].max() <= 1000.0


# ── compute_condition_number ──────────────────────────────────────────────────


class TestComputeConditionNumber:
    def test_identity_condition_number_is_one(self):
        """Identity covariance → κ = 1."""
        X = _make_independent(1000)
        kappa = compute_condition_number(X)
        # With finite samples there will be deviation; allow up to 5
        assert kappa < 5.0, f"Expected κ ≈ 1 for independent data, got {kappa:.2f}"

    def test_correlated_data_higher_condition_number(self):
        """High correlation → κ > identity case."""
        X_ind = _make_independent(500)
        X_cor = _make_correlated(500, corr=0.99)
        kappa_ind = compute_condition_number(X_ind)
        kappa_cor = compute_condition_number(X_cor)
        assert kappa_cor > kappa_ind, (
            f"Expected correlated κ ({kappa_cor:.1f}) > "
            f"independent κ ({kappa_ind:.1f})"
        )

    def test_scalar_returns_float(self):
        X = _make_independent(100, p=3)
        result = compute_condition_number(X)
        assert isinstance(result, float)

    def test_single_feature(self):
        """Single feature column → κ = 1."""
        X = _make_independent(50, p=1)
        kappa = compute_condition_number(X)
        assert kappa == 1.0


# ── compute_effective_rank ────────────────────────────────────────────────────


class TestComputeEffectiveRank:
    def test_independent_effective_rank_near_p(self):
        """Independent features → effective rank ≈ p."""
        p = 5
        X = _make_independent(2000, p=p)
        er = compute_effective_rank(X)
        # Should be close to p
        assert er > p * 0.7, f"Expected effective rank ≈ {p}, got {er:.2f}"

    def test_rank1_effective_rank_near_one(self):
        """Rank-1 matrix → effective rank ≈ 1."""
        rng = np.random.default_rng(7)
        v = rng.standard_normal(200)
        X = np.outer(v, np.ones(4))  # all columns identical
        er = compute_effective_rank(X)
        assert er < 1.5, f"Expected effective rank ≈ 1 for rank-1 data, got {er:.2f}"

    def test_effective_rank_between_1_and_p(self):
        X = _make_correlated(300, corr=0.7, p=4)
        er = compute_effective_rank(X)
        assert 1.0 <= er <= 4.0, f"Effective rank {er:.2f} out of [1, 4]"

    def test_higher_corr_lower_effective_rank(self):
        """More correlation → lower effective rank."""
        er_low = compute_effective_rank(_make_correlated(500, corr=0.1, p=5))
        er_high = compute_effective_rank(_make_correlated(500, corr=0.9, p=5))
        assert er_low > er_high, (
            f"Expected low-corr effective rank ({er_low:.2f}) > "
            f"high-corr ({er_high:.2f})"
        )


# ── joint_drift_report ────────────────────────────────────────────────────────


class TestJointDriftReport:
    def test_same_distribution_flags_ok(self):
        """Source == target distribution → all flags should be OK."""
        X = _make_independent(500, p=4)
        features = ["a", "b", "c", "d"]
        df = joint_drift_report(X, X, features)
        assert list(df.columns) == ["feature", "vif_source", "flag"]
        assert (df["flag"] == "OK").all(), f"Expected all OK, got: {df['flag'].tolist()}"

    def test_shifted_covariance_triggers_watch_or_severe(self):
        """Source: highly correlated → VIF_source should trigger flags."""
        rng = np.random.default_rng(99)
        X_src = _make_correlated(500, corr=0.97, rng=rng)  # high VIF in source
        X_tgt = rng.standard_normal((300, 3))  # target not used for VIF
        features = ["f1", "f2", "f3"]
        df = joint_drift_report(X_src, X_tgt, features, delta_vif_warn=2.0, delta_vif_severe=5.0)
        flagged = (df["flag"] != "OK").any()
        assert flagged, f"Expected some flags, all were OK:\n{df}"

    def test_custom_thresholds(self):
        rng = np.random.default_rng(3)
        X_src = rng.standard_normal((300, 3))
        X_tgt = _make_correlated(200, corr=0.9, rng=rng)
        features = ["x", "y", "z"]
        df_strict = joint_drift_report(X_src, X_tgt, features, delta_vif_warn=0.5, delta_vif_severe=1.0)
        df_loose = joint_drift_report(X_src, X_tgt, features, delta_vif_warn=100.0, delta_vif_severe=200.0)
        assert (df_loose["flag"] == "OK").all()
        assert (df_strict["flag"] != "OK").any()

    def test_mismatched_shapes_raises(self):
        X_s = _make_independent(100, p=3)
        X_t = _make_independent(50, p=4)
        with pytest.raises(ValueError, match="features"):
            joint_drift_report(X_s, X_t, ["a", "b", "c"])

    def test_output_shape(self):
        p = 6
        X = _make_independent(200, p=p)
        features = [f"f{i}" for i in range(p)]
        df = joint_drift_report(X, X, features)
        assert len(df) == p

    def test_severe_flag_when_vif_delta_large(self):
        """Source near-perfect collinearity → VIF_source >> threshold → SEVERE."""
        rng = np.random.default_rng(11)
        # Source: near-perfect collinearity between features 0 and 1 → VIF >> 10
        base = rng.standard_normal(1000)
        X_src = np.column_stack([
            base,
            base + rng.standard_normal(1000) * 0.01,
            rng.standard_normal(1000),
        ])
        X_tgt = rng.standard_normal((200, 3))  # target not used for VIF
        features = ["a", "b", "c"]
        df = joint_drift_report(X_src, X_tgt, features, delta_vif_warn=2.0, delta_vif_severe=5.0)
        assert (df["flag"] == "SEVERE").any(), (
            f"Expected at least one SEVERE flag:\n{df}"
        )


# ── mi_matrix_delta ───────────────────────────────────────────────────────────


class TestMIMatrixDelta:
    def test_same_data_delta_near_zero(self):
        """Same data → MI delta ≈ 0."""
        X = _make_independent(300, p=3)
        delta = mi_matrix_delta(X, X)
        assert delta < 1.0, f"Expected small MI delta for identical data, got {delta:.4f}"

    def test_independent_vs_correlated_larger_delta(self):
        """Independent vs correlated → larger MI delta than identical."""
        rng = np.random.default_rng(42)
        X_ind = rng.standard_normal((200, 3))
        X_cor = _make_correlated(200, corr=0.95, rng=rng)
        delta_same = mi_matrix_delta(X_ind, X_ind)
        delta_diff = mi_matrix_delta(X_ind, X_cor)
        assert delta_diff > delta_same, (
            f"Expected diff delta ({delta_diff:.4f}) > same delta ({delta_same:.4f})"
        )

    def test_returns_float(self):
        X = _make_independent(100, p=2)
        assert isinstance(mi_matrix_delta(X, X), float)
