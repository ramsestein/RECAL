"""
Tests para los aligners CORAL y PCA-CORAL.

Comprueba que:
1. La distancia de Frobenius entre las covarianzas alineadas es ≪ la distancia
   antes de alinear (la alineación mejora la similitud distribucional).
2. El output tiene la misma forma que la entrada.
3. Los NaN del nan_mask se restauran correctamente.
4. safe_sqrtm(M) @ safe_sqrtm(M) ≈ M (propiedades matemáticas).
5. safe_invsqrtm(M) @ M @ safe_invsqrtm(M) ≈ I.
"""

from __future__ import annotations

import numpy as np
import pytest

from domain_transfer.align.base import safe_invsqrtm, safe_sqrtm
from domain_transfer.align.coral import CoralAligner
from domain_transfer.align.pca_coral import PCACoralAligner


# ── Utilidades de covarianza ──────────────────────────────────────────────────

def frob_cov_dist(A: np.ndarray, B: np.ndarray) -> float:
    """
    Distancia de Frobenius entre las matrices de covarianza de A y B.
    Mide la disimilitud de segundo orden entre distribuciones.
    """
    return np.linalg.norm(np.cov(A, rowvar=False) - np.cov(B, rowvar=False), "fro")


# ── Tests safe_sqrtm / safe_invsqrtm ─────────────────────────────────────────

class TestMatrixUtils:
    def _spd_matrix(self, p: int = 6) -> np.ndarray:
        rng = np.random.default_rng(0)
        A = rng.standard_normal((p, p))
        return A @ A.T + np.eye(p) * 0.5

    def test_sqrtm_product(self):
        M = self._spd_matrix()
        S = safe_sqrtm(M)
        np.testing.assert_allclose(S @ S, M, atol=1e-8)

    def test_invsqrtm_identity(self):
        M = self._spd_matrix()
        Si = safe_invsqrtm(M)
        I_approx = Si @ M @ Si
        np.testing.assert_allclose(I_approx, np.eye(M.shape[0]), atol=1e-8)

    def test_sqrtm_symmetric(self):
        M = self._spd_matrix()
        S = safe_sqrtm(M)
        np.testing.assert_allclose(S, S.T, atol=1e-10)

    def test_invsqrtm_symmetric(self):
        M = self._spd_matrix()
        Si = safe_invsqrtm(M)
        np.testing.assert_allclose(Si, Si.T, atol=1e-10)


# ── Tests CoralAligner ────────────────────────────────────────────────────────

class TestCoralAligner:
    def test_output_shape(self, synthetic_source, synthetic_target):
        aligner = CoralAligner()
        out = aligner.fit_transform(synthetic_source, synthetic_target)
        assert out.shape == synthetic_target.shape

    def test_reduces_covariance_distance(self, synthetic_source, synthetic_target):
        """CORAL debería reducir la distancia de Frobenius entre covarianzas."""
        before = frob_cov_dist(synthetic_source, synthetic_target)
        aligner = CoralAligner()
        aligned = aligner.fit_transform(synthetic_source, synthetic_target)
        after = frob_cov_dist(synthetic_source, aligned)
        assert after < before, (
            f"CORAL no redujo la distancia de Frobenius: {before:.3f} → {after:.3f}"
        )

    def test_nan_mask_restored(self, synthetic_source, synthetic_target, nan_mask_target):
        aligner = CoralAligner()
        aligner.fit(synthetic_source, synthetic_target)
        aligned = aligner.transform(synthetic_target, nan_mask=nan_mask_target)
        # Las posiciones NaN del mask deben ser NaN en el output
        assert np.all(np.isnan(aligned[nan_mask_target]))
        # Las posiciones no-NaN no deben ser NaN
        assert not np.any(np.isnan(aligned[~nan_mask_target]))

    def test_no_nan_without_mask(self, synthetic_source, synthetic_target):
        aligner = CoralAligner()
        out = aligner.fit_transform(synthetic_source, synthetic_target)
        assert not np.any(np.isnan(out))

    def test_raises_if_not_fitted(self, synthetic_target):
        aligner = CoralAligner()
        with pytest.raises(RuntimeError, match="fitted"):
            aligner.transform(synthetic_target)


# ── Tests PCACoralAligner ─────────────────────────────────────────────────────

class TestPCACoralAligner:
    def test_output_shape(self, synthetic_source, synthetic_target):
        aligner = PCACoralAligner(k=5)
        out = aligner.fit_transform(synthetic_source, synthetic_target)
        assert out.shape == synthetic_target.shape

    def test_reduces_covariance_distance(self, synthetic_source, synthetic_target):
        """PCA-CORAL debería reducir la distancia de Frobenius entre covarianzas."""
        before = frob_cov_dist(synthetic_source, synthetic_target)
        aligner = PCACoralAligner(k=5)
        aligned = aligner.fit_transform(synthetic_source, synthetic_target)
        after = frob_cov_dist(synthetic_source, aligned)
        assert after < before, (
            f"PCA-CORAL no redujo la distancia de Frobenius: {before:.3f} → {after:.3f}"
        )

    def test_nan_mask_restored(self, synthetic_source, synthetic_target, nan_mask_target):
        aligner = PCACoralAligner(k=5)
        aligner.fit(synthetic_source, synthetic_target)
        aligned = aligner.transform(synthetic_target, nan_mask=nan_mask_target)
        assert np.all(np.isnan(aligned[nan_mask_target]))
        assert not np.any(np.isnan(aligned[~nan_mask_target]))

    def test_no_nan_without_mask(self, synthetic_source, synthetic_target):
        aligner = PCACoralAligner(k=5)
        out = aligner.fit_transform(synthetic_source, synthetic_target)
        assert not np.any(np.isnan(out))

    def test_k_clipped_gracefully(self, synthetic_source):
        """Si k > n_features, PCA lo ajusta sin excepciones."""
        small = synthetic_source[:, :3]  # solo 3 features
        small_t = small[:50]
        aligner = PCACoralAligner(k=100)  # k >> p=3 → debe cliparse
        out = aligner.fit_transform(small, small_t)
        assert out.shape == small_t.shape

    def test_raises_if_not_fitted(self, synthetic_target):
        aligner = PCACoralAligner(k=5)
        with pytest.raises(RuntimeError, match="fitted"):
            aligner.transform(synthetic_target)
