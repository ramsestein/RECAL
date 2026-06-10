"""
Pytest fixtures compartidos para el paquete recal.

Los fixtures sintéticos permiten ejecutar los tests de alineación, selección
y bootstrap SIN depender de los CSVs de datos reales.  El test de
reproducción exacta (``test_reproduces_legacy.py``) sí accede a los datasets
y está marcado ``@pytest.mark.slow``.
"""

from __future__ import annotations

import numpy as np
import pytest

# ── Seeds y dimensiones ───────────────────────────────────────────────────────
N_SOURCE = 500   # SNUH sintético
N_TARGET = 100   # Clínic sintético
N_FEAT = 20      # features
RNG_SEED = 42


@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    """Generador reproducible."""
    return np.random.default_rng(RNG_SEED)


@pytest.fixture(scope="session")
def synthetic_source(rng: np.random.Generator) -> np.ndarray:
    """
    Matriz fuente sintética de forma (N_SOURCE, N_FEAT).

    Distribución multivariada con covarianza no diagonal para que CORAL y
    PCA-CORAL tengan algo que alinear.
    """
    mean = np.zeros(N_FEAT)
    A = rng.standard_normal((N_FEAT, N_FEAT))
    cov = A @ A.T / N_FEAT + np.eye(N_FEAT) * 0.5
    return rng.multivariate_normal(mean, cov, size=N_SOURCE)


@pytest.fixture(scope="session")
def synthetic_target(rng: np.random.Generator) -> np.ndarray:
    """
    Matriz destino sintética de forma (N_TARGET, N_FEAT).

    Introducimos un offset de media y rotación de covarianza deliberada para
    simular covariate shift.
    """
    mean = rng.uniform(1.0, 3.0, size=N_FEAT)     # offset de media
    A = rng.standard_normal((N_FEAT, N_FEAT))
    cov = A @ A.T / N_FEAT + np.eye(N_FEAT) * 1.0
    return rng.multivariate_normal(mean, cov, size=N_TARGET)


@pytest.fixture(scope="session")
def nan_mask_target(rng: np.random.Generator) -> np.ndarray:
    """
    Máscara booleana (N_TARGET, N_FEAT) con ~10% de NaN por celda.
    Simula los missing values del dataset Clínic.
    """
    return rng.random(size=(N_TARGET, N_FEAT)) < 0.10


@pytest.fixture(scope="session")
def synthetic_labels_source(rng: np.random.Generator) -> np.ndarray:
    """Etiquetas binarias aleatorias para la fuente (n_pos ≈ 25%)."""
    return (rng.random(N_SOURCE) < 0.25).astype(int)


@pytest.fixture(scope="session")
def synthetic_labels_target(rng: np.random.Generator) -> np.ndarray:
    """Etiquetas binarias aleatorias para el destino (n_pos ≈ 30%)."""
    return (rng.random(N_TARGET) < 0.30).astype(int)
