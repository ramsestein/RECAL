"""
Tests de reproducción exacta de los números del pipeline legado.

Estos tests son LENTOS (acceden a los CSVs reales y al modelo XGBoost) y
están marcados con ``@pytest.mark.slow``.

Para ejecutarlos:
    pytest tests/test_reproduces_legacy.py -v -m slow

Para saltarlos en CI rápido:
    pytest tests/ -m "not slow"

Tolerancias
-----------
AUROC: atol=1e-3  (los decimales publicados tienen 4 dígitos)
"""

from __future__ import annotations

import pytest
from sklearn.metrics import roc_auc_score

pytestmark = pytest.mark.slow


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_pipeline():
    """Carga loaders, model y pair con los parámetros exactos del script x_eval.py."""
    from recal.data.clinic import ClinicLoader
    from recal.data.pairing import CohortPair
    from recal.data.schema import load_schema
    from recal.data.snuh import SNUHLoader
    from recal.model.xgboost_wrapper import XGBoostWrapper

    schema = load_schema()
    snuh = SNUHLoader(schema=schema)
    clinic = ClinicLoader(schema=schema)

    pair_raw = CohortPair(source=snuh, target=clinic)
    pair = pair_raw.filter_target(max_missing_rate=0.5)

    model = XGBoostWrapper(schema=schema)
    return pair, model


# ── Números de referencia ─────────────────────────────────────────────────────

# Verificados contra los parquets (data/processed/) SIN máscara de features.
# Equivalen al pipeline de x_eval.py ejecutado con MASK_FEATURES = [].
#
# Con máscara de 10 features (bottom-10 L_base+SHAP, N=10 del _sweep_n.py):
#   Raw=0.6318, PCA-CORAL k=5=0.7559, GAP=0.033 (publicados en reporte).
# Con máscara de 15 features (bottom-15, BUG corregido en x_eval.py):
#   Raw=0.5202, PCA-CORAL k=5=0.6436, GAP=0.1528 (valor incorrecto).
REF_N_TARGET_FILTERED = 105
REF_N_POS_TARGET = 29
REF_AUROC_RAW = 0.6293
REF_AUROC_PCA_CORAL_K5 = 0.7051
ATOL = 1e-3


# ── Test 1: filtrado ──────────────────────────────────────────────────────────

@pytest.mark.slow
def test_filter_target_count():
    """n_target después del filtrado (< 50% missing) debe ser 105."""
    pair, _ = _load_pipeline()
    assert pair.X_t.shape[0] == REF_N_TARGET_FILTERED, (
        f"Expected {REF_N_TARGET_FILTERED} samples after filter, "
        f"got {pair.X_t.shape[0]}"
    )


@pytest.mark.slow
def test_filter_target_positives():
    """n_pos_target después del filtrado debe ser 29."""
    pair, _ = _load_pipeline()
    assert int(pair.y_t.sum()) == REF_N_POS_TARGET


# ── Test 2: AUROC Raw (sin alineación) ───────────────────────────────────────

@pytest.mark.slow
def test_auroc_raw():
    """AUROC Clínic sin alineación ≈ 0.6293 (sin máscara)."""
    from recal.align.identity import IdentityAligner

    pair, model = _load_pipeline()
    X_aligned = pair.align(IdentityAligner())
    proba = model.predict_proba(X_aligned)
    auroc = roc_auc_score(pair.y_t, proba)
    assert abs(auroc - REF_AUROC_RAW) <= ATOL, (
        f"Raw AUROC {auroc:.4f} deviates from reference {REF_AUROC_RAW} "
        f"by {abs(auroc - REF_AUROC_RAW):.4f} > {ATOL}"
    )


# ── Test 3: AUROC PCA-CORAL k=5 ──────────────────────────────────────────────

@pytest.mark.slow
def test_auroc_pca_coral_k5():
    """AUROC Clínic con PCA-CORAL k=5 ≈ 0.7051 (sin máscara)."""
    from recal.align.pca_coral import PCACoralAligner

    pair, model = _load_pipeline()
    X_aligned = pair.align(PCACoralAligner(k=5, shrinkage=None))  # legacy: no LW shrinkage
    proba = model.predict_proba(X_aligned)
    auroc = roc_auc_score(pair.y_t, proba)
    assert abs(auroc - REF_AUROC_PCA_CORAL_K5) <= ATOL, (
        f"PCA-CORAL k=5 AUROC {auroc:.4f} deviates from reference "
        f"{REF_AUROC_PCA_CORAL_K5} by {abs(auroc - REF_AUROC_PCA_CORAL_K5):.4f} > {ATOL}"
    )
