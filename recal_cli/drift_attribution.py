"""
recal_cli.drift_attribution
============================
Descomposición del drift en recuperable vs irreducible.

Marco conceptual
----------------
Dado el triangle (raw, adapted, oracle):

    auroc_raw    — modelo source sin adaptar en target
    auroc_adapted — modelo source después de RECAL
    auroc_oracle — techo: modelo entrenado nativamente en target (k-fold OOF)

    total_gap        = auroc_oracle  - auroc_raw
    recoverable_gap  = auroc_adapted - auroc_raw
    irreducible_gap  = auroc_oracle  - auroc_adapted
    recovery_ratio   = recoverable_gap / total_gap   (si total_gap ≠ 0)

Propagación de incertidumbre
----------------------------
Los CIs de los gaps se propagan conservadoramente como suma en cuadratura
de los SE individuales (asume independencia entre estimadores):

    SE(gap) ≈ sqrt(SE1^2 + SE2^2)

Para recovery_ratio usamos delta-method:

    SE(r) ≈ |r| * sqrt((SE_rec/recoverable)^2 + (SE_total/total)^2)

cuando total ≈ 0, el ratio se reporta como None (indeterminado).

Feature recovery attribution
-----------------------------
Para cada feature j:
    1. Construir pipeline RECAL con feature j SIN alinear (resto sí)
    2. AUROC_j_unaligned
    3. Diferencia: contribution_j = AUROC_full - AUROC_j_unaligned
    → features con mayor contribution aportan más a la recuperación del gap
"""

from __future__ import annotations

import logging
import warnings

import numpy as np

logger = logging.getLogger(__name__)

# Umbral para considerar total_gap "prácticamente cero"
_GAP_EPS = 1e-6


def _ci_to_se(value: float, ci_lo: float, ci_hi: float, z: float = 1.96) -> float:
    """Convierte [ci_lo, ci_hi] a SE aproximado."""
    if np.isnan(ci_lo) or np.isnan(ci_hi):
        return np.nan
    return (ci_hi - ci_lo) / (2 * z)


def _gap_ci(
    val_a: float, val_b: float,
    se_a: float, se_b: float,
    z: float = 1.96,
) -> tuple[float, float]:
    """CI del gap (val_a - val_b) propagando SE en cuadratura."""
    gap = val_a - val_b
    if np.isnan(se_a) or np.isnan(se_b):
        return np.nan, np.nan
    se_gap = float(np.sqrt(se_a**2 + se_b**2))
    return gap - z * se_gap, gap + z * se_gap


def drift_decomposition(
    auroc_raw: float,
    auroc_adapted: float,
    auroc_oracle: float,
    ci_raw: tuple | None = None,
    ci_adapted: tuple | None = None,
    ci_oracle: tuple | None = None,
    alpha: float = 0.05,
) -> dict:
    """
    Descompone el gap de drift en recuperable vs irreducible.

    Parameters
    ----------
    auroc_raw : float
        AUROC del modelo source sin adaptar en target.
    auroc_adapted : float
        AUROC tras RECAL.
    auroc_oracle : float
        AUROC del oracle entrenado nativamente en target (k-fold OOF).
    ci_raw, ci_adapted, ci_oracle : tuple (lo, hi), optional
        CIs de cada estimador para propagación de incertidumbre.
    alpha : float
        Nivel de significancia (default 0.05 → 95% CI).

    Returns
    -------
    dict con:
        total_gap, recoverable_gap, irreducible_gap : float
        total_gap_ci, recoverable_gap_ci, irreducible_gap_ci : (lo, hi) or (nan, nan)
        recovery_ratio : float or None
        recovery_ratio_ci : (lo, hi) or (nan, nan) or None
        indeterminate : bool — True si total_gap ≈ 0
        note : str
    """

    z = float(__import__("scipy").stats.norm.ppf(1 - alpha / 2))

    # Si no hay oracle, todo es indeterminado
    if auroc_oracle is None:
        return {
            "auroc_raw": float(auroc_raw),
            "auroc_adapted": float(auroc_adapted),
            "auroc_oracle": None,
            "total_gap": None,
            "recoverable_gap": float(auroc_adapted) - float(auroc_raw),
            "irreducible_gap": None,
            "total_gap_ci": (np.nan, np.nan),
            "recoverable_gap_ci": (np.nan, np.nan),
            "irreducible_gap_ci": (np.nan, np.nan),
            "ci_raw": ci_raw,
            "ci_adapted": ci_adapted,
            "ci_oracle": None,
            "recovery_ratio": None,
            "recovery_ratio_ci": None,
            "indeterminate": True,
            "note": "Oracle no disponible; recovery_ratio indeterminado.",
        }

    total_gap = float(auroc_oracle) - float(auroc_raw)
    recoverable_gap = float(auroc_adapted) - float(auroc_raw)
    irreducible_gap = float(auroc_oracle) - float(auroc_adapted)

    # SE de cada estimador
    se_raw = _ci_to_se(*((auroc_raw,) + tuple(ci_raw)), z=z) if ci_raw else np.nan
    se_adapted = _ci_to_se(*((auroc_adapted,) + tuple(ci_adapted)), z=z) if ci_adapted else np.nan
    se_oracle = _ci_to_se(*((auroc_oracle,) + tuple(ci_oracle)), z=z) if ci_oracle else np.nan

    # CIs de los gaps
    total_gap_ci = _gap_ci(auroc_oracle, auroc_raw, se_oracle, se_raw, z=z)
    recoverable_gap_ci = _gap_ci(auroc_adapted, auroc_raw, se_adapted, se_raw, z=z)
    irreducible_gap_ci = _gap_ci(auroc_oracle, auroc_adapted, se_oracle, se_adapted, z=z)

    # Recovery ratio
    indeterminate = abs(total_gap) < _GAP_EPS
    if indeterminate:
        recovery_ratio = None
        recovery_ratio_ci = None
        note = (
            f"total_gap={total_gap:.6f} ≈ 0 (< {_GAP_EPS}): "
            "recovery_ratio indeterminado. El modelo source ya alcanza el techo."
        )
        logger.warning(note)
    else:
        recovery_ratio = float(recoverable_gap / total_gap)
        # Delta method para el ratio
        if not any(np.isnan(x) for x in [se_adapted, se_raw, se_oracle]):
            se_rec = float(np.sqrt(se_adapted**2 + se_raw**2))
            se_total = float(np.sqrt(se_oracle**2 + se_raw**2))
            rec_abs = abs(recoverable_gap) if abs(recoverable_gap) > _GAP_EPS else _GAP_EPS
            tot_abs = abs(total_gap)
            se_ratio = abs(recovery_ratio) * float(
                np.sqrt((se_rec / rec_abs)**2 + (se_total / tot_abs)**2)
            )
            recovery_ratio_ci = (
                recovery_ratio - z * se_ratio,
                recovery_ratio + z * se_ratio,
            )
        else:
            recovery_ratio_ci = (np.nan, np.nan)
        note = (
            f"total_gap={total_gap:+.4f}  recoverable={recoverable_gap:+.4f}  "
            f"irreducible={irreducible_gap:+.4f}  recovery_ratio={recovery_ratio:.3f}"
        )

    return {
        "auroc_raw": auroc_raw,
        "auroc_adapted": auroc_adapted,
        "auroc_oracle": auroc_oracle,
        "total_gap": total_gap,
        "recoverable_gap": recoverable_gap,
        "irreducible_gap": irreducible_gap,
        "total_gap_ci": total_gap_ci,
        "recoverable_gap_ci": recoverable_gap_ci,
        "irreducible_gap_ci": irreducible_gap_ci,
        "recovery_ratio": recovery_ratio,
        "recovery_ratio_ci": recovery_ratio_ci,
        "indeterminate": indeterminate,
        "note": note,
    }


# ── Feature recovery attribution ────────────────────────────────────────────

def feature_recovery_attribution(
    auto_adapter,
    pair,
    y_target: np.ndarray,
    top_n: int = 20,
) -> list[dict]:
    """
    Atribución del recovery por feature.

    Para cada feature j activa en la pipeline RECAL:
        AUROC_full    — pipeline completa (línea base)
        AUROC_j_off   — pipeline con feature j SIN alinear (resto sí)
        contribution  = AUROC_full - AUROC_j_off

    Features con contribution > 0 contribuyen positivamente al recovery.

    Parameters
    ----------
    auto_adapter : AutoAdapter
        AutoAdapter ya fitted sobre el par.
    pair : CohortPair
        Par usado para el fit (ya filtrado).
    y_target : np.ndarray
        Labels del target.
    top_n : int
        Limitar la atribución a las top_n features del pipeline
        (por combined_score ascendente). Si 0 o None, todas las features.

    Returns
    -------
    list[dict] — ordenada por |contribution| descendente, con:
        feature, auroc_full, auroc_j_off, contribution, in_mask, in_pca
    """
    from sklearn.metrics import roc_auc_score

    config = auto_adapter._config
    profile = auto_adapter._profile

    if config is None or profile is None:
        logger.warning("AutoAdapter no fitted; no se puede computar feature attribution.")
        return []

    # AUROC full (pipeline completa)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scores_full = auto_adapter.predict(pair)
    try:
        auroc_full = float(roc_auc_score(y_target, scores_full))
    except ValueError:
        logger.warning("feature_recovery_attribution: no se puede calcular AUROC full.")
        return []

    # Features a evaluar: las que sí entran a la pipeline
    # (no enmascaradas, activas en CORAL)
    schema = auto_adapter._schema
    masked = set(config.mask_features) if config.apply_mask else set()
    qt_feats = set(config.quantile_features) if config.apply_quantile else set()
    woe_feats = set(config.woe_features) if config.apply_woe else set()

    # Candidatos: features que entran a PCA-CORAL o a algún aligner
    candidates = [f for f in schema if f not in masked]

    if top_n and top_n > 0:
        # Ordenar por combined_score ascendente (más prioritarias)
        feat_scores = {f.name: f.combined_score for f in profile.features}
        candidates = sorted(candidates, key=lambda f: feat_scores.get(f, 0.0))[:top_n]

    results = []

    for feat in candidates:
        try:
            contribution, auroc_j_off = _attribution_one_feature(
                auto_adapter, pair, feat, schema, config, y_target, auroc_full
            )
            results.append({
                "feature": feat,
                "auroc_full": auroc_full,
                "auroc_j_off": auroc_j_off,
                "contribution": contribution,
                "in_mask": feat in masked,
                "in_qt": feat in qt_feats,
                "in_woe": feat in woe_feats,
                "in_pca": feat not in masked and config.apply_pca_coral,
            })
        except Exception as e:
            logger.debug("Attribution feature %s failed: %s", feat, e)

    results.sort(key=lambda x: abs(x["contribution"]), reverse=True)
    return results


def _attribution_one_feature(
    auto_adapter,
    pair,
    feature: str,
    schema: list[str],
    config,
    y_target: np.ndarray,
    auroc_full: float,
) -> tuple[float, float]:
    """
    Compute AUROC con feature `feature` excluida de todo alineamiento.
    Devuelve (contribution, auroc_j_off).
    """
    from sklearn.metrics import roc_auc_score


    feat2idx = {f: i for i, f in enumerate(schema)}
    feat_idx = feat2idx.get(feature)
    if feat_idx is None:
        raise ValueError(f"Feature {feature!r} no en schema.")

    # Reproducir pipeline manualmente con feature j bloqueada (sin alinear)
    working_pair = pair
    if config.apply_mask and config.mask_features:
        working_pair = working_pair.mask_features(config.mask_features)

    X_t = working_pair.X_t_imp.copy()
    X_s = working_pair.X_s_imp.copy()
    idx_corr = working_pair.idx_corr
    nan_mask_t = working_pair.nan_mask_t

    local_feat2idx = {f: i for i, f in enumerate(working_pair.schema)}
    local_feat_idx = local_feat2idx.get(feature)

    # WOE (sin feature j)
    if config.apply_woe and auto_adapter._fitted_woe is not None:
        woe_feats_active = [f for f in config.woe_features if f != feature]
        woe_idx = [local_feat2idx[f] for f in woe_feats_active if f in local_feat2idx]
        woe_idx_corr = [j for j in woe_idx if j in idx_corr]
        if woe_idx_corr:
            X_t[:, woe_idx_corr] = auto_adapter._fitted_woe.transform(
                X_t[:, woe_idx_corr]
            )

    # QT (sin feature j)
    if config.apply_quantile and auto_adapter._fitted_qt is not None:
        qt_feats_active = [f for f in config.quantile_features if f != feature]
        qt_idx = [local_feat2idx[f] for f in qt_feats_active if f in local_feat2idx]
        qt_idx_corr = [j for j in qt_idx if j in idx_corr]
        if qt_idx_corr:
            X_t[:, qt_idx_corr] = auto_adapter._fitted_qt.transform(
                X_t[:, qt_idx_corr],
                nan_mask=nan_mask_t[:, qt_idx_corr],
            )

    # Imputar NaN con media source
    mu_s = working_pair.mu_s
    X_t = np.where(np.isnan(X_t), mu_s[np.newaxis, :], X_t)
    X_t = np.nan_to_num(X_t, nan=0.0)

    # PCA-CORAL excluyendo feature j
    if config.apply_pca_coral and auto_adapter._fitted_aligner is not None:
        # idx_corr sin el idx de feature j
        idx_corr_no_j = [i for i in idx_corr if i != local_feat_idx]
        if idx_corr_no_j:
            X_s_corr = np.nan_to_num(X_s[:, idx_corr_no_j], nan=0.0)
            X_t_corr = np.nan_to_num(X_t[:, idx_corr_no_j], nan=0.0)
            nan_mask_corr = nan_mask_t[:, idx_corr_no_j]

            k = config.pca_coral_k
            k = min(k, X_s_corr.shape[1], X_s_corr.shape[0] - 1, X_t_corr.shape[0] - 1)
            if k >= 1:
                from recal.align.pca_coral import PCACoralAligner as _PCA
                aligner_j = _PCA(k=k, reg_pca=1e-6, random_state=42)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    aligner_j.fit(X_s_corr, X_t_corr)
                    X_t_aligned = aligner_j.transform(X_t_corr, nan_mask=nan_mask_corr)
                X_t[:, idx_corr_no_j] = X_t_aligned

    # Restaurar NaN
    X_t[nan_mask_t] = np.nan

    scores_j = auto_adapter._model.predict_proba(X_t)

    if config.apply_calibration and auto_adapter._fitted_calibrator is not None:
        scores_j = auto_adapter._fitted_calibrator.predict_proba(scores_j)

    auroc_j_off = float(roc_auc_score(y_target, scores_j))
    contribution = auroc_full - auroc_j_off
    return contribution, auroc_j_off
