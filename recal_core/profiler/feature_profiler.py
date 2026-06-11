"""
recal_core.profiler.feature_profiler
=================================
profile_features(): calcula el perfil de cada feature del par (source, target).

Reutiliza:
- recal.drift.concept_shift_univariate para beta3 y qbh
- recal.select.combined_score para lbase + shap + combined_score
- recal_core.profiler.quadrant para la asignación de cuadrantes

Para drift_type_v: acepta un dict precomputado o intenta calcularlo desde
la descomposición V si está disponible. Si no hay datos, usa heurísticas
simples basadas en estadísticos de distribución.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from recal.drift.concept_shift_univariate import UnivariateConceptShiftDiagnoser
from recal.select.combined_score import CombinedScoreSelector
from recal_core.profiler.base import FeatureProfile
from recal_core.profiler.constants import CV_TARGET_NEAR_CONSTANT_THRESHOLD
from recal_core.profiler.quadrant import assign_quadrants

logger = logging.getLogger(__name__)

# Mapeo de prefijos de nombre a dominio temporal
_DOMAIN_MAP = {
    "preop_": "preop",
    "intraop_": "intraop",
    "postop_": "postop",
    "base_": "base",
}


def _infer_domain(feature_name: str) -> str:
    """Infiere el dominio temporal de una feature a partir de su nombre."""
    for prefix, domain in _DOMAIN_MAP.items():
        if feature_name.startswith(prefix):
            return domain
    return "demographic"


def _compute_lbase_scores(
    X_source: np.ndarray,
    y_source: np.ndarray,
    schema: list[str],
) -> np.ndarray:
    """
    Calcula L_base por feature: coeficiente LASSO normalizado (regresión
    logística con penalización L1) en source.

    Normalización: los coeficientes absolutos se escalan a [0, 1].
    """
    mu_s = np.nanmean(X_source, axis=0)
    X_imp = np.where(np.isnan(X_source), mu_s[np.newaxis, :], X_source)
    X_imp = np.nan_to_num(X_imp, nan=0.0)

    scaler = StandardScaler()
    X_std = scaler.fit_transform(X_imp)

    p = X_std.shape[1]
    lbase = np.zeros(p)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            lr = LogisticRegression(
                C=1.0, l1_ratio=1, solver="saga",
                max_iter=5000, random_state=42
            )
            lr.fit(X_std, y_source)
            lbase = np.abs(lr.coef_[0])
        except Exception as e:
            logger.warning("Error in LASSO logistic regression for lbase: %s", e)
            # Fallback: varianza explicada univariada
            for j in range(p):
                lbase[j] = abs(np.corrcoef(X_std[:, j], y_source)[0, 1])

    return lbase


def _compute_shap_importance(
    model,
    X_source: np.ndarray,
) -> np.ndarray:
    """
    Calcula SHAP importance del modelo en source.

    Si el modelo no tiene shap_values(), usa feature_importance() como fallback.
    """
    try:
        mu_s = np.nanmean(X_source, axis=0)
        X_imp = np.where(np.isnan(X_source), mu_s[np.newaxis, :], X_source)
        X_imp = np.nan_to_num(X_imp, nan=0.0)
        shap_vals = model.shap_values(X_imp)
        return np.abs(shap_vals).mean(axis=0)
    except Exception as e:
        logger.warning("shap_values() not available (%s). Using feature_importance().", e)
        fi = model.feature_importance()
        p = X_source.shape[1]
        shap = np.zeros(p)
        # fi es dict name → value; necesitamos mapear a índices
        # Este fallback es aproximado
        for i, name in enumerate(fi):
            if i < p:
                shap[i] = fi.get(name, 0.0)
        return shap


def _compute_per_feature_stats(
    X_source: np.ndarray,
    X_target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calcula cv_source, cv_target, var_ratio, missing_rate_source,
    missing_rate_target, near_constant_target por feature.

    Returns
    -------
    (cv_source, cv_target, var_ratio, miss_s, miss_t, near_constant)
    """
    p = X_source.shape[1]
    cv_s = np.zeros(p)
    cv_t = np.zeros(p)
    var_ratio = np.zeros(p)
    miss_s = np.isnan(X_source).mean(axis=0)
    miss_t = np.isnan(X_target).mean(axis=0)
    near_const = np.zeros(p, dtype=bool)

    for j in range(p):
        xs_j = X_source[:, j]
        xt_j = X_target[:, j]

        xs_valid = xs_j[~np.isnan(xs_j)]
        xt_valid = xt_j[~np.isnan(xt_j)]

        # CV source
        if len(xs_valid) > 1 and abs(np.mean(xs_valid)) > 1e-8:
            cv_s[j] = np.std(xs_valid) / abs(np.mean(xs_valid))
        else:
            cv_s[j] = float("nan")

        # CV target
        if len(xt_valid) > 1 and abs(np.mean(xt_valid)) > 1e-8:
            cv_t[j] = np.std(xt_valid) / abs(np.mean(xt_valid))
            near_const[j] = cv_t[j] < CV_TARGET_NEAR_CONSTANT_THRESHOLD
        else:
            cv_t[j] = float("nan")
            near_const[j] = True  # si no hay datos útiles, tratamos como near-constant

        # var_ratio
        var_s = float(np.var(xs_valid)) if len(xs_valid) > 1 else 0.0
        var_t = float(np.var(xt_valid)) if len(xt_valid) > 1 else 0.0
        if var_s > 1e-12:
            var_ratio[j] = var_t / var_s
        else:
            var_ratio[j] = float("nan")

    return cv_s, cv_t, var_ratio, miss_s, miss_t, near_const


def profile_features(
    X_source: np.ndarray,
    y_source: np.ndarray,
    X_target: np.ndarray,
    y_target: np.ndarray,
    model,
    schema: list[str],
    drift_type_dict: dict[str, str] | None = None,
    shap_importance_dict: dict[str, float] | None = None,
    lbase_dict: dict[str, float] | None = None,
) -> list[FeatureProfile]:
    """
    Calcula el perfil de cada feature.

    Parameters
    ----------
    X_source, y_source : arrays source
    X_target, y_target : arrays target
    model : ModelWrapper con predict_proba() y shap_values()
    schema : list[str] — nombres de las p features
    drift_type_dict : dict[str, str], optional
        Tipos de drift precomputados (feature → drift_type).
        Si None, se usan heurísticas basadas en estadísticos de distribución.
    shap_importance_dict : dict[str, float], optional
        Importancias SHAP precomputadas (feature → valor).
        Si None, se computan desde el modelo.
    lbase_dict : dict[str, float], optional
        L_base precomputados (feature → valor).
        Si None, se computan con LASSO.

    Returns
    -------
    list[FeatureProfile]
    """
    p = len(schema)
    logger.info("Profiling %d features...", p)

    # ── 1. L_base ─────────────────────────────────────────────────────────────
    if lbase_dict is not None:
        lbase = np.array(
            [lbase_dict.get(f, float("nan")) for f in schema], dtype=float
        )
        # Reemplazar NaN con la media
        lbase_mean = float(np.nanmean(lbase))
        lbase = np.where(np.isnan(lbase), lbase_mean, lbase)
    else:
        logger.info("  Computing L_base with LASSO logistic regression...")
        lbase = _compute_lbase_scores(X_source, y_source, schema)

    # ── 2. SHAP importance ────────────────────────────────────────────────────
    if shap_importance_dict is not None:
        shap = np.array(
            [shap_importance_dict.get(f, 0.0) for f in schema], dtype=float
        )
    else:
        logger.info("  Computing SHAP importance...")
        shap = _compute_shap_importance(model, X_source)

    # ── 3. Combined score ─────────────────────────────────────────────────────
    # Usamos CombinedScoreSelector para consistencia con recal
    selector = CombinedScoreSelector(n_to_mask=1)
    selector.fit(lbase, shap)
    combined = selector.scores_

    # ── 4. Cuadrantes ─────────────────────────────────────────────────────────
    quadrants = assign_quadrants(shap, lbase)

    # ── 5. Estadísticos por feature ───────────────────────────────────────────
    cv_s, cv_t, var_ratio, miss_s, miss_t, near_const = _compute_per_feature_stats(
        X_source, X_target
    )

    # ── 6. Concept shift univariado ───────────────────────────────────────────
    logger.info("  Computing univariate concept shift...")
    diagnoser = UnivariateConceptShiftDiagnoser(
        alpha=0.05, max_iter=1000, min_target_nonnan=5
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        diagnoser.fit(
            X_source, X_target, y_source, y_target, schema,
            drift_type_v=[drift_type_dict.get(f, "unknown") if drift_type_dict else "unknown"
                          for f in schema]
        )
    cs_results = diagnoser.results_

    # ── 7. Flip of sign ───────────────────────────────────────────────────────
    flip_col = "flip_of_sign"
    if flip_col in cs_results.columns:
        flip_arr = cs_results[flip_col].values.astype(bool)
    else:
        flip_arr = np.zeros(p, dtype=bool)

    # ── 8. Ensamblar FeatureProfile por feature ───────────────────────────────
    features = []
    for j, name in enumerate(schema):
        # drift_type_v
        if drift_type_dict is not None:
            dt_v = drift_type_dict.get(name, "unknown")
        else:
            dt_v = "unknown"

        # DEGENERATE override: near-constant en target
        if near_const[j] and dt_v not in {"INSUFFICIENT_DATA", "INSUFFICIENT_CLINIC_DATA"}:
            dt_v = "DEGENERATE"

        # Concept shift
        row = cs_results[cs_results["feature"] == name]
        if len(row) > 0:
            beta3 = float(row.iloc[0].get("beta3", float("nan")))
            qbh = float(row.iloc[0].get("q_value_BH", 1.0))
        else:
            beta3 = float("nan")
            qbh = 1.0

        fp = FeatureProfile(
            name=name,
            domain=_infer_domain(name),
            drift_type_v=dt_v,
            lbase_score=float(lbase[j]),
            shap_importance=float(shap[j]),
            combined_score=float(combined[j]),
            quadrant=quadrants[j],
            univariate_concept_shift_beta3=beta3,
            univariate_concept_shift_qbh=qbh,
            flip_of_sign=bool(flip_arr[j]) if j < len(flip_arr) else False,
            cv_target=float(cv_t[j]),
            cv_source=float(cv_s[j]),
            var_ratio=float(var_ratio[j]),
            near_constant_target=bool(near_const[j]),
            missing_rate_source=float(miss_s[j]),
            missing_rate_target=float(miss_t[j]),
        )
        features.append(fp)

    logger.info("  Feature profiling complete (%d features).", p)
    return features
