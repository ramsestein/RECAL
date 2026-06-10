"""
recal_cli.oracle
=================
Oracle: modelo entrenado nativamente en target via k-fold.

Propósito
---------
El oracle establece el techo de AUROC alcanzable con re-entrenamiento completo
en el target. Compararlo con raw y adapted permite descomponer el gap:

    total_gap = auroc_oracle - auroc_raw
    recoverable_gap = auroc_adapted - auroc_raw
    irreducible_gap = auroc_oracle - auroc_adapted

ADVERTENCIA: El oracle es una herramienta de medición de techo, NO un modelo
para deployment clínico. Está entrenado con los datos target (outcomes incluidos),
lo que lo hace inadecuado para uso prospectivo.

Notas de implementación
-----------------------
- Se infiere la familia del modelo source (XGBoost si es XGBoostWrapper).
- BYOM (bring-your-own-model) no reconocido → XGBoost con warning.
- CIs vía DeLong (preferido para AUROC) con fallback a bootstrap.
- DeLong: variance del AUROC sin asumir distribución normal de los scores,
  aprovechando la estructura de rango del estadístico Mann-Whitney.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

logger = logging.getLogger(__name__)

# ── DeLong AUROC CI ──────────────────────────────────────────────────────────


def _delong_auroc_variance(y_true: np.ndarray, scores: np.ndarray) -> float:
    """
    Varianza del AUROC vía DeLong et al. (1988).

    Complejidad O(n log n) usando la representación de Mann-Whitney.

    Returns
    -------
    float
        Varianza del estimador AUROC. Para CI: z * sqrt(var).
    """
    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]
    n1 = len(pos_idx)
    n0 = len(neg_idx)

    if n1 == 0 or n0 == 0:
        return np.nan

    scores_pos = scores[pos_idx]
    scores_neg = scores[neg_idx]

    # Componentes de varianza Vx (pos) y Vy (neg)
    # Vx_i = (1/n0) * sum_j I(scores_neg_j < scores_pos_i) + 0.5*I(scores_neg_j == scores_pos_i)
    # Vy_j = (1/n1) * sum_i I(scores_pos_i > scores_neg_j) + 0.5*I(...)

    # Vectorizado: broadcasting O(n1 * n0)
    diff = scores_pos[:, np.newaxis] - scores_neg[np.newaxis, :]  # (n1, n0)
    indicators = np.where(diff > 0, 1.0, np.where(diff == 0, 0.5, 0.0))

    Vx = indicators.mean(axis=1)  # (n1,)  — media sobre neg para cada pos
    Vy = indicators.mean(axis=0)  # (n0,)  — media sobre pos para cada neg

    _auroc = float(np.mean(Vx))

    # Varianza de DeLong
    var = (
        (np.var(Vx, ddof=1) / n1) if n1 > 1 else 0.0
    ) + (
        (np.var(Vy, ddof=1) / n0) if n0 > 1 else 0.0
    )
    return float(var)


def _delong_auroc_ci(
    y_true: np.ndarray,
    scores: np.ndarray,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """
    AUROC con CI de DeLong al nivel (1-alpha).

    Returns
    -------
    auroc, ci_lo, ci_hi
    """
    from scipy import stats as scipy_stats

    auroc = float(roc_auc_score(y_true, scores))
    var = _delong_auroc_variance(y_true, scores)

    if np.isnan(var) or var <= 0:
        return auroc, np.nan, np.nan

    se = float(np.sqrt(var))
    z = float(scipy_stats.norm.ppf(1 - alpha / 2))
    lo = max(0.0, auroc - z * se)
    hi = min(1.0, auroc + z * se)
    return auroc, lo, hi


def _bootstrap_auroc_ci(
    y_true: np.ndarray,
    scores: np.ndarray,
    n_boot: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Fallback bootstrap CI del AUROC."""
    auroc = float(roc_auc_score(y_true, scores))
    rng = np.random.default_rng(seed)
    pos = np.where(y_true == 1)[0]
    neg = np.where(y_true == 0)[0]
    if len(pos) == 0 or len(neg) == 0:
        return auroc, np.nan, np.nan
    aucs = []
    for _ in range(n_boot):
        ip = rng.choice(pos, size=len(pos), replace=True)
        in_ = rng.choice(neg, size=len(neg), replace=True)
        idx = np.concatenate([ip, in_])
        try:
            aucs.append(roc_auc_score(y_true[idx], scores[idx]))
        except ValueError:
            pass
    if not aucs:
        return auroc, np.nan, np.nan
    lo = float(np.percentile(aucs, 100 * alpha / 2))
    hi = float(np.percentile(aucs, 100 * (1 - alpha / 2)))
    return auroc, lo, hi


def _auroc_with_ci(
    y_true: np.ndarray,
    scores: np.ndarray,
    alpha: float = 0.05,
    n_boot: int = 1000,
) -> tuple[float, float, float]:
    """
    AUROC con CI — intenta DeLong, cae a bootstrap si scipy no disponible.
    """
    try:
        return _delong_auroc_ci(y_true, scores, alpha=alpha)
    except ImportError:
        logger.warning("scipy no disponible; usando bootstrap para CI de AUROC.")
        return _bootstrap_auroc_ci(y_true, scores, n_boot=n_boot, alpha=alpha)
    except Exception as e:
        logger.warning("DeLong failed (%s); falling back to bootstrap.", e)
        return _bootstrap_auroc_ci(y_true, scores, n_boot=n_boot, alpha=alpha)


# ── Inferencia de familia de modelo ─────────────────────────────────────────

def _infer_model_family(model) -> str:
    """Inferir la familia del modelo source."""
    cls_name = type(model).__name__
    module = type(model).__module__ or ""

    if "XGBoost" in cls_name or "xgboost" in module.lower():
        return "xgboost"
    if "RandomForest" in cls_name:
        return "random_forest"
    if "Logistic" in cls_name or "LogisticRegression" in cls_name:
        return "logistic"
    if "GradientBoosting" in cls_name:
        return "gradient_boosting"

    # Intentar acceder al estimador interno (wrapper BYOM)
    for attr in ("_model", "estimator", "_estimator", "base_model"):
        inner = getattr(model, attr, None)
        if inner is not None:
            inner_name = type(inner).__name__
            if "XGBoost" in inner_name or "xgb" in inner_name.lower():
                return "xgboost"

    logger.warning(
        "Could not infer model family (%s). "
        "Defaulting to XGBoost for oracle.",
        cls_name,
    )
    return "xgboost"


def _build_oracle_model(model_family: str):
    """Construye un estimador fresco para el oracle."""
    if model_family == "xgboost":
        try:
            from xgboost import XGBClassifier
            return XGBClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                use_label_encoder=False,
                eval_metric="logloss",
                verbosity=0,
                random_state=42,
            )
        except ImportError:
            logger.warning("xgboost not available; oracle uses GradientBoostingClassifier.")
            from sklearn.ensemble import GradientBoostingClassifier
            return GradientBoostingClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42
            )
    elif model_family == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=200, max_depth=None, random_state=42)
    elif model_family == "logistic":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(max_iter=1000, random_state=42)
    elif model_family == "gradient_boosting":
        from sklearn.ensemble import GradientBoostingClassifier
        return GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42
        )
    else:
        logger.warning("Unknown model family %r; oracle uses XGBoost.", model_family)
        from xgboost import XGBClassifier
        return XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            use_label_encoder=False, eval_metric="logloss",
            verbosity=0, random_state=42,
        )


# ── API pública ──────────────────────────────────────────────────────────────

def fit_target_oracle(
    X_target: np.ndarray,
    y_target: np.ndarray,
    source_model=None,
    model_family: str | None = None,
    cv: int = 5,
    alpha: float = 0.05,
    random_state: int = 42,
) -> dict:
    """
    Entrena un oracle nativamente en el target via k-fold OOF y devuelve AUROC con CI.

    El oracle establece el techo de rendimiento alcanzable entrenando desde cero
    en el target. Es una herramienta de medición del gap irreducible, NO un
    modelo para deployment.

    Parameters
    ----------
    X_target : np.ndarray (n, p)
        Features del target (pueden tener NaN — se imputan con media).
    y_target : np.ndarray (n,)
        Labels binarios del target.
    source_model : object, optional
        Modelo source; se usa para inferir la familia si model_family es None.
    model_family : str, optional
        Familia del modelo: 'xgboost', 'random_forest', 'logistic',
        'gradient_boosting'. Si None, se infiere de source_model.
    cv : int
        Número de folds del k-fold estratificado.
    alpha : float
        Nivel de significancia para los CIs (default 0.05 → 95% CI).
    random_state : int

    Returns
    -------
    dict con:
        auroc : float — AUROC OOF
        ci_lo, ci_hi : float — CI DeLong (o bootstrap fallback)
        model_family : str
        n_folds : int
        n_target : int
        n_events : int
        oof_scores : np.ndarray (n,) — scores OOF
        warning : str or None
    """
    n = len(y_target)
    n_events = int(y_target.sum())
    warning = None

    # Inferir familia
    if model_family is None:
        if source_model is not None:
            model_family = _infer_model_family(source_model)
        else:
            model_family = "xgboost"
            warning = "model_family not specified and source_model not provided; defaulting to XGBoost."
            logger.warning(warning)

    # Imputación simple (media de cada columna) para modelos que no aceptan NaN
    X = X_target.copy().astype(float)
    col_means = np.nanmean(X, axis=0)
    col_means = np.where(np.isnan(col_means), 0.0, col_means)
    nan_mask = np.isnan(X)
    X[nan_mask] = np.take(col_means, np.where(nan_mask)[1])

    # Reducir cv si no hay suficientes eventos
    effective_cv = cv
    if n_events < cv:
        effective_cv = max(2, n_events)
        msg = (
            f"n_events_target={n_events} < cv={cv}; reducing to cv={effective_cv}."
        )
        logger.warning(msg)
        warning = (warning + " " if warning else "") + msg

    skf = StratifiedKFold(n_splits=effective_cv, shuffle=True, random_state=random_state)
    oof_scores = np.full(n, np.nan, dtype=float)

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y_target)):
        if y_target[train_idx].sum() < 2 or y_target[test_idx].sum() < 1:
            logger.warning("Oracle fold %d: insufficient events, skip.", fold_idx)
            continue

        estimator = _build_oracle_model(model_family)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                estimator.fit(X[train_idx], y_target[train_idx])
                preds = estimator.predict_proba(X[test_idx])[:, 1]
                oof_scores[test_idx] = preds
            except Exception as e:
                logger.warning("Oracle fold %d failed: %s", fold_idx, e)

    # AUROC sobre los OOF completos
    valid = ~np.isnan(oof_scores)
    if valid.sum() == 0 or y_target[valid].sum() == 0:
        logger.error("Oracle: could not compute OOF scores.")
        return {
            "auroc": np.nan, "ci_lo": np.nan, "ci_hi": np.nan,
            "model_family": model_family, "n_folds": effective_cv,
            "n_target": n, "n_events": n_events, "oof_scores": oof_scores,
            "warning": "Oracle failed: invalid OOF scores.",
        }

    auroc, ci_lo, ci_hi = _auroc_with_ci(y_target[valid], oof_scores[valid], alpha=alpha)

    return {
        "auroc": auroc,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "model_family": model_family,
        "n_folds": effective_cv,
        "n_target": n,
        "n_events": n_events,
        "oof_scores": oof_scores,
        "warning": warning,
    }
