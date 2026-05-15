"""
adapt_cli.cross_validate
=========================
K-fold CV honesto para evaluar el optimismo del sweep ADAPT.

En cada fold:
    1. Split estratificado del target (train/test)
    2. AutoAdapter completo SOLO en train (profile + design[sweep] + fit + calibración)
    3. Predict en test (el aligner ve X_test pero NUNCA y_test)
    4. Métricas (AUROC, P, R, F1) sobre los predicts de test

Métricas finales = pool de scores out-of-fold + agregación con CI bootstrap.

Esto reporta lo que un colaborador externo obtendría si aplicase el wrapper
a una nueva muestra del mismo target.
"""

from __future__ import annotations

import logging
import warnings
from copy import copy
from typing import Optional

import numpy as np
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold

from adapt.pipeline.auto_adapter import AutoAdapter
from domain_transfer.data.pairing import CohortPair

logger = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_pair_with_target_subset(pair: CohortPair, idx: np.ndarray) -> CohortPair:
    """Devuelve un CohortPair con target restringido a las filas `idx`."""
    new = object.__new__(CohortPair)
    new._source = pair._source
    new._target = pair._target
    new.schema = pair.schema
    new._X_s = pair._X_s
    new._y_s = pair._y_s
    new._X_t = pair._X_t[idx]
    new._y_t = pair._y_t[idx]
    new._p = pair._p
    return new


def _classification_metrics(y_true: np.ndarray, scores: np.ndarray) -> dict:
    """AUROC + P/R/F1 al umbral de Youden."""
    try:
        auroc = float(roc_auc_score(y_true, scores))
    except ValueError:
        return {"auroc": np.nan, "threshold": np.nan,
                "precision": np.nan, "recall": np.nan, "f1": np.nan}
    fpr, tpr, thr = roc_curve(y_true, scores)
    j = int(np.argmax(tpr - fpr))
    t = float(thr[j])
    y_pred = (scores >= t).astype(int)
    return {
        "auroc":     auroc,
        "threshold": t,
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, zero_division=0)),
        "f1":        float(f1_score(y_true, y_pred, zero_division=0)),
    }


def _bootstrap_auroc_ci(
    y_true: np.ndarray,
    scores: np.ndarray,
    n_boot: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """CI bootstrap estratificado del AUROC."""
    rng = np.random.default_rng(seed)
    pos = np.where(y_true == 1)[0]
    neg = np.where(y_true == 0)[0]
    if len(pos) == 0 or len(neg) == 0:
        return (np.nan, np.nan)
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
        return (np.nan, np.nan)
    return (
        float(np.percentile(aucs, 100 * alpha / 2)),
        float(np.percentile(aucs, 100 * (1 - alpha / 2))),
    )


# ── API pública ───────────────────────────────────────────────────────────────

def cross_validate_adapt(
    pair: CohortPair,
    model,
    schema: list[str],
    drift_type_dict: Optional[dict] = None,
    shap_dict: Optional[dict] = None,
    lbase_dict: Optional[dict] = None,
    n_splits: int = 5,
    pca_k: int = 5,
    max_n_sweep: int = 30,
    apply_qt_override: Optional[bool] = None,
    random_state: int = 42,
    verbose: bool = True,
) -> dict:
    """
    Stratified k-fold CV honesto.

    Parameters
    ----------
    pair : CohortPair
        Par fuente/target (ya filtrado por filter_target si corresponde).
    model, schema, drift_type_dict, shap_dict, lbase_dict
        Argumentos para construir AutoAdapter en cada fold.
    n_splits : int
    pca_k, max_n_sweep
        Hiperparámetros del sweep del Designer.
    apply_qt_override : bool, optional
        Si se pasa, fuerza apply_quantile en el config tras design.
    random_state : int
    verbose : bool

    Returns
    -------
    dict con:
        n_splits, n_target, n_events
        per_fold : lista de dicts (fold_idx, n_test, mask_n, metrics)
        oof_metrics : métricas sobre concatenación de scores out-of-fold
        oof_auroc_ci : CI bootstrap del AUROC OOF
        skipped : n_folds que no se pudieron correr (eventos insuficientes)
    """
    y = pair.y_t
    n = len(y)
    n_pos = int(y.sum())

    if n_pos < n_splits:
        logger.warning(
            "n_eventos=%d < n_splits=%d. Reduciendo n_splits.",
            n_pos, n_splits,
        )
        n_splits = max(2, n_pos)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    oof_scores = np.full(n, np.nan, dtype=float)
    per_fold = []
    skipped = 0

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(np.zeros(n), y)):
        if y[train_idx].sum() < 2 or y[test_idx].sum() < 1:
            logger.warning("Fold %d: eventos insuficientes en train/test, skip.", fold_idx)
            skipped += 1
            continue

        pair_train = _make_pair_with_target_subset(pair, train_idx)
        pair_test = _make_pair_with_target_subset(pair, test_idx)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            aa = AutoAdapter(
                model=model,
                schema=schema,
                drift_type_dict=drift_type_dict,
                shap_dict=shap_dict,
                lbase_dict=lbase_dict,
            )
            aa.profile(pair_train)
            cfg = aa.design(pair_train, pca_k=pca_k, max_n_sweep=max_n_sweep)
            if apply_qt_override is not None:
                cfg.apply_quantile = bool(apply_qt_override)
                if not apply_qt_override:
                    cfg.quantile_features = []
            aa.fit(pair_train)
            scores_test = aa.predict(pair_test)

        oof_scores[test_idx] = scores_test

        m_fold = _classification_metrics(y[test_idx], scores_test)
        per_fold.append({
            "fold": fold_idx,
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "n_events_test": int(y[test_idx].sum()),
            "mask_n": int(cfg.mask_n),
            **m_fold,
        })

        if verbose:
            print(
                f"  fold {fold_idx + 1}/{n_splits}: n_test={len(test_idx)}, "
                f"events={int(y[test_idx].sum())}, mask_N={cfg.mask_n}, "
                f"AUROC={m_fold['auroc']:.3f}, F1={m_fold['f1']:.3f}"
            )

    # OOF métricas (sobre scores concatenados de todos los folds)
    valid = ~np.isnan(oof_scores)
    oof_m = _classification_metrics(y[valid], oof_scores[valid])
    auroc_lo, auroc_hi = _bootstrap_auroc_ci(y[valid], oof_scores[valid])

    # Promedio entre folds (alternativa sensata)
    if per_fold:
        mean_metrics = {
            k: float(np.nanmean([f[k] for f in per_fold]))
            for k in ("auroc", "precision", "recall", "f1")
        }
        std_metrics = {
            f"{k}_std": float(np.nanstd([f[k] for f in per_fold]))
            for k in ("auroc", "precision", "recall", "f1")
        }
    else:
        mean_metrics, std_metrics = {}, {}

    return {
        "n_splits":     n_splits,
        "n_target":     n,
        "n_events":     n_pos,
        "skipped":      skipped,
        "per_fold":     per_fold,
        "oof_metrics":  oof_m,
        "oof_auroc_ci": (auroc_lo, auroc_hi),
        "mean_metrics": mean_metrics,
        "std_metrics":  std_metrics,
        "oof_scores":   oof_scores,
    }
