"""
recal_core.designer.rules
=====================
Reglas determinísticas para la selección de componentes RECAL.

Cada función es una regla pura DriftProfile → decisión con docstring
explicando la justificación empírica. Todas las funciones retornan
(decisión, razón_legible).

IMPORTANTE: No añadir reglas sin justificación en el docstring.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np

from recal_core.profiler.base import DriftProfile
from recal_core.profiler.constants import (
    CALIBRATION_HETEROGENEITY_PVALUE,
    CALIBRATION_SLOPE_RECAL_THRESHOLD,
    CV_TARGET_QT_MINIMUM,
    N_EVENTS_ISOTONIC,
    N_EVENTS_MINIMUM_CALIBRATION,
    N_EVENTS_MINIMUM_MASK,
    N_EVENTS_MINIMUM_WOE,
    N_EVENTS_SOURCE_MINIMUM_WOE,
    PCA_CORAL_K_RANGE_MIN,
    SHAP_WOE_MINIMUM,
    VAR_RATIO_QT_LOWER,
    VAR_RATIO_QT_UPPER,
)

logger = logging.getLogger(__name__)


# ── Regla 1: Máscara ──────────────────────────────────────────────────────────

def should_mask_features(profile: DriftProfile) -> tuple[bool, str]:
    """
    Aplicar máscara si hay suficientes eventos target para validarla.

    Justificación empírica (exp_extend): con n_events_target < 20, el sweep de N
    se vuelve inestable y la máscara puede eliminar señal sin poder detectarlo.
    Con ≥20 eventos, el elbow del combined_score en source es fiable.
    """
    if profile.n_target_events < N_EVENTS_MINIMUM_MASK:
        return False, (
            f"n_target_events={profile.n_target_events} < {N_EVENTS_MINIMUM_MASK}: "
            f"insufficient to validate mask"
        )
    return True, f"n_target_events={profile.n_target_events} >= {N_EVENTS_MINIMUM_MASK}: mask activated"


def select_mask_n(
    profile: DriftProfile,
    pair=None,
    model=None,
    pca_k: int = 5,
    max_n_sweep: int = 30,
) -> tuple[int, str, list[dict]]:
    """
    Selección de N por mini-sweep con PCA-CORAL en target (si pair y model disponibles).

    Cuando pair y model se proporcionan, barre N=0..min(max_n_sweep, p//4)
    aplicando mask + PCA-CORAL k=pca_k y elige el N que maximiza AUROC en
    target. Esto es legítimo en domain adaptation porque el target está
    disponible en el momento del fit (Clínic tiene outcomes).

    Fallback sin pair/model: elbow del combined_score en source (heurístico,
    independiente de y_target pero menos preciso).

    Cap: máximo 25% de las features para evitar eliminar demasiada señal.

    Returns
    -------
    n_opt : int
    reason : str
    sweep_history : list[dict]  — [{n, auroc}, ...] si se hizo sweep; [] en fallback
    """
    sorted_features = sorted(profile.features, key=lambda f: f.combined_score)
    p = len(sorted_features)
    max_n = max(1, min(max_n_sweep, p // 4))

    if pair is not None and model is not None:
        n_opt, reason, sweep_history = _sweep_mask_n(pair, model, sorted_features, max_n, pca_k)
        return n_opt, reason, sweep_history

    # Fallback: elbow
    scores = np.array([f.combined_score for f in sorted_features])
    n_elbow = _compute_elbow(scores)
    n_ponzonous = len(profile.ponzonous_features())
    n_elbow = max(n_elbow, min(1, n_ponzonous))
    n_elbow = min(n_elbow, max_n)
    return n_elbow, f"elbow of combined score in source: N={n_elbow} (D_ponzonous={n_ponzonous})", []


def _sweep_mask_n(
    pair, model, sorted_features, max_n: int, pca_k: int
) -> tuple[int, str, list[dict]]:
    """
    Barre N=0..max_n y devuelve el N que maximiza AUROC target con PCA-CORAL.

    Cada paso: mask las bottom-N features → PCA-CORAL k=pca_k → predict → AUROC.

    Returns
    -------
    best_n : int
    reason : str
    sweep_history : list[dict]  — [{n, auroc}, ...] para todos los N probados
    """
    import warnings

    from sklearn.metrics import roc_auc_score

    from recal.align.pca_coral import PCACoralAligner

    feat_names = [f.name for f in sorted_features]
    best_n, best_auroc = 0, -1.0
    sweep_history: list[dict] = []

    for n in range(0, max_n + 1):
        try:
            masked = pair if n == 0 else pair.mask_features(feat_names[:n])
            idx_corr = masked.idx_corr
            X_s = np.nan_to_num(masked.X_s_imp[:, idx_corr], nan=0.0)
            X_t = np.nan_to_num(masked.X_t_imp[:, idx_corr], nan=0.0)

            k = min(pca_k, X_s.shape[1], X_s.shape[0] - 1, X_t.shape[0] - 1)
            if k < 1:
                continue

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                aligner = PCACoralAligner(k=k, reg_pca=1e-6, random_state=42)
                aligner.fit(X_s, X_t)
                X_t_aligned = aligner.transform(X_t, nan_mask=np.zeros_like(X_t, dtype=bool))

            # Reconstruir X_t completo para predict:
            # - base = X_t original con NaN (XGBoost fue entrenado con NaN nativos)
            # - solo sobreescribir las columnas alineadas (idx_corr)
            X_t_full = masked.X_t.copy()
            X_t_full[:, idx_corr] = X_t_aligned

            scores = model.predict_proba(X_t_full)
            auroc = float(roc_auc_score(masked.y_t, scores))
            sweep_history.append({"n": n, "auroc": auroc})

            if auroc > best_auroc:
                best_auroc = auroc
                best_n = n
        except Exception:
            continue

    reason = (
        f"mini-sweep PCA-CORAL k={pca_k} on target: N={best_n} "
        f"(AUROC={best_auroc:.4f}, sweep 0..{max_n})"
    )
    logger.info("  [mask_n_sweep] %s", reason)
    return best_n, reason, sweep_history


def _compute_elbow(scores: np.ndarray) -> int:
    """
    Calcula el punto de elbow de una curva ordenada como el índice de
    máxima segunda derivada (curvatura).

    Si no hay curva suficiente (< 3 puntos), devuelve 1.
    """
    n = len(scores)
    if n < 3:
        return 1
    d1 = np.diff(scores)
    d2 = np.diff(d1)
    if len(d2) == 0:
        return 1
    elbow_idx = int(np.argmax(np.abs(d2))) + 1
    return max(1, elbow_idx)


# ── Regla 2: QuantileTransform ────────────────────────────────────────────────

def should_apply_quantile_transform_per_feature(
    profile: DriftProfile,
) -> tuple[dict[str, bool], str]:
    """
    Aplicar QT a feature j solo si:
    - drift_type_v(j) en {NONLINEAR_DRIFT, PARTIAL_RECOVERY}, Y
    - cv_target(j) >= CV_TARGET_QT_MINIMUM (no near-constant), Y
    - var_ratio(j) < VAR_RATIO_QT_LOWER o > VAR_RATIO_QT_UPPER (varianza desplazada).

    Justificación: exp_extend confirmó que QT no aporta en SNUH→Clínic porque
    las features NONLINEAR_DRIFT son o bien near-constant en target
    (SpO2_median CV=1.1%, Na_median CV=1.5%) o cargan PC1-5 con peso similar
    a todas las demás features, lo que hace que CORAL absorba la variación
    introducida por QT. Aplicar QT en ambos casos es contraproducente.
    """
    decisions: dict[str, bool] = {}
    for f in profile.features:
        apply = (
            f.drift_type_v in {"NONLINEAR_DRIFT", "PARTIAL_RECOVERY"}
            and not np.isnan(f.cv_target)
            and f.cv_target >= CV_TARGET_QT_MINIMUM
            and not np.isnan(f.var_ratio)
            and (f.var_ratio < VAR_RATIO_QT_LOWER or f.var_ratio > VAR_RATIO_QT_UPPER)
        )
        decisions[f.name] = apply

    n_apply = sum(decisions.values())
    return decisions, (
        f"QT on {n_apply} features after filtering near-constants (CV<{CV_TARGET_QT_MINIMUM}) "
        f"and moderate ratios ({VAR_RATIO_QT_LOWER}–{VAR_RATIO_QT_UPPER})"
    )


# ── Regla 3: WOE ─────────────────────────────────────────────────────────────

def should_apply_woe_per_feature(
    profile: DriftProfile,
) -> tuple[dict[str, bool], str]:
    """
    Aplicar WOE a feature j solo si:
    - n_source_events >= N_EVENTS_SOURCE_MINIMUM_WOE, Y
    - n_target_events >= N_EVENTS_MINIMUM_WOE, Y
    - drift_type_v(j) en {STABLE, LINEAR_RECOVERABLE}, Y
    - shap_importance(j) >= SHAP_WOE_MINIMUM (filtra noise features).

    Justificación: exp_extend mostró que WOE empeora con n_events_target=29
    incluso aplicado a features STABLE+LINEAR_RECOVERABLE. El umbral 30 es
    conservador y previene aplicación en régimen donde la evidencia indica daño.
    El mínimo en source garantiza estimaciones robustas del mapa WOE.
    """
    if profile.n_source_events < N_EVENTS_SOURCE_MINIMUM_WOE:
        return (
            {f.name: False for f in profile.features},
            f"n_source_events={profile.n_source_events} < {N_EVENTS_SOURCE_MINIMUM_WOE}: skip WOE"
        )
    if profile.n_target_events < N_EVENTS_MINIMUM_WOE:
        return (
            {f.name: False for f in profile.features},
            f"n_target_events={profile.n_target_events} < {N_EVENTS_MINIMUM_WOE}: skip WOE"
        )

    decisions: dict[str, bool] = {}
    for f in profile.features:
        apply = (
            f.drift_type_v in {"STABLE", "LINEAR_RECOVERABLE"}
            and f.shap_importance >= SHAP_WOE_MINIMUM
        )
        decisions[f.name] = apply

    n_apply = sum(decisions.values())
    return decisions, (
        f"WOE on {n_apply} STABLE/LINEAR_RECOVERABLE features with SHAP >= {SHAP_WOE_MINIMUM}"
    )


# ── Regla 4: PCA-CORAL ────────────────────────────────────────────────────────

def should_apply_pca_coral(profile: DriftProfile) -> tuple[bool, str]:
    """
    Aplicar PCA-CORAL casi siempre (default robusto).

    PCA-CORAL es el aligner default porque:
    1. Es seguro en cualquier régimen p/n (PCA resuelve el problema de
       rango de CORAL cuando p ≈ n).
    2. Validado en SNUH→Clínic con k=5 → AUROC 0.629 → 0.706.

    Excepción documentada: si p/n_target < 0.2, CORAL global directo podría
    funcionar igualmente bien. Por ahora siempre aplicamos PCA-CORAL.

    Siempre retorna True — la excepción es un TO-DO para v0.2.
    """
    return True, "PCA-CORAL is the default robust aligner under any p/n ratio"


def select_pca_coral_k(profile: DriftProfile) -> tuple[int, str]:
    """
    k seleccionado por varianza explicada acumulada en source, con techo
    en sqrt(n_target_obs) para garantizar buen condicionamiento.

    Justificación: SNUH→Clínic con n_target=105 → max_k=10.
    El óptimo observado en exp_extend es k=5.
    La heurística sqrt(n_target) garantiza que la covarianza latente k×k
    no sea rank-deficient en target (Kritchman & Nadler, 2008).

    Heurística de selección por varianza:
    - Tomar el k donde la varianza explicada acumulada supera el 80%
      (umbral estándar en compresión PCA).
    - Si la curva no llega al 80% dentro del rango, usar el codo de la
      curva de varianza incremental.
    - Recortar a max_k = max(2, floor(sqrt(n_target))).
    """
    max_k = max(PCA_CORAL_K_RANGE_MIN, int(np.sqrt(profile.n_target_obs)))

    # Usar la curva de varianza explicada del perfil global
    var_explained = profile.pca_variance_explained
    if var_explained:
        # Buscar primer k donde varianza acumulada >= 0.80
        k_cv = len(var_explained)  # default: usar todos
        for i, v in enumerate(var_explained):
            if v >= 0.80:
                k_cv = i + 1
                break
        k_cv = max(PCA_CORAL_K_RANGE_MIN, k_cv)
    else:
        k_cv = 5  # fallback por defecto (validado en SNUH→Clínic)

    k_final = min(k_cv, max_k)
    k_final = max(PCA_CORAL_K_RANGE_MIN, k_final)

    return k_final, (
        f"k_cv={k_cv} (var≥80% in source), clipped to sqrt(n_target)={max_k} → k={k_final}"
    )


def select_alignment_strategy(
    profile: DriftProfile,
    pair=None,
    model=None,
    k_heuristic: int = 5,
) -> tuple[int, bool, str]:
    """
    Selecciona la mejor estrategia de alineación comparando PCA-CORAL vs
    CORAL puro mediante mini-sweep en target.

    Cuando pair y model están disponibles, barre:
      - PCA-CORAL con k = k_heuristic - 1, k_heuristic, k_heuristic + 1
      - CORAL puro (full rank, k=-1)
    Eligiendo la que maximiza AUROC en target.

    Sin pair/model: usa PCA-CORAL con k_heuristic (seguro para cualquier p/n).

    Returns
    -------
    k : int
        k para PCA-CORAL, o -1 si CORAL puro es mejor.
    use_coral_pure : bool
        True si CORAL puro fue seleccionado.
    reason : str
    """
    if pair is None or model is None:
        return k_heuristic, False, f"No target sweep available: PCA-CORAL k={k_heuristic}"

    from sklearn.metrics import roc_auc_score
    from recal.align.pca_coral import PCACoralAligner
    from recal.align.coral import CoralAligner

    idx_corr = pair.idx_corr
    X_s = np.nan_to_num(pair.X_s_imp[:, idx_corr], nan=0.0)
    X_t = np.nan_to_num(pair.X_t_imp[:, idx_corr], nan=0.0)

    p_eff = X_s.shape[1]
    n_t = X_t.shape[0]

    # Candidatos: PCA-CORAL con k alrededor del heurístico + CORAL puro
    candidates = []
    for delta in [-1, 0, 1]:
        k_candidate = max(PCA_CORAL_K_RANGE_MIN, min(k_heuristic + delta, p_eff - 1, n_t - 1))
        if k_candidate >= PCA_CORAL_K_RANGE_MIN and k_candidate < p_eff and k_candidate < n_t:
            candidates.append(("pca_coral", k_candidate))
    candidates.append(("coral_pure", -1))

    # Eliminar duplicados
    seen = set()
    unique_candidates = []
    for kind, k_val in candidates:
        key = (kind, k_val)
        if key not in seen:
            seen.add(key)
            unique_candidates.append((kind, k_val))

    results = []
    best_k = k_heuristic
    best_coral_pure = False
    best_auroc = -1.0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        for kind, k_val in unique_candidates:
            try:
                if kind == "coral_pure":
                    aligner = CoralAligner(reg=1e-4, shrinkage="auto")
                else:
                    aligner = PCACoralAligner(k=k_val, reg_pca=1e-6, random_state=42)

                aligner.fit(X_s, X_t)
                X_t_aligned = aligner.transform(X_t, nan_mask=np.zeros_like(X_t, dtype=bool))

                X_t_full = pair.X_t.copy()
                X_t_full[:, idx_corr] = X_t_aligned

                scores = model.predict_proba(X_t_full)
                auroc = float(roc_auc_score(pair.y_t, scores))

                results.append((kind, k_val, auroc))

                if auroc > best_auroc:
                    best_auroc = auroc
                    best_k = k_val
                    best_coral_pure = (kind == "coral_pure")
            except Exception:
                continue

    if best_coral_pure:
        reason = (
            f"Alignment sweep on target: CORAL pure selected (AUROC={best_auroc:.4f}, "
            f"competing k={[f'{k}({a:.3f})' for _, k, a in results if k != -1]})"
        )
    else:
        reason = (
            f"Alignment sweep on target: PCA-CORAL k={best_k} selected (AUROC={best_auroc:.4f}, "
            f"coral_pure AUROC={next((a for kd, k, a in results if kd == 'coral_pure'), float('nan')):.3f})"
        )

    return best_k, best_coral_pure, reason


# ── Regla 5: Calibración ─────────────────────────────────────────────────────

def should_recalibrate(profile: DriftProfile) -> tuple[bool, str]:
    """
    Recalibrar si:
    - n_events_target >= N_EVENTS_MINIMUM_CALIBRATION, Y
    - |baseline_calibration_slope - 1| > CALIBRATION_SLOPE_RECAL_THRESHOLD.

    Justificación: con n_events_target < 20 los coeficientes Platt LOO son
    inestables (CIs explosivos). Con slope ya cerca de 1.0, recalibrar
    introduce más ruido que señal.
    En SNUH→Clínic slope=9.06, claramente roto → recalibración obligatoria.
    """
    if profile.n_target_events < N_EVENTS_MINIMUM_CALIBRATION:
        return False, (
            f"n_target_events={profile.n_target_events} < {N_EVENTS_MINIMUM_CALIBRATION}: "
            f"LOO calibration unstable, skip"
        )
    slope = profile.baseline_calibration_slope
    if np.isnan(slope):
        return False, "slope=NaN: cannot evaluate, skip calibration"

    deviation = abs(slope - 1.0)
    if deviation <= CALIBRATION_SLOPE_RECAL_THRESHOLD:
        return False, (
            f"slope={slope:.2f}: |slope-1|={deviation:.2f} <= {CALIBRATION_SLOPE_RECAL_THRESHOLD}: "
            f"calibration acceptable, skip"
        )
    return True, (
        f"slope={slope:.2f}: |slope-1|={deviation:.2f} > {CALIBRATION_SLOPE_RECAL_THRESHOLD}: "
        f"recalibration required"
    )


def select_calibration_method(profile: DriftProfile) -> tuple[str, str]:
    """
    Platt LOO global como default.
    Isotónica solo si n_events_target >= N_EVENTS_ISOTONIC.
    Estratificada solo si test de heterogeneidad p < CALIBRATION_HETEROGENEITY_PVALUE.

    Justificación del default Platt LOO: es el método más robusto con n pequeño
    (2 parámetros, LOO evita overfitting).
    Isotónica: más flexible pero requiere n >> p donde p = número de quiebres.
    Estratificada: útil cuando la heterogeneidad por nivel de riesgo es
    significativa (p < 0.05 en el test de Hosmer-Lemeshow estratificado).
    """
    if profile.n_target_events >= N_EVENTS_ISOTONIC:
        return "isotonic_loo", (
            f"n_events_target={profile.n_target_events} >= {N_EVENTS_ISOTONIC}: "
            f"isotonic regression viable"
        )

    # Test de heterogeneidad simplificado: comparar slopes en terciles del score predicho
    p_het = _test_calibration_heterogeneity(profile)
    if p_het < CALIBRATION_HETEROGENEITY_PVALUE:
        return "platt_stratified", (
            f"slope heterogeneity p={p_het:.3f} < {CALIBRATION_HETEROGENEITY_PVALUE}: "
            f"stratified calibration by score"
        )

    return "platt_loo", "Platt LOO global: robust default with small n"


def _test_calibration_heterogeneity(profile: DriftProfile) -> float:
    """
    Test de heterogeneidad de slopes de calibración.

    Con los datos disponibles en el DriftProfile (solo estadísticos agregados),
    no podemos ejecutar el test exacto. Retornamos 1.0 (sin heterogeneidad
    significativa) como decisión conservadora — preferimos Platt LOO global
    salvo que haya evidencia clara de heterogeneidad.

    Nota: en el AutoAdapter.fit(), si se detecta heterogeneidad post-hoc
    (analizando los residuos de calibración por tercil), se puede re-ejecutar
    el selector.
    """
    # OPEN QUESTION OQ-1: implementar test de heterogeneidad real en recal_core.
    # Por ahora: conservador → siempre retornamos 1.0 → Platt LOO global.
    return 1.0
