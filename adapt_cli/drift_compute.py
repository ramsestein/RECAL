"""
adapt_cli.drift_compute
========================
Cálculo on-the-fly de la **descomposición de drift por feature** descrita en
el Experimento V del REPORT.md. Para cada feature `v`:

  1. Entrena XGBoost y LASSO que predicen `v` desde el resto (solo en source).
  2. Evalúa esos predictores sobre target con tres versiones de X:
        - raw (sin alinear)
        - CORAL global
        - PCA-CORAL k=5
  3. Calcula las pérdidas (drops) y las recuperaciones (% que CORAL recupera).
  4. Clasifica la feature en una de:
        STABLE | NONLINEAR_DRIFT | LINEAR_RECOVERABLE | PARTIAL_RECOVERY |
        CONCEPT_RELATIONAL | INSUFFICIENT_CLINIC_DATA | LOW_VARIANCE_TARGET

Outputs un dict con las claves que `AutoAdapter` espera:

    {
        "drift_type_dict": {feat: str, ...},
        "shap_dict":       {feat: float, ...},
        "lbase_dict":      {feat: float, ...},
        "df":              pd.DataFrame  # tabla completa por feature
    }

Cacheable a CSV.

Adaptado de ``legacy/scripts/v_drift_decomposition.py`` (Exp V) eliminando
todas las dependencias específicas de SNUH/Clínic, figuras y stats globales.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LassoCV, LogisticRegression
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

# ── Hiperparámetros (idénticos a Exp V) ──────────────────────────────────────
SEED = 42
XGB_N_EST = 200
XGB_MAX_DEPTH = 5
XGB_LR = 0.05
MIN_DROP = 0.05
CV_THRESHOLD = 0.02
RECOVERABLE_THRESHOLD = 70.0
PARTIAL_THRESHOLD = 30.0
PCA_K = 5


# ── Utilidades numéricas ─────────────────────────────────────────────────────
def _safe_sqrtm(M: np.ndarray) -> np.ndarray:
    S = (M + M.T) / 2
    w, V = np.linalg.eigh(S)
    w = np.maximum(w, 1e-10)
    return V @ np.diag(np.sqrt(w)) @ V.T


def _safe_invsqrtm(M: np.ndarray) -> np.ndarray:
    S = (M + M.T) / 2
    w, V = np.linalg.eigh(S)
    w = np.maximum(w, 1e-10)
    return V @ np.diag(1.0 / np.sqrt(w)) @ V.T


def _impute_with_mu(X: np.ndarray, mu: np.ndarray) -> np.ndarray:
    return np.where(np.isnan(X), mu[np.newaxis, :], X)


def _detect_vtype(x: np.ndarray) -> str:
    v = x[~np.isnan(x)]
    if len(v) == 0:
        return "all_nan"
    u = np.unique(v)
    if len(u) == 2:
        return "binary"
    if len(u) <= 10 and all(val % 1 == 0 for val in u):
        return "ordinal"
    return "continuous"


def _clip_rec(val):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return np.nan
    return float(np.clip(val, -200.0, 200.0))


def _compute_rec(after, before, drop):
    if drop <= 0 or np.isnan(after) or np.isnan(before):
        return np.nan
    return _clip_rec((after - before) / drop * 100)


# ── SHAP importance (gain normalizado del booster XGBoost) ───────────────────
def _shap_importance_from_model(model, schema: list[str]) -> dict[str, float]:
    """
    Devuelve dict {feature: gain_normalizado}. Soporta XGBoostWrapper, booster
    XGBoost directo, o cualquier modelo con .feature_importance() o
    .shap_values(X). Si nada funciona, devuelve ceros (no rompe el cálculo).
    """
    # 1) Intentar el método estándar de XGBoostWrapper / wrapper minimal
    try:
        fi = model.feature_importance()
        if isinstance(fi, dict) and any(v > 0 for v in fi.values()):
            total = sum(fi.values()) or 1.0
            out = {f: 0.0 for f in schema}
            for k, v in fi.items():
                if k in out:
                    out[k] = float(v) / total
                elif isinstance(k, str) and k.startswith("f") and k[1:].isdigit():
                    idx = int(k[1:])
                    if idx < len(schema):
                        out[schema[idx]] = float(v) / total
            return out
    except Exception:
        pass

    # 2) Booster directo (atributo .booster_ o .booster)
    booster = getattr(model, "booster_", None) or getattr(model, "booster", None)
    if booster is not None:
        try:
            imp = booster.get_score(importance_type="gain")
            total = sum(imp.values()) or 1.0
            out = {f: 0.0 for f in schema}
            for k, v in imp.items():
                if k in out:
                    out[k] = float(v) / total
                elif k.startswith("f") and k[1:].isdigit():
                    idx = int(k[1:])
                    if idx < len(schema):
                        out[schema[idx]] = float(v) / total
            return out
        except Exception:
            pass

    logger.warning("No se pudo extraer SHAP/gain importance; usando ceros.")
    return {f: 0.0 for f in schema}


# ── Núcleo: cálculo por feature ──────────────────────────────────────────────
def compute_drift_decomposition(
    X_s: np.ndarray,
    X_t: np.ndarray,
    schema: list[str],
    model,
    cache_path: Optional[str | Path] = None,
    verbose: bool = True,
) -> dict:
    """
    Calcula la descomposición de drift por feature.

    Parameters
    ----------
    X_s, X_t : np.ndarray
        Matrices source/target alineadas al schema (NaN permitidos).
    schema : list[str]
        Nombres de features (len == X_s.shape[1] == X_t.shape[1]).
    model : objeto
        El modelo a desplegar; se usa solo para extraer SHAP/gain importance.
    cache_path : str | Path, optional
        Si se proporciona y el archivo existe, lo carga y devuelve sin recomputar.
        Si no existe, calcula y lo escribe.
    verbose : bool
        Imprime progreso por cada 25 features.

    Returns
    -------
    dict con claves: drift_type_dict, shap_dict, lbase_dict, df
    """
    import xgboost as xgb

    # ── Cache ───────────────────────────────────────────────────────────────
    if cache_path is not None:
        cache_path = Path(cache_path)
        if cache_path.exists():
            logger.info("Drift decomposition: cargando cache %s", cache_path)
            df = pd.read_csv(cache_path)
            return _df_to_dicts(df)

    n_feat = len(schema)
    assert X_s.shape[1] == n_feat == X_t.shape[1], "Shape mismatch schema/X"

    # Sanitizar Inf → NaN
    X_s = np.where(np.isinf(X_s), np.nan, X_s.astype(float))
    X_t = np.where(np.isinf(X_t), np.nan, X_t.astype(float))

    # ── Índices: features 100% NaN en target son "structurally absent" ──────
    nan_t_rate = np.isnan(X_t).mean(axis=0)
    idx_missing = [j for j in range(n_feat) if nan_t_rate[j] == 1.0]
    idx_corr = [j for j in range(n_feat) if j not in set(idx_missing)]
    n_corr = len(idx_corr)
    if verbose:
        logger.info("Drift decomposition: %d features, %d evaluables (target tiene %d 100%%-NaN)",
                    n_feat, n_corr, len(idx_missing))

    # ── SHAP importance del modelo principal ────────────────────────────────
    shap_imp = _shap_importance_from_model(model, schema)

    # ── Matrices imputadas (NaN → media source) ─────────────────────────────
    mu_s = np.nanmean(X_s, axis=0)
    X_s_imp = np.nan_to_num(_impute_with_mu(X_s, mu_s), nan=0.0, posinf=0.0, neginf=0.0)
    X_t_imp = np.nan_to_num(_impute_with_mu(X_t, mu_s), nan=0.0, posinf=0.0, neginf=0.0)

    # ── CORAL y PCA-CORAL sobre el subconjunto idx_corr ─────────────────────
    Xs = X_s_imp[:, idx_corr]
    Xt = X_t_imp[:, idx_corr]
    mu_xs = Xs.mean(axis=0)
    mu_xt = Xt.mean(axis=0)
    Sig_s = np.cov(Xs, rowvar=False) + 1e-4 * np.eye(n_corr)
    Sig_t = np.cov(Xt, rowvar=False) + 1e-4 * np.eye(n_corr)
    A_full = _safe_sqrtm(Sig_s) @ _safe_invsqrtm(Sig_t)

    nan_mask = np.isnan(X_t[:, idx_corr])
    Xt_aligned_full = (Xt - mu_xt) @ A_full.T + mu_xs

    X_t_coral = X_t.copy()
    X_t_coral[:, idx_corr] = np.where(nan_mask, np.nan, Xt_aligned_full)

    # PCA-CORAL k=5
    std_xs = np.std(Xs, axis=0).clip(1e-8)
    Xs_std = np.nan_to_num((Xs - mu_xs) / std_xs, nan=0.0, posinf=0.0, neginf=0.0)
    Xt_std = np.nan_to_num((Xt - mu_xs) / std_xs, nan=0.0, posinf=0.0, neginf=0.0)
    k = min(PCA_K, n_corr, Xs_std.shape[0])
    pca = PCA(n_components=k).fit(Xs_std)
    Zs = pca.transform(Xs_std)
    Zt = pca.transform(Xt_std)
    mu_zs, mu_zt = Zs.mean(axis=0), Zt.mean(axis=0)
    Sg_zs = np.cov(Zs, rowvar=False) + 1e-6 * np.eye(k)
    Sg_zt = np.cov(Zt, rowvar=False) + 1e-6 * np.eye(k)
    A_pca = _safe_sqrtm(Sg_zs) @ _safe_invsqrtm(Sg_zt)
    Zt_ali = (Zt - mu_zt) @ A_pca.T + mu_zs
    Xt_pca_ali = pca.inverse_transform(Zt_ali) * std_xs + mu_xs

    X_t_pca = X_t.copy()
    X_t_pca[:, idx_corr] = np.where(nan_mask, np.nan, Xt_pca_ali)

    # Sanear Inf inducidos
    for _m in (X_t_coral, X_t_pca):
        _m[~np.isfinite(_m) & ~np.isnan(_m)] = np.nan

    X_t_imp_coral = np.nan_to_num(_impute_with_mu(X_t_coral, mu_s),
                                  nan=0.0, posinf=0.0, neginf=0.0)
    X_t_imp_pca = np.nan_to_num(_impute_with_mu(X_t_pca, mu_s),
                                nan=0.0, posinf=0.0, neginf=0.0)

    # ── Split source train/val ─────────────────────────────────────────────
    idx_tr, idx_va = train_test_split(np.arange(len(X_s)), test_size=0.2, random_state=SEED)

    xgb_kw = dict(
        n_estimators=XGB_N_EST, max_depth=XGB_MAX_DEPTH, learning_rate=XGB_LR,
        random_state=SEED, verbosity=0, n_jobs=-1,
    )

    records = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        for i, v_idx in enumerate(idx_corr):
            v_name = schema[v_idx]
            if verbose and ((i + 1) % 25 == 0 or i == 0):
                logger.info("  drift [%d/%d] %s", i + 1, n_corr, v_name)

            vtype = _detect_vtype(X_s[:, v_idx])
            if vtype == "all_nan":
                continue

            pred_idx = [j for j in range(n_feat) if j != v_idx]
            y_v = X_s[:, v_idx]
            m_tr = ~np.isnan(y_v[idx_tr])
            m_va = ~np.isnan(y_v[idx_va])
            if m_tr.sum() < 200 or m_va.sum() < 50:
                records.append({"feature": v_name, "v_type": vtype,
                                "drift_type": "INSUFFICIENT_SNUH_DATA"})
                continue

            Xtr_xgb = X_s[idx_tr][m_tr][:, pred_idx]
            Xva_xgb = X_s[idx_va][m_va][:, pred_idx]
            Xtr_las = X_s_imp[idx_tr][m_tr][:, pred_idx]
            Xva_las = X_s_imp[idx_va][m_va][:, pred_idx]
            ytr = y_v[idx_tr][m_tr]
            yva = y_v[idx_va][m_va]

            # XGBoost
            try:
                if vtype == "binary":
                    if len(np.unique(ytr)) < 2:
                        records.append({"feature": v_name, "v_type": vtype,
                                        "drift_type": "SINGLE_CLASS"})
                        continue
                    xmdl = xgb.XGBClassifier(eval_metric="logloss", **xgb_kw)
                    xmdl.fit(Xtr_xgb, ytr.astype(int))
                    X_base = (float(roc_auc_score(yva.astype(int),
                                                  xmdl.predict_proba(Xva_xgb)[:, 1]))
                              if len(np.unique(yva)) > 1 else np.nan)
                else:
                    xmdl = xgb.XGBRegressor(**xgb_kw)
                    xmdl.fit(Xtr_xgb, ytr)
                    X_base = float(r2_score(yva, xmdl.predict(Xva_xgb)))
            except Exception:
                records.append({"feature": v_name, "v_type": vtype,
                                "drift_type": "TRAIN_ERROR_XGB"})
                continue

            # LASSO
            try:
                if vtype == "binary":
                    lmdl = LogisticRegression(
                        penalty="l1", solver="liblinear", C=1.0,
                        max_iter=2000, random_state=SEED,
                    )
                    lmdl.fit(Xtr_las, ytr.astype(int))
                    L_base = (float(roc_auc_score(yva.astype(int),
                                                  lmdl.predict_proba(Xva_las)[:, 1]))
                              if len(np.unique(yva)) > 1 else np.nan)
                else:
                    lmdl = LassoCV(cv=3, n_alphas=15, max_iter=3000,
                                   n_jobs=1, random_state=SEED)
                    lmdl.fit(Xtr_las, ytr)
                    L_base = float(r2_score(yva, lmdl.predict(Xva_las)))
            except Exception:
                records.append({"feature": v_name, "v_type": vtype,
                                "drift_type": "TRAIN_ERROR_LASSO"})
                continue

            if np.isnan(X_base) and np.isnan(L_base):
                records.append({"feature": v_name, "v_type": vtype,
                                "drift_type": "NAN_BASELINE"})
                continue

            # Target availability
            y_vc = X_t[:, v_idx]
            m_c = ~np.isnan(y_vc)
            shap_v = float(shap_imp.get(v_name, 0.0))
            if m_c.sum() < 20 or (vtype == "binary" and len(np.unique(y_vc[m_c])) < 2):
                records.append({
                    "feature": v_name, "v_type": vtype,
                    "X_base": X_base, "L_base": L_base,
                    "drift_type": "INSUFFICIENT_CLINIC_DATA",
                    "shap_importance_main_model": shap_v,
                })
                continue

            yc = y_vc[m_c]

            # LOW_VARIANCE_TARGET pre-filter (solo continuas/ordinales)
            if vtype != "binary":
                cv_t = float(np.std(yc)) / (abs(float(np.mean(yc))) + 1e-8)
                if cv_t < CV_THRESHOLD:
                    records.append({
                        "feature": v_name, "v_type": vtype,
                        "X_base": X_base, "L_base": L_base,
                        "drift_type": "LOW_VARIANCE_TARGET",
                        "shap_importance_main_model": shap_v,
                        "cv_target": round(cv_t, 4),
                    })
                    continue

            xmet, lmet = {}, {}
            try:
                for vn, Xv_x, Xv_l in [
                    ("raw", X_t, X_t_imp),
                    ("coral", X_t_coral, X_t_imp_coral),
                    ("pca", X_t_pca, X_t_imp_pca),
                ]:
                    Xp_x = Xv_x[m_c][:, pred_idx]
                    Xp_l = Xv_l[m_c][:, pred_idx]
                    if vtype == "binary":
                        xmet[vn] = float(roc_auc_score(yc.astype(int),
                                                       xmdl.predict_proba(Xp_x)[:, 1]))
                        lmet[vn] = float(roc_auc_score(yc.astype(int),
                                                       lmdl.predict_proba(Xp_l)[:, 1]))
                    else:
                        xmet[vn] = float(r2_score(yc, xmdl.predict(Xp_x)))
                        lmet[vn] = float(r2_score(yc, lmdl.predict(Xp_l)))
            except Exception:
                records.append({"feature": v_name, "v_type": vtype,
                                "X_base": X_base, "L_base": L_base,
                                "drift_type": "EVAL_ERROR",
                                "shap_importance_main_model": shap_v})
                continue

            X_drop = float(X_base) - xmet["raw"]
            L_drop = float(L_base) - lmet["raw"]
            L_rec_coral = _compute_rec(lmet["coral"], lmet["raw"], L_drop)
            L_rec_pca = _compute_rec(lmet["pca"], lmet["raw"], L_drop)
            X_rec_coral = _compute_rec(xmet["coral"], xmet["raw"], X_drop)
            X_rec_pca = _compute_rec(xmet["pca"], xmet["raw"], X_drop)

            x_dropped = X_drop > MIN_DROP and float(X_base) > MIN_DROP
            l_dropped = L_drop > MIN_DROP and float(L_base) > MIN_DROP

            if not x_dropped:
                dtype = "STABLE"
            elif not l_dropped:
                dtype = "NONLINEAR_DRIFT"
            else:
                recs = [v for v in [L_rec_coral, L_rec_pca]
                        if v is not None and not np.isnan(v)]
                best_l_rec = max(recs) if recs else -999.0
                if best_l_rec >= RECOVERABLE_THRESHOLD:
                    dtype = "LINEAR_RECOVERABLE"
                elif best_l_rec >= PARTIAL_THRESHOLD:
                    dtype = "PARTIAL_RECOVERY"
                else:
                    dtype = "CONCEPT_RELATIONAL"

            records.append({
                "feature": v_name, "v_type": vtype,
                "X_base": X_base, "X_raw": xmet["raw"],
                "X_coral": xmet["coral"], "X_pca": xmet["pca"],
                "X_drop": X_drop, "X_rec_coral": X_rec_coral, "X_rec_pca": X_rec_pca,
                "L_base": L_base, "L_raw": lmet["raw"],
                "L_coral": lmet["coral"], "L_pca": lmet["pca"],
                "L_drop": L_drop, "L_rec_coral": L_rec_coral, "L_rec_pca": L_rec_pca,
                "drift_type": dtype,
                "shap_importance_main_model": shap_v,
            })

    df = pd.DataFrame(records)

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_path, index=False)
        logger.info("Drift decomposition: cache guardado en %s", cache_path)

    return _df_to_dicts(df)


def _df_to_dicts(df: pd.DataFrame) -> dict:
    """Construye los dicts que AutoAdapter consume."""
    drift_type_dict = dict(zip(df["feature"], df["drift_type"]))
    shap_dict = (
        dict(zip(df["feature"], pd.to_numeric(df["shap_importance_main_model"],
                                              errors="coerce").fillna(0.0)))
        if "shap_importance_main_model" in df.columns else {}
    )
    if "L_base" in df.columns:
        lbase_raw = pd.to_numeric(df["L_base"], errors="coerce")
        mean_l = float(np.nanmean(lbase_raw)) if lbase_raw.notna().any() else 0.0
        lbase_filled = lbase_raw.fillna(mean_l)
        lbase_dict = dict(zip(df["feature"], lbase_filled))
    else:
        lbase_dict = {}

    return {
        "drift_type_dict": drift_type_dict,
        "shap_dict": shap_dict,
        "lbase_dict": lbase_dict,
        "df": df,
    }
