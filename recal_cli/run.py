"""
recal_cli.run
==============
CLI principal — orquesta carga, adaptación, evaluación y salida.

Uso:
    python -m recal_cli.run --config configs/example.yaml
    python -m recal_cli.run --config configs/example.yaml --no-cv
    python -m recal_cli.run --config configs/example.yaml --override recal_core.pca_k=8
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Forzar UTF-8 en stdout/stderr (Windows usa cp1252 por defecto)
try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass

import joblib
import numpy as np

from recal.data.pairing import CohortPair
from recal_cli.config_schema import FullConfig, load_config
from recal_cli.cross_validate import (
    _bootstrap_auroc_ci,
    _classification_metrics,
    cross_validate_adapt,
)
from recal_cli.data_loader import GenericCohortLoader, load_schema_from_file
from recal_cli.model_loader import load_model
from recal_core.pipeline.auto_adapter import AutoAdapter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("recal_cli")


# ── DeLong paired AUROC test ──────────────────────────────────────────────────

def _delong_roc_test(
    y_true: np.ndarray,
    prob_a: np.ndarray,
    prob_b: np.ndarray,
) -> tuple[float, float, float, float]:
    """
    DeLong et al. (1988) paired AUROC significance test.

    Returns
    -------
    (z_stat, p_two_sided, auc_a, auc_b)
    """
    y = np.asarray(y_true, dtype=int)
    a = np.asarray(prob_a, dtype=float)
    b = np.asarray(prob_b, dtype=float)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    m, n = len(pos_idx), len(neg_idx)
    if m < 2 or n < 2:
        return float("nan"), float("nan"), float("nan"), float("nan")

    def _placement(pos_scores: np.ndarray, neg_scores: np.ndarray):
        """Structural components V10 (per positive) and V01 (per negative)."""
        m_, n_ = len(pos_scores), len(neg_scores)
        V10 = np.zeros(m_)
        V01 = np.zeros(n_)
        for i in range(m_):
            for j in range(n_):
                if pos_scores[i] > neg_scores[j]:
                    V10[i] += 1.0
                elif pos_scores[i] == neg_scores[j]:
                    V10[i] += 0.5
                    V01[j] += 0.5
                else:
                    V01[j] += 1.0
        V10 /= n_
        V01 /= m_
        return V10, V01

    V10_a, V01_a = _placement(a[pos_idx], a[neg_idx])
    V10_b, V01_b = _placement(b[pos_idx], b[neg_idx])
    auc_a = float(V10_a.mean())
    auc_b = float(V10_b.mean())

    if m > 1:
        S10 = np.cov(np.vstack([V10_a, V10_b]), ddof=1)
    else:
        S10 = np.zeros((2, 2))
    if n > 1:
        S01 = np.cov(np.vstack([V01_a, V01_b]), ddof=1)
    else:
        S01 = np.zeros((2, 2))

    var_diff = (
        (S10[0, 0] + S10[1, 1] - 2 * S10[0, 1]) / m
        + (S01[0, 0] + S01[1, 1] - 2 * S01[0, 1]) / n
    )
    if var_diff <= 1e-15:
        return float("nan"), float("nan"), auc_a, auc_b

    from scipy import stats as _sp_stats
    z = float((auc_a - auc_b) / np.sqrt(var_diff))
    p = float(2.0 * _sp_stats.norm.sf(abs(z)))
    return z, p, auc_a, auc_b


# ── Wrapper serializable ──────────────────────────────────────────────────────

class AdaptedModelWrapper:
    """
    Wrapper persistible: encapsula AutoAdapter ya fitted + metadata para
    poder hacer predict_proba en datos nuevos del mismo target.

    NOTA: Solo es válido para datos extraídos de la misma distribución que
    el target original. NO sustituye un re-entrenamiento.
    """

    def __init__(self, auto_adapter: AutoAdapter, schema: list[str], metadata: dict):
        self.auto_adapter = auto_adapter
        self.schema = schema
        self.metadata = metadata

    def predict_proba(self, X) -> np.ndarray:
        """X: DataFrame o ndarray (n, p) con columnas en orden de schema."""
        # No podemos correr la pipeline sin un source... el wrapper solo
        # almacena un snapshot del aligner ya fitted para inspección/auditoría.
        raise NotImplementedError(
            "Para predict en nuevos datos del target, re-ejecutar el pipeline "
            "con CohortPair (source + nuevo target). Este wrapper guarda "
            "los componentes fitted para auditoría."
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info("Wrapper saved to %s", path)

    @staticmethod
    def load(path: str | Path) -> AdaptedModelWrapper:
        return joblib.load(path)


# ── Pipeline principal ────────────────────────────────────────────────────────

def _apply_overrides(cfg: FullConfig, overrides: list[str]) -> None:
    """Aplica overrides tipo `seccion.clave=valor`."""
    for ov in overrides:
        if "=" not in ov:
            raise ValueError(f"Invalid override: {ov} (expected section.key=value)")
        key, val = ov.split("=", 1)
        parts = key.strip().split(".")
        if len(parts) != 2:
            raise ValueError(f"Invalid override: {ov}")
        section, attr = parts
        section_obj = getattr(cfg, section, None)
        if section_obj is None or not hasattr(section_obj, attr):
            raise ValueError(f"Unknown section/attribute: {key}")
        # Tipo del campo destino
        cur = getattr(section_obj, attr)
        if isinstance(cur, bool):
            new = val.lower() in ("true", "1", "yes", "y")
        elif isinstance(cur, int):
            new = int(val)
        elif isinstance(cur, float):
            new = float(val)
        else:
            new = val
        setattr(section_obj, attr, new)
        logger.info("Override: %s = %r", key, new)


def _stamp_path(p: str | None, ts: str) -> str | None:
    """Inserta el timestamp antes de la extensión: foo.html → foo_20260519_120000.html"""
    if p is None:
        return None
    path = Path(p)
    return str(path.with_stem(f"{path.stem}_{ts}"))


def run(cfg: FullConfig, run_cv: bool = True, skip_expensive: bool = False) -> dict:
    """Ejecuta el pipeline completo y devuelve un dict de resultados."""
    import datetime
    out = {}

    # Aplicar timestamp a los paths de salida para evitar sobreescrituras
    if cfg.output.timestamp:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        cfg.output.report = _stamp_path(cfg.output.report, ts)
        cfg.output.recal_model = _stamp_path(cfg.output.recal_model, ts)
        cfg.output.metrics_json = _stamp_path(cfg.output.metrics_json, ts)
        print(f"  Run timestamp: {ts}")

    # 1. Schema (lo cargamos primero para pasarlo al modelo)
    print("\n[1/6] Loading schema…")
    schema_path = cfg.source.schema or cfg.target.schema
    if schema_path is None:
        raise ValueError(
            "You must specify 'schema' in source or target (path to a JSON/TXT file with features)."
        )
    schema = load_schema_from_file(schema_path)
    print(f"  Schema: {len(schema)} features")

    # 2. Modelo (con schema)
    print("\n[2/6] Loading model…")
    model = load_model(
        cfg.model.path,
        model_type=cfg.model.type,
        custom_loader=cfg.model.custom_loader,
        schema=schema,
    )
    print(f"  Model: {cfg.model.path} (n_features={getattr(model, 'n_features_in_', '?')})")
    if hasattr(model, "n_features_in_") and model.n_features_in_ is not None:
        if len(schema) != model.n_features_in_:
            logger.warning(
                "schema (%d) ≠ model n_features (%d)",
                len(schema), model.n_features_in_,
            )

    # ── Pipeline preprocessor (auto-detect or explicit) ───────────────────
    pipeline_preprocessor = None
    pipeline_path = cfg.model.pipeline  # explicit path from config

    if pipeline_path is None:
        # Auto-detect: buscar *_pipeline.json junto al modelo
        model_path = Path(cfg.model.path)
        candidates = list(model_path.parent.glob("*_pipeline.json"))
        if len(candidates) == 1:
            pipeline_path = str(candidates[0])
        elif len(candidates) > 1:
            # Preferir el que comparte prefijo con el modelo
            model_stem = model_path.stem
            for c in candidates:
                if c.stem.startswith(model_stem):
                    pipeline_path = str(c)
                    break
            if pipeline_path is None:
                logger.warning(
                    "Multiple _pipeline.json found: %s — using none. "
                    "Set model.pipeline in config to pick one.",
                    [c.name for c in candidates],
                )

    if pipeline_path:
        from recal_cli.pipeline_preprocessor import PipelinePreprocessor
        pipeline_preprocessor = PipelinePreprocessor(pipeline_path)
        print(f"  Pipeline: {pipeline_path} "
              f"({pipeline_preprocessor.n_features_in} → {pipeline_preprocessor.n_features_out} features)")

    # 3. Datos
    print("\n[3/6] Loading datasets…")
    src = GenericCohortLoader(cfg.source.path, schema, cfg.source.outcome_col,
                              cfg.source.label_positive_value,
                              unit_corrections=cfg.source.unit_corrections,
                              pipeline_preprocessor=pipeline_preprocessor)
    tgt = GenericCohortLoader(cfg.target.path, schema, cfg.target.outcome_col,
                              cfg.target.label_positive_value,
                              unit_corrections=cfg.target.unit_corrections,
                              pipeline_preprocessor=pipeline_preprocessor)
    src.load()
    tgt.load()  # eager
    print(f"  Source ({cfg.output.source_name}): {src.metadata()}")
    print(f"  Target ({cfg.output.target_name}): {tgt.metadata()}")

    pair = CohortPair(src, tgt)
    pair_filt = pair.filter_target(max_missing_rate=cfg.recal_core.max_missing_rate)
    print(f"  After filter_target({cfg.recal_core.max_missing_rate}): n_target={len(pair_filt.y_t)}")

    # 4. AutoAdapter — ajuste in-sample
    print("\n[4/6] Fitting AutoAdapter (in-sample)…")

    # Drift decomposition: opcional. Si hay CSV precomputado, lo cargamos;
    # si no, lo computamos on-the-fly (~2-3 min para 100 features) y cacheamos.
    drift_kwargs = {}
    df_drift = None  # will be set below if CSV is loaded
    if cfg.recal_core.drift_csv:
        drift_path = Path(cfg.recal_core.drift_csv)
    else:
        drift_path = None

    if drift_path is not None and drift_path.exists():
        import pandas as pd
        df_drift = pd.read_csv(drift_path).set_index("feature")
        if "drift_type" in df_drift.columns:
            drift_kwargs["drift_type_dict"] = df_drift["drift_type"].to_dict()
        if "shap_importance_main_model" in df_drift.columns:
            drift_kwargs["shap_dict"] = pd.to_numeric(
                df_drift["shap_importance_main_model"], errors="coerce"
            ).to_dict()
        if "L_base" in df_drift.columns:
            lbase_raw = pd.to_numeric(df_drift["L_base"], errors="coerce")
            mean_l = float(lbase_raw.mean()) if lbase_raw.notna().any() else 0.0
            drift_kwargs["lbase_dict"] = lbase_raw.fillna(mean_l).to_dict()
        print(f"  drift_csv (cache): {drift_path.name} ({len(df_drift)} features)")
    else:
        print("  drift_csv not found → computing decomposition on-the-fly "
              "(this may take several minutes)…")
        from recal_cli.drift_compute import compute_drift_decomposition
        result = compute_drift_decomposition(
            X_s=pair_filt.X_s,
            X_t=pair_filt.X_t,
            schema=schema,
            model=model,
            cache_path=drift_path,  # si era None, no cachea
        )
        drift_kwargs["drift_type_dict"] = result["drift_type_dict"]
        drift_kwargs["shap_dict"] = result["shap_dict"]
        drift_kwargs["lbase_dict"] = result["lbase_dict"]
        from collections import Counter
        cnt = Counter(result["drift_type_dict"].values())
        print(f"  Drift types: {dict(cnt)}")

    # Combined score per feature (used to sort section 9 in the HTML report)
    feature_combined_scores: dict = {}
    try:
        if df_drift is not None:
            import pandas as _pd_cs
            _shap = _pd_cs.to_numeric(
                df_drift["shap_importance_main_model"], errors="coerce"
            ).fillna(0.0).values
            _lbase = _pd_cs.to_numeric(
                df_drift["L_base"], errors="coerce"
            ).abs().fillna(0.0).values
            def _norm01(arr):
                lo_, hi_ = float(arr.min()), float(arr.max())
                return (arr - lo_) / max(hi_ - lo_, 1e-12)
            _combined = _norm01(_lbase) + _norm01(_shap)
            feature_combined_scores = dict(zip(df_drift.index, _combined.tolist()))
    except Exception as _e:
        logger.warning("Could not compute feature combined scores: %s", _e)

    aa = AutoAdapter(model=model, schema=schema, **drift_kwargs)
    aa.profile(pair_filt)

    # ── Joint drift (covariance structure) ───────────────────────────────────
    joint_drift_dict: dict | None = None
    try:
        from sklearn.covariance import LedoitWolf

        from recal_cli.joint_drift import (
            compute_condition_number,
            compute_effective_rank,
            joint_drift_report,
        )

        # Impute NaN with source column means before computing VIF
        src_means = np.nanmean(pair_filt.X_s, axis=0)
        X_s_imp = np.where(np.isnan(pair_filt.X_s), src_means, pair_filt.X_s)
        X_t_imp = np.where(np.isnan(pair_filt.X_t), src_means, pair_filt.X_t)

        # Standardise (unit-variance) so VIF is comparable across features
        std = np.maximum(np.std(X_s_imp, axis=0), 1e-8)
        X_s_std = (X_s_imp - src_means) / std
        X_t_std = (X_t_imp - src_means) / std

        print("\n[2.5] Joint drift analysis…")
        vif_df = joint_drift_report(
            X_s_std, X_t_std, schema,
            delta_vif_warn=cfg.joint_drift.delta_vif_warn,
            delta_vif_severe=cfg.joint_drift.delta_vif_severe,
        )
        severe_share = float((vif_df["flag"] == "SEVERE").mean())

        lw_s = float(LedoitWolf().fit(X_s_std).shrinkage_)
        lw_t = float(LedoitWolf().fit(X_t_std).shrinkage_)

        mi_delta = None
        if cfg.joint_drift.compute_mi_matrix:
            from recal_cli.joint_drift import mi_matrix_delta
            mi_delta = mi_matrix_delta(X_s_std, X_t_std)

        severe_cnt = int((vif_df["flag"] == "SEVERE").sum())
        watch_cnt = int((vif_df["flag"] == "WATCH").sum())
        print(f"  VIF flags: {severe_cnt} SEVERE, {watch_cnt} WATCH — severe share={severe_share:.1%}")

        joint_drift_dict = {
            "vif_table": vif_df,
            "condition_source": compute_condition_number(X_s_std),
            "condition_target": compute_condition_number(X_t_std),
            "eff_rank_source": compute_effective_rank(X_s_std),
            "eff_rank_target": compute_effective_rank(X_t_std),
            "lw_coef_source": lw_s,
            "lw_coef_target": lw_t,
            "severe_share": severe_share,
            "severe_share_threshold": cfg.joint_drift.severe_share_threshold,
            "mi_delta": mi_delta,
            "compute_mi_matrix": cfg.joint_drift.compute_mi_matrix,
        }
    except Exception as e:
        logger.warning("Joint drift analysis failed: %s", e)

    cfg_design = aa.design(pair_filt, pca_k=cfg.recal_core.pca_k, max_n_sweep=cfg.recal_core.max_n_sweep)
    if not cfg.recal_core.apply_qt:
        cfg_design.apply_quantile = False
        cfg_design.quantile_features = []
        print("  apply_qt=False → disabling QT")
    aa.fit(pair_filt)
    scores_raw = model.predict_proba(pair_filt.X_t)
    scores_adapted = aa.predict(pair_filt)

    # Source domain metrics (reference).
    try:
        scores_src = model.predict_proba(pair_filt.X_s)
        m_source = _classification_metrics(pair_filt.y_s, scores_src)
    except Exception as e:
        logger.warning("Could not compute source metrics: %s", e)
        m_source = None

    # Additional source metrics for the report (CI, calibration slope, ECE)
    if m_source is not None:
        try:
            lo, hi = _bootstrap_auroc_ci(pair_filt.y_s, scores_src, n_boot=500)
            m_source["auroc_ci_lo"] = lo
            m_source["auroc_ci_hi"] = hi
        except Exception as e:
            logger.warning("Source bootstrap CI failed: %s", e)
        try:
            from recal_core.profiler.global_profiler import _calibration_metrics as _cal_metrics
            slope_s, ece_s, _ = _cal_metrics(pair_filt.y_s, model, pair_filt.X_s)
            m_source["calibration_slope"] = slope_s
            m_source["ece"] = ece_s
        except Exception as e:
            logger.warning("Source calibration metrics failed: %s", e)

    m_raw = _classification_metrics(pair_filt.y_t, scores_raw)
    m_adapted = _classification_metrics(pair_filt.y_t, scores_adapted)

    # Bootstrap CI for raw and adapted on target
    try:
        lo_raw, hi_raw = _bootstrap_auroc_ci(pair_filt.y_t, scores_raw, n_boot=500)
        m_raw["auroc_ci_lo"] = lo_raw
        m_raw["auroc_ci_hi"] = hi_raw
    except Exception as _e:
        lo_raw, hi_raw = None, None
        logger.warning("Raw bootstrap CI failed: %s", _e)
    try:
        lo_ada, hi_ada = _bootstrap_auroc_ci(pair_filt.y_t, scores_adapted, n_boot=500)
        m_adapted["auroc_ci_lo"] = lo_ada
        m_adapted["auroc_ci_hi"] = hi_ada
    except Exception as _e:
        lo_ada, hi_ada = None, None
        logger.warning("Adapted bootstrap CI failed: %s", _e)

    # ── Statistical significance tests (DeLong + bootstrap z-test) ───────────
    significance_tests: dict | None = None
    try:
        z_dl, p_dl, auc_ada, auc_raw = _delong_roc_test(
            pair_filt.y_t, scores_adapted, scores_raw
        )
        n_tests = 2  # adapted_vs_raw + adapted_vs_source
        p_dl_corr = min(1.0, p_dl * n_tests) if p_dl == p_dl else float("nan")

        sig_vs_source: dict | None = None
        if m_source is not None and lo_ada is not None:
            lo_src = m_source.get("auroc_ci_lo")
            hi_src = m_source.get("auroc_ci_hi")
            if lo_src is not None and hi_src is not None:
                se_ada = (hi_ada - lo_ada) / (2 * 1.96)
                se_src = (hi_src - lo_src) / (2 * 1.96)
                se_diff = float(np.sqrt(se_ada ** 2 + se_src ** 2))
                delta_src = m_adapted["auroc"] - m_source["auroc"]
                if se_diff > 1e-9:
                    from scipy import stats as _sp_stats2
                    z_src = float(delta_src / se_diff)
                    p_src = float(2.0 * _sp_stats2.norm.sf(abs(z_src)))
                    sig_vs_source = {
                        "auc_source": m_source["auroc"],
                        "auc_adapted": m_adapted["auroc"],
                        "delta": delta_src,
                        "z": z_src,
                        "p_raw": p_src,
                        "p_bonferroni": min(1.0, p_src * n_tests),
                        "ci_source": [lo_src, hi_src],
                        "ci_adapted": [lo_ada, hi_ada],
                        "note": (
                            f"Independent samples (source n={len(pair_filt.y_s)}"
                            f" vs target n={len(pair_filt.y_t)}); "
                            "SE approximated from bootstrap 95% CI."
                        ),
                    }

        significance_tests = {
            "adapted_vs_raw": {
                "auc_adapted": auc_ada,
                "auc_raw": auc_raw,
                "delta": auc_ada - auc_raw,
                "z_delong": z_dl,
                "p_raw": p_dl,
                "p_bonferroni": p_dl_corr,
                "ci_adapted": [lo_ada, hi_ada] if lo_ada is not None else None,
                "ci_raw": [lo_raw, hi_raw] if lo_raw is not None else None,
                "method": "DeLong (1988) paired test",
            },
            "adapted_vs_source": sig_vs_source,
            "correction": "Bonferroni (k=2)",
            "alpha": 0.05,
        }
    except Exception as _e:
        logger.warning("Significance tests failed: %s", _e)

    print("\n  ── In-sample metrics ──")
    if m_source is not None:
        print(f"    Source  : AUROC={m_source['auroc']:.4f}  P={m_source['precision']:.3f}  "
              f"R={m_source['recall']:.3f}  F1={m_source['f1']:.3f}  "
              f"(n={len(pair_filt.y_s)}, modelo original)")
    print(f"    Raw     : AUROC={m_raw['auroc']:.4f}  P={m_raw['precision']:.3f}  "
          f"R={m_raw['recall']:.3f}  F1={m_raw['f1']:.3f}  thr={m_raw['threshold']:.3f}")
    print(f"    Adapted : AUROC={m_adapted['auroc']:.4f}  P={m_adapted['precision']:.3f}  "
          f"R={m_adapted['recall']:.3f}  F1={m_adapted['f1']:.3f}  thr={m_adapted['threshold']:.3f}")
    print(f"    ΔAUROC  : {m_adapted['auroc'] - m_raw['auroc']:+.4f}")
    print(f"    Mask N  : {cfg_design.mask_n}")

    out["in_sample"] = {
        "source": m_source,
        "raw": m_raw, "adapted": m_adapted,
        "mask_n": int(cfg_design.mask_n),
        "n_target": int(len(pair_filt.y_t)),
        "n_events": int(pair_filt.y_t.sum()),
        "n_source": int(len(pair_filt.y_s)),
        "n_source_events": int(pair_filt.y_s.sum()),
    }

    # 5. Honest cross-validation
    if run_cv and cfg.overfitting_check.enabled and cfg.overfitting_check.method == "kfold":
        print(f"\n[5/6] Honest CV k={cfg.overfitting_check.n_splits}…")
        cv_res = cross_validate_adapt(
            pair=pair_filt,
            model=model,
            schema=schema,
            n_splits=cfg.overfitting_check.n_splits,
            pca_k=cfg.recal_core.pca_k,
            max_n_sweep=cfg.recal_core.max_n_sweep,
            apply_qt_override=None if cfg.recal_core.apply_qt else False,
            random_state=cfg.overfitting_check.random_state,
            verbose=True,
        )
        oof = cv_res["oof_metrics"]
        lo, hi = cv_res["oof_auroc_ci"]
        print("\n  ── Honest CV (OOF aggregate) ──")
        print(f"    AUROC OOF: {oof['auroc']:.4f}  [95% CI: {lo:.3f}, {hi:.3f}]")
        print(f"    P/R/F1   : {oof['precision']:.3f} / {oof['recall']:.3f} / {oof['f1']:.3f}")

        gap = m_adapted["auroc"] - oof["auroc"]
        print("\n  ── Optimism ──")
        print(f"    in-sample AUROC: {m_adapted['auroc']:.4f}")
        print(f"    OOF AUROC      : {oof['auroc']:.4f}")
        print(f"    gap (overfit)  : {gap:+.4f}")
        if gap > 0.05:
            print("    WARNING: gap > 0.05 — consider regularisation / fewer sweep steps")

        cv_serializable = {k: v for k, v in cv_res.items() if k != "oof_scores"}
        cv_serializable["oof_auroc_ci"] = list(cv_res["oof_auroc_ci"])
        out["cv"] = cv_serializable
    else:
        print("\n[5/6] Honest CV: SKIPPED")
        out["cv"] = None

    # 5.5. Advanced evaluations (oracle, drift attribution, counterfactuals, Brier, YAML audit)
    eval_cfg = cfg.evaluation
    oracle_result = None
    drift_decomp = None
    feature_attr = None
    counterfactuals = None
    calibration_raw = None
    calibration_adapted = None
    calibration_delta_dict = None
    audit_yaml_path = None
    patient_profiles_result = None
    # Note: significance_tests is computed above (DeLong + bootstrap z-test)

    if skip_expensive:
        print("\n[5.5] Advanced evaluations: SKIPPED (--skip-expensive)")
    else:
        print("\n[5.5] Advanced evaluations…")

        # Oracle
        if eval_cfg.oracle_eval:
            try:
                from recal_cli.oracle import fit_target_oracle
                print(f"  Oracle k={eval_cfg.oracle_cv} (target k-fold)…")
                oracle_result = fit_target_oracle(
                    X_target=pair_filt.X_t,
                    y_target=pair_filt.y_t,
                    source_model=model,
                    cv=eval_cfg.oracle_cv,
                )
                print(f"  Oracle AUROC: {oracle_result['auroc']:.4f}  "
                      f"[{oracle_result['ci_lo']:.3f}, {oracle_result['ci_hi']:.3f}]")
                if oracle_result.get("warning"):
                    logger.warning("Oracle: %s", oracle_result["warning"])
            except Exception as e:
                logger.warning("Oracle failed: %s", e)

        # Drift decomposition
        try:
            from recal_cli.drift_attribution import drift_decomposition
            ci_raw = (m_raw.get("auroc_ci_lo", m_raw["auroc"]), m_raw.get("auroc_ci_hi", m_raw["auroc"]))
            ci_adapted = (m_adapted.get("auroc_ci_lo", m_adapted["auroc"]), m_adapted.get("auroc_ci_hi", m_adapted["auroc"]))
            ci_oracle = (
                (oracle_result["ci_lo"], oracle_result["ci_hi"])
                if oracle_result else None
            )
            auroc_oracle = oracle_result["auroc"] if oracle_result else None
            drift_decomp = drift_decomposition(
                auroc_raw=m_raw["auroc"],
                auroc_adapted=m_adapted["auroc"],
                auroc_oracle=auroc_oracle,
                ci_raw=ci_raw,
                ci_adapted=ci_adapted,
                ci_oracle=ci_oracle,
            )
            if drift_decomp and not drift_decomp.get("indeterminate"):
                print(f"  Recovery ratio: {drift_decomp['recovery_ratio']:.3f}  "
                      f"[{drift_decomp['recovery_ratio_ci'][0]:.3f}, {drift_decomp['recovery_ratio_ci'][1]:.3f}]")
        except Exception as e:
            logger.warning("Drift decomposition failed: %s", e)

        # Feature attribution
        if eval_cfg.feature_attribution and oracle_result is not None:
            try:
                from recal_cli.drift_attribution import feature_recovery_attribution
                print(f"  Feature attribution (top {eval_cfg.feature_attribution_top_n})…")
                feature_attr = feature_recovery_attribution(
                    auto_adapter=aa,
                    pair=pair_filt,
                    y_target=pair_filt.y_t,
                    top_n=eval_cfg.feature_attribution_top_n,
                )
            except Exception as e:
                logger.warning("Feature attribution failed: %s", e)

        # Counterfactuals
        if eval_cfg.counterfactual_alternatives > 0:
            try:
                from recal_cli.counterfactuals import run_counterfactual_sensitivity
                print(f"  Counterfactual sensitivity ({eval_cfg.counterfactual_alternatives} alternativas)…")
                counterfactuals = run_counterfactual_sensitivity(
                    auto_adapter=aa,
                    pair=pair_filt,
                    model=model,
                    schema=schema,
                    n_alternatives=eval_cfg.counterfactual_alternatives,
                )
            except Exception as e:
                logger.warning("Counterfactuals failed: %s", e)

        # Brier decomposition
        if eval_cfg.brier_decompose:
            try:
                from recal_cli.calibration_decomposition import brier_decompose, brier_delta
                calibration_raw = brier_decompose(pair_filt.y_t, scores_raw)
                calibration_adapted = brier_decompose(pair_filt.y_t, scores_adapted)
                calibration_delta_dict = brier_delta(calibration_raw, calibration_adapted)
                print(f"  Brier raw={calibration_raw['brier_score']:.4f}  "
                      f"adapted={calibration_adapted['brier_score']:.4f}  "
                      f"Δ={calibration_delta_dict['delta_brier_score']:+.4f}")
            except Exception as e:
                logger.warning("Brier decomposition failed: %s", e)

        # Audit YAML
        try:
            import dataclasses

            from recal_cli.audit_serializer import serialize_run_audit
            config_dict = dataclasses.asdict(cfg) if hasattr(cfg, "__dataclass_fields__") else {}
            audit_doc = serialize_run_audit(
                schema=schema,
                pair=pair_filt,
                auto_adapter=aa,
                oracle_result=oracle_result,
                drift_decomp=drift_decomp,
                counterfactuals=counterfactuals,
                calibration_raw=calibration_raw,
                calibration_adapted=calibration_adapted,
                calibration_delta=calibration_delta_dict,
                feature_attribution=feature_attr,
                config_dict=config_dict,
            )
            audit_yaml_path = audit_doc.get("_audit_path")
            out["audit_yaml_path"] = audit_yaml_path
            print(f"  Audit YAML → {audit_yaml_path}")
        except Exception as e:
            logger.warning("Audit YAML failed: %s", e)

        # Patient profile clustering
        try:
            from recal_cli.patient_profiles import analyze_patient_profiles
            print("  Patient profile clustering…")

            # Build a compact VIF summary from the joint-drift step so that the
            # PCA component selection can be motivated by multicollinearity.
            vif_summary_for_profiles: dict | None = None
            if joint_drift_dict is not None and "vif_table" in joint_drift_dict:
                _vt = joint_drift_dict["vif_table"]
                vif_summary_for_profiles = {
                    "n_severe":   int((_vt["flag"] == "SEVERE").sum()),
                    "n_watch":    int((_vt["flag"] == "WATCH").sum()),
                    "n_features": len(_vt),
                }

            patient_profiles_result = analyze_patient_profiles(
                X_s=pair_filt.X_s,
                y_s=pair_filt.y_s,
                X_t=pair_filt.X_t,
                y_t=pair_filt.y_t,
                model=model,
                features=schema,
                vif_summary=vif_summary_for_profiles,
            )
            top_cluster = patient_profiles_result["gap_ranking"][0]
            print(
                f"  Profiles: {patient_profiles_result['n_clusters']} clusters "
                f"(PCA k={patient_profiles_result['pca_k_used']}, "
                f"var={patient_profiles_result['pca_variance_explained']:.1%}) — "
                f"highest gap cluster {top_cluster[0]}: |Δscore|={top_cluster[1]:.4f}"
            )
        except Exception as e:
            logger.warning("Patient profile analysis failed: %s", e)

    out["oracle"] = oracle_result
    out["drift_decomp"] = drift_decomp
    out["feature_attribution"] = feature_attr
    out["counterfactuals"] = counterfactuals
    out["calibration_raw"] = calibration_raw
    out["calibration_adapted"] = calibration_adapted
    out["calibration_delta"] = calibration_delta_dict

    # 6. Outputs
    print("\n[6/6] Saving outputs…")
    if cfg.output.metrics_json:
        Path(cfg.output.metrics_json).parent.mkdir(parents=True, exist_ok=True)
        with open(cfg.output.metrics_json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, default=str)
        print(f"  Metrics → {cfg.output.metrics_json}")

    if cfg.output.recal_model:
        wrapper = AdaptedModelWrapper(
            auto_adapter=aa,
            schema=schema,
            metadata={
                "source_name": cfg.output.source_name,
                "target_name": cfg.output.target_name,
                "in_sample": out["in_sample"],
                "cv": out.get("cv"),
            },
        )
        try:
            wrapper.save(cfg.output.recal_model)
        except Exception as e:
            logger.warning("Could not serialise adapted model wrapper: %s", e)

    if cfg.output.report:
        try:
            from recal_core.reporter.html_report import generate_html_report
            Path(cfg.output.report).parent.mkdir(parents=True, exist_ok=True)
            generate_html_report(
                profile=aa._profile,
                config=aa._config,
                y_true=pair_filt.y_t,
                scores_before=scores_raw,
                scores_after=scores_adapted,
                source_name=cfg.output.source_name,
                target_name=cfg.output.target_name,
                output_path=cfg.output.report,
                cv_results=out.get("cv"),
                in_sample_metrics={
                    "source": m_source,
                    "raw": m_raw,
                    "adapted": m_adapted,
                    "n_source": int(len(pair_filt.y_s)),
                    "n_source_events": int(pair_filt.y_s.sum()),
                },
                oracle_results=oracle_result,
                drift_decomp=drift_decomp,
                feature_attribution=feature_attr,
                counterfactuals=counterfactuals,
                brier_decomp_raw=calibration_raw,
                brier_decomp_adapted=calibration_adapted,
                brier_delta=calibration_delta_dict,
                audit_yaml_path=audit_yaml_path,
                feature_log=aa.feature_log,
                joint_drift_data=joint_drift_dict,
                patient_profiles=patient_profiles_result,
                significance_tests=significance_tests,
                feature_combined_scores=feature_combined_scores,
            )
            print(f"  HTML report → {cfg.output.report}")
        except Exception as e:
            logger.warning("Could not generate HTML report: %s", e)
            import traceback
            traceback.print_exc()

    print("\nDone.\n")
    return out


# ── Entry point ───────────────────────────────────────────────────────────────

def main(argv=None):
    p = argparse.ArgumentParser(
        prog="recal_cli",
        description="Adapta un modelo binario a un dataset target sin re-entrenar.",
    )
    p.add_argument("--config", required=True, help="Ruta al YAML de configuración.")
    p.add_argument("--no-cv", action="store_true", help="Disable honest cross-validation.")
    p.add_argument("--override", action="append", default=[],
                   help="Override seccion.clave=valor (repetible).")
    p.add_argument("--skip-expensive", action="store_true",
                   help="Salta oracle, atribución y counterfactuals (más rápido).")
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    if args.override:
        _apply_overrides(cfg, args.override)

    run(cfg, run_cv=not args.no_cv, skip_expensive=args.skip_expensive)


if __name__ == "__main__":
    sys.exit(main() or 0)
