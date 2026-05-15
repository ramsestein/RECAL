"""
adapt_cli.run
==============
CLI principal — orquesta carga, adaptación, evaluación y salida.

Uso:
    python -m adapt_cli.run --config configs/example.yaml
    python -m adapt_cli.run --config configs/example.yaml --no-cv
    python -m adapt_cli.run --config configs/example.yaml --override adapt.pca_k=8
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

from adapt.pipeline.auto_adapter import AutoAdapter
from domain_transfer.data.pairing import CohortPair

from adapt_cli.config_schema import FullConfig, load_config
from adapt_cli.cross_validate import (
    _classification_metrics,
    cross_validate_adapt,
)
from adapt_cli.data_loader import GenericCohortLoader, load_schema_from_file
from adapt_cli.model_loader import load_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("adapt_cli")


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
        logger.info("Wrapper guardado en %s", path)

    @staticmethod
    def load(path: str | Path) -> "AdaptedModelWrapper":
        return joblib.load(path)


# ── Pipeline principal ────────────────────────────────────────────────────────

def _apply_overrides(cfg: FullConfig, overrides: list[str]) -> None:
    """Aplica overrides tipo `seccion.clave=valor`."""
    for ov in overrides:
        if "=" not in ov:
            raise ValueError(f"Override inválido: {ov} (esperado seccion.clave=valor)")
        key, val = ov.split("=", 1)
        parts = key.strip().split(".")
        if len(parts) != 2:
            raise ValueError(f"Override inválido: {ov}")
        section, attr = parts
        section_obj = getattr(cfg, section, None)
        if section_obj is None or not hasattr(section_obj, attr):
            raise ValueError(f"Sección/atributo desconocido: {key}")
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


def run(cfg: FullConfig, run_cv: bool = True) -> dict:
    """Ejecuta el pipeline completo y devuelve un dict de resultados."""
    out = {}

    # 1. Schema (lo cargamos primero para pasarlo al modelo)
    print("\n[1/6] Cargando schema…")
    schema_path = cfg.source.schema or cfg.target.schema
    if schema_path is None:
        raise ValueError(
            "Debes especificar 'schema' en source o target (path a JSON/TXT con features)."
        )
    schema = load_schema_from_file(schema_path)
    print(f"  Schema: {len(schema)} features")

    # 2. Modelo (con schema)
    print("\n[2/6] Cargando modelo…")
    model = load_model(
        cfg.model.path,
        model_type=cfg.model.type,
        custom_loader=cfg.model.custom_loader,
        schema=schema,
    )
    print(f"  Modelo: {cfg.model.path} (n_features={getattr(model, 'n_features_in_', '?')})")
    if hasattr(model, "n_features_in_") and model.n_features_in_ is not None:
        if len(schema) != model.n_features_in_:
            logger.warning(
                "schema (%d) ≠ n_features del modelo (%d)",
                len(schema), model.n_features_in_,
            )

    # 3. Datos
    print("\n[3/6] Cargando datasets…")
    src = GenericCohortLoader(cfg.source.path, schema, cfg.source.outcome_col,
                              cfg.source.label_positive_value,
                              unit_corrections=cfg.source.unit_corrections)
    tgt = GenericCohortLoader(cfg.target.path, schema, cfg.target.outcome_col,
                              cfg.target.label_positive_value,
                              unit_corrections=cfg.target.unit_corrections)
    src.load(); tgt.load()  # eager
    print(f"  Source ({cfg.output.source_name}): {src.metadata()}")
    print(f"  Target ({cfg.output.target_name}): {tgt.metadata()}")

    pair = CohortPair(src, tgt)
    pair_filt = pair.filter_target(max_missing_rate=cfg.adapt.max_missing_rate)
    print(f"  Tras filter_target({cfg.adapt.max_missing_rate}): n_target={len(pair_filt.y_t)}")

    # 4. AutoAdapter — ajuste in-sample
    print("\n[4/6] Ajustando AutoAdapter (in-sample)…")

    # Drift decomposition: opcional. Si hay CSV precomputado, lo cargamos;
    # si no, lo computamos on-the-fly (~2-3 min para 100 features) y cacheamos.
    drift_kwargs = {}
    if cfg.adapt.drift_csv:
        drift_path = Path(cfg.adapt.drift_csv)
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
        print("  drift_csv no encontrado → computando descomposición on-the-fly "
              "(esto puede tardar varios minutos)…")
        from adapt_cli.drift_compute import compute_drift_decomposition
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

    aa = AutoAdapter(model=model, schema=schema, **drift_kwargs)
    aa.profile(pair_filt)
    cfg_design = aa.design(pair_filt, pca_k=cfg.adapt.pca_k, max_n_sweep=cfg.adapt.max_n_sweep)
    if not cfg.adapt.apply_qt:
        cfg_design.apply_quantile = False
        cfg_design.quantile_features = []
        print("  apply_qt=False → desactivando QT")
    aa.fit(pair_filt)
    scores_raw = model.predict_proba(pair_filt.X_t)
    scores_adapted = aa.predict(pair_filt)

    # Métricas del modelo original en su dominio de origen (referencia).
    try:
        scores_src = model.predict_proba(pair_filt.X_s)
        m_source = _classification_metrics(pair_filt.y_s, scores_src)
    except Exception as e:
        logger.warning("No se pudieron calcular métricas en source: %s", e)
        m_source = None

    m_raw = _classification_metrics(pair_filt.y_t, scores_raw)
    m_adapted = _classification_metrics(pair_filt.y_t, scores_adapted)

    print("\n  ── Métricas in-sample ──")
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

    # 5. CV honesto
    if run_cv and cfg.overfitting_check.enabled and cfg.overfitting_check.method == "kfold":
        print(f"\n[5/6] CV honesto k={cfg.overfitting_check.n_splits}…")
        cv_res = cross_validate_adapt(
            pair=pair_filt,
            model=model,
            schema=schema,
            n_splits=cfg.overfitting_check.n_splits,
            pca_k=cfg.adapt.pca_k,
            max_n_sweep=cfg.adapt.max_n_sweep,
            apply_qt_override=None if cfg.adapt.apply_qt else False,
            random_state=cfg.overfitting_check.random_state,
            verbose=True,
        )
        oof = cv_res["oof_metrics"]
        lo, hi = cv_res["oof_auroc_ci"]
        print("\n  ── CV honesto (OOF agregado) ──")
        print(f"    AUROC OOF: {oof['auroc']:.4f}  [95% CI: {lo:.3f}, {hi:.3f}]")
        print(f"    P/R/F1   : {oof['precision']:.3f} / {oof['recall']:.3f} / {oof['f1']:.3f}")

        gap = m_adapted["auroc"] - oof["auroc"]
        print(f"\n  ── Optimismo ──")
        print(f"    in-sample AUROC: {m_adapted['auroc']:.4f}")
        print(f"    OOF AUROC      : {oof['auroc']:.4f}")
        print(f"    gap (overfit)  : {gap:+.4f}")
        if gap > 0.05:
            print("    ADVERTENCIA: gap > 0.05 — considerar regularización/menos sweep")

        cv_serializable = {k: v for k, v in cv_res.items() if k != "oof_scores"}
        cv_serializable["oof_auroc_ci"] = list(cv_res["oof_auroc_ci"])
        out["cv"] = cv_serializable
    else:
        print("\n[5/6] CV honesto: SKIPPED")
        out["cv"] = None

    # 6. Outputs
    print("\n[6/6] Guardando outputs…")
    if cfg.output.metrics_json:
        Path(cfg.output.metrics_json).parent.mkdir(parents=True, exist_ok=True)
        with open(cfg.output.metrics_json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, default=str)
        print(f"  Métricas → {cfg.output.metrics_json}")

    if cfg.output.adapted_model:
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
            wrapper.save(cfg.output.adapted_model)
        except Exception as e:
            logger.warning("No se pudo serializar el wrapper: %s", e)

    if cfg.output.report:
        try:
            from adapt.reporter.html_report import generate_html_report
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
            )
            print(f"  Reporte HTML → {cfg.output.report}")
        except Exception as e:
            logger.warning("No se pudo generar reporte: %s", e)
            import traceback; traceback.print_exc()

    print("\nOK. Listo.\n")
    return out


# ── Entry point ───────────────────────────────────────────────────────────────

def main(argv=None):
    p = argparse.ArgumentParser(
        prog="adapt_cli",
        description="Adapta un modelo binario a un dataset target sin re-entrenar.",
    )
    p.add_argument("--config", required=True, help="Ruta al YAML de configuración.")
    p.add_argument("--no-cv", action="store_true", help="Desactiva CV honesto.")
    p.add_argument("--override", action="append", default=[],
                   help="Override seccion.clave=valor (repetible).")
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    if args.override:
        _apply_overrides(cfg, args.override)

    run(cfg, run_cv=not args.no_cv)


if __name__ == "__main__":
    sys.exit(main() or 0)
