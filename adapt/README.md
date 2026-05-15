# ADAPT — Auto-Domain-Adaptation Pipeline Toolkit

> Meta-framework de transferencia de dominio para modelos clínicos.  
> Construido sobre `domain_transfer/` sin modificarlo.

---

## ¿Qué es ADAPT?

ADAPT orquesta los componentes del paquete `domain_transfer/` en una pipeline auto-configurada:

```
Profiler (diagnóstico) → Designer (selección de componentes) → Pipeline (alineación + calibración)
```

Dado un modelo entrenado en **source** (SNUH) y una cohorte **target** (Clínic), ADAPT:

1. **Perfila** el drift entre source y target (MMD², concept shift, cuadrantes de features).
2. **Diseña** la pipeline óptima mediante reglas determinísticas (sin búsqueda en target).
3. **Ejecuta** la pipeline: máscara → WOE → QuantileTransform → PCA-CORAL → calibración.
4. **Reporta** un HTML autocontenido con figuras y tablas.

---

## Caso benchmark: SNUH → Clínic

| Métrica | Raw | ADAPT |
|---|---|---|
| AUROC | 0.6293 | ~0.706 |
| Calibration slope | 9.06 | ~0.7 |
| ECE | ~0.15 | <0.05 |

**Decisiones del Designer (sin tocar un solo parámetro):**

| Componente | Decisión | Justificación |
|---|---|---|
| Máscara | ✓ N=4 | n_events=29 ≥ 20; elbow combined_score |
| QuantileTransform | ✗ | Features NONLINEAR_DRIFT son near-constant (CV<2%) en Clínic |
| WOE | ✗ | n_events=29 < 30 |
| PCA-CORAL | ✓ k=5 | var≥80% en 5 PCs source; sqrt(105)=10 → k=5 |
| Calibración | ✓ Platt LOO | slope=9.06 >> 0.5; n_events=29 < 500 (no isotónica) |

---

## Instalación

ADAPT se usa como paquete interno. No tiene dependencias extras; usa
`numpy`, `scipy`, `sklearn`, `xgboost`, `matplotlib`.

```bash
# Desde la raíz del proyecto
pip install -e .
```

---

## Uso rápido

```python
from domain_transfer.data.schema import load_schema
from domain_transfer.data.snuh import SNUHLoader
from domain_transfer.data.clinic import ClinicLoader
from domain_transfer.data.pairing import CohortPair
from domain_transfer.model.xgboost_wrapper import XGBoostWrapper
from adapt.pipeline.auto_adapter import AutoAdapter
from adapt.reporter.html_report import generate_html_report
import pandas as pd

# 1. Cargar datos
schema = load_schema()
model = XGBoostWrapper(schema=schema)
pair = CohortPair(
    source=SNUHLoader(schema=schema),
    target=ClinicLoader(schema=schema),
).filter_target(max_missing_rate=0.5)

# 2. Cargar datos precomputados (opcional pero recomendado)
df_drift = pd.read_csv("results/v/v_drift_decomposition.csv")
drift_type_dict = dict(zip(df_drift["feature"], df_drift["drift_type"]))
shap_dict = dict(zip(df_drift["feature"], df_drift["shap_importance_main_model"]))
lbase_dict = dict(zip(df_drift["feature"], df_drift["L_base"]))

# 3. Ejecutar ADAPT
aa = AutoAdapter(
    model=model,
    schema=schema,
    drift_type_dict=drift_type_dict,
    shap_dict=shap_dict,
    lbase_dict=lbase_dict,
)
proba = aa.auto_adapt(pair)

# 4. Reporte
scores_raw = model.predict_proba(pair.X_t_imp)
html = generate_html_report(
    profile=aa.profile_,
    config=aa.config_,
    y_true=pair.y_t,
    scores_before=scores_raw,
    scores_after=proba,
    source_name="SNUH",
    target_name="Clínic",
    output_path="reports/adapt/snuh_to_clinic.html",
)
print(aa.config_.summary())
```

---

## Estructura del paquete

```
adapt/
├── __init__.py
├── profiler/
│   ├── constants.py       # 18 thresholds con justificaciones empíricas
│   ├── base.py            # FeatureProfile, DriftProfile dataclasses
│   ├── quadrant.py        # Asignación de cuadrantes A/B/C/D
│   ├── global_profiler.py # MMD², Fisher, AUROC CI, calibration slope
│   ├── feature_profiler.py# L_base, SHAP, combined score, concept shift
│   └── profiler.py        # Clase Profiler (combina global + features)
├── designer/
│   ├── base.py            # AdapterConfig dataclass con rationale
│   ├── rules.py           # 5 reglas determinísticas con justificaciones
│   └── selector.py        # ComponentSelector: profile → AdapterConfig
├── pipeline/
│   └── auto_adapter.py    # AutoAdapter: profile/design/fit/predict/auto_adapt
├── reporter/
│   ├── tables.py          # Tablas Markdown (global, decisiones, features, eval)
│   ├── figures.py         # 4 figuras matplotlib (cuadrantes, calibración, ...)
│   └── html_report.py     # HTML autocontenido con figuras en base64
└── tests/
    ├── test_validation_snuh_clinic.py  # Test E2E benchmark (SLOW)
    ├── test_profiler.py                # Tests unitarios del Profiler
    ├── test_designer_rules.py          # Tests unitarios de las 5 reglas
    ├── test_auto_adapter.py            # Tests de integración (sintéticos)
    └── test_dataset_shift_invariance.py# Invarianza determinismo y escalado
```

---

## Filosofía de diseño

### Sin búsqueda en target
Las reglas del Designer NO usan métricas de target (AUROC, ECE, slope) para
seleccionar hiperparámetros. Esto garantiza evaluación honesta: el AUROC
reportado post-ADAPT es una estimación no sesgada del rendimiento en producción.

### Todos los thresholds en `constants.py`
Cada umbral tiene una justificación empírica en el docstring y está centralizado
en `adapt/profiler/constants.py`. Cambiar un umbral requiere documentar por qué.

### domain_transfer/ es solo una librería
ADAPT no modifica ni reescribe ningún componente de `domain_transfer/`.
Solo los importa y orquesta.

---

## Tests

```bash
# Tests unitarios rápidos (sin datos reales)
pytest adapt/tests/ -v -k "not slow"

# Tests E2E con datos reales (requiere SNUH + Clínic)
pytest adapt/tests/test_validation_snuh_clinic.py -v -m slow

# Todos los tests
pytest adapt/tests/ -v
```

---

## Preguntas abiertas

Ver [OPEN_QUESTIONS.md](OPEN_QUESTIONS.md).

---

## Changelog

Ver [CHANGELOG.md](CHANGELOG.md).
