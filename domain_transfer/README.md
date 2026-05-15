# domain_transfer — Package Overview

## What each technique corrects

| Técnica | Shift corregido | Requiere labels target | Coste computacional |
|---------|----------------|----------------------|--------------------|
| `IdentityAligner` | Ninguno (baseline) | No | — |
| `AdaBNAligner` | Covariate shift P(X): media y varianza (1er y 2º momento marginal) | No | Muy bajo |
| `CoralAligner` | Covariate shift P(X): covarianza global de 2º orden | No | Bajo |
| `PCACoralAligner` | Covariate shift P(X): covarianza proyectada en subespacio principal | No | Bajo |
| `OTAligner` | Covariate shift P(X): distribución conjunta completa (Sinkhorn) | No | Medio-alto |
| `SelectiveAligner` | Subconjunto de features — decorator sobre cualquier aligner | No | Hereda del base |
| `QuantileTransformAligner` | Covariate shift P(X): forma de distribución marginal per-feature | No | Bajo |
| Feature masking (`mask_features`) | Features ruidosas o no transferibles (elimina contribución individual) | No (usa scores de source) | Bajo |
| `WOEEncoder` | Escala log-odds de source → normaliza features STABLE/LINEAR_RECOVERABLE | Sí (y_s en source, siempre disponible) | Bajo |
| `StratifiedPlattRecalibrator` | Probability shift P(Y\|X) por subgrupos — recalibra scores en target | Sí (requiere y_t) | Bajo |
| Re-entrenamiento en target (fuera del scope) | Concept shift P(Y\|X) — relación feature→label diferente | Sí | Alto |

### Notas

- **Covariate shift** P(X) ≠ P(X): la distribución de features cambia entre
  hospitales, pero la relación P(Y|X) se asume estable.  Los aligners
  corrigen esto sin acceder a los labels del hospital destino.

- **Probability shift**: aunque P(Y|X) sea idéntica, la prevalencia diferente
  (SNUH 25.7 % vs Clínic 27.6 %) puede desplazar la escala de probabilidades
  predichas.  Requiere y_t para recalibrar.

- **Concept shift**: la relación entre features y label es distinta.  Ningún
  aligner de esta librería lo resuelve; requiere re-entrenamiento o
  fine-tuning en datos etiquetados del destino.

- `spearman_flip` (inversión del signo de Spearman entre source y target) es
  un síntoma de concept shift puntual en una feature.  Solo es computable con
  labels target → reside en `MetaDriftAnalyzer` (diagnóstico), no en
  `MetaDriftPredictor` (operacional).

## Módulos

```
domain_transfer/
├── align/          # Aligners (B2) — transforman X_t hacia P(X_s)
│   ├── identity.py             # IdentityAligner: baseline sin alineación
│   ├── adabn.py                # AdaBNAligner: normalización por media/std
│   ├── coral.py                # CoralAligner: covarianza global 2.º orden
│   ├── pca_coral.py            # PCACoralAligner: covarianza en subespacio PCA (default k=5)
│   ├── optimal_transport.py    # OTAligner: transporte óptimo Sinkhorn
│   ├── selective.py            # SelectiveAligner: decorator por sub-conjunto de features
│   └── quantile_transform.py   # QuantileTransformAligner: matching cuantil-a-cuantil [exp_extend]
├── select/         # Feature selectors (B3) — puntúan y enmascaran features
│   ├── meta_drift.py       # MetaDriftPredictor: scorer operacional, sin labels target
│   ├── combined_score.py   # CombinedScoreSelector: combina L_base + SHAP + d_j
│   ├── sweep.py            # sweep_mask_n: búsqueda del N óptimo
│   └── woe_encoder.py      # WOEEncoder: codificación Weight-of-Evidence [exp_extend]
├── drift/          # Drift diagnostics (B3) — análisis retrospectivo con labels target
│   ├── analyzer.py                     # MetaDriftAnalyzer: diagnóstico, incluye spearman_flip
│   └── concept_shift_univariate.py     # UnivariateConceptShiftDiagnoser: logistic+interaction [exp_extend]
├── calibration/    # Calibración de probabilidades (B6)
│   └── stratified_platt.py     # StratifiedPlattRecalibrator: Platt por subgrupos [exp_extend]
├── data/           # Loaders y CohortPair (B1)
├── model/          # ModelWrapper y registry (B1)
└── eval/           # Evaluación y comparación (B5)
```

## Framework adaptador (exp_extend)

Los módulos marcados `[exp_extend]` implementan el patrón **wrapper/black-box adapter**: 
extienden el framework sin modificar los pipelines existentes.

```
       ┌─────────────────────────────────────────────────────────────┐
       │                    CohortPair.align()                       │
       │                                                             │
       │   X_s_imp ──► fit(Aligner)                                  │
       │   X_t_imp ──► transform() ──► X_t_aligned                  │
       └─────────────────────────────────────────────────────────────┘
              ▲              ▲                ▲
      PCACoralAligner  SelectiveAligner  QuantileTransformAligner
       (covariate)      (selective)        (marginal shape)

       ┌─────────────────────────────────────────────────────────────┐
       │                WOEEncoder (select/)                         │
       │   Adapta la escala de features STABLE a log-odds source     │
       │   Implementa interfaz Aligner → composable con pair.align() │
       └─────────────────────────────────────────────────────────────┘

       ┌─────────────────────────────────────────────────────────────┐
       │           StratifiedPlattRecalibrator (calibration/)        │
       │   Post-hoc: recalibra P(y=1|x) por subgrupos del target     │
       │   Requiere y_t → diagnóstico / exploración únicamente       │
       └─────────────────────────────────────────────────────────────┘

       ┌─────────────────────────────────────────────────────────────┐
       │        UnivariateConceptShiftDiagnoser (drift/)             │
       │   Logistic + cohort interaction → detecta β₃ ≠ 0           │
       │   Requiere y_t → diagnóstico retrospectivo únicamente       │
       └─────────────────────────────────────────────────────────────┘
```

### Principios del adaptador

1. **Sin modificar el pipeline existente**: los nuevos módulos se añaden como
   componentes intercambiables, no como parches al código base.
2. **Interfaz Aligner unificada**: `fit(X_s, X_t) → self` + `transform(X_t, nan_mask) → ndarray`.
   Todo lo que implemente esta interfaz funciona con `pair.align()`.
3. **Separación diagnóstico / operacional**: los módulos que requieren `y_t`
   viven en `drift/` o `calibration/`, nunca en `select/`.
4. **Composabilidad**: `SelectiveAligner` permite aplicar cualquier aligner
   solo al subconjunto de features relevante (p.ej., `NONLINEAR_DRIFT`).
