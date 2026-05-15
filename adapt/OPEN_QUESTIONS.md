# OPEN QUESTIONS — ADAPT

Preguntas abiertas e issues conocidos para versiones futuras.

---

## OQ-1: Heterogeneidad de calibración (estratos)

**Archivo:** `adapt/designer/rules.py::_test_calibration_heterogeneity()`  
**Estado:** Conservador (siempre devuelve p=1.0)

**Descripción:**  
La función `_test_calibration_heterogeneity()` actualmente devuelve 1.0 siempre,
lo que significa que `select_calibration_method()` nunca elige `stratified_platt`.
Esto equivale a asumir "sin heterogeneidad" y usar siempre `platt_loo`.

Para SNUH→Clínic, esto es correcto (n_events=29 → no hay poder estadístico para detectar
heterogeneidad por estratos). Pero en cohortes más grandes podría subestimarse la ganancia
de la calibración estratificada.

**Plan v0.2:**  
Implementar Hosmer-Lemeshow test by stratum usando `StratifiedPlattRecalibrator._strata_fn`
en `domain_transfer/calibration/stratified_platt.py`. Usar chi2 con k-1 grados de libertad.

---

## OQ-2: PCACoralAligner doble fit en `_get_aligned_scores()`

**Archivo:** `adapt/pipeline/auto_adapter.py::_get_aligned_scores()`  
**Estado:** Ineficiencia menor, no bug (resultado determinista)

**Descripción:**  
`_get_aligned_scores()` es llamado tanto en `fit()` (para datos de calibración)
como en `predict()`. El `PCACoralAligner` se re-fittea en ambas llamadas con los
mismos datos, por lo que el resultado es idéntico pero el cómputo es doble.

**Plan v0.2:**  
Añadir bandera `self._aligner_fitted: bool` y saltarse `.fit()` si ya está fiteado.

```python
if not self._aligner_fitted:
    self._fitted_aligner.fit(X_s_corr, X_t_corr)
    self._aligner_fitted = True
X_t_aligned = self._fitted_aligner.transform(X_t_corr, nan_mask=nan_mask_corr)
```

---

## OQ-3: L_base computado vs precomputado (CSV)

**Archivo:** `adapt/profiler/feature_profiler.py::_compute_lbase_scores()`  
**Estado:** Divergencia semántica

**Descripción:**  
`results/v/v_drift_decomposition.csv` contiene `L_base` definido como el AUROC de
un predictor univariado XGBoost entrenado en source y evaluado en source (capacidad
predictiva del feature). El `_compute_lbase_scores()` de ADAPT usa LASSO logístico,
que mide correlación lineal con el outcome.

Para el pipeline de producción, se recomienda usar `lbase_dict` con los valores del CSV.
La implementación de LASSO es un fallback para cuando no hay CSV disponible.

**Plan v0.2:**  
- Renombrar `lbase_score` a `lbase_score_approx` en el fallback.
- Documentar la diferencia en el docstring.
- Añadir flag `lbase_method: Literal["xgboost_auroc", "lasso_approx"]` al `FeatureProfile`.

---

## OQ-4: Parámetro `drift_type_v` en `UnivariateConceptShiftDiagnoser`

**Archivo:** `adapt/profiler/feature_profiler.py`  
**Estado:** Por verificar en tests de integración

**Descripción:**  
`feature_profiler.py` pasa `drift_type_v=[drift_type]` como kwarg a
`UnivariateConceptShiftDiagnoser.fit()`. Si esta clase no acepta ese parámetro,
el fallback maneja el error (beta3=0, qbh=1.0, flip=False), pero no reporta warning.

**Plan v0.2:**  
Verificar la firma del constructor en `domain_transfer/drift/concept_shift_univariate.py`.
Si el parámetro no existe, eliminarlo de la llamada y calcular drift_type_v internamente.

---

## OQ-5: Extensión multi-target

**Archivo:** `adapt/pipeline/auto_adapter.py`  
**Estado:** No implementado

**Descripción:**  
La pipeline actual asume un único target hospital. Para transferencia multi-target
(SNUH → [Clínic, HUGTiP, HJD]), cada target necesita su propio `Profiler` y `AutoAdapter`.
No hay mecanismo de agrupación ni de transfer jerárquico.

**Plan v0.3:**  
Añadir `MultiTargetAutoAdapter(targets: dict[str, CohortPair])` que ejecute ADAPT
en paralelo y combine los reportes en un único HTML comparativo.

---

## OQ-6: Robustez del cálculo de MMD² en `global_profiler.py`

**Archivo:** `adapt/profiler/global_profiler.py`  
**Estado:** Subóptimo en alta dimensionalidad

**Descripción:**  
El cálculo de MMD² usa el kernel RBF con bandwidth por defecto (median heuristic).
Con p >> n_target, el cálculo se vuelve inestable. Se debería proyectar a las primeras
k PCs antes de calcular MMD².

**Plan v0.2:**  
Pre-proyectar en las primeras `min(k_pca, n_target//2)` componentes PCA antes del
cálculo del kernel. Añadir `pca_before_mmd: bool = True` como parámetro de constants.

---

## OQ-7: Reporter sin Jinja2 (mantenibilidad)

**Archivo:** `adapt/reporter/html_report.py`  
**Estado:** Funcional, pero frágil para HTML complejo

**Descripción:**  
El HTML se genera por concatenación de strings en Python. Funciona pero es difícil
de mantener si se quiere añadir más secciones.

**Plan v0.2:**  
Evaluar usar `jinja2` como dependencia opcional. Si no está instalado, fallback al
generador actual.
