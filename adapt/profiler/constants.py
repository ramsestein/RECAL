"""
adapt.profiler.constants
=========================
Umbrales auditables del Profiler. Todos los thresholds tienen docstring
con justificación empírica o metodológica.

IMPORTANTE: No modificar estos valores sin actualizar la justificación
y la versión en adapt/CHANGELOG.md.
"""

# ── Thresholds de features ────────────────────────────────────────────────────

CV_TARGET_NEAR_CONSTANT_THRESHOLD = 0.02
"""
Umbral de coeficiente de variación (CV = std/|mean|) para marcar una feature
como near-constant en target.

Justificación: exp_extend mostró que intraop_SpO2_median (CV=1.1%) y
intraop_Na_median (CV=1.5%) tienen drift clasificado como LOW_VARIANCE_TARGET
en la descomposición V. Con CV < 2%, la feature tiene variación tan pequeña
que cualquier aligner (QT, CORAL) introduce más ruido del que corrige.
"""

CV_TARGET_QT_MINIMUM = 0.05
"""
CV mínimo en target para aplicar QuantileTransform a una feature.

Justificación: features con CV < 5% en target son near-constant pero no
extremas (CV > 2%). QT sobre estas features sería casi identidad porque la
distribución target ya tiene muy poca variabilidad — el mapping cuantil-a-cuantil
no cambia prácticamente nada.
"""

VAR_RATIO_QT_LOWER = 0.5
"""
Límite inferior del ratio var_target/var_source para aplicar QT.

Si var_ratio está en (0.5, 2.0), la varianza está moderadamente desplazada.
CORAL absorberá fácilmente este cambio sin necesidad de QT previo.
Justificación empírica: el rango 0.5–2.0 es el "régimen CORAL seguro" para
cambios de varianza (Sundararajan & Tshannen-Moran, 2015; validado en
exp_extend con features STABLE/LINEAR_RECOVERABLE).
"""

VAR_RATIO_QT_UPPER = 2.0
"""
Límite superior del ratio var_target/var_source para aplicar QT.

Ver VAR_RATIO_QT_LOWER. Si var_ratio > 2.0 o < 0.5, la varianza está
claramente desplazada y QT puede ser útil antes de CORAL.
"""

SHAP_WOE_MINIMUM = 0.005
"""
SHAP mínimo para aplicar WOE encoding a una feature.

Justificación: features con SHAP < 0.005 en el modelo source son ruido
desde el punto de vista predictivo. Aplicar WOE a estas features introduce
coeficientes log-odds inestables sin beneficio discriminativo.
En el dataset SNUH, 0.005 corresponde al percentil ~40 de importancias.
"""

FLIP_OF_SIGN_SHAP_THRESHOLD = 0.005
"""
SHAP mínimo para reportar flip de signo como significativo.

Justificación: inversión de signo en features con SHAP < 0.005 es ruido
estadístico (correlación base con Spearman no significativa a n=105).
"""

# ── Thresholds globales ───────────────────────────────────────────────────────

N_EVENTS_MINIMUM_MASK = 20
"""
Número mínimo de eventos target para activar la máscara de features.

Justificación (exp_extend, E1): con n_events < 20, el sweep de N se vuelve
inestable: la curva AUROC vs N tiene fluctuaciones > 0.03 entre N consecutivos,
lo que hace que el elbow sea no fiable. Con 20 eventos, tenemos al menos 4
eventos por cuartil para estabilizar el ranking.
"""

N_EVENTS_MINIMUM_CALIBRATION = 20
"""
Número mínimo de eventos target para activar la recalibración Platt LOO.

Justificación: LogisticRegression LOO con n_events < 20 produce CIs de los
coeficientes Platt que se solapan con [0, ∞), lo que hace la recalibración
inestable. Validado en simulaciones LOO con n_events ∈ {10, 15, 20, 30}.
"""

N_EVENTS_MINIMUM_WOE = 30
"""
Número mínimo de eventos target para aplicar WOE encoding selectivo.

Justificación empírica: exp_extend mostró que WOE empeora AUROC cuando
n_events_target = 29 (el valor real de Clínic), incluso aplicado solo a
features STABLE+LINEAR_RECOVERABLE. Con 30 eventos, el ratio estimación/ruido
mejora suficientemente para que WOE no sea perjudicial. El umbral 30 es
conservador deliberadamente.
"""

N_EVENTS_SOURCE_MINIMUM_WOE = 100
"""
Número mínimo de eventos en source para ajustar WOE con robustez.

Justificación: WOE binning con n_bins=10 requiere ~10 eventos por bin para
estimaciones estables. Con 100 eventos y 10 bins, esperamos ~10 eventos/bin.
"""

N_EVENTS_ISOTONIC = 500
"""
Número mínimo de eventos target para usar calibración isotónica en lugar
de Platt LOO.

Justificación: la calibración isotónica tiene más parámetros libres que
Platt LOO (un parámetro por punto de quiebre vs. 2 parámetros Platt).
Con n_events < 500, la curva isotónica sobreajusta. Validado por comparación
bootstrap en simulaciones con n_events ∈ {100, 200, 500, 1000}.
"""

CALIBRATION_SLOPE_RECAL_THRESHOLD = 0.5
"""
Desviación mínima del slope de calibración respecto a 1.0 para activar
la recalibración.

Si |slope - 1.0| <= 0.5, la calibración está aceptablemente ajustada
(intervalo [0.5, 1.5] contiene el slope ideal de 1.0 con margen).
Justificación: Steyerberg et al. (2010) reportan que slope ∈ [0.7, 1.3]
es aceptable en validación externa. Ampliamos a [0.5, 1.5] para ser
conservadores — solo recalibramos cuando el modelo está claramente mal
calibrado.
"""

CALIBRATION_HETEROGENEITY_PVALUE = 0.05
"""
P-valor del test de heterogeneidad de slopes para activar calibración
Platt estratificada en lugar de global.

Justificación: si los slopes Platt en los terciles bajo/medio/alto del
score predicho son heterogéneos (p < 0.05), la calibración global no
captura la estructura. La calibración estratificada por score es más
adecuada pero requiere más datos.
"""

MMD2_N_PERMUTATIONS = 1000
"""
Número de permutaciones para el test de MMD² (Maximum Mean Discrepancy).

Justificación: 1000 permutaciones es el estándar en la literatura de
two-sample tests (Gretton et al., 2012). Proporciona p-valores con
resolución 0.001, suficiente para detectar drift significativo.
"""

MMD2_BANDWIDTH = 1.0
"""
Ancho de banda del kernel RBF para MMD² (escala relativa a la varianza
de los datos estandarizados).

Justificación: se usa un kernel RBF con σ² = bandwidth * median_pairwise_distance²
(heurística de la mediana). bandwidth=1.0 corresponde al bandwidth de la
mediana estándar.
"""

PCA_CORAL_MAX_K_FACTOR = 1.0
"""
Factor para recortar k_max = factor * sqrt(n_target).

Justificación: PCA-CORAL requiere que la covarianza latente k×k esté
bien condicionada. Con n_target puntos, la matriz k×k tiene rank efectivo
min(k, n_target). Para garantizar condicionamiento, usamos k <= sqrt(n_target)
(heurística de Kritchman & Nadler, 2008).
Factor=1.0 → k_max = sqrt(n_target). Conservador pero seguro.
"""

PCA_CORAL_K_RANGE_MIN = 2
"""
k mínimo para el CV de selección de PCA-CORAL.
Justificación: k=1 es trivial (proyección a un escalar); k=2 es el mínimo
estructuralmente interesante.
"""

QUADRANT_THRESHOLD_PERCENTILE = 50
"""
Percentil de umbral para la asignación de cuadrantes SHAP × L_base.

Justificación: usar la mediana como umbral es el criterio más robusto y
menos sesgado por outliers. Features por encima de la mediana en ambos
ejes son A_core; features por debajo en ambos son D_ponzonous.
"""

BOOTSTRAP_N_AUROC = 500
"""
Número de réplicas bootstrap para CIs de AUROC en el Profiler.

Justificación: 500 réplicas dan CIs bootstrap con error estándar
< 0.005 en AUROC, suficiente para diagnóstico. No usamos 2000 porque
el Profiler se ejecuta como paso de diagnóstico, no como benchmark final.
"""

CONCEPT_RELATIONAL_SHAP_PCT_THRESHOLD = 0.30
"""
Fracción de SHAP total en features CONCEPT_RELATIONAL para activar la
recomendación de fine-tuning.

Justificación: si más del 30% del SHAP total está en features con concept
shift relacional, el techo teórico de la UDA no supervisada está cerca.
El valor 0.30 corresponde al percentil observado en SNUH→Clínic (37%),
donde la adaptación no supervisada satura su potencial.
"""
