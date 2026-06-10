"""
recal.align.quantile_transform
==========================================
QuantileTransformAligner: alineación de distribución marginal per-feature
mediante QuantileTransformer de sklearn.

Motivación
----------
PCA-CORAL corrige covarianza de 2.º orden pero no cambios de forma de la
distribución marginal (e.g., bimodal → unimodal, skew inverso).
QuantileTransformer mapea cada feature independientemente a una distribución
de referencia uniforme o normal, cancelando diferencias de percentil a
percentil.

Aplica mejor a:
- NONLINEAR_DRIFT: desplazamiento de forma no lineal (escalas, outliers)
- PARTIAL_RECOVERY: recuperación parcial con corrección más agresiva

Limitaciones
------------
- Transforma feature a feature independientemente → no corrige correlaciones.
- Puede romper relaciones bivariadas si las features tienen correlación alta.
- Conservador: solo corrige features en ``features`` (si se especifica).
- Los NaN son restaurados tras la transformación.

Composabilidad
--------------
QuantileTransformAligner implementa la misma interfaz Aligner que
PCACoralAligner → es intercambiable directamente:

>>> qt = QuantileTransformAligner(output_distribution='uniform')
>>> X_aligned = pair.align(qt)

Para combinar con SelectiveAligner:

>>> from recal.align.selective import SelectiveAligner
>>> sel = SelectiveAligner(
...     base_aligner=QuantileTransformAligner(),
...     feature_indices=nonlinear_drift_idx,
... )
>>> X_aligned = pair.align(sel)
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
from sklearn.preprocessing import QuantileTransformer

from recal.align.base import Aligner, _restore_nan

logger = logging.getLogger(__name__)


class QuantileTransformAligner(Aligner):
    """
    Aligner por quantile-matching de la distribución marginal.

    El transformer se ajusta sobre la distribución **source** y se aplica al
    target, de modo que la función de distribución acumulada del target se
    alinea a la del source.

    Parameters
    ----------
    output_distribution : {'uniform', 'normal'}
        Distribución de referencia a usar como espacio intermedio.
        - 'uniform': mapea source y target a U[0, 1].  Más robusto a outliers.
        - 'normal': mapea source y target a N(0, 1).  Más conveniente para
          aligners lineales downstream.
        Default 'uniform'.
    n_quantiles : int or None
        Número de cuantiles a usar en el transformer.  Si None, se usa el
        mínimo entre 1000 y n_source.  Mayor = más preciso pero más lento.
    subsample : int
        Submuestreo para el cálculo de cuantiles (parámetro de sklearn).
        Default 100_000 (sin efecto para n_source ≤ 100_000).
    random_state : int
        Semilla aleatoria para reproducibilidad.
    """

    def __init__(
        self,
        output_distribution: Literal["uniform", "normal"] = "uniform",
        n_quantiles: int | None = None,
        subsample: int = 100_000,
        random_state: int = 42,
    ) -> None:
        self.output_distribution = output_distribution
        self.n_quantiles = n_quantiles
        self.subsample = subsample
        self.random_state = random_state

        self._transformers: list[QuantileTransformer] = []
        self._fitted = False

    # ─────────────────────────────────────────────────────────────────────────

    def fit(
        self,
        X_source: np.ndarray,
        X_target: np.ndarray,
    ) -> QuantileTransformAligner:
        """
        Ajusta un QuantileTransformer por feature sobre la distribución source.

        Parameters
        ----------
        X_source : np.ndarray (n_s, q)
            Feature matrix source.  Debe estar imputada (sin NaN).
        X_target : np.ndarray (n_t, q)
            Feature matrix target (no se usa para el fit; solo para validar
            dimensiones).

        Returns
        -------
        self
        """
        n_s, q = X_source.shape
        if X_target.shape[1] != q:
            raise ValueError(
                f"X_source tiene {q} columnas pero X_target tiene "
                f"{X_target.shape[1]}."
            )

        n_q = self.n_quantiles if self.n_quantiles is not None else min(1000, n_s)

        self._transformers = []
        for j in range(q):
            qt = QuantileTransformer(
                output_distribution=self.output_distribution,
                n_quantiles=n_q,
                subsample=self.subsample,
                random_state=self.random_state,
            )
            qt.fit(X_source[:, j : j + 1])
            self._transformers.append(qt)

        self._q = q
        self._fitted = True
        logger.debug(
            "QuantileTransformAligner: ajustado sobre %d features, "
            "n_quantiles=%d, output='%s'.",
            q, n_q, self.output_distribution,
        )
        return self

    # ─────────────────────────────────────────────────────────────────────────

    def transform(
        self,
        X_target: np.ndarray,
        nan_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Transforma X_target mapeando cada feature a la distribución source.

        El proceso es:
        1. Mapear target_j → espacio de referencia con ``transform``.
        2. Mapear de vuelta a escala source con ``inverse_transform``.
        3. Restaurar NaN en las posiciones originales.

        Parameters
        ----------
        X_target : np.ndarray (n_t, q)
            Target imputado (sin NaN).
        nan_mask : np.ndarray (n_t, q), optional
            Máscara de NaN originales.  Si se proporciona, los NaN se
            restauran tras la transformación.

        Returns
        -------
        X_aligned : np.ndarray (n_t, q)
        """
        if not self._fitted:
            raise RuntimeError(
                "QuantileTransformAligner debe ser ajustado antes de transform()."
            )
        if X_target.shape[1] != self._q:
            raise ValueError(
                f"X_target tiene {X_target.shape[1]} columnas pero el aligner "
                f"fue ajustado con {self._q}."
            )

        X_out = X_target.copy()
        for j, qt in enumerate(self._transformers):
            col = X_target[:, j : j + 1]
            # Mapear a referencia (uniform / normal) y volver a escala source
            X_out[:, j] = qt.inverse_transform(qt.transform(col)).ravel()

        return _restore_nan(X_out, nan_mask)
