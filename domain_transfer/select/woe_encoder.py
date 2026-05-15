"""
domain_transfer.select.woe_encoder
=====================================
WOEEncoder: codificación Weight-of-Evidence (WoE) per-feature.

Motivación
----------
En el contexto de transferencia de dominio, las features STABLE y
LINEAR_RECOVERABLE mantienen una relación monótona con el outcome en ambas
cohortes.  Para estas features, el WoE encoding transforma cada feature
a la escala log-odds del outcome en source:

    WoE_j(x) = log[P(x_j ≤ x | y=1) / P(x_j ≤ x | y=0)]

Esto tiene dos efectos:
1. Comprime escalas heterogéneas a una escala comparable (log-odds).
2. Hace la relación feature→outcome aproximadamente lineal, lo que facilita
   la transferencia cuando la relación es estable.

NO aplica a CONCEPT_RELATIONAL ni NONLINEAR_DRIFT, donde la relación
feature→outcome cambia entre cohortes.  Aplicar WoE allí introduciría
ruido de mala especificación.

Implementación
--------------
- Discretiza cada feature en ``n_bins`` bins de igual frecuencia (quantiles)
  estimados sobre la fuente (source).
- Calcula WoE por bin usando solo source (sin leakage de target).
- Aplica suavizado de Laplace (``smoothing``) para evitar log(0).
- Transforma target con el mapa aprendido en source.
- Los NaN se tratan como una categoría especial o se restauran (configurable).
- Compatibilidad total con la interfaz Aligner → usable en pair.align().

Referencia
----------
WoE encoding clásico (usado en scorecards de crédito desde 1980s; adaptado
aquí como preprocesamiento de domain transfer).
Thomas, L. C. (2009). Consumer Credit Models. Oxford University Press.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from domain_transfer.align.base import Aligner, _restore_nan

logger = logging.getLogger(__name__)


class WOEEncoder(Aligner):
    """
    Encoder Weight-of-Evidence por feature, ajustado en source y aplicado
    a target.

    Implementa la interfaz Aligner para compatibilidad con pair.align() y
    SelectiveAligner.

    Parameters
    ----------
    n_bins : int
        Número de bins de igual frecuencia (quantile bins) para la
        discretización.  Default 10.
    smoothing : float
        Pseudo-conteo de Laplace para evitar log(0).  Se suma al numerador
        y denominador:
            WoE = log[(n_pos_bin + α) / (n_neg_bin + α)]
                - log[(n_pos_total + α) / (n_neg_total + α)]
        Default 0.5.
    nan_woe : float or None
        Valor WoE asignado a los NaN de target.  Si None, los NaN se
        restauran (comportamiento por defecto con nan_mask).
        Default None (restaurar NaN).
    """

    def __init__(
        self,
        n_bins: int = 10,
        smoothing: float = 0.5,
        nan_woe: Optional[float] = None,
    ) -> None:
        if n_bins < 2:
            raise ValueError(f"n_bins debe ser ≥ 2; recibido: {n_bins}")
        if smoothing <= 0:
            raise ValueError(f"smoothing debe ser > 0; recibido: {smoothing}")
        self.n_bins = n_bins
        self.smoothing = smoothing
        self.nan_woe = nan_woe

        # Por feature: lista de (bordes, woe_values)
        # bordes: array de longitud (n_bins+1) incluyendo −inf/+inf
        # woe_values: array de longitud n_bins
        self._bin_edges: list[np.ndarray] = []
        self._woe_values: list[np.ndarray] = []
        self._global_woe: list[float] = []  # WoE global (fallback para fuera de rango)
        self._q: int = 0
        self._fitted = False

    # ─────────────────────────────────────────────────────────────────────────

    def fit(
        self,
        X_source: np.ndarray,
        X_target: np.ndarray,
        y_source: Optional[np.ndarray] = None,
    ) -> "WOEEncoder":
        """
        Ajusta los bins y WoE sobre la distribución source.

        Parameters
        ----------
        X_source : np.ndarray (n_s, q)
            Feature matrix source.  Debe estar imputada (sin NaN).
        X_target : np.ndarray (n_t, q)
            Feature matrix target (no se usa; solo para validar dimensiones).
        y_source : np.ndarray (n_s,), optional
            Labels binarios source.  Si None, no se puede ajustar el WoE.
            **Requerido** — la firma Aligner.fit() no toma y_source, por lo
            que se recomienda llamar a fit_supervised() directamente y usar
            transform() a través de pair.align().

        Returns
        -------
        self

        Raises
        ------
        ValueError
            Si y_source es None (requiere labels de source).
        """
        if y_source is None:
            raise ValueError(
                "WOEEncoder requiere y_source (labels source).  "
                "La interfaz Aligner estándar no pasa labels; usa "
                "fit_supervised(X_source, X_target, y_source) directamente."
            )
        return self.fit_supervised(X_source, X_target, y_source)

    def fit_supervised(
        self,
        X_source: np.ndarray,
        X_target: np.ndarray,
        y_source: np.ndarray,
    ) -> "WOEEncoder":
        """
        Ajusta los bins y WoE con labels source explícitos.

        Alternativa a fit() para no necesitar sobrecargar la signatura Aligner.
        Compatible con SelectiveAligner si se pasa el aligner ya ajustado.
        """
        n_s, q = X_source.shape
        if X_target.shape[1] != q:
            raise ValueError(
                f"X_source tiene {q} columnas pero X_target tiene "
                f"{X_target.shape[1]}."
            )
        if len(y_source) != n_s:
            raise ValueError(
                f"y_source tiene {len(y_source)} filas pero X_source tiene {n_s}."
            )

        alpha = self.smoothing
        n_pos_total = float(y_source.sum()) + alpha
        n_neg_total = float((1 - y_source).sum()) + alpha

        self._bin_edges = []
        self._woe_values = []
        self._global_woe = []

        for j in range(q):
            edges, woe_vals = self._fit_feature(
                X_source[:, j], y_source, alpha, n_pos_total, n_neg_total
            )
            self._bin_edges.append(edges)
            self._woe_values.append(woe_vals)

            # WoE global: basado en toda la feature (para fallback)
            global_woe = float(
                np.log((n_pos_total / n_neg_total))
            )  # simplificado; no usamos aquí la corrección
            self._global_woe.append(global_woe)

        self._q = q
        self._fitted = True
        logger.debug(
            "WOEEncoder: ajustado sobre %d features, %d bins, "
            "smoothing=%.2f.",
            q, self.n_bins, alpha,
        )
        return self

    def _fit_feature(
        self,
        x_j: np.ndarray,
        y_s: np.ndarray,
        alpha: float,
        n_pos_total: float,
        n_neg_total: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Calcula los bordes y valores WoE para una feature.

        Devuelve (edges, woe_values):
        - edges: (n_bins+1,) con ±inf en extremos
        - woe_values: (n_bins,)
        """
        n_bins = self.n_bins
        quantiles = np.linspace(0, 100, n_bins + 1)
        raw_edges = np.percentile(x_j, quantiles)

        # Colapsar bordes duplicados (puede pasar con distribuciones discretas)
        unique_edges = np.unique(raw_edges)
        if len(unique_edges) < 2:
            # Feature sin variación → asignamos WoE=0 (sin información)
            edges = np.array([-np.inf, np.inf])
            woe_vals = np.array([0.0])
            return edges, woe_vals

        # Ajustar bins con bordes únicos
        actual_bins = len(unique_edges) - 1
        edges = np.concatenate([[-np.inf], unique_edges[1:-1], [np.inf]])
        woe_vals = np.zeros(actual_bins)

        for b in range(actual_bins):
            lo = edges[b]
            hi = edges[b + 1]
            if b == 0:
                mask_bin = x_j <= hi
            elif b == actual_bins - 1:
                mask_bin = x_j > lo
            else:
                mask_bin = (x_j > lo) & (x_j <= hi)

            n_pos_bin = float(y_s[mask_bin].sum()) + alpha
            n_neg_bin = float((1 - y_s[mask_bin]).sum()) + alpha

            # WoE = log(Distribution of Events / Distribution of Non-Events)
            dist_ev = n_pos_bin / n_pos_total
            dist_nev = n_neg_bin / n_neg_total
            woe_vals[b] = float(np.log(np.maximum(dist_ev / dist_nev, 1e-12)))

        return edges, woe_vals

    # ─────────────────────────────────────────────────────────────────────────

    def transform(
        self,
        X_target: np.ndarray,
        nan_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Transforma X_target a la escala WoE aprendida en source.

        Parameters
        ----------
        X_target : np.ndarray (n_t, q)
            Target imputado (sin NaN).
        nan_mask : np.ndarray (n_t, q), optional
            Máscara de NaN originales.  Si se proporciona, los NaN se restauran.

        Returns
        -------
        X_woe : np.ndarray (n_t, q)
        """
        if not self._fitted:
            raise RuntimeError(
                "WOEEncoder debe ser ajustado antes de transform()."
            )
        if X_target.shape[1] != self._q:
            raise ValueError(
                f"X_target tiene {X_target.shape[1]} columnas pero el encoder "
                f"fue ajustado con {self._q}."
            )

        X_out = np.zeros_like(X_target, dtype=float)
        for j in range(self._q):
            X_out[:, j] = self._transform_feature(
                X_target[:, j], j, nan_mask[:, j] if nan_mask is not None else None
            )

        # Restaurar NaN
        return _restore_nan(X_out, nan_mask)

    def _transform_feature(
        self,
        x_j: np.ndarray,
        j: int,
        nan_mask_j: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Mapea los valores de x_j a WoE."""
        edges = self._bin_edges[j]
        woe_vals = self._woe_values[j]
        n_bins = len(woe_vals)

        result = np.zeros(len(x_j), dtype=float)
        for b in range(n_bins):
            lo = edges[b]
            hi = edges[b + 1]
            if b == 0:
                mask_bin = x_j <= hi
            elif b == n_bins - 1:
                mask_bin = x_j > lo
            else:
                mask_bin = (x_j > lo) & (x_j <= hi)
            result[mask_bin] = woe_vals[b]

        # NaN → nan_woe si especificado, o 0.0 (se restaurará después)
        if nan_mask_j is not None:
            result[nan_mask_j] = self.nan_woe if self.nan_woe is not None else 0.0

        return result

    # ─────────────────────────────────────────────────────────────────────────

    def information_value(self) -> np.ndarray:
        """
        Calcula el Information Value (IV) por feature.

        IV_j = Σ_b (dist_events_b - dist_nonevents_b) * WoE_b

        Un IV alto indica mayor poder predictivo de la feature.
        Referencia: IV > 0.3 = fuerte, 0.1-0.3 = media, < 0.1 = débil.

        Returns
        -------
        iv : np.ndarray, shape (q,)
        """
        if not self._fitted:
            raise RuntimeError("Llama a fit_supervised() primero.")
        # No podemos recalcular sin X_source; devolvemos array vacío con NaN
        # La implementación completa requiere almacenar dist_events/nonevents
        logger.warning(
            "information_value() no disponible tras fit(); necesita re-acceso "
            "a X_source e y_source.  Usa WOEEncoder._iv_ si fue calculado."
        )
        if hasattr(self, "_iv_"):
            return self._iv_
        return np.full(self._q, np.nan)

    def fit_supervised_with_iv(
        self,
        X_source: np.ndarray,
        X_target: np.ndarray,
        y_source: np.ndarray,
    ) -> "WOEEncoder":
        """
        Como fit_supervised, pero también calcula y almacena el IV por feature.
        Útil para selección de features antes de WoE encoding.
        """
        self.fit_supervised(X_source, X_target, y_source)

        alpha = self.smoothing
        n_pos_total = float(y_source.sum()) + alpha
        n_neg_total = float((1 - y_source).sum()) + alpha

        iv = np.zeros(self._q)
        for j in range(self._q):
            edges = self._bin_edges[j]
            woe_vals = self._woe_values[j]
            n_bins = len(woe_vals)
            x_j = X_source[:, j]
            iv_j = 0.0
            for b in range(n_bins):
                lo = edges[b]
                hi = edges[b + 1]
                if b == 0:
                    mask_bin = x_j <= hi
                elif b == n_bins - 1:
                    mask_bin = x_j > lo
                else:
                    mask_bin = (x_j > lo) & (x_j <= hi)
                n_pos_b = float(y_source[mask_bin].sum()) + alpha
                n_neg_b = float((1 - y_source[mask_bin]).sum()) + alpha
                dist_ev = n_pos_b / n_pos_total
                dist_nev = n_neg_b / n_neg_total
                iv_j += (dist_ev - dist_nev) * woe_vals[b]
            iv[j] = iv_j

        self._iv_ = iv
        return self
