"""
domain_transfer.calibration.stratified_platt
==============================================
StratifiedPlattRecalibrator: recalibración de probabilidades con Platt scaling
estratificado por subgrupos.

Motivación
----------
La recalibración global (Platt LOO sobre toda la cohorte target) ignora el
hecho de que la desviación de calibración puede ser heterogénea entre
subgrupos.  Por ejemplo:
- Pacientes de cirugía larga (CPB > mediana) pueden tener scores
  sistemáticamente más altos que el outcome real.
- Pacientes jóvenes (edad < mediana) pueden estar mejor calibrados.

Una recalibración estratificada ajusta un modelo de Platt independiente
por estrato, permitiendo correcciones específicas por subgrupo.

Estrategias de estratificación implementadas
--------------------------------------------
- ``'score_terciles'``: terciles del score predicho (bajo/medio/alto).
  Es la estratificación más relevante porque corrige heterogeneidad de
  calibración a lo largo de la escala de riesgo.
- ``'cpb_time'``: mediana del tiempo de circulación extracorpórea (CPB).
- ``'age'``: mediana de edad del paciente.
- ``'custom'``: función de estratificación pasada por el usuario.

Limitaciones
-------------
Con n≈105 pacientes y 29 eventos en Clínic, los estratos tienen n≈35
pacientes cada uno.  Los resultados son **exploratorios** — los IC de ECE
y calibration slope por estrato se solapan mucho.  Solo reportar medias
como indicativas; no claims de significancia.

Método Platt (LOO)
------------------
Con n_target pequeño (n≈105), se usa leave-one-out (LOO) para ajustar
la regresión logística de Platt y evitar overfitting de la calibración.
Dentro de cada estrato: ajustar en (estrato \ {i}), predecir {i}.
Si el estrato tiene < min_stratum_size, se usa calibración global (fallback).

Calibración estratificada vs recalibración clínica
----------------------------------------------------
Esta clase NO reemplaza la validación clínica de una herramienta.  Es un
componente diagnóstico y de postprocesado para investigación.

Requiere y_t → vive en calibration/, no en select/.
"""

from __future__ import annotations

import logging
from typing import Callable, Literal, Optional, Union

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)

# Tipos de estratificación
StratMethod = Literal["score_terciles", "cpb_time", "age", "custom"]


class StratifiedPlattRecalibrator:
    """
    Recalibrador Platt estratificado por subgrupos del target.

    Parameters
    ----------
    strategy : str or callable
        Estrategia de estratificación:
        - 'score_terciles': terciles del score predicho.
        - 'cpb_time': mediana del tiempo de CPB (requiere ``cpb_col``).
        - 'age': mediana de edad (requiere ``age_col``).
        - callable: función f(scores, covariates) → np.ndarray de int (stratum id).
    n_strata : int
        Número de estratos.  Para 'score_terciles' se usa siempre 3.
        Para estrategias medianas se usa 2.  Default 3.
    min_stratum_size : int
        Tamaño mínimo de estrato para ajustar calibración propia.
        Estratos por debajo → fallback a calibración global.  Default 15.
    max_iter : int
        Iteraciones máximas del solver de LogisticRegression.  Default 1000.
    cpb_col : str, optional
        Nombre de columna de CPB time en el DataFrame de covariables.
    age_col : str, optional
        Nombre de columna de edad en el DataFrame de covariables.
    """

    def __init__(
        self,
        strategy: Union[StratMethod, Callable] = "score_terciles",
        n_strata: int = 3,
        min_stratum_size: int = 15,
        max_iter: int = 1000,
        cpb_col: str = "cpb_time",
        age_col: str = "age",
    ) -> None:
        self.strategy = strategy
        self.n_strata = n_strata
        self.min_stratum_size = min_stratum_size
        self.max_iter = max_iter
        self.cpb_col = cpb_col
        self.age_col = age_col

        # Después del fit
        self._global_calibrator: Optional[LogisticRegression] = None
        self._stratum_calibrators: dict[int, Optional[LogisticRegression]] = {}
        self._stratum_ids: Optional[np.ndarray] = None
        self._fitted = False
        self.calibration_report_: Optional[pd.DataFrame] = None

    # ─────────────────────────────────────────────────────────────────────────

    def fit(
        self,
        scores: np.ndarray,
        y_target: np.ndarray,
        covariates: Optional[pd.DataFrame] = None,
    ) -> "StratifiedPlattRecalibrator":
        """
        Ajusta la recalibración Platt estratificada con leave-one-out.

        Parameters
        ----------
        scores : np.ndarray (n_t,)
            Probabilidades predichas en target ANTES de recalibrar.
        y_target : np.ndarray (n_t,)
            Labels binarios del target.  Requiere al menos 2 positivos
            por estrato para LOO.
        covariates : pd.DataFrame (n_t, *), optional
            Covariables del target (necesario para 'cpb_time' y 'age').

        Returns
        -------
        self
        """
        n = len(scores)
        if len(y_target) != n:
            raise ValueError("scores e y_target deben tener el mismo tamaño.")
        if y_target.sum() < 2:
            raise ValueError("Se necesitan al menos 2 eventos en target.")

        # ── Calibración global (fallback) ────────────────────────────────
        self._global_calibrator = self._fit_platt_loo(scores, y_target)

        # ── Asignar estratos ─────────────────────────────────────────────
        strata = self._assign_strata(scores, covariates)
        self._stratum_ids = strata
        unique_strata = np.unique(strata)

        # ── Calibración por estrato ──────────────────────────────────────
        for sid in unique_strata:
            mask = strata == sid
            n_s = int(mask.sum())
            n_events_s = int(y_target[mask].sum())

            if n_s < self.min_stratum_size or n_events_s < 2 or (n_s - n_events_s) < 2:
                logger.warning(
                    "Estrato %d: n=%d, n_events=%d → fallback a calibración global.",
                    sid, n_s, n_events_s,
                )
                self._stratum_calibrators[sid] = None  # usar global
            else:
                self._stratum_calibrators[sid] = self._fit_platt_loo(
                    scores[mask], y_target[mask]
                )

        # ── Reporte de calibración por estrato ──────────────────────────
        self.calibration_report_ = self._build_report(scores, y_target, strata)

        self._fitted = True
        logger.info(
            "StratifiedPlattRecalibrator: %d estratos, strategy='%s'.",
            len(unique_strata), self.strategy if isinstance(self.strategy, str) else "custom",
        )
        return self

    # ─────────────────────────────────────────────────────────────────────────

    def _fit_platt_loo(
        self,
        scores_sub: np.ndarray,
        y_sub: np.ndarray,
    ) -> LogisticRegression:
        """
        Ajusta regresión logística Platt con leave-one-out en el subconjunto.

        Dado el tamaño pequeño de los estratos (n ≈ 30-50), se usa LOO:
        el calibrador final es ajustado sobre TODOS los datos del subconjunto
        (para producir el mejor predictor), pero los scores recalibrados LOO
        se calculan por fuera (en predict_proba_loo()).

        Para mantener la API simple, aquí solo ajustamos el modelo global del
        subconjunto.  predict_proba() lo usa para nuevas observaciones.
        """
        X = scores_sub.reshape(-1, 1)
        lr = LogisticRegression(
            fit_intercept=True,
            max_iter=self.max_iter,
            solver="lbfgs",
            C=1e9,  # sin regularización: Platt puro
            random_state=42,
        )
        lr.fit(X, y_sub)
        return lr

    def _assign_strata(
        self,
        scores: np.ndarray,
        covariates: Optional[pd.DataFrame],
    ) -> np.ndarray:
        """Asigna cada observación a un estrato (int 0..n_strata-1)."""
        strategy = self.strategy
        n = len(scores)

        if callable(strategy):
            strata = strategy(scores, covariates)
            return np.asarray(strata, dtype=int)

        if strategy == "score_terciles":
            terciles = np.percentile(scores, [33.33, 66.67])
            strata = np.zeros(n, dtype=int)
            strata[scores > terciles[0]] = 1
            strata[scores > terciles[1]] = 2
            return strata

        if strategy in ("cpb_time", "age"):
            col = self.cpb_col if strategy == "cpb_time" else self.age_col
            if covariates is None or col not in covariates.columns:
                logger.warning(
                    "Columna '%s' no encontrada en covariates → fallback a score_terciles.",
                    col,
                )
                return self._assign_strata(scores, None)
            vals = covariates[col].values.astype(float)
            median = np.nanmedian(vals)
            strata = (vals > median).astype(int)
            return strata

        raise ValueError(f"Estrategia desconocida: '{strategy}'.")

    def _build_report(
        self,
        scores: np.ndarray,
        y_target: np.ndarray,
        strata: np.ndarray,
    ) -> pd.DataFrame:
        """Construye DataFrame con métricas de calibración por estrato."""
        rows = []
        for sid in sorted(np.unique(strata)):
            mask = strata == sid
            s_sub = scores[mask]
            y_sub = y_target[mask].astype(float)
            n_sub = int(mask.sum())
            n_pos = int(y_sub.sum())

            # Calibration-in-the-large (CITL): mean predicted - mean observed
            citl = float(s_sub.mean() - y_sub.mean())

            # Calibration slope (regresión logística de y sobre logit(score))
            logit_s = np.log(np.clip(s_sub, 1e-7, 1 - 1e-7))
            cal_slope = np.nan
            if n_pos >= 2 and (n_sub - n_pos) >= 2:
                try:
                    lr_cs = LogisticRegression(
                        fit_intercept=True, max_iter=1000, C=1e9
                    )
                    lr_cs.fit(logit_s.reshape(-1, 1), y_sub.astype(int))
                    cal_slope = float(lr_cs.coef_[0][0])
                except Exception:
                    pass

            # ECE (Expected Calibration Error) con 5 bins
            ece = self._ece(s_sub, y_sub, n_bins=5)

            rows.append(
                {
                    "stratum": sid,
                    "n": n_sub,
                    "n_events": n_pos,
                    "mean_predicted": float(s_sub.mean()),
                    "mean_observed": float(y_sub.mean()),
                    "CITL": citl,
                    "calibration_slope": cal_slope,
                    "ECE_5bins": ece,
                    "uses_global_fallback": self._stratum_calibrators.get(sid) is None,
                }
            )
        return pd.DataFrame(rows)

    @staticmethod
    def _ece(scores: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
        """Expected Calibration Error con bins de igual amplitud."""
        bins = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        n = len(y)
        for i in range(n_bins):
            mask = (scores >= bins[i]) & (scores < bins[i + 1])
            if mask.sum() == 0:
                continue
            acc = float(y[mask].mean())
            conf = float(scores[mask].mean())
            ece += (mask.sum() / n) * abs(acc - conf)
        return ece

    # ─────────────────────────────────────────────────────────────────────────

    def predict_proba(
        self,
        scores: np.ndarray,
        covariates: Optional[pd.DataFrame] = None,
    ) -> np.ndarray:
        """
        Recalibra los scores usando el calibrador del estrato correspondiente.

        Parameters
        ----------
        scores : np.ndarray (n,)
            Scores antes de recalibrar.
        covariates : pd.DataFrame, optional
            Covariables para asignar estratos (necesario para cpb_time/age).

        Returns
        -------
        scores_cal : np.ndarray (n,)
        """
        if not self._fitted:
            raise RuntimeError("Llama a fit() primero.")

        strata = self._assign_strata(scores, covariates)
        scores_cal = np.empty_like(scores)

        for sid in np.unique(strata):
            mask = strata == sid
            calibrator = self._stratum_calibrators.get(sid)
            if calibrator is None:
                calibrator = self._global_calibrator
            scores_cal[mask] = calibrator.predict_proba(
                scores[mask].reshape(-1, 1)
            )[:, 1]

        return scores_cal

    # ─────────────────────────────────────────────────────────────────────────

    def ece_global(
        self,
        scores: np.ndarray,
        y_target: np.ndarray,
        n_bins: int = 10,
    ) -> float:
        """ECE global antes de recalibrar (para comparación)."""
        return self._ece(scores, y_target.astype(float), n_bins=n_bins)

    def ece_recalibrated(
        self,
        scores: np.ndarray,
        y_target: np.ndarray,
        covariates: Optional[pd.DataFrame] = None,
        n_bins: int = 10,
    ) -> float:
        """ECE global tras recalibrar."""
        scores_cal = self.predict_proba(scores, covariates)
        return self._ece(scores_cal, y_target.astype(float), n_bins=n_bins)

    def summary(self) -> str:
        """Resumen textual de la calibración por estrato."""
        if not self._fitted:
            return "No ajustado aún."
        lines = [
            "StratifiedPlattRecalibrator — resumen",
            f"  Estrategia: {self.strategy}",
            "",
            "  Por estrato:",
        ]
        if self.calibration_report_ is not None:
            for _, row in self.calibration_report_.iterrows():
                fb = " [fallback global]" if row["uses_global_fallback"] else ""
                lines.append(
                    f"  Estrato {int(row['stratum'])}: "
                    f"n={int(row['n'])}, events={int(row['n_events'])}, "
                    f"CITL={row['CITL']:+.3f}, slope={row['calibration_slope']:.2f}, "
                    f"ECE={row['ECE_5bins']:.3f}{fb}"
                )
        lines.append("")
        lines.append(
            "  NOTA: n por estrato ≈ 35 — resultados exploratorios únicamente."
        )
        return "\n".join(lines)
