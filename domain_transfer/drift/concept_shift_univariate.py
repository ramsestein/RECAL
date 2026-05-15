"""
domain_transfer.drift.concept_shift_univariate
================================================
UnivariateConceptShiftDiagnoser: regresión logística pooled con interacción
de cohorte para detectar concept shift univariado per-feature.

Modelo por feature j
---------------------
P(y=1 | x_j, c) = sigmoid(β₀ + β₁·x_j + β₂·c + β₃·x_j·c)

donde c=0 para SNUH (source) y c=1 para Clínic (target).

- β₁: pendiente en source (relación feature→outcome en SNUH)
- β₃: **diferencia de pendiente** en target respecto a source
  → H₀: β₃ = 0 (relación estable entre cohortes)
  → β₃ ≠ 0 ⟹ concept shift univariado significativo

Corrección múltiple
--------------------
Los p-valores de β₃ se corrigen con Benjamini-Hochberg (BH) sobre los 107
tests (uno por feature).  Se reportan los q-valores ajustados.

Limitaciones esperadas
-----------------------
Con solo 29 eventos en target (Clínic), los β₃ de la mayoría de features no
alcanzarán q<0.05 tras BH.  El análisis es principalmente exploratorio para
el ranking de |β₃| y para validar la taxonomía CONCEPT_RELATIONAL de la
descomposición V.

Requiere labels target (y_t) → vive en ``drift/``, no en ``select/``.
"""

from __future__ import annotations

import logging
import warnings
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import norm as scipy_norm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────

def _bh_correction(p_values: np.ndarray) -> np.ndarray:
    """
    Benjamini-Hochberg FDR correction.

    Parameters
    ----------
    p_values : np.ndarray, shape (m,)

    Returns
    -------
    q_values : np.ndarray, shape (m,)
        Adjusted p-values (q-values), same order as input.
    """
    m = len(p_values)
    order = np.argsort(p_values)
    ranked = np.empty(m, dtype=int)
    ranked[order] = np.arange(1, m + 1)
    q = p_values * m / ranked
    # Enforce monotonicity (cumulative min from the right)
    q_sorted = q[order]
    for i in range(m - 2, -1, -1):
        q_sorted[i] = min(q_sorted[i], q_sorted[i + 1])
    q[order] = q_sorted
    return np.clip(q, 0.0, 1.0)


class UnivariateConceptShiftDiagnoser:
    """
    Detecta concept shift univariado per-feature mediante regresión logística
    con interacción de cohorte.

    Parameters
    ----------
    alpha : float
        Nivel de significancia nominal para reportar features significativas
        (sobre los q-valores BH).  Default 0.05.
    max_iter : int
        Iteraciones máximas del solver de LogisticRegression.  Default 1000.
    ci_level : float
        Nivel de confianza para los intervalos de β₁ y β₃ (z score).
        Default 0.95 → z=1.96.
    min_target_nonnan : int
        Número mínimo de observaciones target no-NaN en una feature para que
        sea incluida en el análisis.  Features por debajo del umbral reciben
        p_value=1.0 y son marcadas como 'insufficient_data'.
    """

    def __init__(
        self,
        alpha: float = 0.05,
        max_iter: int = 1000,
        ci_level: float = 0.95,
        min_target_nonnan: int = 10,
    ) -> None:
        self.alpha = alpha
        self.max_iter = max_iter
        self.ci_level = ci_level
        self.min_target_nonnan = min_target_nonnan
        self.results_: Optional[pd.DataFrame] = None

    # ─────────────────────────────────────────────────────────────────────────

    def fit(
        self,
        X_s: np.ndarray,
        X_t: np.ndarray,
        y_s: np.ndarray,
        y_t: np.ndarray,
        schema: list[str],
        drift_type_v: Optional[list[str]] = None,
    ) -> "UnivariateConceptShiftDiagnoser":
        """
        Ajusta la regresión logística con interacción de cohorte para cada feature.

        Parameters
        ----------
        X_s : np.ndarray (n_s, p)
            Feature matrix source (puede contener NaN).
        X_t : np.ndarray (n_t, p)
            Feature matrix target (puede contener NaN).
        y_s : np.ndarray (n_s,)
            Labels binarios source.
        y_t : np.ndarray (n_t,)
            Labels binarios target.
        schema : list[str]
            Nombres de las p features.
        drift_type_v : list[str], optional
            Categoría de drift de la descomposición V (para cross-referencia).
            Si None, se omite la columna.

        Returns
        -------
        self
        """
        p = X_s.shape[1]
        assert p == len(schema), "schema debe tener p elementos"

        z_crit = scipy_norm.ppf(1 - (1 - self.ci_level) / 2)
        rows = []

        for j, feat in enumerate(schema):
            row = self._fit_one_feature(
                j, feat, X_s, X_t, y_s, y_t, z_crit
            )
            rows.append(row)

        df = pd.DataFrame(rows)

        # ── BH correction ─────────────────────────────────────────────────
        p_vals = df["p_value"].values
        df["q_value_BH"] = _bh_correction(p_vals)

        # ── Cross-reference con taxonomía V ───────────────────────────────
        if drift_type_v is not None:
            df["drift_type_v"] = drift_type_v
        else:
            df["drift_type_v"] = "unknown"

        # ── Significancia ─────────────────────────────────────────────────
        df["significant_BH"] = df["q_value_BH"] < self.alpha

        self.results_ = df

        # Resumen en log
        n_sig = df["significant_BH"].sum()
        n_flip = (df["flip_of_sign"] == True).sum()  # noqa: E712
        logger.info(
            "UnivariateConceptShift: %d/%d features q<%.2f tras BH. "
            "%d flip-of-sign.",
            n_sig, p, self.alpha, n_flip,
        )
        if n_sig == 0:
            logger.warning(
                "Ninguna feature alcanza q<%.2f tras BH. "
                "Potencia estadística limitada por n_events_target=%d.",
                self.alpha, int(y_t.sum()),
            )

        return self

    # ─────────────────────────────────────────────────────────────────────────

    def _fit_one_feature(
        self,
        j: int,
        feat: str,
        X_s: np.ndarray,
        X_t: np.ndarray,
        y_s: np.ndarray,
        y_t: np.ndarray,
        z_crit: float,
    ) -> dict:
        """Ajusta el modelo de interacción para la feature j y devuelve un dict."""

        xs_j = X_s[:, j].astype(float)
        xt_j = X_t[:, j].astype(float)

        # Filtrar NaN
        valid_s = ~np.isnan(xs_j)
        valid_t = ~np.isnan(xt_j)
        n_s_obs = int(valid_s.sum())
        n_t_obs = int(valid_t.sum())

        _nan_row = {
            "feature": feat,
            "n_source_obs": n_s_obs,
            "n_target_obs": n_t_obs,
            "beta1_source": np.nan,
            "beta1_ci_low": np.nan,
            "beta1_ci_high": np.nan,
            "beta3_interaction": np.nan,
            "beta3_ci_low": np.nan,
            "beta3_ci_high": np.nan,
            "p_value": 1.0,
            "flip_of_sign": False,
            "status": "insufficient_data",
        }

        # Datos insuficientes en target
        if n_t_obs < self.min_target_nonnan:
            return _nan_row
        # Necesitamos al menos 2 clases en cada cohorte para ajustar
        if len(np.unique(y_s[valid_s])) < 2 or len(np.unique(y_t[valid_t])) < 2:
            return _nan_row

        # ── Pool source + target ───────────────────────────────────────────
        x_all = np.concatenate([xs_j[valid_s], xt_j[valid_t]])
        c_all = np.concatenate([
            np.zeros(n_s_obs, dtype=float),
            np.ones(n_t_obs, dtype=float),
        ])
        y_all = np.concatenate([y_s[valid_s], y_t[valid_t]]).astype(float)

        # Escalar x para estabilidad numérica
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x_all.reshape(-1, 1)).ravel()

        # Diseño: [intercept x c x*c] — sklearn sin intercept automático
        # X_design = [x, c, x*c]
        X_design = np.column_stack([x_scaled, c_all, x_scaled * c_all])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                lr = LogisticRegression(
                    fit_intercept=True,
                    max_iter=self.max_iter,
                    solver="lbfgs",
                    C=1.0,  # regularización leve para estabilidad
                    random_state=42,
                )
                lr.fit(X_design, y_all)
            except Exception as exc:
                logger.debug("Feature %s: convergence error: %s", feat, exc)
                return {**_nan_row, "status": "convergence_error"}

        coef = lr.coef_[0]   # [β₁, β₂, β₃]
        beta1 = float(coef[0])
        beta3 = float(coef[2])

        # ── Errores estándar via hessiana ──────────────────────────────────
        # H = X^T W X donde W = diag(p*(1-p)), p = predicted proba
        proba = lr.predict_proba(X_design)[:, 1]
        W = proba * (1 - proba) + 1e-12
        # Diseño aumentado con intercept (para que coincida con sklearn)
        X_aug = np.column_stack([np.ones(len(X_design)), X_design])
        H = (X_aug * W[:, np.newaxis]).T @ X_aug
        try:
            cov_mat = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            return {**_nan_row, "status": "singular_hessian"}

        se = np.sqrt(np.clip(np.diag(cov_mat), 0, None))
        # Índices en H: 0=intercept, 1=beta1, 2=beta2, 3=beta3
        se_beta1 = float(se[1]) if len(se) > 1 else np.nan
        se_beta3 = float(se[3]) if len(se) > 3 else np.nan

        # CI para β₁
        beta1_ci_low = beta1 - z_crit * se_beta1
        beta1_ci_high = beta1 + z_crit * se_beta1

        # CI y p-valor (test Wald) para β₃
        beta3_ci_low = beta3 - z_crit * se_beta3
        beta3_ci_high = beta3 + z_crit * se_beta3

        # p = 2 * (1 - Φ(|β₃| / SE))
        if np.isnan(se_beta3) or se_beta3 < 1e-12:
            p_val = 1.0
        else:
            z_stat = abs(beta3) / se_beta3
            p_val = float(2 * (1 - scipy_norm.cdf(z_stat)))

        # Flip of sign: signo de β₁ opuesto a signo de (β₁ + β₃)
        beta1_target = beta1 + beta3
        flip = bool(np.sign(beta1) != np.sign(beta1_target) and abs(beta1) > 1e-9)

        return {
            "feature": feat,
            "n_source_obs": n_s_obs,
            "n_target_obs": n_t_obs,
            "beta1_source": beta1,
            "beta1_ci_low": beta1_ci_low,
            "beta1_ci_high": beta1_ci_high,
            "beta3_interaction": beta3,
            "beta3_ci_low": beta3_ci_low,
            "beta3_ci_high": beta3_ci_high,
            "p_value": p_val,
            "flip_of_sign": flip,
            "status": "ok",
        }

    # ─────────────────────────────────────────────────────────────────────────

    def top_concept_shift(self, n: int = 20, q_threshold: float = 1.0) -> pd.DataFrame:
        """
        Devuelve las n features con mayor |β₃| ordenadas por |β₃|.

        Parameters
        ----------
        n : int
            Número de features a devolver.
        q_threshold : float
            Filtrar por q_value_BH < q_threshold.  Por defecto 1.0 (sin filtro).
        """
        if self.results_ is None:
            raise RuntimeError("Llama a fit() primero.")
        df = self.results_[self.results_["status"] == "ok"].copy()
        if q_threshold < 1.0:
            df = df[df["q_value_BH"] < q_threshold]
        df["abs_beta3"] = df["beta3_interaction"].abs()
        return (
            df.sort_values("abs_beta3", ascending=False)
            .head(n)
            .drop(columns=["abs_beta3"])
            .reset_index(drop=True)
        )

    def summary(self) -> str:
        """Resumen en texto del diagnóstico."""
        if self.results_ is None:
            return "No ajustado aún."
        df = self.results_
        n_total = len(df)
        n_ok = (df["status"] == "ok").sum()
        n_sig = df["significant_BH"].sum()
        n_flip = (df["flip_of_sign"] == True).sum()  # noqa: E712
        n_events_t = df["n_target_obs"].sum()  # aproximado
        lines = [
            f"UnivariateConceptShiftDiagnoser — resumen",
            f"  Features analizadas:       {n_ok}/{n_total} (status='ok')",
            f"  Features q<{self.alpha} (BH):     {n_sig} de {n_ok}",
            f"  Features flip-of-sign:     {n_flip}",
            f"  Nota: potencia estadística limitada por n_events_target ≈ "
            f"{int(df['n_target_obs'].max())} obs target (muchas features "
            f"con datos limitados en Clínic).",
        ]
        return "\n".join(lines)
