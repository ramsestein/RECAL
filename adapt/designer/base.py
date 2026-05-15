"""
adapt.designer.base
====================
Dataclass AdapterConfig — configuración de la pipeline ADAPT.

Cada decisión queda registrada en el campo ``rationale`` con la explicación
textual de la regla aplicada (clave = nombre del componente).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AdapterConfig:
    """
    Configuración completa de la pipeline ADAPT.

    Attributes
    ----------
    apply_mask : bool
        Si True, enmascarar las peores N features.
    mask_features : list[str]
        Nombres de las features a enmascarar.
    mask_n : int
        Cardinalidad de la máscara.
    mask_selection_method : str
        Método de selección del N: 'elbow_source' | 'cv_pseudo_target' | 'fixed'

    apply_quantile : bool
        Si True, aplicar QuantileTransform selectivo.
    quantile_features : list[str]
        Features elegibles para QT.
    quantile_output_distribution : str
        Distribución de salida: 'uniform' | 'normal'

    apply_woe : bool
        Si True, aplicar WOE encoding selectivo.
    woe_features : list[str]
        Features elegibles para WOE.
    woe_n_bins : int
        Número de bins para discretización.

    apply_pca_coral : bool
        Si True, aplicar PCA-CORAL.
    pca_coral_k : int
        Número de componentes PCA-CORAL.
    pca_coral_k_selection_method : str
        Método de selección de k: 'cv_source' | 'sqrt_n_target' | 'fixed'

    apply_calibration : bool
        Si True, recalibrar las probabilidades.
    calibration_method : str
        Método de calibración: 'platt_loo' | 'platt_stratified' | 'isotonic_loo'
    calibration_strata_fn : str, optional
        Función de estratificación para calibración estratificada.

    rationale : dict[str, str]
        Justificación por componente. Clave = nombre del componente,
        valor = explicación de la decisión tomada.
    """

    # Máscara
    apply_mask: bool = False
    mask_features: list = field(default_factory=list)
    mask_n: int = 0
    mask_selection_method: str = "elbow_source"

    # QuantileTransform
    apply_quantile: bool = False
    quantile_features: list = field(default_factory=list)
    quantile_output_distribution: str = "uniform"

    # WOE
    apply_woe: bool = False
    woe_features: list = field(default_factory=list)
    woe_n_bins: int = 10

    # PCA-CORAL
    apply_pca_coral: bool = True
    pca_coral_k: int = 5
    pca_coral_k_selection_method: str = "sqrt_n_target"

    # Calibración
    apply_calibration: bool = True
    calibration_method: str = "platt_loo"
    calibration_strata_fn: Optional[str] = None

    # Justificaciones
    rationale: dict = field(default_factory=dict)

    def summary(self) -> str:
        """Resumen legible de la configuración."""
        lines = ["AdapterConfig:"]
        lines.append(f"  mask:       {self.apply_mask} (N={self.mask_n}, method={self.mask_selection_method})")
        lines.append(f"  quantile:   {self.apply_quantile} ({len(self.quantile_features)} features, dist={self.quantile_output_distribution})")
        lines.append(f"  woe:        {self.apply_woe} ({len(self.woe_features)} features, bins={self.woe_n_bins})")
        lines.append(f"  pca_coral:  {self.apply_pca_coral} (k={self.pca_coral_k}, method={self.pca_coral_k_selection_method})")
        lines.append(f"  calibration:{self.apply_calibration} ({self.calibration_method})")
        if self.rationale:
            lines.append("  Rationale:")
            for key, reason in self.rationale.items():
                lines.append(f"    [{key}] {reason}")
        return "\n".join(lines)
