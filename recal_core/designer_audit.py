"""
recal_core.designer_audit
=====================
Trazabilidad completa de las decisiones del Designer.

Cada decisión queda registrada como un DesignerDecision con:
- El paso (mask_selection, pca_coral_k, calibration_method…)
- El criterio usado
- Las alternativas evaluadas (con métricas)
- La decisión final con justificación

El DesignerAuditTrail acumula todas las decisiones y se puede serializar.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class AlternativeChoice:
    """Una alternativa evaluada durante la selección."""
    choice: Any
    metric_name: str
    metric_value: float | None
    selected: bool

    def to_dict(self) -> dict:
        return {
            "choice": self.choice,
            "metric_name": self.metric_name,
            "metric_value": (
                round(float(self.metric_value), 6)
                if self.metric_value is not None and self.metric_value == self.metric_value
                else None
            ),
            "selected": self.selected,
        }


@dataclass
class DesignerDecision:
    """
    Registro de una decisión del Designer.

    Attributes
    ----------
    step : str
        Nombre del paso: mask_activate, mask_n, mask_features, quantile,
        woe, pca_coral_activate, pca_coral_k, calibration_activate,
        calibration_method.
    criterion : str
        Criterio de selección usado.
    alternatives : list[AlternativeChoice]
        Todas las opciones evaluadas, con la elegida marcada.
    final_choice : Any
        La decisión tomada.
    justification : str
        Texto legible explicando por qué se tomó esta decisión.
    """
    step: str
    criterion: str
    alternatives: list = field(default_factory=list)
    final_choice: Any = None
    justification: str = ""

    def to_dict(self) -> dict:
        return {
            "step": self.step,
            "criterion": self.criterion,
            "alternatives": [
                a.to_dict() if isinstance(a, AlternativeChoice) else a
                for a in self.alternatives
            ],
            "final_choice": self.final_choice,
            "justification": self.justification,
        }


class DesignerAuditTrail:
    """
    Acumula todas las decisiones del Designer para un run.

    Expuesto como ``pipeline.audit``.

    Uso:
        audit = DesignerAuditTrail()
        audit.record(DesignerDecision(step="mask_n", ...))
        audit.to_dict()   # serializable
    """

    def __init__(self) -> None:
        self._decisions: list[DesignerDecision] = []

    def record(self, decision: DesignerDecision) -> None:
        """Registra una decisión."""
        self._decisions.append(decision)

    @property
    def decisions(self) -> list[DesignerDecision]:
        return list(self._decisions)

    def get(self, step: str) -> DesignerDecision | None:
        """Devuelve la primera decisión con el step dado, o None."""
        for d in self._decisions:
            if d.step == step:
                return d
        return None

    def to_dict(self) -> list[dict]:
        return [d.to_dict() for d in self._decisions]

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def summary(self) -> str:
        lines = ["=== Designer Audit Trail ==="]
        for d in self._decisions:
            n_alt = len(d.alternatives)
            lines.append(
                f"  [{d.step}] chosen={d.final_choice!r}  "
                f"({n_alt} alternatives)  — {d.justification[:80]}"
            )
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"DesignerAuditTrail(n_decisions={len(self._decisions)})"
