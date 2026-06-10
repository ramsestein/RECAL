"""recal_core.reporter — Tablas, figuras y reporte HTML."""

from recal_core.reporter.html_report import generate_html_report
from recal_core.reporter.tables import (
    make_decisions_table,
    make_eval_table,
    make_features_table,
    make_global_table,
)

__all__ = [
    "make_global_table",
    "make_decisions_table",
    "make_features_table",
    "make_eval_table",
    "generate_html_report",
]
