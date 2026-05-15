"""adapt.reporter — Tablas, figuras y reporte HTML."""

from adapt.reporter.tables import (
    make_global_table,
    make_decisions_table,
    make_features_table,
    make_eval_table,
)
from adapt.reporter.html_report import generate_html_report

__all__ = [
    "make_global_table",
    "make_decisions_table",
    "make_features_table",
    "make_eval_table",
    "generate_html_report",
]
