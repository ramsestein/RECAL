"""
RECAL — Meta-framework auto-adapter sobre recal.

Tagline: "LoRA-style wrapper sin fine-tuning, explicable y auto-configurado".

RECAL es un meta-framework que orquesta los componentes de recal
en una pipeline auto-configurada a partir del par (source, target):

    1. Profiler   — diagnostica el par cohorte-cohorte
    2. Designer   — decide qué componentes activar con reglas determinísticas
    3. AutoAdapter — ejecuta la pipeline elegida
    4. Reporter   — genera reportes con métricas y recomendaciones

Uso rápido
----------
    from recal_core import AutoAdapter
    adapter = AutoAdapter(source_model, X_source, y_source, schema)
    result = adapter.auto_adapt(X_target, y_target)

Componentes
-----------
    recal_core.profiler.Profiler
    recal_core.designer.ComponentSelector
    recal_core.pipeline.AutoAdapter
    recal_core.reporter.HTMLReporter
"""

from recal_core.pipeline.auto_adapter import AutoAdapter

__version__ = "0.1.0"
__all__ = ["AutoAdapter"]
