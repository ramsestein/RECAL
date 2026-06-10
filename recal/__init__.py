"""
recal
===============
Modular pipeline for clinical domain adaptation of predictive models.

Packages:
  data         — CohortLoaders, schema, CohortPair
  model        — ModelWrapper abstractions and XGBoost implementation
  align        — Domain alignment algorithms (CORAL, PCA-CORAL, AdaBN, OT, …)
  select       — Feature selectors (SHAP, L-base, combined, flip-of-sign, meta)
  drift        — Drift decomposition, statistical tests, shift characterisation
  eval         — AUROC/calibration metrics with BCa bootstrap CI
  calibration  — Post-hoc calibration (Platt, isotonic)
  cli          — Typer CLI entry point
"""

__version__ = "0.1.0"
