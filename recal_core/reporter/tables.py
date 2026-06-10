"""
recal_core.reporter.tables
======================
Markdown tables for the RECAL report.

Tables produced:
- Global pair metrics (source, target)
- Source-domain reference performance of the original model
- Designer decisions: component -> decision + rationale
- Top-N features by combined score / quadrant
- Evaluation metrics before / after the pipeline
"""

from __future__ import annotations

import numpy as np

from recal_core.designer.base import AdapterConfig
from recal_core.profiler.base import DriftProfile


def _fmt(x, fmt=".4f") -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return format(x, fmt)


def make_global_table(profile: DriftProfile) -> str:
    """Markdown table with global metrics for the source/target pair."""
    rows = [
        ("n_source", f"{profile.n_source_obs:,}", "Source observations"),
        ("n_target", f"{profile.n_target_obs:,}", "Target observations"),
        ("events_source", f"{profile.n_source_events} ({100*profile.prevalence_source:.1f}%)", "Positive events in source"),
        ("events_target", f"{profile.n_target_events} ({100*profile.prevalence_target:.1f}%)", "Positive events in target"),
        ("prevalence_shift_p", _fmt(profile.prevalence_shift_pvalue, ".3f"), "Fisher test: prevalence difference"),
        ("mmd2", _fmt(profile.mmd2_source_target, ".4f"), "MMD² source → target (RBF)"),
        ("mmd2_p", _fmt(profile.mmd2_pvalue, ".3f"), "MMD² permutation p-value"),
        ("pca_var_explained", _fmt(profile.pca_variance_explained[-1] if profile.pca_variance_explained else None, ".3f"), "Cumulative variance, 5 source PCs"),
        ("baseline_auroc", _fmt(profile.baseline_auroc, ".4f"), "Source model AUROC on target (unadapted)"),
        ("baseline_auroc_ci", f"[{_fmt(profile.baseline_auroc_ci_low, '.4f')}, {_fmt(profile.baseline_auroc_ci_high, '.4f')}]", "Bootstrap 95% CI (500 replicates)"),
        ("calibration_slope", _fmt(profile.baseline_calibration_slope, ".2f"), "Platt calibration slope"),
        ("ece", _fmt(profile.baseline_ece, ".4f"), "Expected Calibration Error (10 bins)"),
        ("citl", _fmt(profile.baseline_citl, ".4f"), "Calibration-in-the-Large"),
    ]
    header = "| Metric | Value | Description |\n|---|---|---|\n"
    body = "\n".join(f"| `{k}` | {v} | {d} |" for k, v, d in rows)
    return header + body


def make_source_table(
    source_metrics: dict | None,
    n_source: int | None = None,
    n_source_events: int | None = None,
    source_name: str = "Source",
) -> str:
    """Markdown table reporting the original model performance on its source domain."""
    if source_metrics is None:
        return (
            "| Metric | Value |\n|---|---|\n"
            "| Status | Source-domain metrics unavailable |\n"
        )

    prev = "—"
    if n_source and n_source_events:
        prev = f"{(n_source_events / n_source * 100):.1f}%"

    rows = [
        ("Dataset", source_name),
        ("n samples", f"{n_source:,}" if n_source else "—"),
        ("n events", f"{n_source_events:,}" if n_source_events is not None else "—"),
        ("Prevalence", prev),
        ("AUROC (original model)", _fmt(source_metrics.get("auroc"), ".4f")),
        ("Precision @ Youden", _fmt(source_metrics.get("precision"), ".3f")),
        ("Recall @ Youden", _fmt(source_metrics.get("recall"), ".3f")),
        ("F1 @ Youden", _fmt(source_metrics.get("f1"), ".3f")),
        ("Decision threshold", _fmt(source_metrics.get("threshold"), ".3f")),
    ]
    header = "| Metric | Value |\n|---|---|\n"
    body = "\n".join(f"| {k} | {v} |" for k, v in rows)
    return header + body


def make_decisions_table(config: AdapterConfig) -> str:
    """Markdown table with Designer decisions and their rationale."""
    rows = []

    # Mask
    rows.append((
        "Feature mask",
        "✓ Enabled" if config.apply_mask else "✗ Disabled",
        f"N={config.mask_n}" if config.apply_mask else "—",
        config.rationale.get("mask_activate", ""),
    ))
    if config.apply_mask:
        feats = ", ".join(config.mask_features[:5])
        if len(config.mask_features) > 5:
            feats += f" (+{len(config.mask_features)-5} more)"
        rows.append((
            "  Masked features",
            feats,
            config.mask_selection_method,
            config.rationale.get("mask_n", ""),
        ))

    # QT
    rows.append((
        "QuantileTransform",
        "✓ Enabled" if config.apply_quantile else "✗ Disabled",
        f"{len(config.quantile_features)} features" if config.apply_quantile else "—",
        config.rationale.get("quantile", ""),
    ))

    # WOE
    rows.append((
        "WOE encoding",
        "✓ Enabled" if config.apply_woe else "✗ Disabled",
        f"{len(config.woe_features)} features" if config.apply_woe else "—",
        config.rationale.get("woe", ""),
    ))

    # PCA-CORAL
    rows.append((
        "PCA-CORAL",
        "✓ Enabled" if config.apply_pca_coral else "✗ Disabled",
        f"k={config.pca_coral_k}" if config.apply_pca_coral else "—",
        config.rationale.get("pca_coral_k", config.rationale.get("pca_coral_activate", "")),
    ))

    # Calibration
    rows.append((
        "Calibration",
        "✓ Enabled" if config.apply_calibration else "✗ Disabled",
        config.calibration_method if config.apply_calibration else "—",
        config.rationale.get("calibration_method", config.rationale.get("calibration_activate", "")),
    ))

    header = "| Component | Decision | Parameter | Rationale |\n|---|---|---|---|\n"
    body = "\n".join(f"| {c} | {d} | {p} | {j} |" for c, d, p, j in rows)
    return header + body


def make_features_table(
    profile: DriftProfile,
    n_top: int = 20,
    sort_by: str = "combined_score",
) -> str:
    """Markdown table with the top-N features by combined score (ascending)."""
    features = sorted(profile.features, key=lambda f: getattr(f, sort_by, 0))
    features = features[:n_top]

    header = (
        "| Feature | Domain | Drift type | L_base | SHAP | Combined | Quadrant | "
        "CV_target | Near-const | Flip |\n"
        "|---|---|---|---|---|---|---|---|---|---|\n"
    )
    rows = []
    for f in features:
        rows.append(
            f"| `{f.name}` | {f.domain} | {f.drift_type_v} | "
            f"{_fmt(f.lbase_score, '.3f')} | {_fmt(f.shap_importance, '.4f')} | "
            f"{_fmt(f.combined_score, '.4f')} | {f.quadrant} | "
            f"{_fmt(f.cv_target, '.2f')} | {'✓' if f.near_constant_target else ''} | "
            f"{'✓' if f.flip_of_sign else ''} |"
        )
    return header + "\n".join(rows)


def make_eval_table(
    auroc_before: float,
    auroc_after: float,
    auroc_ci_before: tuple | None = None,
    auroc_ci_after: tuple | None = None,
    slope_before: float | None = None,
    slope_after: float | None = None,
    ece_before: float | None = None,
    ece_after: float | None = None,
    n_target: int | None = None,
    n_events: int | None = None,
    auroc_source: float | None = None,
    auroc_ci_source: tuple | None = None,
    slope_source: float | None = None,
    ece_source: float | None = None,
) -> str:
    """Markdown table comparing metrics before/after the pipeline (with source as reference)."""

    def ci_str(ci):
        if ci is None:
            return "—"
        return f"[{_fmt(ci[0], '.4f')}, {_fmt(ci[1], '.4f')}]"

    rows = [
        ("n target", "—", f"{n_target}" if n_target else "—", "—", "—"),
        ("n events", "—", f"{n_events}" if n_events else "—", "—", "—"),
        ("AUROC", _fmt(auroc_source, ".4f"), _fmt(auroc_before, ".4f"),
         _fmt(auroc_after, ".4f"), "higher is better"),
        ("AUROC 95% CI", ci_str(auroc_ci_source), ci_str(auroc_ci_before), ci_str(auroc_ci_after), "Bootstrap 500"),
        ("Calibration slope", _fmt(slope_source, ".2f"), _fmt(slope_before, ".2f"), _fmt(slope_after, ".2f"), "Ideal = 1.0"),
        ("ECE", _fmt(ece_source, ".4f"), _fmt(ece_before, ".4f"), _fmt(ece_after, ".4f"), "lower is better"),
    ]

    header = (
        "| Metric | Source (reference) | Target raw | Target RECAL | Note |\n"
        "|---|---|---|---|---|\n"
    )
    body = "\n".join(f"| {m} | {s} | {b} | {a} | {n} |" for m, s, b, a, n in rows)
    return header + body
