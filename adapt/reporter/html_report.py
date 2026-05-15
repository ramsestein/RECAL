"""
adapt.reporter.html_report
===========================
Generates a self-contained HTML report (figures embedded as base64) for the
ADAPT source/target pair. The report is written in English.

Usage:
    from adapt.reporter.html_report import generate_html_report

    html = generate_html_report(
        profile=aa.profile_,
        config=aa.config_,
        y_true=pair.y_t,
        scores_before=scores_raw,
        scores_after=scores_adapted,
        source_name="SNUH",
        target_name="Clinic",
        output_path="reports/adapt/snuh_to_clinic.html",
    )
"""

from __future__ import annotations

import base64
import io
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from adapt.profiler.base import DriftProfile
from adapt.designer.base import AdapterConfig
from adapt.reporter import tables, figures

logger = logging.getLogger(__name__)


def _fig_to_base64(fig) -> str:
    """Convert a matplotlib figure to a base64 string ready to embed in HTML."""
    import matplotlib.pyplot as plt
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return b64


def _md_to_html_table(md: str) -> str:
    """Convert a Markdown pipe-table to HTML (no external dependency)."""
    lines = [l.strip() for l in md.strip().split("\n") if l.strip()]
    if not lines:
        return ""

    rows_html = []
    is_header = True
    for line in lines:
        if line.startswith("|---") or set(line.replace("|", "").replace("-", "").replace(" ", "")) == set():
            continue  # separator row
        cells = [c.strip() for c in line.strip("|").split("|")]
        if is_header:
            row_html = "<tr>" + "".join(f"<th>{c}</th>" for c in cells) + "</tr>"
            is_header = False
        else:
            row_html = "<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>"
        rows_html.append(row_html)

    return f"<table>\n{''.join(rows_html)}\n</table>"


_HTML_CSS = """
<style>
  body { font-family: 'Segoe UI', Arial, sans-serif; margin: 32px; background: #fafafa; color: #222; }
  h1 { color: #1565C0; border-bottom: 2px solid #1565C0; padding-bottom: 8px; }
  h2 { color: #0D47A1; margin-top: 32px; }
  h3 { color: #1976D2; }
  table { border-collapse: collapse; width: 100%; margin: 16px 0; font-size: 0.88em; }
  th { background: #1565C0; color: white; padding: 7px 10px; text-align: left; }
  td { padding: 5px 10px; border-bottom: 1px solid #e0e0e0; }
  tr:nth-child(even) { background: #f5f5f5; }
  code { background: #e8eaf6; padding: 1px 4px; border-radius: 3px; font-size: 0.9em; }
  .badge-ok { background: #4CAF50; color: white; padding: 2px 8px; border-radius: 10px; }
  .badge-warn { background: #FF9800; color: white; padding: 2px 8px; border-radius: 10px; }
  .badge-off { background: #9E9E9E; color: white; padding: 2px 8px; border-radius: 10px; }
  .fig-container { text-align: center; margin: 24px 0; }
  .fig-container img { max-width: 100%; border: 1px solid #ccc; border-radius: 4px; }
  .meta { font-size: 0.8em; color: #777; margin-bottom: 24px; }
  .summary-box { background: #E3F2FD; border-left: 4px solid #1565C0; padding: 12px 16px; margin: 16px 0; border-radius: 4px; }
  .source-box  { background: #F1F8E9; border-left: 4px solid #558B2F; padding: 12px 16px; margin: 16px 0; border-radius: 4px; }
  .gap-good { color: #2E7D32; font-weight: bold; }
  .gap-warn { color: #E65100; font-weight: bold; }
  .gap-bad  { color: #C62828; font-weight: bold; }
</style>
"""


def _render_source_section(
    source_metrics: Optional[dict],
    n_source: Optional[int],
    n_source_events: Optional[int],
    source_name: str,
    baseline_auroc_target: Optional[float],
) -> str:
    """Section 0: original model performance on its source domain (reference)."""
    if not source_metrics:
        return ""

    tbl = tables.make_source_table(
        source_metrics=source_metrics,
        n_source=n_source,
        n_source_events=n_source_events,
        source_name=source_name,
    )

    auroc_src = source_metrics.get("auroc")
    gap_html = ""
    if auroc_src is not None and baseline_auroc_target is not None:
        gap = float(auroc_src) - float(baseline_auroc_target)
        gap_html = (
            f"<p><strong>Source → Target AUROC gap (unadapted):</strong> "
            f"<span class='gap-warn'>{gap:+.4f}</span> "
            f"(source={auroc_src:.4f} − raw target={baseline_auroc_target:.4f})</p>"
        )

    return f"""
<h2>0. Original Model — Source-Domain Reference</h2>
<div class="source-box">
This section reports the unmodified model's performance on the dataset it was
originally trained / validated on. It establishes the upper bound any
domain-transfer pipeline can realistically approach.
</div>
{_md_to_html_table(tbl)}
{gap_html}
"""


def _render_cv_section(in_sample: Optional[dict], cv: Optional[dict]) -> str:
    """Section 3.3: in-sample vs honest CV comparison."""
    if not cv or not in_sample:
        return ""

    oof = cv.get("oof_metrics", {})
    lo, hi = cv.get("oof_auroc_ci", (None, None))
    adapted = in_sample.get("adapted", {})

    gap = (adapted.get("auroc", 0) or 0) - (oof.get("auroc", 0) or 0)
    if abs(gap) < 0.02:
        gap_class, gap_label = "gap-good", "OK (small gap)"
    elif abs(gap) < 0.05:
        gap_class, gap_label = "gap-warn", "Moderate optimism"
    else:
        gap_class, gap_label = "gap-bad", "High optimism — suspicious"

    rows = ""
    for f in cv.get("per_fold", []):
        rows += (
            f"<tr><td>{f['fold']+1}</td><td>{f['n_test']}</td>"
            f"<td>{f['n_events_test']}</td><td>{f['mask_n']}</td>"
            f"<td>{f['auroc']:.3f}</td><td>{f['precision']:.3f}</td>"
            f"<td>{f['recall']:.3f}</td><td>{f['f1']:.3f}</td></tr>"
        )

    ci_str = (
        f"[{lo:.3f}, {hi:.3f}]"
        if lo is not None and not (lo != lo)
        else "—"
    )

    return f"""
<h3>3.3 Honest validation (Stratified k-fold CV)</h3>
<p>
Each fold re-runs the full sweep + fit on the train split and predicts on the
held-out test split (which the pipeline never sees). The P/R/F1 decision
threshold is selected per fold via Youden. The OOF row aggregates the
out-of-fold scores.
</p>

<table>
<tr><th>Metric</th><th>In-sample (full target)</th><th>Honest CV (OOF)</th><th>Δ (overfit gap)</th></tr>
<tr><td>AUROC</td>
    <td>{adapted.get('auroc', 0):.4f}</td>
    <td>{oof.get('auroc', 0):.4f} {ci_str}</td>
    <td class="{gap_class}">{gap:+.4f} ({gap_label})</td></tr>
<tr><td>Precision</td><td>{adapted.get('precision', 0):.3f}</td>
    <td>{oof.get('precision', 0):.3f}</td><td>—</td></tr>
<tr><td>Recall</td><td>{adapted.get('recall', 0):.3f}</td>
    <td>{oof.get('recall', 0):.3f}</td><td>—</td></tr>
<tr><td>F1</td><td>{adapted.get('f1', 0):.3f}</td>
    <td>{oof.get('f1', 0):.3f}</td><td>—</td></tr>
</table>

<h4>Per-fold detail ({cv.get('n_splits', '?')} splits)</h4>
<table>
<tr><th>Fold</th><th>n_test</th><th>events</th><th>mask_N</th>
    <th>AUROC</th><th>P</th><th>R</th><th>F1</th></tr>
{rows}
</table>
"""


def generate_html_report(
    profile: DriftProfile,
    config: AdapterConfig,
    y_true: np.ndarray,
    scores_before: np.ndarray,
    scores_after: Optional[np.ndarray] = None,
    source_name: str = "Source",
    target_name: str = "Target",
    output_path: Optional[str] = None,
    auroc_after: Optional[float] = None,
    slope_after: Optional[float] = None,
    ece_after: Optional[float] = None,
    auroc_ci_after: Optional[tuple] = None,
    cv_results: Optional[dict] = None,
    in_sample_metrics: Optional[dict] = None,
) -> str:
    """
    Generate a self-contained HTML report for the ADAPT pair.

    Parameters
    ----------
    profile : DriftProfile
    config : AdapterConfig
    y_true : np.ndarray
        Target ground-truth labels.
    scores_before : np.ndarray
        Raw model probabilities (no adaptation).
    scores_after : np.ndarray, optional
        Probabilities after the ADAPT pipeline.
    source_name, target_name : str
        Names used throughout the report.
    output_path : str, optional
        If given, write the HTML to this file.
    auroc_after, slope_after, ece_after, auroc_ci_after : float / tuple, optional
        Post-pipeline metrics. If None they are computed from `scores_after`.
    in_sample_metrics : dict, optional
        Dict with keys:
            - "source"            : metrics of the original model on its
                                    source dataset (dict with auroc/precision/...).
            - "raw"               : raw-on-target metrics.
            - "adapted"           : adapted-on-target metrics.
            - "n_source"          : number of source samples.
            - "n_source_events"   : number of positive events in source.

    Returns
    -------
    str
        The full HTML document as a string.
    """
    from adapt.profiler.global_profiler import (
        _bootstrap_auroc_from_scores as _bootstrap_auroc,
        _calibration_slope, _ece_score
    )

    timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    in_sample_metrics = in_sample_metrics or {}
    source_metrics = in_sample_metrics.get("source")
    n_source = in_sample_metrics.get("n_source")
    n_source_events = in_sample_metrics.get("n_source_events")

    # Compute post-pipeline metrics if not provided
    if scores_after is not None:
        if auroc_after is None:
            auroc_after, ci_lo, ci_hi = _bootstrap_auroc(y_true, scores_after, n_boot=200)
            auroc_ci_after = (ci_lo, ci_hi)
        if slope_after is None:
            slope_after, _ = _calibration_slope(y_true, scores_after)
        if ece_after is None:
            ece_after = _ece_score(y_true, scores_after)

    # ── Figures ───────────────────────────────────────────────────────────────
    fig1 = figures.figure_quadrant_map(profile)
    b64_fig1 = _fig_to_base64(fig1)

    fig2 = figures.figure_calibration_curve(y_true, scores_before, scores_after)
    b64_fig2 = _fig_to_base64(fig2)

    fig3 = figures.figure_combined_score_bar(profile)
    b64_fig3 = _fig_to_base64(fig3)

    fig4 = figures.figure_missing_rates(profile)
    b64_fig4 = _fig_to_base64(fig4)

    # ── Tables ────────────────────────────────────────────────────────────────
    tbl_global = tables.make_global_table(profile)
    tbl_decisions = tables.make_decisions_table(config)
    tbl_features = tables.make_features_table(profile)
    tbl_eval = tables.make_eval_table(
        auroc_before=profile.baseline_auroc,
        auroc_after=auroc_after,
        auroc_ci_before=(profile.baseline_auroc_ci_low, profile.baseline_auroc_ci_high),
        auroc_ci_after=auroc_ci_after,
        slope_before=profile.baseline_calibration_slope,
        slope_after=slope_after,
        ece_before=profile.baseline_ece,
        ece_after=ece_after,
        n_target=profile.n_target_obs,
        n_events=profile.n_target_events,
        auroc_source=(source_metrics or {}).get("auroc"),
    )

    src_section = _render_source_section(
        source_metrics=source_metrics,
        n_source=n_source,
        n_source_events=n_source_events,
        source_name=source_name,
        baseline_auroc_target=profile.baseline_auroc,
    )

    # ── HTML ──────────────────────────────────────────────────────────────────
    def badge(active: bool) -> str:
        if active:
            return '<span class="badge-ok">✓ Enabled</span>'
        return '<span class="badge-off">✗ Disabled</span>'

    src_auroc_str = (
        f"{source_metrics['auroc']:.4f}" if source_metrics and source_metrics.get("auroc") is not None else "—"
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ADAPT Report — {source_name} → {target_name}</title>
{_HTML_CSS}
</head>
<body>

<h1>ADAPT — Domain Transfer Report</h1>
<p class="meta">
  Generated: {timestamp} |
  Transfer: <strong>{source_name} → {target_name}</strong> |
  ADAPT v0.1.0
</p>

<div class="summary-box">
  <strong>Executive summary:</strong>
  n_target={profile.n_target_obs} ({profile.n_target_events} events) |
  AUROC on source (original model)={src_auroc_str} |
  AUROC target raw={profile.baseline_auroc:.4f} |
  AUROC target post-ADAPT={f'{auroc_after:.4f}' if auroc_after else '—'} |
  Slope baseline={profile.baseline_calibration_slope:.2f} |
  Slope post={f'{slope_after:.2f}' if slope_after else '—'} |
  Mask: {badge(config.apply_mask)} N={config.mask_n} |
  QT: {badge(config.apply_quantile)} |
  WOE: {badge(config.apply_woe)} |
  PCA-CORAL: {badge(config.apply_pca_coral)} k={config.pca_coral_k} |
  Calibration: {badge(config.apply_calibration)} {config.calibration_method}
</div>

{src_section}

<h2>1. Pair Diagnostics</h2>
<h3>1.1 Global metrics</h3>
{_md_to_html_table(tbl_global)}

<h3>1.2 Feature quadrant map</h3>
<div class="fig-container">
  <img src="data:image/png;base64,{b64_fig1}" alt="Feature quadrant map">
</div>

<h3>1.3 Top-20 features by combined score</h3>
{_md_to_html_table(tbl_features)}

<div class="fig-container">
  <img src="data:image/png;base64,{b64_fig3}" alt="Combined score top-20">
</div>

<h3>1.4 Missing rates: source vs target</h3>
<div class="fig-container">
  <img src="data:image/png;base64,{b64_fig4}" alt="Missing rates">
</div>

<h2>2. Designer Decisions</h2>
{_md_to_html_table(tbl_decisions)}

<h2>3. Pipeline Evaluation</h2>
<h3>3.1 Metrics: source reference / target raw / target adapted</h3>
{_md_to_html_table(tbl_eval)}

<h3>3.2 Reliability diagram</h3>
<div class="fig-container">
  <img src="data:image/png;base64,{b64_fig2}" alt="Calibration curve">
</div>

{_render_cv_section(in_sample_metrics, cv_results)}

<h2>4. Designer Rationale</h2>
<ul>
{''.join(f"<li><strong>[{k}]</strong> {v}</li>" for k, v in config.rationale.items())}
</ul>

<p class="meta">
  ADAPT is a research tool. It does not replace independent clinical
  validation of any predictive model.
</p>

</body>
</html>"""

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(html, encoding="utf-8")
        logger.info("HTML report saved to: %s", out)

    return html
