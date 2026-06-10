"""
recal_core.reporter.html_report
===========================
Generates a self-contained HTML report (figures embedded as base64) for the
RECAL source/target pair. The report is written in English.

Usage:
    from recal_core.reporter.html_report import generate_html_report

    html = generate_html_report(
        profile=aa.profile_,
        config=aa.config_,
        y_true=pair.y_t,
        scores_before=scores_raw,
        scores_after=scores_adapted,
        source_name="SNUH",
        target_name="Clinic",
        output_path="reports/recal_core/snuh_to_clinic.html",
    )
"""

from __future__ import annotations

import base64
import io
import logging
from datetime import datetime
from pathlib import Path

import numpy as np

from recal_core.designer.base import AdapterConfig
from recal_core.profiler.base import DriftProfile
from recal_core.reporter import figures, tables

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
  .badge-severe { background: #C62828; color: white; padding: 2px 8px; border-radius: 10px; }
  .fig-container { text-align: center; margin: 24px 0; }
  .fig-container img { max-width: 100%; border: 1px solid #ccc; border-radius: 4px; }
  .meta { font-size: 0.8em; color: #777; margin-bottom: 24px; }
  .summary-box { background: #E3F2FD; border-left: 4px solid #1565C0; padding: 12px 16px; margin: 16px 0; border-radius: 4px; }
  .source-box  { background: #F1F8E9; border-left: 4px solid #558B2F; padding: 12px 16px; margin: 16px 0; border-radius: 4px; }
  .warn-box    { background: #FFF3E0; border-left: 4px solid #E65100; padding: 12px 16px; margin: 16px 0; border-radius: 4px; }
  .severe-box  { background: #FFEBEE; border-left: 4px solid #C62828; padding: 12px 16px; margin: 16px 0; border-radius: 4px; }
  .gap-good { color: #2E7D32; font-weight: bold; }
  .gap-warn { color: #E65100; font-weight: bold; }
  .gap-bad  { color: #C62828; font-weight: bold; }
  .scalar-highlight { font-size: 1.2em; font-weight: bold; color: #0D47A1; }
</style>
"""


def _render_source_section(
    source_metrics: dict | None,
    n_source: int | None,
    n_source_events: int | None,
    source_name: str,
    baseline_auroc_target: float | None,
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


def _render_cv_section(in_sample: dict | None, cv: dict | None) -> str:
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


def _render_patient_profiles_section(patient_profiles: dict | None) -> str:
    """Render Section 6: Patient Profile Clustering."""
    if not patient_profiles:
        return ""

    import pandas as pd

    cluster_df: pd.DataFrame = patient_profiles.get("cluster_table")
    fig_b64: str = patient_profiles.get("fig_b64", "")
    gap_ranking = patient_profiles.get("gap_ranking", [])
    n_clusters = patient_profiles.get("n_clusters", "?")
    pca_k_used = patient_profiles.get("pca_k_used", "?")
    pca_var = patient_profiles.get("pca_variance_explained")
    pca_rationale = patient_profiles.get("pca_k_rationale", "")
    var_threshold = patient_profiles.get("variance_threshold", 0.90)
    cluster_feature_stats: dict = patient_profiles.get("cluster_feature_stats", {})

    pca_str = f"{pca_var * 100:.1f}%" if pca_var is not None else "—"

    # ── Cluster table ─────────────────────────────────────────────────────────
    table_rows = ""
    if cluster_df is not None and len(cluster_df) > 0:
        for _, row in cluster_df.iterrows():
            gap = row["Gap |Δscore|"]
            if gap > 0.15:
                gap_cls = "gap-bad"
            elif gap > 0.07:
                gap_cls = "gap-warn"
            else:
                gap_cls = "gap-good"
            table_rows += (
                f"<tr>"
                f"<td><strong>{int(row['Cluster'])}</strong></td>"
                f"<td>{int(row['n source'])}</td>"
                f"<td>{int(row['n target'])}</td>"
                f"<td>{row['% target']:.1f}%</td>"
                f"<td>{row['Score source']:.4f}</td>"
                f"<td>{row['Score target']:.4f}</td>"
                f"<td class='{gap_cls}'>{gap:.4f}</td>"
                f"<td>{row['AKI rate source']:.3f}</td>"
                f"<td>{row['AKI rate target']:.3f}</td>"
                f"<td><small>{row['Top features']}</small></td>"
                f"</tr>\n"
            )

    cluster_table_html = f"""
<table>
<tr>
  <th>Cluster</th><th>n source</th><th>n target</th><th>% target</th>
  <th>Score source</th><th>Score target</th><th>Gap |Δscore|</th>
  <th>AKI rate source</th><th>AKI rate target</th><th>Top features</th>
</tr>
{table_rows}
</table>
"""

    # ── Interpretation box ────────────────────────────────────────────────────
    if gap_ranking:
        top_c, top_gap = gap_ranking[0]
        low_c, low_gap = gap_ranking[-1]
        interp = (
            f"Cluster <strong>{top_c}</strong> shows the highest prediction gap "
            f"(|Δscore|&nbsp;=&nbsp;{top_gap:.4f}): the model generalises poorly "
            f"to the target patients in this phenotype. "
            f"Cluster <strong>{low_c}</strong> has the smallest gap "
            f"({low_gap:.4f}), indicating good transferability for that subgroup."
        )
    else:
        interp = "No gap ranking available."

    # ── Cluster phenotype detail (means / medians per top feature) ─────────────
    def _fmt(v) -> str:
        """Format a float to 3 significant figures, or '—' if NaN."""
        import math
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return "—"
        if abs(v) >= 100:
            return f"{v:.1f}"
        if abs(v) >= 1:
            return f"{v:.3g}"
        return f"{v:.3g}"

    phenotype_blocks = ""
    if cluster_df is not None and cluster_feature_stats:
        for _, row in cluster_df.iterrows():
            cid = int(row["Cluster"])
            gap = row["Gap |Δscore|"]
            if gap > 0.15:
                badge_cls = "gap-bad"
            elif gap > 0.07:
                badge_cls = "gap-warn"
            else:
                badge_cls = "gap-good"

            feat_rows = ""
            for fs in cluster_feature_stats.get(cid, []):
                feat_rows += (
                    f"<tr>"
                    f"<td><code>{fs['feature']}</code></td>"
                    f"<td>{_fmt(fs['mean_src'])}</td>"
                    f"<td>{_fmt(fs['median_src'])}</td>"
                    f"<td>{_fmt(fs['mean_tgt'])}</td>"
                    f"<td>{_fmt(fs['median_tgt'])}</td>"
                    f"</tr>\n"
                )

            n_src = int(row["n source"])
            n_tgt = int(row["n target"])
            phenotype_blocks += f"""
<details style="margin-bottom:0.6em">
  <summary style="cursor:pointer;font-weight:600">
    Cluster {cid}
    &nbsp;<span class="{badge_cls}" style="padding:2px 6px;border-radius:4px;font-size:0.82em">
      gap&nbsp;{gap:.4f}
    </span>
    &nbsp;<span style="font-weight:normal;font-size:0.88em;color:#555">
      n_src={n_src} · n_tgt={n_tgt} · AKI src={row['AKI rate source']:.3f} · AKI tgt={row['AKI rate target']:.3f}
    </span>
  </summary>
  <table style="margin-top:0.4em;font-size:0.88em">
    <tr>
      <th>Feature</th>
      <th>Mean (source)</th><th>Median (source)</th>
      <th>Mean (target)</th><th>Median (target)</th>
    </tr>
    {feat_rows}
  </table>
</details>
"""

    fig_html = f'<img src="data:image/png;base64,{fig_b64}" alt="Patient profile UMAP">' if fig_b64 else ""

    return f"""
<h2>6. Patient Profile Analysis</h2>
<div class="summary-box">
  <strong>Methodology:</strong> Source and target patients are jointly projected
  via PCA and then clustered with K-Means ({n_clusters}&nbsp;clusters).
  UMAP is used only for the 2-D visualisation.<br>
  <strong>PCA rationale:</strong> <em>{pca_rationale}</em><br>
  The number of components is chosen automatically as the minimum needed to
  explain {var_threshold * 100:.0f}% of the combined variance.
  High feature multicollinearity (elevated VIF) reduces the number of
  independent directions in the data, so fewer components are sufficient —
  making PCA decollinearisation especially valuable in that regime.<br>
  <strong>Result:</strong> k={pca_k_used} components, {pca_str} variance explained.
  The <em>gap</em> column shows |mean&nbsp;score<sub>source</sub>&nbsp;−&nbsp;mean&nbsp;score<sub>target</sub>|
  within each cluster; a large gap flags phenotypes where predictions diverge
  most between populations.
</div>

<h3>6.1 Per-cluster summary (sorted by prediction gap)</h3>
{cluster_table_html}

<div class="summary-box">{interp}</div>

<h3>6.2 Cluster phenotype detail</h3>
<p style="margin-top:0;color:#555;font-size:0.9em">
  Top-5 features driving each cluster centroid (ranked by absolute standardised weight).
  Values are in the original feature units (imputed with source means where missing).
  Click a cluster to expand.
</p>
{phenotype_blocks}

<h3>6.3 UMAP projection — source vs target by cluster</h3>
<div class="fig-container">
{fig_html}
</div>
"""


def _render_joint_drift_section(joint_drift_data: dict) -> str:
    """
    Render Section 5: Joint Drift (covariance structure analysis).

    Parameters
    ----------
    joint_drift_data : dict
        Keys:
        - "vif_table"            : pd.DataFrame [feature, vif_source, vif_target,
                                   delta_vif, flag]
        - "condition_source"     : float
        - "condition_target"     : float
        - "eff_rank_source"      : float
        - "eff_rank_target"      : float
        - "lw_coef_source"       : float or None  (LW shrinkage coeff applied)
        - "lw_coef_target"       : float or None
        - "severe_share"         : float  (fraction of features flagged SEVERE)
        - "severe_share_threshold" : float
        - "mi_delta"             : float or None  (Frobenius norm of MI-matrix diff)
        - "compute_mi_matrix"    : bool
    """

    if not joint_drift_data:
        return ""

    vif_df = joint_drift_data.get("vif_table")
    cond_s = joint_drift_data.get("condition_source")
    cond_t = joint_drift_data.get("condition_target")
    er_s = joint_drift_data.get("eff_rank_source")
    er_t = joint_drift_data.get("eff_rank_target")
    lw_s = joint_drift_data.get("lw_coef_source")
    lw_t = joint_drift_data.get("lw_coef_target")
    severe_share = joint_drift_data.get("severe_share", 0.0)
    sev_threshold = joint_drift_data.get("severe_share_threshold", 0.20)
    mi_delta = joint_drift_data.get("mi_delta")

    # ── VIF table ─────────────────────────────────────────────────────────────
    vif_rows = ""
    if vif_df is not None and len(vif_df) > 0:
        for _, row in vif_df.iterrows():
            flag = row["flag"]
            if flag == "SEVERE":
                badge_html = '<span class="badge-severe">SEVERE</span>'
            elif flag == "WATCH":
                badge_html = '<span class="badge-warn">WATCH</span>'
            else:
                badge_html = '<span class="badge-ok">OK</span>'
            vif_rows += (
                f"<tr><td>{row['feature']}</td>"
                f"<td>{row['vif_source']:.2f}</td>"
                f"<td>{badge_html}</td></tr>\n"
            )

    vif_section = f"""
<h3>5.1 VIF by feature (source)</h3>
<p><small>Flag thresholds: WATCH ≥ 5, SEVERE ≥ 10. Computed on the source cohort only;
target VIF is omitted because small target cohorts make OLS underdetermined.</small></p>
<table>
<tr><th>Feature</th><th>VIF source</th><th>Flag</th></tr>
{vif_rows}
</table>
""" if vif_rows else "<p><em>VIF table not available.</em></p>"

    # ── Scalar metrics ─────────────────────────────────────────────────────────
    def _fmt(v) -> str:
        if v is None:
            return "—"
        if v == float("inf") or v != v:
            return "∞"
        return f"{v:.2f}"

    scalar_section = f"""
<h3>5.2 Covariance structure scalars</h3>
<table>
<tr><th>Metric</th><th>Source</th><th>Target</th></tr>
<tr>
  <td>Condition number κ (covariance)</td>
  <td class="scalar-highlight">{_fmt(cond_s)}</td>
  <td class="scalar-highlight">{_fmt(cond_t)}</td>
</tr>
<tr>
  <td>Effective rank (Shannon entropy of eigenvalues)</td>
  <td class="scalar-highlight">{_fmt(er_s)}</td>
  <td class="scalar-highlight">{_fmt(er_t)}</td>
</tr>
</table>
"""

    # ── LW shrinkage reported ─────────────────────────────────────────────────
    lw_section = ""
    if lw_s is not None or lw_t is not None:
        lw_section = f"""
<h3>5.3 Ledoit-Wolf shrinkage applied by CORAL</h3>
<table>
<tr><th>Cohort</th><th>LW shrinkage coefficient α</th></tr>
<tr><td>Source</td><td>{_fmt(lw_s)}</td></tr>
<tr><td>Target</td><td>{_fmt(lw_t)}</td></tr>
</table>
<p><em>α=0 → no shrinkage (sample covariance); α=1 → fully shrunk to diagonal.
A high α for the target cohort indicates small n_target and that shrinkage is
actively regularising the covariance estimate.</em></p>
"""

    # ── MI matrix delta ───────────────────────────────────────────────────────
    mi_section = ""
    if joint_drift_data.get("compute_mi_matrix") and mi_delta is not None:
        mi_section = f"""
<h3>5.4 Mutual information matrix delta</h3>
<p>Frobenius norm of (MI_matrix_source − MI_matrix_target):
<span class="scalar-highlight">{mi_delta:.4f}</span></p>
<p><em>A large value indicates that the pairwise dependencies between features
differ substantially between cohorts, signalling joint drift beyond
marginal distributional change.</em></p>
"""

    # ── Go/No-go block ────────────────────────────────────────────────────────
    severe_pct = severe_share * 100.0
    if severe_share > sev_threshold:
        gonogo_html = f"""
<div class="severe-box">
  <strong>⚠ Joint Drift: NO-GO for transfer without retraining.</strong>
  {severe_pct:.1f}% of features have |ΔVIF| &gt; SEVERE threshold
  (threshold: {sev_threshold * 100:.0f}%).
  The covariance structure of the target cohort departs substantially from
  source. Consider retraining the base model on pooled or target-only data
  before applying domain transfer.
</div>
"""
    else:
        gonogo_html = f"""
<div class="summary-box">
  <strong>✓ Joint Drift: GO.</strong>
  Only {severe_pct:.1f}% of features have |ΔVIF| &gt; SEVERE threshold
  (threshold: {sev_threshold * 100:.0f}%). Covariance structure shift is
  within acceptable bounds for domain transfer.
</div>
"""

    return f"""
<h2>5. Joint Drift Analysis</h2>
<p>This section characterises changes in the <em>covariance structure</em>
between source and target — information invisible to marginal (per-feature)
drift metrics.  A Variance Inflation Factor (VIF) is computed per feature
in each cohort and the change |ΔVIF| is flagged against configurable thresholds.</p>

{gonogo_html}
{vif_section}
{scalar_section}
{lw_section}
{mi_section}
"""


# ── Nuevas secciones de auditabilidad ─────────────────────────────────────────


def _render_drift_attribution_section(
    drift_decomp: dict | None,
    oracle_result: dict | None,
    feature_attribution: list | None,
) -> str:
    """Section: Drift Attribution — raw/adapted/oracle + gaps + feature attribution."""
    if drift_decomp is None and oracle_result is None:
        return ""

    oracle_warning = (
        '<div class="severe-box"><strong>⚠ CLINICAL NOTE:</strong> '
        'The oracle (model trained natively on the target cohort) is <em>exclusively</em> '
        'a measurement tool to quantify the recoverable gap. '
        '<strong>It is NOT a model suitable for clinical deployment</strong> — '
        'it has not been prospectively validated and is trained on the evaluation set.'
        '</div>'
    )

    rows = []
    if drift_decomp:
        def _fmt(v):
            return f"{v:.4f}" if v is not None and v == v else "—"
        def _fmt_ci(ci):
            if ci and len(ci) == 2 and all(x is not None for x in ci):
                return f"[{ci[0]:.3f}, {ci[1]:.3f}]"
            return "—"

        rows.append(f"<tr><td>AUROC raw (no adaptation)</td><td>{_fmt(drift_decomp.get('auroc_raw'))}</td>"
                    f"<td>{_fmt_ci(drift_decomp.get('ci_raw'))}</td></tr>")
        rows.append(f"<tr><td>AUROC adapted</td><td>{_fmt(drift_decomp.get('auroc_adapted'))}</td>"
                    f"<td>{_fmt_ci(drift_decomp.get('ci_adapted'))}</td></tr>")
        rows.append(f"<tr><td>AUROC oracle (measurement only)</td><td>{_fmt(drift_decomp.get('auroc_oracle'))}</td>"
                    f"<td>{_fmt_ci(drift_decomp.get('ci_oracle'))}</td></tr>")
        rows.append(f"<tr><td>Total gap (oracle − raw)</td>"
                    f"<td>{_fmt(drift_decomp.get('total_gap'))}</td>"
                    f"<td>{_fmt_ci(drift_decomp.get('total_gap_ci'))}</td></tr>")
        rows.append(f"<tr><td>Recoverable gap (adapted − raw)</td>"
                    f"<td>{_fmt(drift_decomp.get('recoverable_gap'))}</td>"
                    f"<td>{_fmt_ci(drift_decomp.get('recoverable_gap_ci'))}</td></tr>")
        rows.append(f"<tr><td>Irreducible gap (oracle − adapted)</td>"
                    f"<td>{_fmt(drift_decomp.get('irreducible_gap'))}</td>"
                    f"<td>{_fmt_ci(drift_decomp.get('irreducible_gap_ci'))}</td></tr>")

        rr = drift_decomp.get("recovery_ratio")
        rr_ci = drift_decomp.get("recovery_ratio_ci")
        rr_str = f"{rr:.3f} {_fmt_ci(rr_ci)}" if rr is not None else "Indeterminate"
        rows.append(f"<tr><td><strong>Recovery ratio</strong></td>"
                    f"<td colspan='2'><strong>{rr_str}</strong></td></tr>")
        if drift_decomp.get("note"):
            rows.append(f"<tr><td colspan='3'><em>{drift_decomp['note']}</em></td></tr>")

    gap_table = f"""
<table>
<thead><tr><th>Metric</th><th>Value</th><th>95% CI</th></tr></thead>
<tbody>{"".join(rows)}</tbody>
</table>
""" if rows else ""

    attr_section = ""
    if feature_attribution:
        attr_rows = "".join(
            f"<tr><td>{r.get('feature','—')}</td>"
            f"<td>{r.get('contribution', 0.0):+.4f}</td>"
            f"<td>{r.get('auroc_j_off', r.get('auroc_without', 0.0)):.4f}</td></tr>"
            for r in feature_attribution
        )
        attr_section = f"""
<h3>6.2 Per-feature attribution</h3>
<p>Contribution of each feature to the recoverable gap
(positive = feature helps recovery; negative = feature hurts it).</p>
<table>
<thead><tr><th>Feature</th><th>AUROC contribution</th><th>AUROC without this feature</th></tr></thead>
<tbody>{attr_rows}</tbody>
</table>
"""

    return f"""
<h2>6. Drift Attribution</h2>
{oracle_warning}
{gap_table}
{attr_section}
"""


def _render_designer_audit_section(audit) -> str:
    """Section: Designer Audit Trail."""
    if audit is None:
        return ""

    try:
        decisions = audit.to_dict() if hasattr(audit, "to_dict") else []
    except Exception:
        return ""

    if not decisions:
        return ""

    rows_html = []
    for d in decisions:
        step = d.get("step", "—")
        criterion = d.get("criterion", "—")
        final = str(d.get("final_choice", "—"))
        justification = d.get("justification", "—")

        alts = d.get("alternatives", [])
        if alts:
            alt_parts = []
            for a in alts[:6]:
                sel_mark = " ✓" if a.get("selected") else ""
                metric = a.get("metric_value")
                metric_str = f" ({metric:.4f})" if metric is not None and metric == metric else ""
                alt_parts.append(f"<em>{a.get('choice','?')}{metric_str}{sel_mark}</em>")
            alts_str = " | ".join(alt_parts)
            if len(alts) > 6:
                alts_str += f" (+{len(alts)-6})"
        else:
            alts_str = "—"

        rows_html.append(
            f"<tr>"
            f"<td><code>{step}</code></td>"
            f"<td>{criterion}</td>"
            f"<td>{alts_str}</td>"
            f"<td><strong>{final}</strong></td>"
            f"<td>{justification}</td>"
            f"</tr>"
        )

    return f"""
<h2>7. Designer Audit Trail</h2>
<p>Each row corresponds to one Designer decision. The
<em>Alternatives</em> column lists all evaluated options (✓ = chosen).
The criterion is the business rule governing the decision.</p>
<table>
<thead><tr>
  <th>Step</th><th>Criterion</th><th>Alternatives</th>
  <th>Chosen</th><th>Justification</th>
</tr></thead>
<tbody>{"".join(rows_html)}</tbody>
</table>
"""


def _render_counterfactuals_section(counterfactuals: dict | None) -> str:
    """Section: Counterfactual Sensitivity."""
    if not counterfactuals:
        return ""

    sections_html = []
    section_titles = {
        "mask_n": "Number of masked features (mask_n)",
        "pca_coral_k": "PCA-CORAL components (k)",
        "calibration": "Calibration method",
    }

    for key, title in section_titles.items():
        entries = counterfactuals.get(key)
        if not entries:
            continue
        rows = "".join(
            "<tr>"
            f"<td>{'★ ' if e.get('selected') else ''}{e.get('choice','—')}</td>"
            f"<td>{'{:.4f}'.format(e['auroc']) if e.get('auroc') is not None else '—'}</td>"
            "</tr>"
            for e in entries
        )
        sections_html.append(f"""
<h3>8.{len(sections_html)+1} {title}</h3>
<table>
<thead><tr><th>Value (★ = selected)</th><th>AUROC target</th></tr></thead>
<tbody>{rows}</tbody>
</table>
""")

    if not sections_html:
        return ""

    return f"""
<h2>8. Counterfactual Sensitivity</h2>
<p>Sensitivity analysis: what AUROC would have been obtained with
alternative design decisions? Rows marked ★ correspond to the
actually selected configuration.</p>
{"".join(sections_html)}
"""


def _render_significance_section(sig: dict | None) -> str:
    """Section 3.4: Statistical significance tests (DeLong + bootstrap z-test)."""
    if not sig:
        return ""

    correction = sig.get("correction", "Bonferroni (k=2)")
    alpha = sig.get("alpha", 0.05)

    def _p_fmt(p):
        if p != p or p is None:  # NaN
            return "—"
        if p < 0.001:
            return "<0.001"
        return f"{p:.3f}"

    def _sig_badge(p_corr):
        if p_corr != p_corr or p_corr is None:
            return ""
        if p_corr < alpha:
            return '<span class="badge-severe" style="background:#2E7D32">p&lt;α</span>'
        return '<span class="badge-ok" style="background:#9E9E9E">n.s.</span>'

    def _ci_fmt(ci):
        if ci is None:
            return "—"
        return f"[{ci[0]:.3f}, {ci[1]:.3f}]"

    rows = []
    avr = sig.get("adapted_vs_raw")
    if avr:
        ci_ada = _ci_fmt(avr.get("ci_adapted"))
        ci_raw = _ci_fmt(avr.get("ci_raw"))
        rows.append(
            f"<tr>"
            f"<td>Adapted vs Raw (target)</td>"
            f"<td>{avr.get('auc_adapted', 0):.4f}&nbsp;{ci_ada}</td>"
            f"<td>{avr.get('auc_raw', 0):.4f}&nbsp;{ci_raw}</td>"
            f"<td>{avr.get('delta', 0):+.4f}</td>"
            f"<td>DeLong (1988) paired</td>"
            f"<td>{_p_fmt(avr.get('p_raw'))}</td>"
            f"<td>{_p_fmt(avr.get('p_bonferroni'))}</td>"
            f"<td>{_sig_badge(avr.get('p_bonferroni'))}</td>"
            f"</tr>"
        )

    avs = sig.get("adapted_vs_source")
    if avs:
        ci_ada_s = _ci_fmt(avs.get("ci_adapted"))
        ci_src = _ci_fmt(avs.get("ci_source"))
        note = avs.get("note", "")
        p_corr_avs = avs.get("p_bonferroni")
        # For equivalence: green = fail to reject H₀ (p > α), grey = significant difference
        def _equiv_badge(p_c):
            if p_c != p_c or p_c is None:
                return ""
            if p_c >= alpha:
                return '<span class="badge-ok" style="background:#2E7D32;color:#fff">n.s. ✓</span>'
            return '<span class="badge-severe" style="background:#C62828">diff. ✗</span>'
        rows.append(
            f"<tr>"
            f"<td>Adapted (target) vs Original (source-domain)<br><small><em>Goal: no significant difference</em></small></td>"
            f"<td>{avs.get('auc_adapted', 0):.4f}&nbsp;{ci_ada_s}</td>"
            f"<td>{avs.get('auc_source', 0):.4f}&nbsp;{ci_src}</td>"
            f"<td>{avs.get('delta', 0):+.4f}</td>"
            f"<td>Bootstrap z-test (indep.)</td>"
            f"<td>{_p_fmt(avs.get('p_raw'))}</td>"
            f"<td>{_p_fmt(p_corr_avs)}</td>"
            f"<td>{_equiv_badge(p_corr_avs)}</td>"
            f"</tr>"
        )
        if note:
            rows.append(
                f'<tr><td colspan="8"><small><em>⚠ {note}</em></small></td></tr>'
            )
    elif sig.get("adapted_vs_raw"):  # no source CI available
        rows.append(
            '<tr><td colspan="8"><small><em>'
            'Adapted vs Source comparison not available (bootstrap CI for source missing).</em></small></td></tr>'
        )

    if not rows:
        return ""

    return f"""
<h3>3.4 Statistical significance</h3>
<p>
H₀ tests for the two key comparisons. Multiple-testing correction: <strong>{correction}</strong>, α&nbsp;=&nbsp;{alpha}.<br>
Row 1 (Adapted vs Raw): green badge = adapted <em>is</em> significantly better (corrected p &lt; α).<br>
Row 2 (Adapted vs Original): green badge = adapted is <em>not</em> significantly different from source (corrected p ≥ α — equivalence goal).
</p>
<table>
<thead><tr>
  <th>Comparison</th><th>Model A (AUROC [95% CI])</th><th>Model B (AUROC [95% CI])</th>
  <th>ΔAUROC</th><th>Test</th><th>p (raw)</th><th>p (corrected)</th><th>Sig.</th>
</tr></thead>
<tbody>{chr(10).join(rows)}</tbody>
</table>
"""


def _render_feature_log_section(feature_log: dict | None, combined_scores: dict | None = None) -> str:
    """Section: Per-feature log."""
    if not feature_log:
        return ""

    # Sort by combined score descending; features not in combined_scores go last
    feats_sorted = sorted(
        feature_log.keys(),
        key=lambda f: -(combined_scores.get(f, -1.0) if combined_scores else 0.0),
    )

    rows = []
    for feat in feats_sorted:
        info = feature_log[feat]
        combined_val = combined_scores.get(feat) if combined_scores else None
        src = info.get("source_dist_summary", {})
        tgt = info.get("target_dist_summary", {})
        post = info.get("post_align_dist_summary", {})
        ks_pre = info.get("ks_stat_pre")
        ks_post = info.get("ks_stat_post")
        method = info.get("alignment_method", "—")

        def _fv(d, k):
            v = d.get(k)
            return f"{v:.3f}" if v is not None else "—"

        rows.append(
            f"<tr>"
            f"<td><code>{feat}</code></td>"
            f"<td>{f'{combined_val:.3f}' if combined_val is not None else '—'}</td>"
            f"<td>{method}</td>"
            f"<td>{_fv(src,'mean')} ± {_fv(src,'std')}</td>"
            f"<td>{_fv(tgt,'mean')} ± {_fv(tgt,'std')}</td>"
            f"<td>{f'{ks_pre:.3f}' if ks_pre is not None else '—'}</td>"
            f"<td>{_fv(post,'mean')} ± {_fv(post,'std')}</td>"
            f"<td>{f'{ks_post:.3f}' if ks_post is not None else '—'}</td>"
            f"</tr>"
        )

    if not rows:
        return ""

    total = len(feature_log)

    return f"""
<h2>9. Per-feature Log</h2>
<p>Distribution statistics per feature before and after alignment ({total} features total, sorted by combined score descending).
KS = Kolmogorov-Smirnov statistic between source and target (0 = identical, 1 = maximum drift).
A lower KS post than pre indicates that alignment reduced marginal drift.
Combined score = normalize(|L_base|)&#x2009;+&#x2009;normalize(SHAP importance) — higher means more informative for transfer.</p>
<table>
<thead><tr>
  <th>Feature</th><th>Combined ↑</th><th>Method</th>
  <th>Source (μ±σ)</th><th>Target pre (μ±σ)</th><th>KS pre</th>
  <th>Target post (μ±σ)</th><th>KS post</th>
</tr></thead>
<tbody>{"".join(rows)}</tbody>
</table>
"""


def _render_calibration_decomp_section(
    brier_raw: dict | None,
    brier_adapted: dict | None,
    brier_delta_dict: dict | None,
    audit_yaml_path: str | None,
) -> str:
    """Section: Calibration Decomposition (Murphy) + YAML download."""
    parts = []

    if brier_raw or brier_adapted:
        parts.append("""
<h2>10. Calibration Decomposition (Murphy)</h2>
<p>
  Brier score decomposed following Murphy (1973):
  <strong>BS = Reliability − Resolution + Uncertainty</strong><br/>
  <em>Reliability</em> measures miscalibration (lower = better calibrated). <br/>
  <em>Resolution</em> measures discrimination sharpness (higher = better). <br/>
  <em>Uncertainty</em> is irreducible given the observed outcome distribution.
</p>
""")

        def _fv(d, k):
            if d is None:
                return "—"
            v = d.get(k)
            return f"{v:.4f}" if v is not None and v == v else "—"

        rows = [
            f"<tr><th>Brier Score</th><td>{_fv(brier_raw,'brier_score')}</td>"
            f"<td>{_fv(brier_adapted,'brier_score')}</td>"
            f"<td>{_fv(brier_delta_dict,'delta_brier_score')}</td></tr>",
            f"<tr><th>Reliability ↓</th><td>{_fv(brier_raw,'reliability')}</td>"
            f"<td>{_fv(brier_adapted,'reliability')}</td>"
            f"<td>{_fv(brier_delta_dict,'delta_reliability')}</td></tr>",
            f"<tr><th>Resolution ↑</th><td>{_fv(brier_raw,'resolution')}</td>"
            f"<td>{_fv(brier_adapted,'resolution')}</td>"
            f"<td>{_fv(brier_delta_dict,'delta_resolution')}</td></tr>",
            f"<tr><th>Uncertainty (fixed)</th><td>{_fv(brier_raw,'uncertainty')}</td>"
            f"<td>{_fv(brier_adapted,'uncertainty')}</td><td>—</td></tr>",
        ]

        interp = ""
        if brier_delta_dict and brier_delta_dict.get("interpretation"):
            interp = f"<p><em>Interpretation: {brier_delta_dict['interpretation']}</em></p>"

        parts.append(f"""
<table>
<thead><tr><th>Component</th><th>Raw</th><th>Adapted</th><th>Δ (adapted − raw)</th></tr></thead>
<tbody>{"".join(rows)}</tbody>
</table>
{interp}
""")

    # YAML download link
    if audit_yaml_path:
        parts.append(f"""
<h3>Audit YAML</h3>
<p>Full execution record (input hashes, decision trail, feature log, dependency versions):
<a href="{audit_yaml_path}" download>Download audit YAML</a>
<br/><small>Path: <code>{audit_yaml_path}</code></small>
</p>
""")

    return "".join(parts)


def generate_html_report(
    profile: DriftProfile,
    config: AdapterConfig,
    y_true: np.ndarray,
    scores_before: np.ndarray,
    scores_after: np.ndarray | None = None,
    source_name: str = "Source",
    target_name: str = "Target",
    output_path: str | None = None,
    auroc_after: float | None = None,
    slope_after: float | None = None,
    ece_after: float | None = None,
    auroc_ci_after: tuple | None = None,
    cv_results: dict | None = None,
    in_sample_metrics: dict | None = None,
    joint_drift_data: dict | None = None,
    # ── Nuevas secciones de auditabilidad ─────────────────────────────────────
    oracle_results: dict | None = None,
    drift_decomp: dict | None = None,
    feature_attribution: list | None = None,
    counterfactuals: dict | None = None,
    brier_decomp_raw: dict | None = None,
    brier_decomp_adapted: dict | None = None,
    brier_delta: dict | None = None,
    audit_yaml_path: str | None = None,
    feature_log: dict | None = None,
    patient_profiles: dict | None = None,
    significance_tests: dict | None = None,
    feature_combined_scores: dict | None = None,
) -> str:
    """
    Generate a self-contained HTML report for the RECAL pair.

    Parameters
    ----------
    profile : DriftProfile
    config : AdapterConfig
    y_true : np.ndarray
        Target ground-truth labels.
    scores_before : np.ndarray
        Raw model probabilities (no adaptation).
    scores_after : np.ndarray, optional
        Probabilities after the RECAL pipeline.
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
    joint_drift_data : dict, optional
        Output of :func:`recal_cli.joint_drift.joint_drift_report` plus scalar
        metrics.  If provided, Section 5 "Joint Drift" is included.
        Expected keys: ``vif_table``, ``condition_source``, ``condition_target``,
        ``eff_rank_source``, ``eff_rank_target``, ``lw_coef_source``,
        ``lw_coef_target``, ``severe_share``, ``severe_share_threshold``,
        ``mi_delta`` (or None), ``compute_mi_matrix``.

    Returns
    -------
    str
        The full HTML document as a string.
    """
    from recal_core.profiler.global_profiler import _bootstrap_auroc_from_scores as _bootstrap_auroc
    from recal_core.profiler.global_profiler import _calibration_slope, _ece_score

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
        auroc_ci_source=(
            ((source_metrics or {}).get("auroc_ci_lo"), (source_metrics or {}).get("auroc_ci_hi"))
            if (source_metrics or {}).get("auroc_ci_lo") is not None else None
        ),
        slope_source=(source_metrics or {}).get("calibration_slope"),
        ece_source=(source_metrics or {}).get("ece"),
    )

    src_section = _render_source_section(
        source_metrics=source_metrics,
        n_source=n_source,
        n_source_events=n_source_events,
        source_name=source_name,
        baseline_auroc_target=profile.baseline_auroc,
    )

    joint_drift_section = _render_joint_drift_section(joint_drift_data or {})
    patient_profiles_section = _render_patient_profiles_section(patient_profiles)

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
<title>RECAL Report — {source_name} → {target_name}</title>
{_HTML_CSS}
</head>
<body>

<h1>RECAL — Domain Transfer Report</h1>
<p class="meta">
  Generated: {timestamp} |
  Transfer: <strong>{source_name} → {target_name}</strong> |
  RECAL v0.1.0
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

{_render_significance_section(significance_tests)}

{joint_drift_section}

{patient_profiles_section}

{_render_drift_attribution_section(drift_decomp, oracle_results, feature_attribution)}

{_render_designer_audit_section(getattr(config, 'audit', None))}

{_render_counterfactuals_section(counterfactuals)}

{_render_feature_log_section(feature_log, combined_scores=feature_combined_scores)}

{_render_calibration_decomp_section(brier_decomp_raw, brier_decomp_adapted, brier_delta, audit_yaml_path)}

<h2>4. Designer Rationale</h2>
<ul>
{''.join(f"<li><strong>[{k}]</strong> {v}</li>" for k, v in config.rationale.items())}
</ul>

<p class="meta">
  RECAL is a research tool. It does not replace independent clinical
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
