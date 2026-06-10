"""
recal_cli.patient_profiles
===========================
Identifies patient archetypes via unsupervised clustering and quantifies
the model-prediction gap between source and target populations per cluster.

Philosophy
----------
Following RECAL's covariate-drift philosophy, "drift" is not uniform across
patients.  Some patient phenotypes transfer well (low gap) while others expose
the model to out-of-distribution inputs (high gap).  This module makes that
heterogeneity explicit by:

  1. Standardising the shared feature space (source statistics).
  2. Reducing to PCA components selected to explain ``variance_threshold`` of
     the combined variance.  High multicollinearity (elevated VIF) means that
     fewer components are required to reach the threshold — in that regime PCA
     is especially powerful at decollinearising the space before clustering.
  3. Clustering the combined source + target population (K-Means).
  4. Projecting to 2D with UMAP for visual inspection.
  5. Per-cluster: measuring domain overlap, mean model score, AKI rate,
     and prediction gap |ȳ_source − ȳ_target|.
  6. Characterising each cluster by the top features driving it.

The module is fully dataset-agnostic: it only requires a feature matrix, label
vector, a ``predict_proba`` model, and an optional VIF summary dict produced by
``recal_cli.joint_drift.joint_drift_report``.

Exported function
-----------------
analyze_patient_profiles(
    X_s, y_s, X_t, y_t, model, features,
    n_clusters=6, pca_k_max=50, variance_threshold=0.90,
    vif_summary=None, umap_neighbors=15, random_state=42
) -> dict
"""

from __future__ import annotations

import base64
import io
import logging
from collections.abc import Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── PCA component selection ───────────────────────────────────────────────────

def _select_pca_k(
    X_combined: np.ndarray,
    pca_k_max: int,
    variance_threshold: float,
    vif_summary: dict | None,
    random_state: int,
) -> tuple[int, str]:
    """
    Fit a full PCA (up to *pca_k_max* components) on *X_combined* and return
    the minimum ``k`` whose cumulative explained variance reaches
    *variance_threshold*, together with a human-readable rationale.

    The connection to VIF
    ---------------------
    When many features are highly collinear (VIF ≫ 1), the covariance matrix
    has a few dominant eigenvalues and many near-zero ones.  The threshold k is
    therefore *smaller* under high multicollinearity — PCA is more effective at
    compressing the space.  The rationale string reflects this.
    """
    from sklearn.decomposition import PCA

    p = X_combined.shape[1]
    k_cap = min(pca_k_max, p, X_combined.shape[0] - 1)

    pca_full = PCA(n_components=k_cap, random_state=random_state)
    pca_full.fit(X_combined)
    cumvar = np.cumsum(pca_full.explained_variance_ratio_)

    # Find minimum k ≥ 1 that reaches the threshold
    hits = np.where(cumvar >= variance_threshold)[0]
    k_selected = int(hits[0] + 1) if len(hits) > 0 else k_cap
    var_at_k = float(cumvar[k_selected - 1])

    # VIF context
    vif_clause = ""
    if vif_summary:
        n_severe = vif_summary.get("n_severe", 0)
        n_watch  = vif_summary.get("n_watch",  0)
        n_total  = vif_summary.get("n_features", p)
        n_high   = n_severe + n_watch
        if n_high > 0:
            vif_clause = (
                f" {n_high}/{n_total} features show elevated multicollinearity "
                f"(VIF>5: {n_watch} WATCH, VIF>10: {n_severe} SEVERE), "
                f"which increases PCA compression efficiency."
            )

    rationale = (
        f"k={k_selected} components explain {var_at_k * 100:.1f}% of combined "
        f"source+target variance (threshold={variance_threshold * 100:.0f}%, "
        f"p={p} features, cap={k_cap}).{vif_clause}"
    )
    return k_selected, rationale


# ── Main entry point ─────────────────────────────────────────────────────────

def analyze_patient_profiles(
    X_s: np.ndarray,
    y_s: np.ndarray,
    X_t: np.ndarray,
    y_t: np.ndarray,
    model,
    features: Sequence[str],
    n_clusters: int = 6,
    pca_k_max: int = 50,
    variance_threshold: float = 0.90,
    vif_summary: dict | None = None,
    umap_neighbors: int = 15,
    random_state: int = 42,
) -> dict:
    """
    Cluster combined source + target patients and measure the model-prediction
    gap per profile.

    Parameters
    ----------
    X_s, y_s : source feature matrix and outcome vector.
    X_t, y_t : target feature matrix and outcome vector.
    model     : fitted model with a ``predict_proba(X) -> np.ndarray`` method.
    features  : feature names (length == X_s.shape[1]).
    n_clusters : number of K-Means clusters (patient archetypes).
    pca_k_max  : hard upper bound on the number of PCA components.  The actual
                 k is chosen as the minimum needed to reach *variance_threshold*.
    variance_threshold : fraction of variance the PCA projection must explain
                 (default 0.90 = 90 %).  Increase for a more faithful (but
                 higher-dimensional) representation; decrease for speed.
    vif_summary : optional dict produced by the joint-drift step with keys
                 ``n_severe``, ``n_watch``, ``n_features``.  Used only to enrich
                 the PCA rationale narrative; does not change the algorithm.
    umap_neighbors : UMAP ``n_neighbors`` parameter.
    random_state : seed for reproducibility.

    Returns
    -------
    dict with keys:
        ``fig_b64``               – base64-encoded PNG of the UMAP scatter.
        ``cluster_table``         – pd.DataFrame with per-cluster statistics.
        ``gap_ranking``           – list of (cluster_id, gap) sorted descending.
        ``n_clusters``            – number of clusters used.
        ``pca_k_used``            – actual PCA k selected.
        ``pca_variance_explained``– fraction of variance explained by PCA.
        ``pca_k_rationale``       – human-readable rationale for k selection.
        ``variance_threshold``    – threshold passed in.
    """
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    features = list(features)
    _p = len(features)

    # ── 1. Impute NaN with source column means ───────────────────────────────
    src_means = np.nanmean(X_s, axis=0)
    X_s_imp = np.where(np.isnan(X_s), src_means, X_s)
    X_t_imp = np.where(np.isnan(X_t), src_means, X_t)

    # ── 2. Standardise with source statistics ────────────────────────────────
    scaler = StandardScaler()
    scaler.fit(X_s_imp)
    X_s_std = scaler.transform(X_s_imp)
    X_t_std = scaler.transform(X_t_imp)

    # ── 3. PCA: auto-select k via variance threshold ──────────────────────────
    n_s, n_t = len(X_s_std), len(X_t_std)
    X_combined = np.vstack([X_s_std, X_t_std])
    k, pca_rationale = _select_pca_k(
        X_combined, pca_k_max, variance_threshold, vif_summary, random_state
    )
    pca = PCA(n_components=k, random_state=random_state)
    X_pca = pca.fit_transform(X_combined)
    X_pca_s = X_pca[:n_s]
    X_pca_t = X_pca[n_s:]

    # ── 4. K-Means clustering ─────────────────────────────────────────────────
    km = KMeans(n_clusters=n_clusters, n_init=20, random_state=random_state)
    labels_all = km.fit_predict(X_pca)
    labels_s = labels_all[:n_s]
    labels_t = labels_all[n_s:]

    # ── 5. Model predictions ──────────────────────────────────────────────────
    scores_s = model.predict_proba(X_s_imp)
    scores_t = model.predict_proba(X_t_imp)

    # ── 6. Per-cluster summary ────────────────────────────────────────────────
    rows = []
    for c in range(n_clusters):
        mask_s = labels_s == c
        mask_t = labels_t == c

        n_c_s = int(mask_s.sum())
        n_c_t = int(mask_t.sum())
        n_c = n_c_s + n_c_t

        mean_score_s = float(scores_s[mask_s].mean()) if n_c_s > 0 else float("nan")
        mean_score_t = float(scores_t[mask_t].mean()) if n_c_t > 0 else float("nan")
        gap = abs(mean_score_s - mean_score_t) if (n_c_s > 0 and n_c_t > 0) else float("nan")

        aki_rate_s = float(y_s[mask_s].mean()) if n_c_s > 0 else float("nan")
        aki_rate_t = float(y_t[mask_t].mean()) if n_c_t > 0 else float("nan")

        pct_target = n_c_t / n_c if n_c > 0 else float("nan")

        # Top 5 features by absolute standardised centroid value
        centroid_std = km.cluster_centers_[c]          # in PCA space
        centroid_orig = pca.inverse_transform(centroid_std.reshape(1, -1)).ravel()   # back to std space
        top_idx = np.argsort(np.abs(centroid_orig))[::-1][:5]
        top_feats = ", ".join(features[i] for i in top_idx[:3])

        # Mean & median of top features in raw (imputed, non-standardised) space
        feat_stats = []
        for fi in top_idx:
            src_vals = X_s_imp[mask_s, fi] if n_c_s > 0 else np.array([])
            tgt_vals = X_t_imp[mask_t, fi] if n_c_t > 0 else np.array([])
            feat_stats.append({
                "feature": features[fi],
                "mean_src": float(np.mean(src_vals)) if len(src_vals) > 0 else float("nan"),
                "median_src": float(np.median(src_vals)) if len(src_vals) > 0 else float("nan"),
                "mean_tgt": float(np.mean(tgt_vals)) if len(tgt_vals) > 0 else float("nan"),
                "median_tgt": float(np.median(tgt_vals)) if len(tgt_vals) > 0 else float("nan"),
            })

        rows.append({
            "Cluster": c + 1,
            "n source": n_c_s,
            "n target": n_c_t,
            "% target": round(pct_target * 100, 1),
            "Score source": round(mean_score_s, 4),
            "Score target": round(mean_score_t, 4),
            "Gap |Δscore|": round(gap, 4),
            "AKI rate source": round(aki_rate_s, 3),
            "AKI rate target": round(aki_rate_t, 3),
            "Top features": top_feats,
            "_feat_stats": feat_stats,
        })

    cluster_df = pd.DataFrame(rows).sort_values("Gap |Δscore|", ascending=False).reset_index(drop=True)
    gap_ranking = [(int(r["Cluster"]), float(r["Gap |Δscore|"])) for _, r in cluster_df.iterrows()]

    # Extract per-cluster feature stats before dropping the helper column
    cluster_feature_stats = {
        int(r["Cluster"]): r["_feat_stats"]
        for _, r in cluster_df.iterrows()
    }
    cluster_df = cluster_df.drop(columns=["_feat_stats"])

    # ── 7. UMAP 2D projection ─────────────────────────────────────────────────
    # Subsample source for UMAP readability (target always shown in full)
    max_src_umap = 800
    if n_s > max_src_umap:
        rng = np.random.default_rng(random_state)
        src_idx = rng.choice(n_s, size=max_src_umap, replace=False)
    else:
        src_idx = np.arange(n_s)

    X_umap_input = np.vstack([X_pca_s[src_idx], X_pca_t])
    labels_umap = np.concatenate([labels_s[src_idx], labels_t])
    domain_umap = np.array(["Source"] * len(src_idx) + ["Target"] * n_t)

    try:
        import umap
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=umap_neighbors,
            min_dist=0.3,
            random_state=random_state,
        )
        embedding = reducer.fit_transform(X_umap_input)
    except ImportError:
        logger.warning("umap-learn not available; falling back to PCA 2D projection.")
        embedding = X_umap_input[:, :2]

    fig_b64 = _make_umap_figure(embedding, domain_umap, labels_umap, n_clusters)

    return {
        "fig_b64": fig_b64,
        "cluster_table": cluster_df,
        "cluster_feature_stats": cluster_feature_stats,
        "gap_ranking": gap_ranking,
        "n_clusters": n_clusters,
        "pca_k_used": k,
        "pca_variance_explained": float(pca.explained_variance_ratio_.sum()),
        "pca_k_rationale": pca_rationale,
        "variance_threshold": variance_threshold,
    }


# ── Figure ────────────────────────────────────────────────────────────────────

def _make_umap_figure(
    embedding: np.ndarray,
    domain: np.ndarray,
    cluster_labels: np.ndarray,
    n_clusters: int,
) -> str:
    """Return base64-encoded PNG of a dual-panel UMAP figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    CLUSTER_COLORS = plt.cm.tab10(np.linspace(0, 1, max(n_clusters, 10)))
    DOMAIN_MARKERS = {"Source": "o", "Target": "^"}
    DOMAIN_SIZES   = {"Source": 8, "Target": 60}
    DOMAIN_ALPHA   = {"Source": 0.35, "Target": 0.95}

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Patient Profile Clustering — UMAP projection", fontsize=13)

    # Left panel: coloured by cluster
    ax = axes[0]
    ax.set_title("Coloured by cluster", fontsize=11)
    for c in range(n_clusters):
        for dom in ("Source", "Target"):
            mask = (cluster_labels == c) & (domain == dom)
            if mask.sum() == 0:
                continue
            ax.scatter(
                embedding[mask, 0], embedding[mask, 1],
                c=[CLUSTER_COLORS[c]],
                marker=DOMAIN_MARKERS[dom],
                s=DOMAIN_SIZES[dom],
                alpha=DOMAIN_ALPHA[dom],
                linewidths=0,
            )
    # Legend: clusters
    cluster_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=CLUSTER_COLORS[c],
               markersize=8, label=f"Cluster {c + 1}")
        for c in range(n_clusters)
    ]
    ax.legend(handles=cluster_handles, fontsize=7, loc="best", framealpha=0.7)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.axis("off")

    # Right panel: coloured by domain
    ax = axes[1]
    ax.set_title("Coloured by domain", fontsize=11)
    domain_colors = {"Source": "#4477AA", "Target": "#EE6677"}
    for dom in ("Source", "Target"):
        mask = domain == dom
        ax.scatter(
            embedding[mask, 0], embedding[mask, 1],
            c=domain_colors[dom],
            marker=DOMAIN_MARKERS[dom],
            s=DOMAIN_SIZES[dom],
            alpha=DOMAIN_ALPHA[dom],
            linewidths=0,
            label=dom,
        )
    ax.legend(fontsize=9, loc="best", framealpha=0.7)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.axis("off")

    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")
