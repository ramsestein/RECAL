"""
domain_transfer.align
=====================
Domain alignment algorithms for covariate shift correction.

Available aligners
------------------
- :class:`~domain_transfer.align.identity.IdentityAligner` — Raw baseline (no alignment).
- :class:`~domain_transfer.align.coral.CoralAligner` — Full CORAL (requires n >> p).
- :class:`~domain_transfer.align.pca_coral.PCACoralAligner` — PCA-CORAL (default k=5; best result).
- :class:`~domain_transfer.align.adabn.AdaBNAligner` — Diagonal AdaBN (per-feature mean/std).
- :class:`~domain_transfer.align.optimal_transport.OTAligner` — Sinkhorn OT (requires POT).
- :class:`~domain_transfer.align.selective.SelectiveAligner` — Decorator restricting alignment to a feature subset.
- :class:`~domain_transfer.align.quantile_transform.QuantileTransformAligner` — Per-feature quantile matching.

Quick reference
---------------
>>> from domain_transfer.align import PCACoralAligner, SelectiveAligner, QuantileTransformAligner
>>> aligner = SelectiveAligner(PCACoralAligner(k=5), feature_indices=bottom_10_idx)
>>> X_aligned = pair.align(aligner)
"""

from domain_transfer.align.adabn import AdaBNAligner
from domain_transfer.align.base import Aligner, safe_invsqrtm, safe_sqrtm
from domain_transfer.align.coral import CoralAligner
from domain_transfer.align.identity import IdentityAligner
from domain_transfer.align.optimal_transport import OTAligner
from domain_transfer.align.pca_coral import PCACoralAligner
from domain_transfer.align.quantile_transform import QuantileTransformAligner
from domain_transfer.align.selective import SelectiveAligner

__all__ = [
    "Aligner",
    "safe_sqrtm",
    "safe_invsqrtm",
    "IdentityAligner",
    "CoralAligner",
    "PCACoralAligner",
    "AdaBNAligner",
    "OTAligner",
    "SelectiveAligner",
    "QuantileTransformAligner",
]
