"""
recal.align
=====================
Domain alignment algorithms for covariate shift correction.

Available aligners
------------------
- :class:`~recal.align.identity.IdentityAligner` — Raw baseline (no alignment).
- :class:`~recal.align.coral.CoralAligner` — Full CORAL (requires n >> p).
- :class:`~recal.align.pca_coral.PCACoralAligner` — PCA-CORAL (default k=5; best result).
- :class:`~recal.align.adabn.AdaBNAligner` — Diagonal AdaBN (per-feature mean/std).
- :class:`~recal.align.optimal_transport.OTAligner` — Sinkhorn OT (requires POT).
- :class:`~recal.align.selective.SelectiveAligner` — Decorator restricting alignment to a feature subset.
- :class:`~recal.align.quantile_transform.QuantileTransformAligner` — Per-feature quantile matching.

Quick reference
---------------
>>> from recal.align import PCACoralAligner, SelectiveAligner, QuantileTransformAligner
>>> aligner = SelectiveAligner(PCACoralAligner(k=5), feature_indices=bottom_10_idx)
>>> X_aligned = pair.align(aligner)
"""

from recal.align.adabn import AdaBNAligner
from recal.align.base import Aligner, safe_invsqrtm, safe_sqrtm
from recal.align.coral import CoralAligner
from recal.align.identity import IdentityAligner
from recal.align.optimal_transport import OTAligner
from recal.align.pca_coral import PCACoralAligner
from recal.align.quantile_transform import QuantileTransformAligner
from recal.align.selective import SelectiveAligner

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
