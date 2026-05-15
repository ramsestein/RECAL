"""
domain_transfer.align.optimal_transport
=========================================
Optimal Transport alignment using the Sinkhorn regularised OT plan (POT).

Background
----------
Optimal Transport finds the minimum-cost mapping that transports the target
distribution to the source distribution.  The Sinkhorn regularised variant
adds an entropy term to make the OT problem strictly convex and efficiently
solvable with the Sinkhorn–Knopp algorithm.

We use ``ot.da.SinkhornTransport`` from the POT library with the exact
same convention as the legacy ``w_alignment_eval.py`` script:

    transport.fit(Xs=X_target, Xt=X_source)

Note: the POT convention swaps source/target relative to our nomenclature —
we pass *our* target as ``Xs`` and *our* source as ``Xt``.  Internally, OT
then maps ``Xs → Xt``, i.e., maps our target distribution toward the source.

Because fitting OT is O(n_t² × n_s) in memory and time, a random sub-sample
of ``max_src_samples`` source samples is used when n_s > ``max_src_samples``.

Limitations
-----------
* Computationally expensive: O(n² log n) per feature dimension.
* Sensitive to outliers in both cohorts.
* The Sinkhorn approximation quality depends on the regularisation parameter
  ``reg``.  Small reg → better approximation but slower convergence / more
  iterations; large reg → blurrier transport but faster.

References
----------
Flamary, R., et al. (2021). POT: Python Optimal Transport.
https://pythonot.github.io/
"""

from __future__ import annotations

import logging

import numpy as np

from domain_transfer.align.base import Aligner, _restore_nan

logger = logging.getLogger(__name__)


class OTAligner(Aligner):
    """
    Sinkhorn Optimal Transport alignment (via POT ``ot.da.SinkhornTransport``).

    Parameters
    ----------
    reg : float
        Regularisation parameter for the Sinkhorn algorithm.
        Default 0.1 (matches legacy script).
    max_src_samples : int
        Maximum number of source samples to use when fitting the transport
        plan.  Source is randomly sub-sampled if n_s > max_src_samples.
        Default 1000.
    random_state : int
        Seed for reproducible sub-sampling.
    """

    def __init__(
        self,
        reg: float = 0.1,
        max_src_samples: int = 1000,
        random_state: int = 42,
    ) -> None:
        self.reg = reg
        self.max_src_samples = max_src_samples
        self.random_state = random_state
        self._transport = None

    def fit(self, X_source: np.ndarray, X_target: np.ndarray) -> "OTAligner":
        """
        Fit the Sinkhorn transport plan.

        Parameters
        ----------
        X_source : np.ndarray
            Shape (n_s, q).  No NaN.
        X_target : np.ndarray
            Shape (n_t, q).  No NaN.
        """
        try:
            import ot  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "POT (Python Optimal Transport) is required for OTAligner. "
                "Install it with:  pip install POT"
            ) from exc

        rng = np.random.default_rng(self.random_state)

        # Sub-sample source if needed (memory / time)
        Xs_fit = X_source
        if X_source.shape[0] > self.max_src_samples:
            idx = rng.choice(X_source.shape[0], self.max_src_samples, replace=False)
            Xs_fit = X_source[idx]
            logger.info(
                "OTAligner.fit: sub-sampled source from %d → %d samples.",
                X_source.shape[0],
                self.max_src_samples,
            )

        # POT convention: Xs=our_target, Xt=our_source → maps target→source
        self._transport = ot.da.SinkhornTransport(reg_e=self.reg)
        self._transport.fit(Xs=X_target, Xt=Xs_fit)
        return self

    def transform(
        self,
        X_target: np.ndarray,
        nan_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Apply the learned OT transport map to the target matrix.
        """
        if self._transport is None:
            raise RuntimeError("OTAligner must be fitted before transform.")

        X_aligned = self._transport.transform(Xs=X_target)
        return _restore_nan(X_aligned, nan_mask)
