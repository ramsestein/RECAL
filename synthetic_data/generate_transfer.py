"""Generate a TRANSFERRED synthetic dataset with recoverable domain shift.

Generates a fresh 150-row target cohort (unseen by source model), applies
moderate covariate shift (mean/scale) plus mild concept shift (weaker
coefficients), recomputes F3/FA, and builds the target.  This mirrors the
drift pattern used in RECAL's own e2e tests.
"""
from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.RandomState(43)

# ── 1. Generate a fresh target cohort (never seen by source model) ─────────
data_dir = Path(__file__).parent
n_target = 150

target_rng = np.random.RandomState(44)
base = target_rng.randn(n_target, 20).astype(np.float32)
n = base.shape[0]

# ── 2. Apply covariate shift (mean + scale) ─────────────────────────────────
# Mean shifts on the first two features (like RECAL's own e2e test)
base[:, 0] += 1.5
base[:, 1] -= 1.0

# Scale shifts on a few others
base[:, 2] *= 1.4
base[:, 5] *= 0.7
base[:, 8] *= 1.3
base[:, 12] *= 0.6
base[:, 15] *= 1.2
base[:, 18] *= 0.8

# ── 3. Recompute F3 (same random triples as source) ─────────────────────────
cols = {}
for i in range(20):
    cols[f"X{i:02d}"] = base[:, i]

rng_orig = np.random.RandomState(42)
for j in range(10):
    idx = rng_orig.choice(20, 3, replace=False)
    a, b, c = base[:, idx[0]], base[:, idx[1]], base[:, idx[2]]
    func = (
        0.5 * a
        + 0.3 * b
        + 0.2 * c
        + 0.1 * a * b
        + 0.05 * target_rng.randn(n)
    )
    cols[f"F3_{j:02d}"] = func.astype(np.float32)

# ── 4. Recompute FA (same weights as source) ──────────────────────────────
for j in range(10):
    weights = rng_orig.randn(20)
    weights /= np.linalg.norm(weights)
    linear = base @ weights
    func = (
        0.6 * linear
        + 0.3 * linear ** 2
        + 0.1 * target_rng.randn(n)
    )
    cols[f"FA_{j:02d}"] = func.astype(np.float32)

# ── 5. Fresh noise columns ──────────────────────────────────────────────────
for j in range(10):
    cols[f"N{j:02d}"] = target_rng.randn(n).astype(np.float32)

# ── 6. Recompute target with MILD concept shift (weaker coefficients) ───────
# The same functional form but with attenuated coefficients simulates a
# scenario where the biological/clinical signal is slightly different.
x0, x1, x2 = cols["X00"], cols["X01"], cols["X02"]
f3_0, f3_1, f3_2 = cols["F3_00"], cols["F3_01"], cols["F3_02"]
fa_0, fa_1 = cols["FA_00"], cols["FA_01"]

target = (
    1.5 * x0          # was 2.0  (-25%)
    - 1.2 * x1        # was -1.5 (-20%)
    + 0.6 * x2        # was 0.8  (-25%)
    + 0.9 * f3_0      # was 1.2  (-25%)
    - 0.7 * f3_1      # was -0.9 (-22%)
    + 0.4 * f3_2      # was 0.5  (-20%)
    + 0.75 * fa_0     # was 1.0  (-25%)
    - 0.5 * fa_1      # was -0.7 (-29%)
    + 0.45 * (x0 ** 2)  # was 0.6 (-25%)
    - 0.3 * (x1 * x2)   # was -0.4 (-25%)
    + 0.22 * (f3_0 * f3_1)  # was 0.3 (-27%)
    + 0.15 * (fa_0 * fa_1)  # was 0.2 (-25%)
    + 0.08 * (x0 * f3_0 * fa_0)  # was 0.1 (-20%)
    + 0.30 * target_rng.randn(n)  # more noise (was 0.05)
)

# Sigmoid + adaptive threshold to match source prevalence (~0.59)
target_prob = 1.0 / (1.0 + np.exp(-target))
# Use the 41st percentile as threshold to get ~59% prevalence
threshold = float(np.percentile(target_prob, 41.0))
target_binary = (target_prob >= threshold).astype(int)
cols["target"] = target_binary

# ── 7. Extra mild scaling on 10 features ─────────────────────────────────────
feature_names = [c for c in cols if c != "target"]
perturbed = rng.choice(feature_names, 10, replace=False)
for fname in perturbed:
    factor = rng.uniform(0.7, 1.4)
    cols[fname] = (cols[fname] * factor).astype(np.float32)
print(f"Mildly scaled 10 features: {perturbed[:5]}...")

# ── Save ─────────────────────────────────────────────────────────────────────
df = pd.DataFrame(cols)
out_path = data_dir / "synthetic_transfer.csv"
df.to_csv(out_path, index=False)

print(f"Saved: {out_path}")
print(f"Shape: {df.shape}")
print(f"Target prevalence: {target_binary.mean():.3f} ({target_binary.sum()}/{len(target_binary)})")
