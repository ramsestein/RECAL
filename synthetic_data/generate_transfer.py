"""Generate a TRANSFERRED synthetic dataset by applying domain-shift transformations.

Reads the original synthetic_features.csv, applies monotonic/non-linear transforms
to the 20 base columns, recomputes the engineered features (F3, FA), recomputes
the polynomial target, and finally applies a non-linear warp:
    new_target = sin(target ** log(target)) * 42
"""
from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.RandomState(43)

# ── 1. Load original data ───────────────────────────────────────────────────
data_dir = Path(__file__).parent
df_orig = pd.read_csv(data_dir / "synthetic_features.csv")

# Extract the 20 independent columns
base_names = [f"X{i:02d}" for i in range(20)]
base = df_orig[base_names].values.astype(np.float32)

# ── 2. Apply domain-shift transformations to each of the 20 base columns ────
# Each column gets a distinct monotonic or mild non-linear warp
n = base.shape[0]
transformed = np.empty_like(base)

transformed[:, 0]  = base[:, 0] ** 2                     # x^2
transformed[:, 1]  = np.sqrt(np.abs(base[:, 1]))        # sqrt(|x|)
transformed[:, 2]  = base[:, 2] ** 3                     # x^3
transformed[:, 3]  = np.sin(base[:, 3])                  # sin(x)
transformed[:, 4]  = np.exp(base[:, 4] / 5.0)            # exp(x/5)
transformed[:, 5]  = np.log1p(np.abs(base[:, 5]))        # log(|x|+1)
transformed[:, 6]  = np.tanh(base[:, 6] * 2.0)           # tanh(2x)
transformed[:, 7]  = base[:, 7] * 2.0                    # 2x
transformed[:, 8]  = base[:, 8] + 3.0                    # x+3
transformed[:, 9]  = base[:, 9] ** 2 + base[:, 9]        # x^2 + x
transformed[:, 10] = np.abs(base[:, 10])                 # |x|
transformed[:, 11] = base[:, 11] * 0.5                   # x/2
transformed[:, 12] = base[:, 12] ** 3 - base[:, 12]      # x^3 - x
transformed[:, 13] = np.cos(base[:, 13])                 # cos(x)
transformed[:, 14] = base[:, 14] ** 2 * np.sign(base[:, 14])  # x^2 * sign(x)
transformed[:, 15] = np.log1p(np.abs(base[:, 15]))       # log(|x|+1) again
transformed[:, 16] = base[:, 16] * (1.0 + 0.1 * base[:, 16])  # x(1+0.1x)
transformed[:, 17] = base[:, 17] ** 2 + 2 * base[:, 17] + 1  # (x+1)^2
transformed[:, 18] = np.sqrt(base[:, 18] ** 2 + 1.0)     # sqrt(x^2+1)
transformed[:, 19] = base[:, 19] * np.sin(base[:, 19])   # x*sin(x)

# ── 3. Recompute F3 (functions of 3 random columns) ─────────────────────────
# Use the SAME random seed as original so we pick the SAME triples
cols = {}
for i in range(20):
    cols[f"X{i:02d}"] = transformed[:, i]

rng_orig = np.random.RandomState(42)
for j in range(10):
    idx = rng_orig.choice(20, 3, replace=False)
    a, b, c = transformed[:, idx[0]], transformed[:, idx[1]], transformed[:, idx[2]]
    func = (
        0.5 * np.sin(a)
        + 0.3 * np.cos(b)
        + 0.2 * (c ** 2)
        + 0.1 * a * b
        + 0.05 * rng.randn(n)
    )
    cols[f"F3_{j:02d}"] = func.astype(np.float32)

# ── 4. Recompute FA (functions of all 20 columns) ───────────────────────────
for j in range(10):
    weights = rng_orig.randn(20)
    weights /= np.linalg.norm(weights)
    linear = transformed @ weights
    func = (
        0.6 * linear
        + 0.3 * np.tanh(linear)
        + 0.1 * rng.randn(n)
    )
    cols[f"FA_{j:02d}"] = func.astype(np.float32)

# ── 5. Keep noise columns (same random seed = same values) ────────────────────
for j in range(10):
    cols[f"N{j:02d}"] = rng_orig.randn(n).astype(np.float32)

# ── 6. Use ORIGINAL target from source (no concept shift, only covariate shift) ─
target_binary = df_orig["target"].values.astype(int)
cols["target"] = target_binary

# ── 8. Extra domain shift: scale half the features by random factors ────────
feature_names = [c for c in cols if c != "target"]
n_features = len(feature_names)
n_to_perturb = n_features // 2  # 25 features
perturbed = rng.choice(feature_names, n_to_perturb, replace=False)
for fname in perturbed:
    scale_factor = rng.uniform(0.3, 2.5)
    cols[fname] = (cols[fname] * scale_factor).astype(np.float32)
print(f"Perturbed {n_to_perturb} features with random scale factors: {perturbed[:5]}...")

# ── 9. Replace 5 random features with pure noise (no relation to original) ───
feature_names = [c for c in cols if c != "target"]
randomised = rng.choice(feature_names, 5, replace=False)
for fname in randomised:
    cols[fname] = rng.randn(n).astype(np.float32)
print(f"Replaced 5 features with pure noise: {randomised}")

# ── Save ─────────────────────────────────────────────────────────────────────
df = pd.DataFrame(cols)
out_path = data_dir / "synthetic_transfer.csv"
df.to_csv(out_path, index=False)

print(f"Saved: {out_path}")
print(f"Shape: {df.shape}")
print(f"Target prevalence: {target_binary.mean():.3f} ({target_binary.sum()}/{len(target_binary)})")
