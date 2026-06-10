"""Generate a synthetic dataset with engineered features and a polynomial target.

Structure:
- 20 independent random columns (X0-X19)
- 10 functions of 3 random columns each (X20-X29)
- 10 functions of all 20 random columns (X30-X39)
- 10 extra random noise columns (X40-X49)
- 1 polynomial target using:
  * 3 random columns (X0, X1, X2)
  * 3 "function-of-3" columns (X20, X21, X22)
  * 2 "function-of-all" columns (X30, X31)

Total features: 50 + target.
"""
import numpy as np
import pandas as pd
from pathlib import Path

rng = np.random.RandomState(42)
n_samples = 1000

# ── 1. 20 independent random columns ─────────────────────────────────────────
base = rng.randn(n_samples, 20).astype(np.float32)
cols = {}
for i in range(20):
    cols[f"X{i:02d}"] = base[:, i]

# ── 2. 10 functions of 3 random columns each ─────────────────────────────────
# Each uses a different triple of base columns
for j in range(10):
    idx = rng.choice(20, 3, replace=False)
    a, b, c = base[:, idx[0]], base[:, idx[1]], base[:, idx[2]]
    # Non-linear interaction
    func = (
        0.5 * np.sin(a)
        + 0.3 * np.cos(b)
        + 0.2 * (c ** 2)
        + 0.1 * a * b
        + 0.05 * rng.randn(n_samples)
    )
    cols[f"F3_{j:02d}"] = func.astype(np.float32)

# ── 3. 10 functions of all 20 random columns ─────────────────────────────────
# Each is a different random linear combination + non-linear twist
for j in range(10):
    weights = rng.randn(20)
    weights /= np.linalg.norm(weights)
    linear = base @ weights
    # Add a mild non-linearity
    func = (
        0.6 * linear
        + 0.3 * np.tanh(linear)
        + 0.1 * rng.randn(n_samples)
    )
    cols[f"FA_{j:02d}"] = func.astype(np.float32)

# ── 4. 10 extra random noise columns ─────────────────────────────────────────
for j in range(10):
    cols[f"N{j:02d}"] = rng.randn(n_samples).astype(np.float32)

# ── 5. Polynomial target ─────────────────────────────────────────────────────
# Uses:
#   3 random columns  -> X00, X01, X02
#   3 F3 columns      -> F3_00, F3_01, F3_02
#   2 FA columns      -> FA_00, FA_01
x0, x1, x2 = cols["X00"], cols["X01"], cols["X02"]
f3_0, f3_1, f3_2 = cols["F3_00"], cols["F3_01"], cols["F3_02"]
fa_0, fa_1 = cols["FA_00"], cols["FA_01"]

# Polynomial with interactions and squared terms
target = (
    2.0 * x0
    - 1.5 * x1
    + 0.8 * x2
    + 1.2 * f3_0
    - 0.9 * f3_1
    + 0.5 * f3_2
    + 1.0 * fa_0
    - 0.7 * fa_1
    + 0.6 * (x0 ** 2)
    - 0.4 * (x1 * x2)
    + 0.3 * (f3_0 * f3_1)
    + 0.2 * (fa_0 * fa_1)
    + 0.1 * (x0 * f3_0 * fa_0)
    + 0.05 * rng.randn(n_samples)
)

# Normalise to [0,1] via sigmoid, then threshold at 0.5 for binary label
target_prob = 1.0 / (1.0 + np.exp(-target))
target_binary = (target_prob >= 0.5).astype(int)

cols["target"] = target_binary

# ── Save ─────────────────────────────────────────────────────────────────────
df = pd.DataFrame(cols)
out_path = Path(__file__).with_name("synthetic_features.csv")
df.to_csv(out_path, index=False)

print(f"Saved: {out_path}")
print(f"Shape: {df.shape}")
print(f"Target prevalence: {target_binary.mean():.3f} ({target_binary.sum()}/{len(target_binary)})")
print(f"Columns: {list(df.columns)}")
