from __future__ import annotations

import math

import numpy as np


def paired_normal_test_holm(a: np.ndarray, b: np.ndarray) -> dict:
    """
    Paired test: compute differences d = a - b. For large n (>=30), use normal approx:
        z = mean(d) / (std(d)/sqrt(n)),  two-sided p = 2 * (1 - Phi(|z|)).
    Returns dict with t_stat (z here), raw p, Holm-adjusted p across a family-of-one.
    """
    assert a.shape == b.shape
    d = a.astype(float) - b.astype(float)
    d = d[np.isfinite(d)]
    n = d.size
    if n < 3:
        return {
            "t_stat": float("nan"),
            "p": float("nan"),
            "p_adj": float("nan"),
            "n": int(n),
        }

    mean = float(np.mean(d))
    sd = float(np.std(d, ddof=1)) if n > 1 else 0.0
    if sd == 0.0:
        z = float("inf") if mean != 0 else 0.0
        p = 0.0 if mean != 0 else 1.0
    else:
        z = mean / (sd / math.sqrt(n))
        # 2-sided p under N(0,1): p = 2 * (1 - Phi(|z|))
        # Phi via erf: Phi(x) = 0.5*(1+erf(x/sqrt(2)))
        p = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z) / math.sqrt(2.0))))

    # Holm for a single test is the same p
    return {"t_stat": float(z), "p": float(p), "p_adj": float(p), "n": int(n)}


def paired_wilcoxon_sign(a: np.ndarray, b: np.ndarray) -> dict:
    """
    Nonparametric paired test (sign test).
    Returns: z, two-sided p (normal approx), n_effective.
    Ignores ties (zero differences). Suitable for n >= ~10.
    """
    assert a.shape == b.shape
    d = a.astype(float) - b.astype(float)
    d = d[np.isfinite(d) & (d != 0.0)]
    n = d.size
    if n < 5:
        return {"z": float("nan"), "p": float("nan"), "n": int(n)}
    # Count positives under H0: Binomial(n, 0.5)
    k = float((d > 0).sum())
    mean = 0.5 * n
    sd = math.sqrt(0.25 * n)
    if sd == 0.0:
        z = float("inf") if k != mean else 0.0
    else:
        z = (k - mean) / sd
    # Two-sided p under N(0,1)
    p = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z) / math.sqrt(2.0))))
    return {"z": float(z), "p": float(p), "n": int(n)}
