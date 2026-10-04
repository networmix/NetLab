"""Paired Student t inference and Holm correction shared by all reports."""

from __future__ import annotations

import math
from typing import Hashable, TypeVar

import numpy as np
import pandas as pd
from scipy import stats

K = TypeVar("K", bound=Hashable)


def paired_t(a: np.ndarray, b: np.ndarray, alpha: float = 0.05) -> dict:
    """Two-sided paired test and CI; only finite paired observations participate."""
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError("paired_t requires one-dimensional arrays with the same shape")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between zero and one")
    finite = np.isfinite(a) & np.isfinite(b)
    with np.errstate(over="ignore"):
        differences = a[finite].astype(float) - b[finite].astype(float)
    if not np.isfinite(differences).all():
        raise ValueError("Paired differences overflow float64")
    n = len(differences)
    scale = float(np.max(np.abs(differences))) if n else 0.0
    scaled = differences / scale if scale else differences
    mean_scaled = float(scaled.mean()) if n else float("nan")
    mean = mean_scaled * scale if scale else mean_scaled
    result = dict(
        n=n,
        mean_diff=mean,
        t_stat=float("nan"),
        p=float("nan"),
        ci_low=float("nan"),
        ci_high=float("nan"),
        deterministic=False,
    )
    if n < 3:
        return result
    if np.all(differences == differences[0]):
        result.update(
            t_stat=math.copysign(float("inf"), mean) if mean != 0.0 else 0.0,
            p=0.0 if mean != 0.0 else 1.0,
            ci_low=mean,
            ci_high=mean,
            deterministic=True,
        )
    else:
        se_scaled = float(scaled.std(ddof=1)) / math.sqrt(n)
        t = mean_scaled / se_scaled
        margin_scaled = float(stats.t.ppf(1 - alpha / 2, df=n - 1)) * se_scaled
        result.update(
            t_stat=t,
            p=float(2 * stats.t.sf(abs(t), df=n - 1)),
            ci_low=(mean_scaled - margin_scaled) * scale,
            ci_high=(mean_scaled + margin_scaled) * scale,
        )
    return result


def holm_adjust(p_values: list[tuple[K, float]]) -> dict[K, float]:
    """Correct the finite tests in a family; unavailable tests remain NaN."""
    if len({key for key, _ in p_values}) != len(p_values):
        raise ValueError("Holm test identifiers must be unique")
    if any(not math.isnan(p) and not 0 <= p <= 1 for _, p in p_values):
        raise ValueError("p-values must be between zero and one, or NaN")
    valid = sorted(
        ((key, p) for key, p in p_values if math.isfinite(p)), key=lambda x: x[1]
    )
    result = dict.fromkeys((key for key, _ in p_values), float("nan"))
    previous = 0.0
    for index, (key, p) in enumerate(valid):
        if not 0 <= p <= 1:
            raise ValueError("p-values must be between zero and one")
        previous = max(previous, min(1.0, (len(valid) - index) * p))
        result[key] = previous
    return result


def holm_series(series: pd.Series) -> pd.Series:
    adjusted = holm_adjust([(key, float(value)) for key, value in series.items()])
    return pd.Series(adjusted).reindex(series.index)
