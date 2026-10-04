"""Empirical survival curves, with the same inclusive threshold everywhere."""

from __future__ import annotations

from fractions import Fraction
from math import ceil

import numpy as np


def threshold_at_probability(samples: np.ndarray, percent: float) -> float:
    """Largest observed t with P(X >= t) >= percent/100.

    Integer ranks and a decimal rational probability avoid floating-point
    off-by-one errors at exact sample-count boundaries (e.g. 90% of 10).
    """
    values = np.asarray(samples, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Expected a nonempty finite one-dimensional sample")
    if not 0 < percent <= 100:
        raise ValueError("percent must be in (0, 100]")
    rank = len(values) - ceil(len(values) * Fraction(str(percent)) / 100)
    return float(np.partition(values, rank)[rank])


def availability_curve(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(samples, dtype=float)
    values = values[np.isfinite(values)]
    xs, counts = np.unique(values, return_counts=True)
    if not len(xs):
        return xs, xs.copy()
    return xs, (len(values) - np.cumsum(counts) + counts) / len(values)


def curve_on_grid(
    xs: np.ndarray, availability: np.ndarray, grid: np.ndarray
) -> np.ndarray:
    """Evaluate P(sample >= threshold) without linear interpolation of an ECDF."""
    if not len(xs):
        return np.full_like(grid, np.nan, dtype=float)
    indices = np.searchsorted(xs, grid, side="left")
    return np.append(availability, 0.0)[indices]
