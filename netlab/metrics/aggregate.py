"""Cross-seed summaries of scalar estimands.

Failure pattern positions are not paired observations across independent seeds.
Distribution comparisons must evaluate survival curves on common thresholds.
"""

from __future__ import annotations

import numpy as np


def summarize_across_seeds(values_by_seed: dict[int, float]) -> dict:
    values = np.array([float(v) for v in values_by_seed.values() if v is not None])
    values = values[np.isfinite(values)]
    if not len(values):
        return {}
    return {
        "type": "scalar",
        "median": float(np.median(values)),
        "q25": float(np.percentile(values, 25)),
        "q75": float(np.percentile(values, 75)),
    }
