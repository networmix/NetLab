from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class AlphaResult:
    alpha_star: float
    source: str  # MSD workflow step name
    base_total_demand: float  # sum of baseline demand volumes

    def to_jsonable(self) -> dict:
        return {
            "alpha_star": float(self.alpha_star),
            "source": self.source,
            "base_total_demand": float(self.base_total_demand)
            if not np.isnan(self.base_total_demand)
            else None,
        }


def compute_alpha_star(results: dict) -> AlphaResult:
    try:
        msd = results["steps"]["msd_baseline"]["data"]
        alpha = float(msd["alpha_star"])
        base_total = sum(float(demand["volume"]) for demand in msd["base_demands"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "msd_baseline.data requires alpha_star and base_demands[].volume"
        ) from exc
    return AlphaResult(
        alpha_star=alpha, source="msd_baseline", base_total_demand=base_total
    )
