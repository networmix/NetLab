from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .common import nonnegative_number


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
        alpha = nonnegative_number(msd["alpha_star"], "alpha_star")
        demands = msd["base_demands"]
        if not isinstance(demands, list) or not demands:
            raise ValueError("base_demands must be a nonempty list")
        base_total = nonnegative_number(
            sum(nonnegative_number(demand["volume"], "volume") for demand in demands),
            "base_total_demand",
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "msd_baseline.data requires alpha_star and base_demands[].volume"
        ) from exc
    return AlphaResult(
        alpha_star=alpha, source="msd_baseline", base_total_demand=base_total
    )
