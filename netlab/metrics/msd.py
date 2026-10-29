from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from netlab.results import feasible_alpha, require_step

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
    step = require_step(results, "msd_baseline")
    alpha = feasible_alpha(step)
    demands = step["question"]["demands"]
    if not isinstance(demands, list) or not demands:
        raise ValueError("MSD question.demands must be a nonempty list")
    base_total = nonnegative_number(
        sum(
            nonnegative_number(demand["physical_volume"], "physical_volume")
            for demand in demands
        ),
        "base_total_demand",
    )
    return AlphaResult(
        alpha_star=alpha, source="msd_baseline", base_total_demand=base_total
    )
