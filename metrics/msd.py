from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class AlphaResult:
    alpha_star: float
    source: str  # 'msd_baseline' | 'probes' | 'unknown'
    base_total_demand: float  # sum of baseline TM demands (if available), else NaN

    def to_jsonable(self) -> dict:
        return {
            "alpha_star": float(self.alpha_star),
            "source": self.source,
            "base_total_demand": float(self.base_total_demand)
            if not np.isnan(self.base_total_demand)
            else None,
        }


def compute_alpha_star(results: dict) -> AlphaResult:
    msd = results.get("steps", {}).get("msd_baseline", {}).get("data", {}) or {}
    alpha = msd.get("alpha_star", None)
    base_total = np.nan
    try:
        base_demands = msd.get("base_demands", [])
        base_total = float(sum([float(x.get("demand", 0.0)) for x in base_demands]))
    except Exception:
        base_total = np.nan

    if alpha is not None:
        return AlphaResult(
            alpha_star=float(alpha), source="msd_baseline", base_total_demand=base_total
        )

    # Fallback: probe table under tm_placement.metadata.probes (binary search record)
    probes = (
        results.get("steps", {})
        .get("tm_placement", {})
        .get("metadata", {})
        .get("probes", None)
    )
    if isinstance(probes, list) and probes:
        # choose highest feasible alpha OR interpolate on min_placement_ratio around 1.0
        feas = [p for p in probes if bool(p.get("feasible"))]
        if feas:
            best = max(feas, key=lambda r: float(r.get("alpha", 0.0)))
            return AlphaResult(
                alpha_star=float(best.get("alpha", 1.0)),
                source="probes",
                base_total_demand=base_total,
            )
        # else: find the last >=1.0 min_placement_ratio — conservative estimate
        near = [p for p in probes if float(p.get("min_placement_ratio", 0.0)) >= 1.0]
        if near:
            best = max(near, key=lambda r: float(r.get("alpha", 0.0)))
            return AlphaResult(
                alpha_star=float(best.get("alpha", 1.0)),
                source="probes",
                base_total_demand=base_total,
            )

    return AlphaResult(
        alpha_star=float("nan"), source="unknown", base_total_demand=base_total
    )
