from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass
class CostPowerResult:
    capex_total: float
    power_total_w: float
    per_metro: (
        pd.DataFrame
    )  # columns: platform_capex, optics_capex, capex_total, power_total_watts
    # Normalizations:
    per_offered_demand__usd_per_gbit: Optional[float]
    per_offered_demand__watt_per_gbit: Optional[float]
    per_reliable_p999__usd_per_gbit: Optional[float]
    per_reliable_p999__watt_per_gbit: Optional[float]

    def flat_series(self) -> Dict[str, float]:
        return {
            "capex_total": float(self.capex_total),
            "power_total_w": float(self.power_total_w),
            "USD_per_Gbit_offered": float(self.per_offered_demand__usd_per_gbit)
            if self.per_offered_demand__usd_per_gbit is not None
            else np.nan,
            "Watt_per_Gbit_offered": float(self.per_offered_demand__watt_per_gbit)
            if self.per_offered_demand__watt_per_gbit is not None
            else np.nan,
            "USD_per_Gbit_p999": float(self.per_reliable_p999__usd_per_gbit)
            if self.per_reliable_p999__usd_per_gbit is not None
            else np.nan,
            "Watt_per_Gbit_p999": float(self.per_reliable_p999__watt_per_gbit)
            if self.per_reliable_p999__watt_per_gbit is not None
            else np.nan,
        }

    def to_jsonable(self) -> dict:
        return self.flat_series()


def compute_cost_power(
    results: dict,
    offered_at_alpha1: Optional[float] = None,
    reliable_at_p999: Optional[float] = None,
) -> CostPowerResult:
    cp = results.get("steps", {}).get("cost_power", {}).get("data", {}) or {}
    levels = cp.get("levels", {}) or {}
    root = (levels.get("0") or [{}])[0]
    capex_total = float(root.get("capex_total", 0.0))
    power_total = float(root.get("power_total_watts", 0.0))

    level1 = levels.get("1", []) or []
    df = (
        pd.DataFrame(level1)
        if level1
        else pd.DataFrame(
            columns=[
                "path",
                "platform_capex",
                "optics_capex",
                "capex_total",
                "power_total_watts",
            ]
        )
    )

    def safe_div(num, den):
        try:
            den = float(den)
            if den > 0:
                return float(num) / den
        except Exception:
            pass
        return None

    usd_per_offered = safe_div(capex_total, offered_at_alpha1)
    w_per_offered = safe_div(power_total, offered_at_alpha1)

    usd_per_p999 = safe_div(capex_total, reliable_at_p999)
    w_per_p999 = safe_div(power_total, reliable_at_p999)

    return CostPowerResult(
        capex_total=capex_total,
        power_total_w=power_total,
        per_metro=df,
        per_offered_demand__usd_per_gbit=usd_per_offered,
        per_offered_demand__watt_per_gbit=w_per_offered,
        per_reliable_p999__usd_per_gbit=usd_per_p999,
        per_reliable_p999__watt_per_gbit=w_per_p999,
    )


def plot_cost_power(cp: CostPowerResult, save_to: Optional[Path] = None) -> None:
    plt.figure()
    bars = {
        "CapEx (USD)": cp.capex_total,
        "Power (W)": cp.power_total_w,
    }
    names = list(bars.keys())
    vals = list(bars.values())
    plt.bar(names, vals)
    plt.title("Total cost & power")
    for i, v in enumerate(vals):
        plt.text(i, v, f"{v:,.0f}", ha="center", va="bottom", fontsize=8, rotation=0)
    if save_to is not None:
        save_to.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_to)
    plt.close()
