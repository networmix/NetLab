#!/usr/bin/env python3
"""Compute metrics for one simulation result."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from netlab.metrics.bac import BacResult, compute_bac, plot_bac
from netlab.metrics.costpower import (
    CostPowerResult,
    compute_cost_power,
    plot_cost_power,
)
from netlab.metrics.iterops import IterOpsResult, compute_iter_ops
from netlab.metrics.latency import LatencyResult, compute_latency_stretch, plot_latency
from netlab.metrics.msd import AlphaResult, compute_alpha_star
from netlab.metrics.sps import SpsResult, compute_sps

from .validation import (
    _require_steps,
    _validate_alpha_and_base_demands,
    _validate_maxflow_baseline,
    _validate_tm_placement_baseline,
)


def analyze_one_seed(
    results: dict, out_dir: Path, do_plots: bool, *, enable_maxflow: bool = False
) -> Tuple[
    AlphaResult,
    BacResult,
    Optional[BacResult],
    LatencyResult,
    CostPowerResult,
    IterOpsResult,
    Optional[SpsResult],
]:
    _require_steps(results, enable_maxflow)
    expected_total_at_alpha = _validate_alpha_and_base_demands(results)
    _validate_tm_placement_baseline(results, expected_total_at_alpha)
    if enable_maxflow:
        _validate_maxflow_baseline(results)

    alpha = compute_alpha_star(results)

    bac_place = compute_bac(results, step_name="tm_placement", mode="auto")
    bac_max = (
        compute_bac(results, step_name="node_to_node_capacity_matrix", mode="auto")
        if enable_maxflow
        else None
    )

    latency = compute_latency_stretch(results)
    iterops = compute_iter_ops(results)

    sps_res: Optional[SpsResult] = None
    if enable_maxflow:
        sps_res = compute_sps(results)

    offered_alpha_star = None
    if not np.isnan(alpha.base_total_demand) and np.isfinite(alpha.alpha_star):
        offered_alpha_star = float(alpha.base_total_demand * alpha.alpha_star)
    reliable_p999 = bac_place.bw_at_probability_abs.get(99.9, np.nan)
    costpower = compute_cost_power(
        results, offered_demand=offered_alpha_star, reliable_at_p999=reliable_p999
    )

    if do_plots:
        if out_dir:
            out_dir.mkdir(parents=True, exist_ok=True)
        plot_bac(
            bac_place,
            overlay=bac_max if bac_max is not None else None,
            save_to=out_dir / "bac.png",
        )
        plot_latency(latency, save_to=out_dir / "latency.png")
        plot_cost_power(costpower, save_to=out_dir / "costpower.png")

    return alpha, bac_place, bac_max, latency, costpower, iterops, sps_res
