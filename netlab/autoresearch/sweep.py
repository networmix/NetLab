"""Simulate DC-BB group counts and layouts, saving capacity and BAC metrics to JSONL."""

from __future__ import annotations

import json
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

from netlab.artifacts import ensure_run_identity, package_versions, write_text_atomic
from netlab.autoresearch.dcbb_config import DcBbScenarioConfig, get_valid_layouts
from netlab.autoresearch.dcbb_failures import FAILURE_MODE_NAMES
from netlab.autoresearch.scenario_generator import generate_scenario_with_validation
from netlab.autoresearch.scenario_validation import validate_scenario_file
from netlab.autoresearch.structural_analysis import (
    ConfigResult,
    run_structural_analysis,
)
from netlab.metrics.bac import BacResult, compute_bac
from netlab.metrics.common import expand_flow_results
from netlab.simulation import run_simulation


@dataclass
class ResultEntry:
    """One simulation result.

    ``to_dict`` stores each mode under ``bac_<mode>`` with its BAC and failure
    statistics.
    """

    g_abc1: int = 0
    g_xyz1: int = 0
    layout_abc1: str = ""
    layout_xyz1: str = ""
    alpha_star: Optional[float] = None
    bac_combined: Optional[float] = None
    bac_modes: Optional[dict[str, dict]] = (
        None  # mode_name → {auc, pct, flow_bac, failure_stats}
    )
    result_dir: str = ""
    status: str = "pending"
    error: Optional[str] = None
    duration_s: Optional[float] = None
    timestamp: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ResultEntry":
        return cls(**d)


@dataclass
class SweepConfig:
    """Configuration for a sweep (single-side or cross-side)."""

    output_dir: Path
    failure_iterations: int = 200
    timeout_s: int = 300
    seed: int = 42

    def __post_init__(self) -> None:
        if self.failure_iterations < 1 or self.timeout_s <= 0:
            raise ValueError("failure_iterations and timeout_s must be positive")


def _prepare_sweep(config: SweepConfig, mode: str) -> None:
    path = config.output_dir / "sweep.json"
    if not path.exists() and (config.output_dir / "results.jsonl").exists():
        raise ValueError("Sweep has no provenance; use a new output directory")
    ensure_run_identity(
        path,
        {
            "mode": mode,
            "seed": config.seed,
            "failure_iterations": config.failure_iterations,
            "packages": package_versions(),
        },
    )


def _dedup_configs(configs: list[ConfigResult]) -> list[ConfigResult]:
    """Deduplicate by (G, bb_block_rows, bb_block_cols)."""
    seen: dict[tuple[int, int, int], ConfigResult] = {}
    for cfg in configs:
        key = (cfg.g, cfg.bb_block_rows, cfg.bb_block_cols)
        if key not in seen:
            seen[key] = cfg
    return sorted(seen.values(), key=lambda c: (-c.g, c.bb_block_rows, c.bb_block_cols))


def _pick_layout(
    side: str,
    g: int,
    bb_block_rows: int,
    bb_block_cols: int,
    config: DcBbScenarioConfig,
) -> tuple[int, int, int, int]:
    """Pick the first valid full layout for the given (G, BB_block)."""
    if side == "abc1":
        dc_rows, dc_cols = config.abc1_hgrids, config.abc1_fadu_per_hgrid
    else:
        dc_rows, dc_cols = config.xyz1_xsw_per_plane, config.xyz1_xsw_planes
    bb_rows, bb_cols = config.bb_planes, config.bb_devices_per_plane
    gr_bb = bb_rows // bb_block_rows
    gc_bb = bb_cols // bb_block_cols
    for layout in get_valid_layouts(g, dc_rows, dc_cols, bb_rows, bb_cols):
        if layout[2] == gr_bb and layout[3] == gc_bb:
            return layout
    raise ValueError(f"No valid layout for G={g} BB={bb_block_rows}rx{bb_block_cols}c")


def _layout_notation(
    layout: tuple[int, int, int, int],
    dc_rows: int,
    dc_cols: int,
    bb_rows: int,
    bb_cols: int,
) -> str:
    """Convert layout tuple to 'DCrDCc-BBrBBc' notation."""
    gr_dc, gc_dc, gr_bb, gc_bb = layout
    return (
        f"{dc_rows // gr_dc}r{dc_cols // gc_dc}c-{bb_rows // gr_bb}r{bb_cols // gc_bb}c"
    )


def _result_dir_name(
    g_abc1: int,
    nota: str,
    g_xyz1: int,
    notx: str,
) -> str:
    """Deterministic directory name: abc1-g{G}-{notation}__xyz1-g{G}-{notation}."""
    return f"abc1-g{g_abc1}-{nota}__xyz1-g{g_xyz1}-{notx}"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _node_site(node_path: str) -> str:
    """Extract site from node path. 'bb/abc1/...' -> 'abc1', 'abc1/fadu/...' -> 'abc1'."""
    parts = node_path.split("/")
    if parts[0] == "bb" and len(parts) > 1:
        return parts[1]
    return parts[0]


def _link_sites(link_id: str) -> list[str]:
    """Extract sites from link ID. Format: 'node1|node2|hash'."""
    parts = link_id.split("|")
    sites: set[str] = set()
    for p in parts[:2]:
        sites.add(_node_site(p))
    return sorted(sites)


def _dist_summary(values: list[int]) -> dict:
    """Min/max/mean of an integer list."""
    if not values:
        return {"min": 0, "max": 0, "mean": 0.0}
    arr = np.array(values, dtype=int)
    return {
        "min": int(arr.min()),
        "max": int(arr.max()),
        "mean": round(float(arr.mean()), 2),
    }


def _extract_failure_stats(flow_results: list[dict]) -> dict:
    """Summarize failure scope across events by site.

    Uses occurrence_count to weight each unique pattern by how many
    MC iterations produced it.
    """
    # Expand by occurrence_count so each MC iteration is counted
    expanded = expand_flow_results(flow_results)
    n_events = len(expanded)
    if n_events == 0:
        return {"event_count": 0, "nodes_by_site": {}, "links_by_site": {}}

    site_node_counts: dict[str, list[int]] = defaultdict(lambda: [0] * n_events)
    site_link_counts: dict[str, list[int]] = defaultdict(lambda: [0] * n_events)

    for i, fr in enumerate(expanded):
        fs = fr.get("failure_state", {})
        for node in fs.get("excluded_nodes", []):
            site_node_counts[_node_site(node)][i] += 1
        for link in fs.get("excluded_links", []):
            for site in _link_sites(link):
                site_link_counts[site][i] += 1

    return {
        "event_count": n_events,
        "nodes_by_site": {
            s: _dist_summary(v) for s, v in sorted(site_node_counts.items())
        },
        "links_by_site": {
            s: _dist_summary(v) for s, v in sorted(site_link_counts.items())
        },
    }


def _bac_summary(bac: BacResult) -> dict:
    values = np.asarray(bac.series.values, dtype=float)
    ratios = values / bac.offered if bac.offered > 0 else values
    return {
        "auc": bac.auc_normalized,
        "pct": [
            float(value)
            for value in np.percentile(ratios, range(1, 101), method="lower")
        ],
    }


def _extract_step_metrics(results_data: dict, step_name: str) -> dict:
    """Extract BAC and failure-scope metrics from a placement step.

    Returns:
        auc: aggregate BAC AUC
        pct: aggregate percentile distribution, p1-p100
        flow_bac: per-flow BAC {label: {auc, pct}}
        failure_stats: failure scope summary by site
    """
    step_data = results_data.get("steps", {}).get(step_name, {}).get("data", {})
    baseline = step_data.get("baseline")
    flow_results = step_data.get("flow_results", [])

    if not baseline or not flow_results:
        return {}

    bac = compute_bac(results_data, step_name=step_name)
    return {
        **_bac_summary(bac),
        "flow_bac": {label: _bac_summary(flow) for label, flow in bac.per_flow.items()},
        "failure_stats": _extract_failure_stats(flow_results),
    }


def _execute_scenario(
    config: DcBbScenarioConfig,
    timeout_s: int,
    work_dir: Path,
) -> dict:
    """Generate scenario, validate, run ngraph, extract all metrics.

    Returns flat dict with alpha_star, per-mode BAC, status, error, duration_s.
    """
    result: dict = {
        "status": "pending",
        "alpha_star": None,
        "bac_combined": None,
        "bac_modes": {},
        "error": None,
        "duration_s": None,
    }

    t0 = time.time()

    try:
        scenario, expected = generate_scenario_with_validation(config)
    except Exception as e:
        result.update(
            status="error", error=f"generate: {e}", duration_s=time.time() - t0
        )
        return result

    scenario_path = work_dir / "scenario.yml"
    write_text_atomic(
        scenario_path, yaml.dump(scenario, default_flow_style=False, sort_keys=False)
    )

    try:
        errors = validate_scenario_file(scenario_path, expected)
        if errors:
            raise ValueError("; ".join(errors))
    except (ValueError, OSError) as exc:
        result.update(
            status="error", error=f"validation: {exc}", duration_s=time.time() - t0
        )
        return result
    outcome = run_simulation(scenario_path, timeout=timeout_s)
    if not outcome.success:
        result.update(
            status=outcome.status, error=outcome.error, duration_s=time.time() - t0
        )
        return result
    results_data = outcome.results

    try:
        msd = results_data.get("steps", {}).get("msd_baseline", {}).get("data", {})
        result["alpha_star"] = float(msd["alpha_star"])
    except Exception as e:
        result.update(
            status="error", error=f"alpha_star: {e}", duration_s=time.time() - t0
        )
        return result

    try:
        steps = results_data.get("steps", {})
        bac_modes: dict = {}
        for mode in FAILURE_MODE_NAMES:
            step_name = f"tm_{mode}"
            if step_name in steps:
                bac_modes[mode] = _extract_step_metrics(results_data, step_name)

        if "tm_combined" in steps:
            combined_metrics = _extract_step_metrics(results_data, "tm_combined")
            result["bac_combined"] = combined_metrics.get("auc", 0.0)
            bac_modes["combined"] = combined_metrics
        else:
            result["bac_combined"] = 0.0

        result["bac_modes"] = bac_modes
    except Exception as e:
        result.update(status="error", error=f"bac: {e}", duration_s=time.time() - t0)
        return result

    result.update(status="success", duration_s=round(time.time() - t0, 1))
    return result


def run_sweep(
    sweep_config: SweepConfig,
    side: str,
) -> list[ResultEntry]:
    """Sweep one side (fix the other at default)."""
    if side not in {"abc1", "xyz1"}:
        raise ValueError("side must be abc1 or xyz1")
    _prepare_sweep(sweep_config, side)
    output_dir = sweep_config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    results_jsonl = output_dir / "results.jsonl"
    results_dir = output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    default = DcBbScenarioConfig(seed=sweep_config.seed)
    bb_rows, bb_cols = default.bb_planes, default.bb_devices_per_plane

    completed = _load_completed(results_jsonl)

    analysis = run_structural_analysis(default)
    unique = _dedup_configs(analysis[side].configs)

    if side == "abc1":
        fixed_layout = default.layout_xyz1
        fixed_g = default.g_xyz1
        fixed_dc_r, fixed_dc_c = default.xyz1_xsw_per_plane, default.xyz1_xsw_planes
    else:
        fixed_layout = default.layout_abc1
        fixed_g = default.g_abc1
        fixed_dc_r, fixed_dc_c = default.abc1_hgrids, default.abc1_fadu_per_hgrid
    fixed_nota = _layout_notation(
        fixed_layout, fixed_dc_r, fixed_dc_c, bb_rows, bb_cols
    )

    entries: list[ResultEntry] = []
    for cfg in unique:
        layout = _pick_layout(
            side, cfg.g, cfg.bb_block_rows, cfg.bb_block_cols, default
        )
        if side == "abc1":
            dc_r, dc_c = default.abc1_hgrids, default.abc1_fadu_per_hgrid
            nota = _layout_notation(layout, dc_r, dc_c, bb_rows, bb_cols)
            dir_name = _result_dir_name(cfg.g, nota, fixed_g, fixed_nota)
            sc = DcBbScenarioConfig(
                g_abc1=cfg.g,
                layout_abc1=layout,
                failure_iterations=sweep_config.failure_iterations,
                seed=sweep_config.seed,
            )
        else:
            dc_r, dc_c = default.xyz1_xsw_per_plane, default.xyz1_xsw_planes
            nota = _layout_notation(layout, dc_r, dc_c, bb_rows, bb_cols)
            dir_name = _result_dir_name(fixed_g, fixed_nota, cfg.g, nota)
            sc = DcBbScenarioConfig(
                g_xyz1=cfg.g,
                layout_xyz1=layout,
                failure_iterations=sweep_config.failure_iterations,
                seed=sweep_config.seed,
            )

        if dir_name in completed:
            continue

        run_dir = results_dir / dir_name
        run_dir.mkdir(parents=True, exist_ok=True)
        r = _execute_scenario(sc, sweep_config.timeout_s, run_dir)

        if side == "abc1":
            entry = _build_entry(cfg.g, nota, fixed_g, fixed_nota, dir_name, r)
        else:
            entry = _build_entry(fixed_g, fixed_nota, cfg.g, nota, dir_name, r)

        entries.append(entry)
        _append_jsonl(results_jsonl, entry)

    return entries


def run_cross_sweep(sweep_config: SweepConfig) -> list[ResultEntry]:
    """Sweep all cross-side (ABC1 × XYZ1) combinations."""
    _prepare_sweep(sweep_config, "cross")
    output_dir = sweep_config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    results_jsonl = output_dir / "results.jsonl"
    results_dir = output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    default = DcBbScenarioConfig(seed=sweep_config.seed)
    bb_rows, bb_cols = default.bb_planes, default.bb_devices_per_plane

    completed = _load_completed(results_jsonl)

    analysis = run_structural_analysis(default)
    abc1_configs = _dedup_configs(analysis["abc1"].configs)
    xyz1_configs = _dedup_configs(analysis["xyz1"].configs)

    entries: list[ResultEntry] = []
    for a_cfg in abc1_configs:
        layout_a = _pick_layout(
            "abc1", a_cfg.g, a_cfg.bb_block_rows, a_cfg.bb_block_cols, default
        )
        nota_a = _layout_notation(
            layout_a, default.abc1_hgrids, default.abc1_fadu_per_hgrid, bb_rows, bb_cols
        )

        for x_cfg in xyz1_configs:
            layout_x = _pick_layout(
                "xyz1", x_cfg.g, x_cfg.bb_block_rows, x_cfg.bb_block_cols, default
            )
            nota_x = _layout_notation(
                layout_x,
                default.xyz1_xsw_per_plane,
                default.xyz1_xsw_planes,
                bb_rows,
                bb_cols,
            )

            dir_name = _result_dir_name(a_cfg.g, nota_a, x_cfg.g, nota_x)
            if dir_name in completed:
                continue

            sc = DcBbScenarioConfig(
                g_abc1=a_cfg.g,
                layout_abc1=layout_a,
                g_xyz1=x_cfg.g,
                layout_xyz1=layout_x,
                failure_iterations=sweep_config.failure_iterations,
                seed=sweep_config.seed,
            )

            run_dir = results_dir / dir_name
            run_dir.mkdir(parents=True, exist_ok=True)
            r = _execute_scenario(sc, sweep_config.timeout_s, run_dir)

            entry = _build_entry(a_cfg.g, nota_a, x_cfg.g, nota_x, dir_name, r)
            entries.append(entry)
            _append_jsonl(results_jsonl, entry)

    return entries


def _build_entry(
    g_abc1: int,
    nota_a: str,
    g_xyz1: int,
    nota_x: str,
    dir_name: str,
    r: dict,
) -> ResultEntry:
    return ResultEntry(
        g_abc1=g_abc1,
        g_xyz1=g_xyz1,
        layout_abc1=nota_a,
        layout_xyz1=nota_x,
        result_dir=dir_name,
        alpha_star=r["alpha_star"],
        bac_combined=r["bac_combined"],
        bac_modes=r.get("bac_modes"),
        status=r["status"],
        error=r["error"],
        duration_s=r["duration_s"],
        timestamp=_now_iso(),
    )


def _load_completed(results_jsonl: Path) -> set[str]:
    completed: set[str] = set()
    if results_jsonl.exists():
        for line in results_jsonl.read_text().splitlines():
            if line.strip():
                data = json.loads(line)
                if data["status"] == "success":
                    completed.add(data["result_dir"])
    return completed


def _append_jsonl(path: Path, entry: ResultEntry) -> None:
    text = path.read_text() if path.exists() else ""
    write_text_atomic(path, text + json.dumps(entry.to_dict()) + "\n")


def print_results(entries: list[ResultEntry]) -> None:
    """Print ranked results with per-mode BAC matrix."""
    successful = [e for e in entries if e.status == "success"]
    successful.sort(key=lambda e: (e.bac_combined or 0), reverse=True)

    modes = FAILURE_MODE_NAMES
    labels = {
        "lh_path": "LH",
        "plane_group": "PG",
        "plane_site": "PS",
        "dev_index": "DI",
        "2x_plane_site": "2PS",
        "4x_plane_site": "4PS",
        "2x_plane_group": "2PG",
        "2x_dev_index": "2DI",
        "1x_bb": "1BB",
        "2x_bb": "2BB",
        "4x_bb": "4BB",
        "8x_bb": "8BB",
        "bb_avail_2pct": "A2%",
        "bb_avail_5pct": "A5%",
        "bb_avail_10pct": "A10",
        "dcbb_avail": "ADC",
        "xsite_avail": "AXS",
    }

    mode_hdr = "  ".join(f"{labels[m]:>5}" for m in modes)

    print(f"\n{'=' * 160}")
    print(
        f"  Results: {len(successful)} success / {len(entries)} total (showing AUC per mode)"
    )
    print(f"{'=' * 160}")
    print(
        f"  {'G_a':>4}  {'ABC1':<12}  {'G_x':>4}  {'XYZ1':<12}  "
        f"{'alpha':>6}  {'Comb':>5}  {mode_hdr}  {'Time':>5}"
    )
    print(f"  {'-' * 156}")
    for e in successful:
        a = f"{e.alpha_star:.2f}" if e.alpha_star is not None else "   N/A"
        c = f"{e.bac_combined:.3f}" if e.bac_combined is not None else "  N/A"
        d = f"{e.duration_s:.0f}s" if e.duration_s is not None else " N/A"

        bm = e.bac_modes or {}
        mode_vals = "  ".join(
            f"{bm[m]['auc']:.3f}" if m in bm and isinstance(bm[m], dict) else "    -"
            for m in modes
        )

        print(
            f"  {e.g_abc1:>4}  {e.layout_abc1:<12}  {e.g_xyz1:>4}  {e.layout_xyz1:<12}  "
            f"{a}  {c}  {mode_vals}  {d:>5}"
        )

    failed = [e for e in entries if e.status != "success"]
    if failed:
        print(f"\n  Failed: {len(failed)}")
        for e in failed[:5]:
            print(f"    {e.result_dir}: {e.status} — {e.error}")
