"""Phase 2: Systematic simulation sweep over DC-BB configurations.

Runs ngraph on (G, layout) combinations, extracts alpha_star and
per-mode BAC metrics. All results written as flat JSONL entries
to a single results.jsonl file.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

from metrics.bac import compute_bac
from metrics.common import expand_flow_results
from netlab.autoresearch.scenario_generator import (
    FAILURE_MODE_NAMES,
    DcBbScenarioConfig,
    generate_scenario_with_validation,
    get_valid_layouts,
)
from netlab.autoresearch.scenario_validation import validate_inspect_output
from netlab.autoresearch.structural_analysis import (
    ConfigResult,
    run_structural_analysis,
)

# ---------------------------------------------------------------------------
# Result entry — one line per simulation, flat fields, loads into pandas
# ---------------------------------------------------------------------------


@dataclass
class ResultEntry:
    """One simulation result. Flat fields, no nesting.

    BAC fields are dynamically named bac_{mode} for each mode in
    FAILURE_MODE_NAMES. The dataclass uses a dict for storage,
    but to_dict() flattens everything.
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
        d = {
            "g_abc1": self.g_abc1,
            "g_xyz1": self.g_xyz1,
            "layout_abc1": self.layout_abc1,
            "layout_xyz1": self.layout_xyz1,
            "alpha_star": self.alpha_star,
            "bac_combined": self.bac_combined,
            "bac_modes": self.bac_modes,
            "result_dir": self.result_dir,
            "status": self.status,
            "error": self.error,
            "duration_s": self.duration_s,
            "timestamp": self.timestamp,
        }
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ResultEntry":
        fields = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**fields)


@dataclass
class SweepConfig:
    """Configuration for a sweep (single-side or cross-side)."""

    output_dir: Path
    failure_iterations: int = 200
    timeout_s: int = 300
    seed: int = 42


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _find_ngraph() -> str:
    ngraph = shutil.which("ngraph")
    if ngraph is None:
        raise RuntimeError("ngraph binary not found on PATH")
    return ngraph


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


# ---------------------------------------------------------------------------
# Metric extraction
# ---------------------------------------------------------------------------


def _flow_label(flow_source: str) -> str:
    """Extract directional label from flow source field.

    Flow source format: '_src_<source_pattern>|<target_pattern>|<hash>'
    Returns label like 'abc1>xyz1'.
    """
    demand_id = flow_source.removeprefix("_src_").removeprefix("_snk_")
    parts = demand_id.split("|")
    if len(parts) >= 2:
        src_site = parts[0].lstrip("^").split("/")[0]
        dst_site = parts[1].lstrip("^").split("/")[0]
        return f"{src_site}>{dst_site}"
    return demand_id[:20]


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


def _extract_flow_bac(baseline: dict, flow_results: list[dict]) -> dict[str, dict]:
    """Per-flow BAC: AUC + percentile distribution, keyed by directional label."""
    baseline_flows = baseline.get("flows", [])
    if not baseline_flows:
        return {}

    # Map flow source field -> (label, baseline_demand)
    flow_map: dict[str, tuple[str, float]] = {}
    for f in baseline_flows:
        src = f.get("source", "")
        if not src:
            continue
        flow_map[src] = (_flow_label(src), float(f.get("demand", 0)))

    # Expand deduplicated patterns by occurrence_count
    expanded = expand_flow_results(flow_results)

    # Collect per-flow placement ratios: baseline (1.0) + each failure event
    flow_ratios: dict[str, list[float]] = {src: [1.0] for src in flow_map}

    for fr in expanded:
        event_flows = {ef["source"]: ef for ef in fr.get("flows", [])}
        for src, (_label, bl_demand) in flow_map.items():
            if src in event_flows and bl_demand > 0:
                ratio = min(event_flows[src]["placed"] / bl_demand, 1.0)
            else:
                ratio = 0.0
            flow_ratios[src].append(ratio)

    result: dict[str, dict] = {}
    for src, ratios in flow_ratios.items():
        label = flow_map[src][0]
        arr = np.array(ratios)
        result[label] = {
            "auc": round(float(arr.mean()), 6),
            "pct": [round(float(v), 6) for v in np.percentile(arr, range(1, 101))],
        }
    return result


def _extract_step_metrics(results_data: dict, step_name: str) -> dict:
    """Extract comprehensive metrics from a single TM step.

    Returns:
        auc: aggregate BAC AUC (backward compatible)
        pct: aggregate percentile distribution, p1-p100 (backward compatible)
        flow_bac: per-flow BAC {label: {auc, pct}}
        failure_stats: failure scope summary by site
    """
    step_data = results_data.get("steps", {}).get(step_name, {}).get("data", {})
    baseline = step_data.get("baseline")
    flow_results = step_data.get("flow_results", [])

    if not baseline or not flow_results:
        return {}

    # Aggregate BAC (via existing compute_bac)
    bac = compute_bac(results_data, step_name=step_name)
    raw_vals = np.asarray(bac.series.values, dtype=float)
    ratios = raw_vals / bac.offered if bac.offered > 0 else raw_vals
    pcts = [round(float(v), 6) for v in np.percentile(ratios, range(1, 101))]

    return {
        "auc": round(bac.auc_normalized, 6),
        "pct": pcts,
        "flow_bac": _extract_flow_bac(baseline, flow_results),
        "failure_stats": _extract_failure_stats(flow_results),
    }


# ---------------------------------------------------------------------------
# Core execution
# ---------------------------------------------------------------------------


def _execute_scenario(
    config: DcBbScenarioConfig,
    timeout_s: int,
    ngraph_bin: str,
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
    scenario_path.write_text(
        yaml.dump(scenario, default_flow_style=False, sort_keys=False)
    )

    # Validate
    try:
        ir = subprocess.run(
            [ngraph_bin, "inspect", str(scenario_path)],
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        if ir.returncode != 0:
            result.update(
                status="error",
                error=f"inspect: {ir.stderr[-300:]}",
                duration_s=time.time() - t0,
            )
            return result
        ve = validate_inspect_output(ir.stdout, expected)
        if ve:
            result.update(
                status="error",
                error=f"validation: {'; '.join(ve)}",
                duration_s=time.time() - t0,
            )
            return result
    except subprocess.TimeoutExpired:
        result.update(status="timeout", duration_s=time.time() - t0)
        return result

    # Run
    try:
        rr = subprocess.run(
            [ngraph_bin, "run", str(scenario_path), "-o", str(work_dir)],
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        if rr.returncode != 0:
            result.update(
                status="crash",
                error=f"run: {rr.stderr[-300:]}",
                duration_s=time.time() - t0,
            )
            return result
    except subprocess.TimeoutExpired:
        result.update(status="timeout", duration_s=time.time() - t0)
        return result

    # Load results
    results_path = work_dir / "scenario.results.json"
    if not results_path.exists():
        result.update(
            status="error", error="no results file", duration_s=time.time() - t0
        )
        return result
    results_data = json.loads(results_path.read_text())

    # Alpha
    try:
        msd = results_data.get("steps", {}).get("msd_baseline", {}).get("data", {})
        result["alpha_star"] = float(msd.get("alpha_star", 0))
    except Exception as e:
        result.update(
            status="error", error=f"alpha_star: {e}", duration_s=time.time() - t0
        )
        return result

    # Per-mode + combined metrics
    try:
        steps = results_data.get("steps", {})
        bac_modes: dict = {}
        for mode in FAILURE_MODE_NAMES:
            step_name = f"tm_{mode}"
            if step_name in steps:
                bac_modes[mode] = _extract_step_metrics(results_data, step_name)

        combined = "tm_combined" if "tm_combined" in steps else "tm_placement"
        if combined in steps:
            combined_metrics = _extract_step_metrics(results_data, combined)
            result["bac_combined"] = combined_metrics.get("auc", 0.0)
            bac_modes["combined"] = combined_metrics
        else:
            result["bac_combined"] = 0.0

        result["bac_modes"] = bac_modes
    except Exception as e:
        result["error"] = f"bac: {e}"

    result.update(status="success", duration_s=round(time.time() - t0, 1))
    return result


# ---------------------------------------------------------------------------
# Sweep runners
# ---------------------------------------------------------------------------


def run_sweep(
    sweep_config: SweepConfig,
    side: str,
) -> list[ResultEntry]:
    """Sweep one side (fix the other at default)."""
    ngraph_bin = _find_ngraph()
    output_dir = sweep_config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    results_jsonl = output_dir / "results.jsonl"
    results_dir = output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    default = DcBbScenarioConfig(seed=sweep_config.seed)
    bb_rows, bb_cols = default.bb_planes, default.bb_devices_per_plane

    # Load completed
    completed = _load_completed(results_jsonl)

    # Get unique configs for swept side
    analysis = run_structural_analysis(default)
    unique = _dedup_configs(analysis[side].configs)

    # Fixed side notation
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
        r = _execute_scenario(sc, sweep_config.timeout_s, ngraph_bin, run_dir)

        if side == "abc1":
            entry = _build_entry(cfg.g, nota, fixed_g, fixed_nota, dir_name, r)
        else:
            entry = _build_entry(fixed_g, fixed_nota, cfg.g, nota, dir_name, r)

        entries.append(entry)
        _append_jsonl(results_jsonl, entry)

    return entries


def run_cross_sweep(sweep_config: SweepConfig) -> list[ResultEntry]:
    """Sweep all cross-side (ABC1 × XYZ1) combinations."""
    ngraph_bin = _find_ngraph()
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
            r = _execute_scenario(sc, sweep_config.timeout_s, ngraph_bin, run_dir)

            entry = _build_entry(a_cfg.g, nota_a, x_cfg.g, nota_x, dir_name, r)
            entries.append(entry)
            _append_jsonl(results_jsonl, entry)

    return entries


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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
                completed.add(data.get("result_dir", ""))
    return completed


def _append_jsonl(path: Path, entry: ResultEntry) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(entry.to_dict()) + "\n")


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------


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
