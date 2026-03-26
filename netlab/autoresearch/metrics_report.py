"""Programmatic metrics report from ngraph simulation results.

Computes all metrics using the verified pipeline (BAC, latency, alpha,
iterops) and formats them into a structured markdown document. This is
the single source of truth for numbers — the LLM never extracts metrics
from raw results JSON.
"""

from __future__ import annotations


def build_metrics_report(results: dict, step_names: list[str] | None = None) -> str:
    """Build a structured metrics report from simulation results.

    Uses the verified metrics pipeline (same code that passed 252
    hand-calculated assertions on the mini DC-BB scenario).

    Args:
        results: ngraph simulation results dict.
        step_names: TMP step names to analyze. If None, auto-detects
            all TrafficMatrixPlacement steps.

    Returns:
        Markdown-formatted metrics report with exact numbers.
    """
    from metrics.bac import compute_bac
    from metrics.latency import compute_latency_stretch
    from metrics.msd import compute_alpha_star

    lines: list[str] = ["# Metrics Report (machine-generated, verified)\n"]

    # Alpha / MSD
    try:
        alpha = compute_alpha_star(results)
        lines.append("## Capacity")
        lines.append(f"- alpha_star: {alpha.alpha_star}")
        lines.append(f"- source: {alpha.source}")
        if alpha.base_total_demand > 0:
            lines.append(f"- base_total_demand: {alpha.base_total_demand}")
        lines.append("")
    except (ValueError, KeyError):
        pass

    # Detect TMP steps
    if step_names is None:
        step_names = _detect_tmp_steps(results)

    # BAC + Latency per step
    for step_name in step_names:
        step_data = results.get("steps", {}).get(step_name, {}).get("data", {})
        if not step_data or not step_data.get("flow_results"):
            continue

        lines.append(f"## {step_name}\n")

        # Iteration counts
        fr = step_data.get("flow_results", [])
        n_patterns = len(fr)
        n_iters = sum(f.get("occurrence_count", 1) for f in fr)
        lines.append(f"- failure iterations: {n_iters}")
        lines.append(f"- unique patterns: {n_patterns}")

        # Baseline
        baseline = step_data.get("baseline", {})
        if baseline:
            s = baseline.get("summary", {})
            lines.append(
                f"- baseline: placed={s.get('total_placed')}, demand={s.get('total_demand')}, ratio={s.get('overall_ratio')}"
            )

        # Per-pattern summary
        lines.append("- failure patterns:")
        for _idx, f in enumerate(fr):
            s = f.get("summary", {})
            fs = f.get("failure_state", {})
            excl = fs.get("excluded_nodes", []) or [
                lid.split("|")[0] for lid in fs.get("excluded_links", [])
            ]
            excl_str = ", ".join(excl[:3]) if excl else "none"
            lines.append(
                f"  - [{f.get('occurrence_count', 1)}x] "
                f"ratio={s.get('overall_ratio', 0):.4f}, "
                f"placed={s.get('total_placed')}, "
                f"excluded: {excl_str}"
            )

        # BAC
        try:
            bac = compute_bac(results, step_name=step_name)
            lines.append("")
            lines.append("### BAC (Bandwidth Availability Curve)")
            lines.append(f"- AUC: {bac.auc_normalized:.4f}")
            lines.append(f"- offered: {bac.offered}")
            for p in (90.0, 99.0, 99.9):
                bw = bac.bw_at_probability_pct.get(p)
                if bw is not None:
                    lines.append(f"- BW @ {p}% probability: {bw * 100:.1f}% of offered")

            if bac.per_flow:
                lines.append("- per direction:")
                for label, pf in bac.per_flow.items():
                    lines.append(
                        f"  - {label}: AUC={pf.auc_normalized:.4f}, offered={pf.offered}"
                    )
        except (ValueError, KeyError):
            pass

        # Latency (requires step to be named tm_placement for the latency module)
        try:
            lat_data = {"steps": {"tm_placement": results["steps"][step_name]}}
            lat = compute_latency_stretch(lat_data)
            if lat.baseline and lat.failures:
                lines.append("")
                lines.append("### Latency Stretch")
                lines.append(f"- baseline p50: {lat.baseline.get('p50', 'N/A')}")
                lines.append(f"- baseline p99: {lat.baseline.get('p99', 'N/A')}")
                lines.append(f"- baseline WES: {lat.baseline.get('WES', 'N/A')}")
                lines.append(f"- failure p50: {lat.failures.get('p50', 'N/A')}")
                lines.append(f"- failure p99: {lat.failures.get('p99', 'N/A')}")
                lines.append(f"- failure WES: {lat.failures.get('WES', 'N/A')}")
                if lat.derived:
                    td99 = lat.derived.get("TD99")
                    if td99 is not None:
                        lines.append(f"- TD99 (failure/baseline p99 ratio): {td99:.4f}")
                    wes_d = lat.derived.get("WES_delta")
                    if wes_d is not None:
                        lines.append(f"- WES delta: {wes_d:.4f}")
        except (ValueError, KeyError):
            pass

        lines.append("")

    return "\n".join(lines)


def _detect_tmp_steps(results: dict) -> list[str]:
    """Find all TrafficMatrixPlacement steps in results."""
    steps = results.get("steps", {})
    tmp_steps: list[str] = []
    for name, step in steps.items():
        data = step.get("data", {})
        if isinstance(data, dict) and "baseline" in data and "flow_results" in data:
            tmp_steps.append(name)
    return tmp_steps
