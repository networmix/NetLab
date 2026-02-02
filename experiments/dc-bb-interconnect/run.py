#!/usr/bin/env python3
"""DC-BB Interconnect Experiment Runner

Single entry point for running and analyzing DC-BB topology experiments.

Usage:
    ./run.py                           # Run all, default seeds
    ./run.py '*dc16x36*'               # Run topologies matching pattern
    ./run.py '*bb16x4*' '*bb32x4*'     # Multiple patterns
    ./run.py dc16x36_bb16x4_bb16x4_dc16x36_one_to_one  # Exact name
    ./run.py --seeds 42:50             # Seed range (42-49)
    ./run.py --list                    # List available topologies
    ./run.py --metrics                 # Compute metrics only
    ./run.py --compare                 # Print comparison table
"""

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Add project root to path for imports
_project_root = Path(__file__).parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from visualize import DcBbVisualizer  # noqa: E402

from netlab.comparison import (  # noqa: E402
    ComparisonTableBuilder,
    format_network_stats,
    format_percent,
)
from netlab.experiment import ExperimentRunner  # noqa: E402
from netlab.metrics_failure import (  # noqa: E402
    analyze_results,
    extract_alpha_star,
    extract_network_stats,
)
from netlab.scenario import ScenarioMerger, is_failure_policy  # noqa: E402


class DcBbExperimentRunner(ExperimentRunner):
    """DC-BB specific experiment runner."""

    def get_merger(self) -> ScenarioMerger:
        """Configure scenario merger for DC-BB experiments."""
        merger = ScenarioMerger(self.root)
        merger.add_source(
            self.root / "policies",
            "failures",
            filter_fn=is_failure_policy,
        )
        merger.add_source(
            self.root / "demands",
            "demands",
            source_key="demands",
        )
        merger.add_workflow_source(self.root / "workflows")
        return merger

    def compute_metrics(self, scenarios: Optional[List[str]] = None) -> None:
        """Compute metrics for completed runs with DC-BB specific analysis."""
        results_by_scenario = self.get_results_files(scenarios)

        if not results_by_scenario:
            print("No results files found. Run experiments first.")
            return

        total_files = sum(len(f) for f in results_by_scenario.values())
        print(f"Found {total_files} results files")

        for scenario, files in sorted(results_by_scenario.items()):
            print(f"\nProcessing {scenario} ({len(files)} seeds)")
            self._compute_scenario_metrics(scenario, files)

    def _compute_scenario_metrics(
        self, scenario: str, results_files: List[Path]
    ) -> None:
        """Compute metrics for a single scenario."""
        summary = {
            "scenario": scenario,
            "seeds": [],
            "network": {},
            "alpha_star": None,
            "failure_analysis": {},
        }

        for rf in sorted(results_files):
            with open(rf) as f:
                results = json.load(f)

            seed = int(rf.stem.split("__seed")[1].split("_")[0])
            summary["seeds"].append(seed)

            # Get network statistics (from first seed only)
            if not summary["network"]:
                net_stats = extract_network_stats(results)
                if net_stats:
                    summary["network"] = net_stats

                # DC-BB specific: Check BB link distribution
                self._check_bb_distribution(results, summary)

            # Get alpha_star
            alpha = extract_alpha_star(results)
            if alpha:
                summary["alpha_star"] = alpha

            # Extract and merge failure metrics
            analysis = analyze_results(results)
            for step_name, stats in analysis.failure_stats.items():
                if step_name not in summary["failure_analysis"]:
                    summary["failure_analysis"][step_name] = {
                        "iterations": 0,
                        "ratios": [],
                    }
                summary["failure_analysis"][step_name]["iterations"] += stats.iterations
                summary["failure_analysis"][step_name]["ratios"].extend(stats.ratios)

        # Compute final statistics
        for step_name, data in summary["failure_analysis"].items():
            ratios = data["ratios"]
            summary["failure_analysis"][step_name] = {
                "iterations": data["iterations"],
                "min_ratio": min(ratios),
                "avg_ratio": statistics.mean(ratios),
                "max_ratio": max(ratios),
                "std_dev": statistics.stdev(ratios) if len(ratios) > 1 else 0,
            }

        # Find worst failures
        if summary["failure_analysis"]:
            min_ratio = min(
                s["min_ratio"] for s in summary["failure_analysis"].values()
            )
            tolerance = 0.001
            worst_types = [
                name
                for name, stats in summary["failure_analysis"].items()
                if abs(stats["min_ratio"] - min_ratio) <= tolerance
            ]
            summary["worst_failures"] = {
                "types": sorted(worst_types),
                "min_ratio": min_ratio,
            }

        # Save summary
        summary_file = self.results_dir / scenario / "summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)

        # Print summary
        self._print_scenario_summary(summary)
        print(f"\n  Saved: {summary_file}")

    def _check_bb_distribution(self, results: dict, summary: dict) -> None:
        """DC-BB specific: Check if BB link distribution is even."""
        steps = results.get("steps", {})
        graph_data = steps.get("build_graph", {}).get("data", {}).get("graph", {})

        if not graph_data:
            return

        edges = graph_data.get("links", graph_data.get("edges", []))
        bb_link_counts: Dict[str, int] = {}

        for edge in edges:
            if edge.get("link_type") == "dc_bb":
                tgt = edge.get("target", "")
                if "SiteA/BB" in tgt:
                    bb_link_counts[tgt] = bb_link_counts.get(tgt, 0) + 1

        if bb_link_counts:
            unique_counts = set(bb_link_counts.values())
            summary["network"]["bb_even"] = len(unique_counts) == 1
            summary["network"]["bb_links_per_node"] = list(unique_counts)
        else:
            summary["network"]["bb_even"] = True
            summary["network"]["bb_links_per_node"] = []

    def _print_scenario_summary(self, summary: dict) -> None:
        """Print scenario summary to console."""
        net = summary["network"]
        if net:
            print(
                f"\n  Network: {net['node_count']} nodes, {net['link_count']} links, "
                f"{net['total_capacity']:,.0f} capacity"
            )
        print(f"  Alpha*: {summary['alpha_star']}")
        print(f"\n  {'Failure':<20} {'Min':>8} {'Avg':>8} {'Max':>8} {'StdDev':>8}")
        print(f"  {'-' * 52}")
        for name, stats in sorted(summary["failure_analysis"].items()):
            print(
                f"  {name:<20} {stats['min_ratio']:>8.2%} {stats['avg_ratio']:>8.2%} "
                f"{stats['max_ratio']:>8.2%} {stats['std_dev']:>8.4f}"
            )

        if "worst_failures" in summary:
            wf = summary["worst_failures"]
            types_str = ", ".join(wf["types"])
            print(f"\n  Worst case ({wf['min_ratio']:.2%}): {types_str}")

    def generate_comparison(self, scenarios: Optional[List[str]] = None) -> None:
        """Generate comparison table with DC-BB specific columns."""
        if scenarios is None:
            scenarios = self.discover_scenarios()

        # Load summaries
        summaries = {}
        for scenario in scenarios:
            summary_file = self.results_dir / scenario / "summary.json"
            if summary_file.exists():
                with open(summary_file) as f:
                    summaries[scenario] = json.load(f)

        if not summaries:
            print("No summaries found. Run experiments first.")
            return

        # Build comparison table
        builder = ComparisonTableBuilder(summaries)

        # Add standard rows
        builder.add_row("Network", format_network_stats)
        builder.add_row(
            "Alpha*",
            lambda s: s.get("alpha_star"),
            formatter=lambda v: f"{v:.2f}" if v else "?",
        )

        # Add alpha % of max
        alphas = [
            s.get("alpha_star") for s in summaries.values() if s.get("alpha_star")
        ]
        max_alpha = max(alphas) if alphas else 1
        builder.add_row(
            "Alpha* %Max",
            lambda s: s.get("alpha_star"),
            formatter=lambda v: f"{v / max_alpha * 100:.1f}%" if v else "?",
        )

        # DC-BB specific: BB Even row
        builder.add_row(
            "BB Even",
            lambda s: s.get("network", {}).get("bb_even"),
            formatter=lambda v: "Yes" if v is True else ("No" if v is False else "?"),
        )

        # Add failure metric rows with DC-BB labels
        failure_labels = {
            "tm_dc_bb_link": "DC-BB Link",
            "tm_bb_bb_link": "BB-BB Link",
            "tm_dc_node": "DC Node",
            "tm_bb_node": "BB Node",
            "tm_plane": "Plane",
            "tm_dc_row": "DC Row",
        }
        builder.add_metric_rows(
            "failure_analysis",
            "min_ratio",
            label_map=failure_labels,
            formatter=format_percent,
        )

        # Add worst case row
        builder.add_row(
            "Worst",
            lambda s: s.get("worst_failures", {}).get("min_ratio"),
            formatter=format_percent,
            default="N/A",
        )

        # Print and save (transposed: scenarios as rows for better readability)
        builder.print_table(title="TOPOLOGY COMPARISON", transposed=True)

        comparison_file = self.results_dir / "comparison.json"
        builder.to_json(comparison_file)
        print(f"\nSaved: {comparison_file}")

    def generate_visualizations(
        self,
        scenarios: Optional[List[str]] = None,
        split: bool = False,
    ) -> None:
        """Generate SVG visualizations for scenarios.

        Args:
            scenarios: List of scenario names (default: all discovered)
            split: If True, also generate split view with disconnected components stacked
        """
        if scenarios is None:
            scenarios = self.discover_scenarios()

        print("\nGenerating visualizations...")

        for scenario in scenarios:
            scenario_dir = self.results_dir / scenario
            if not scenario_dir.exists():
                print(f"  [{scenario}] No results found, skipping")
                continue

            # Find first results file with build_graph data
            results_file = None
            for seed_dir in sorted(scenario_dir.iterdir()):
                if not seed_dir.is_dir():
                    continue
                for rf in seed_dir.glob("*.results.json"):
                    with open(rf) as f:
                        results = json.load(f)
                    if "build_graph" in results.get("steps", {}):
                        results_file = rf
                        break
                if results_file:
                    break

            if not results_file:
                print(f"  [{scenario}] No build_graph data found, skipping")
                continue

            # Generate visualization using DC-BB specific visualizer
            output_path = scenario_dir / "topology.svg"
            try:
                viz = DcBbVisualizer.from_results(results_file)
                viz.layout()
                viz.render_svg(output_path)
                print(f"  [{scenario}] Generated: {output_path}")

                # Generate split view if requested
                if split:
                    viz_split = DcBbVisualizer.from_results(results_file)
                    viz_split.layout()
                    split_path = scenario_dir / "topology_split.svg"
                    result = viz_split.render_svg_split(split_path)
                    if result:
                        print(f"  [{scenario}] Generated split: {split_path}")
            except Exception as e:
                print(f"  [{scenario}] Error: {e}")


def parse_seeds(seed_args: List[str]) -> List[int]:
    """Parse seed arguments like ['42', '43'] or ['42:50']."""
    seeds = []
    for arg in seed_args:
        if ":" in arg:
            start, end = map(int, arg.split(":"))
            seeds.extend(range(start, end))
        else:
            seeds.append(int(arg))
    return seeds


def filter_scenarios(patterns: List[str], available: List[str]) -> List[str]:
    """Filter scenarios by patterns (glob-style with * and ?).

    Args:
        patterns: List of patterns to match (supports * and ? wildcards)
        available: List of all available scenario names

    Returns:
        List of matching scenario names (preserves order, no duplicates)
    """
    import fnmatch

    matched = []
    seen = set()

    for pattern in patterns:
        # Check if pattern contains wildcards
        if "*" in pattern or "?" in pattern:
            for name in available:
                if fnmatch.fnmatch(name, pattern) and name not in seen:
                    matched.append(name)
                    seen.add(name)
        else:
            # Exact match
            if pattern in available and pattern not in seen:
                matched.append(pattern)
                seen.add(pattern)
            elif pattern not in available:
                print(f"Warning: topology '{pattern}' not found")

    return matched


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "topology",
        nargs="*",
        help="Topology pattern(s) to run (supports * and ? wildcards, e.g., '*dc16x36*', '*bb16x4*')",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=str,
        default=["42", "43", "44"],
        help="Seeds (e.g., 42 43 or 42:50). Default: 42 43 44",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available topologies",
    )
    parser.add_argument(
        "--metrics",
        action="store_true",
        help="Compute metrics only (skip ngraph runs)",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Print comparison table",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-run even if results exist",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate merged scenarios without running ngraph",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate SVG visualization of topology",
    )
    parser.add_argument(
        "--split",
        action="store_true",
        help="Also generate split view with disconnected components stacked (use with --visualize)",
    )
    args = parser.parse_args()

    runner = DcBbExperimentRunner(Path(__file__).parent)

    # List topologies
    if args.list:
        scenarios = runner.discover_scenarios()
        if not scenarios:
            print("No topologies found in topologies/")
        else:
            print(f"\n┌─ Available {len(scenarios)} topology(ies) ─" + "─" * 40)
            for i, t in enumerate(scenarios, 1):
                print(f"│ {i:>2}. {t}")
            print("└" + "─" * 60 + "\n")
        return

    # Parse seeds
    seeds = parse_seeds(args.seeds)

    # Determine which topologies to run
    available = runner.discover_scenarios()
    if not available:
        print("No topologies found. Create topologies in topologies/*/scenario.yml")
        return

    if args.topology:
        scenarios = filter_scenarios(args.topology, available)
        if not scenarios:
            print("No topologies matched the pattern(s).")
            print(f"Available: {', '.join(available)}")
            return
        # Show matched topologies in a nice table
        print(f"\n┌─ Matched {len(scenarios)} topology(ies) ─" + "─" * 40)
        for i, t in enumerate(scenarios, 1):
            print(f"│ {i:>2}. {t}")
        print("└" + "─" * 60 + "\n")
    else:
        scenarios = available

    # Metrics-only mode (no experiment runs)
    if args.metrics:
        runner.compute_metrics(scenarios)
        runner.generate_comparison(scenarios)
        if args.visualize:
            runner.generate_visualizations(scenarios, split=args.split)
        return

    # Compare-only mode (just print existing comparison)
    if args.compare:
        runner.generate_comparison(scenarios)
        return

    # Visualize-only mode (only if no other action flags)
    if args.visualize and not args.force and not args.dry_run:
        runner.generate_visualizations(scenarios, split=args.split)
        return

    # Run experiments
    print(f"Running {len(scenarios)} topology(ies) with seeds {seeds}")
    print("-" * 60)

    for scenario in scenarios:
        print(f"\n[{scenario}]")
        stats = runner.run_scenario(
            scenario,
            seeds,
            force=args.force,
            dry_run=args.dry_run,
        )
        # Count dry_run as a separate category, not failed
        dry_runs = sum(1 for s in stats["seeds"] if s["status"] == "dry_run")
        if args.dry_run:
            print(f"  Summary: {dry_runs} dry-run scenarios generated")
        else:
            print(
                f"  Summary: {stats['ran']} ran, {stats['cached']} cached, {stats['failed']} failed"
            )

    # Compute metrics after runs (skip if dry-run)
    if not args.dry_run:
        print("\n" + "-" * 60)
        print("Computing metrics...")
        runner.compute_metrics(scenarios)
        runner.generate_comparison(scenarios)

        if args.visualize:
            runner.generate_visualizations(scenarios, split=args.split)


if __name__ == "__main__":
    main()
