"""Command-line adapters for the NetLab library services."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .log_config import configure_from_env, set_global_log_level


def _cmd_pipeline(args: argparse.Namespace) -> None:
    from .pipeline import PipelineConfig, discover_configs, run_pipeline

    try:
        config = PipelineConfig(
            masters=discover_configs(args.configs),
            seeds=args.seeds,
            output_dir=args.scenarios_dir,
            graphs_dir=args.graphs_dir,
            build_jobs=args.build_jobs,
            run_jobs=args.run_jobs,
            build_timeout=args.build_timeout,
            run_timeout=args.run_timeout,
            force=args.force,
            force_run=args.force_run,
        )
        result = run_pipeline(config, simulate=args.command == "run")
    except (ValueError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc
    for scenario, outcome in result.simulations.items():
        print(f"{scenario.name}: {outcome.status}")
    for task, error in result.errors.items():
        print(f"{task}: {error}", file=sys.stderr)
    print(
        f"Built {len(result.scenarios)} scenarios; provenance: {args.scenarios_dir / 'provenance.json'}"
    )
    if not result.success:
        raise SystemExit(1)


def main(argv: list[str] | None = None) -> None:
    configure_from_env()
    from netlab.autoresearch.backend import (
        DEFAULT_CLAUDE_MODEL,
        DEFAULT_OPENAI_MODEL,
        SUPPORTED_BACKENDS,
    )

    ap = argparse.ArgumentParser(prog="netlab", description="NetLab CLI")
    ap.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging (overrides NETLAB_LOG_LEVEL)",
    )
    sub = ap.add_subparsers(dest="command", required=True)

    for command in ("build", "run"):
        parser = sub.add_parser(
            command,
            help="Build seeded scenarios"
            if command == "build"
            else "Build and simulate seeded scenarios",
        )
        parser.add_argument(
            "configs", nargs="?", default=Path("topogen_configs"), type=Path
        )
        parser.add_argument(
            "--seeds",
            nargs="+",
            type=int,
            default=[42] if command == "build" else None,
            required=command == "run",
        )
        parser.add_argument("--scenarios-dir", default=Path("scenarios"), type=Path)
        parser.add_argument(
            "--graphs-dir",
            type=Path,
            help="Use existing <master>_integrated_graph.json files instead of generating geography",
        )
        parser.add_argument("--build-jobs", type=int, default=1)
        parser.add_argument(
            "--build-timeout",
            type=float,
            default=None,
            help="Execution deadline in seconds per graph/build task",
        )
        parser.add_argument(
            "--force",
            action="store_true",
            help="Regenerate graphs and repeat simulations",
        )
        if command == "run":
            parser.add_argument("--run-jobs", type=int, default=1)
            parser.add_argument("--run-timeout", type=float, default=600)
            parser.add_argument("--force-run", action="store_true")
        else:
            parser.set_defaults(run_jobs=1, run_timeout=600, force_run=False)
        parser.set_defaults(func=_cmd_pipeline)

    ap_metrics = sub.add_parser(
        "metrics",
        help="Compute metrics over a scenarios root (results JSONs)",
        description=(
            "Analyze *.results.json under the given root, write per-seed outputs, "
            "cross-seed summaries, and project-level summary"
        ),
    )
    ap_metrics.add_argument(
        "scenarios_root",
        nargs="?",
        default=Path("scenarios"),
        type=Path,
        help="Root directory containing *.results.json (e.g., scenarios)",
    )
    ap_metrics.add_argument(
        "--only",
        type=str,
        default="",
        help="Comma-separated scenario stems to include",
    )
    ap_metrics.add_argument(
        "--no-plots", action="store_true", help="Skip PNG chart generation"
    )
    ap_metrics.add_argument(
        "--enable-maxflow",
        action="store_true",
        help="Enable MaxFlow-based metrics (SPS, BAC overlay)",
    )
    ap_metrics.add_argument(
        "--summary",
        action="store_true",
        help="Print summary tables from CSVs and render cross-seed figures",
    )

    def _cmd_metrics(args: argparse.Namespace) -> None:
        from .metrics.batch import run_metrics
        from .metrics.reporting import print_summary_from_csv

        if bool(args.summary):
            # Summary mode: always render cross-seed figures
            print_summary_from_csv(args.scenarios_root, plots=True, quiet=False)
            return
        run_metrics(
            root=args.scenarios_root,
            only=args.only,
            no_plots=bool(args.no_plots),
            enable_maxflow=bool(args.enable_maxflow),
        )
        if not bool(args.no_plots):
            print_summary_from_csv(args.scenarios_root, plots=True, quiet=True)

    ap_metrics.set_defaults(func=_cmd_metrics)

    ap_test = sub.add_parser(
        "test",
        help="Paired t-tests between two scenarios (A vs B)",
        description=(
            "Run paired t-tests on per-seed metrics between two scenarios using outputs under <root>_metrics."
        ),
    )
    ap_test.add_argument(
        "scenarios_root",
        nargs="?",
        default=Path("scenarios"),
        type=Path,
        help="Root directory containing *.results.json (e.g., scenarios)",
    )
    ap_test.add_argument("scenario_a", type=str, help="Scenario A name")
    ap_test.add_argument("scenario_b", type=str, help="Scenario B name")
    ap_test.add_argument(
        "--alpha", type=float, default=0.05, help="Significance level for tests"
    )

    def _cmd_test(args: argparse.Namespace) -> None:
        from netlab.metrics.comparisons import compare_scenarios

        root: Path = args.scenarios_root
        out_root = root.parent / f"{root.name}_metrics"
        insights = compare_scenarios(
            out_root, alpha=args.alpha, scenarios=(args.scenario_a, args.scenario_b)
        )
        if not insights:
            print("(no project insights available)")
            return
        a = args.scenario_a
        b = args.scenario_b
        alpha = float(args.alpha)
        matches = [
            r
            for r in insights
            if (r.get("scen_a") == a and r.get("scen_b") == b)
            or (r.get("scen_a") == b and r.get("scen_b") == a)
        ]
        if not matches:
            print(f"No common-seed paired results found for {a} vs {b}.")
            return
        print(f"Paired t-tests for {a} vs {b} (alpha={alpha}):")
        print(f"metric  n  mean_diff  [{100 * (1 - alpha):g}% CI]  t  p  p_adj  det")
        for r in sorted(matches, key=lambda x: x.get("metric", "zzz")):
            n = int(r.get("n", 0))
            md = r.get("mean_diff", float("nan"))
            ci_l = r.get("ci_low", float("nan"))
            ci_h = r.get("ci_high", float("nan"))
            t_stat = r.get("t_stat", float("nan"))
            p = r.get("p", float("nan"))
            p_adj = r.get("p_adj", float("nan"))
            det = "✓" if r.get("deterministic") else ""
            print(
                f"{r.get('metric')}  {n}  {md:.4g}  [{ci_l:.4g}, {ci_h:.4g}]  "
                f"{t_stat:.3f}  {p:.4f}  {p_adj:.4f}  {det}"
            )

    ap_test.set_defaults(func=_cmd_test)

    ap_auto = sub.add_parser(
        "autoresearch",
        help="Autonomous research loop (init, run)",
        description="Autonomous research: scaffold projects and run experiment loops",
    )
    auto_sub = ap_auto.add_subparsers(dest="auto_command", required=True)

    ap_auto_init = auto_sub.add_parser(
        "init",
        help="Scaffold a new autoresearch project directory",
        description="Create project dir with default templates, copy base scenario",
    )
    ap_auto_init.add_argument(
        "--base-scenario",
        type=Path,
        required=True,
        help="Path to the base scenario YAML (will be copied into the project)",
    )
    ap_auto_init.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory for the new autoresearch project",
    )

    def _cmd_autoresearch_init(args: argparse.Namespace) -> None:
        from netlab.autoresearch.cli import autoresearch_init

        autoresearch_init(args)

    ap_auto_init.set_defaults(func=_cmd_autoresearch_init)

    ap_auto_run = auto_sub.add_parser(
        "run",
        help="Run the autoresearch experiment loop",
        description="Load project, construct runner, execute experiment loop",
    )
    ap_auto_run.add_argument(
        "project_dir",
        type=Path,
        help="Path to the autoresearch project directory",
    )
    ap_auto_run.add_argument(
        "--backend",
        type=str,
        default="mock",
        choices=list(SUPPORTED_BACKENDS),
        help="LLM backend to use (default: mock)",
    )
    ap_auto_run.add_argument(
        "--model",
        type=str,
        default=None,
        help=(
            "Model identifier for the selected backend. "
            f"Defaults: claude-cli={DEFAULT_CLAUDE_MODEL}, "
            "codex-cli=CLI default, "
            f"openai={DEFAULT_OPENAI_MODEL}."
        ),
    )
    ap_auto_run.add_argument(
        "--openai-base-url",
        type=str,
        default=None,
        help=(
            "Base URL for the OpenAI-compatible backend "
            "(default: $OPENAI_BASE_URL or https://api.openai.com)."
        ),
    )
    ap_auto_run.add_argument(
        "--backend-bin",
        type=str,
        default=None,
        help=(
            "Executable path for CLI-backed LLMs "
            "(default: $CLAUDE_BIN/$CODEX_BIN, PATH, or current venv)."
        ),
    )
    ap_auto_run.add_argument(
        "--max-experiments",
        type=int,
        default=50,
        help="Maximum number of experiments to run (default: 50)",
    )
    ap_auto_run.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Timeout in seconds for each ngraph run (default: 600)",
    )
    ap_auto_run.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    def _cmd_autoresearch_run(args: argparse.Namespace) -> None:
        from netlab.autoresearch.cli import autoresearch_run

        autoresearch_run(args)

    ap_auto_run.set_defaults(func=_cmd_autoresearch_run)

    ap_auto_sa = auto_sub.add_parser(
        "structural-analysis",
        help="Enumerate layouts and evaluate connection retention under failures",
    )
    ap_auto_sa.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Save results to JSON file (default: print to stdout)",
    )

    def _cmd_autoresearch_structural_analysis(args: argparse.Namespace) -> None:
        from netlab.autoresearch.structural_analysis import (
            print_summary,
            run_structural_analysis,
            save_results,
        )

        results = run_structural_analysis()
        print_summary(results)
        if args.output:
            save_results(results, args.output)
            print(f"\nResults saved to {args.output}")

    ap_auto_sa.set_defaults(func=_cmd_autoresearch_structural_analysis)

    ap_auto_sweep = auto_sub.add_parser(
        "sweep",
        help="Sweep one DC side (fix other at default), extract alpha + per-mode BAC",
    )
    ap_auto_sweep.add_argument(
        "side",
        choices=["abc1", "xyz1"],
        help="DC side to sweep (other side held at default)",
    )
    ap_auto_sweep.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for results.jsonl and result directories",
    )
    ap_auto_sweep.add_argument("--iterations", type=int, default=200)
    ap_auto_sweep.add_argument("--timeout", type=int, default=300)
    ap_auto_sweep.add_argument("--seed", type=int, default=42)

    def _cmd_autoresearch_sweep(args: argparse.Namespace) -> None:
        from netlab.autoresearch.sweep import SweepConfig, print_results, run_sweep

        config = SweepConfig(
            output_dir=args.output_dir,
            failure_iterations=args.iterations,
            timeout_s=args.timeout,
            seed=args.seed,
        )
        entries = run_sweep(config, side=args.side)
        print_results(entries)

    ap_auto_sweep.set_defaults(func=_cmd_autoresearch_sweep)

    ap_auto_xsweep = auto_sub.add_parser(
        "cross-sweep",
        help="Sweep all ABC1 × XYZ1 combinations, extract alpha + per-mode BAC",
    )
    ap_auto_xsweep.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for results.jsonl and result directories",
    )
    ap_auto_xsweep.add_argument("--iterations", type=int, default=200)
    ap_auto_xsweep.add_argument("--timeout", type=int, default=300)
    ap_auto_xsweep.add_argument("--seed", type=int, default=42)

    def _cmd_autoresearch_cross_sweep(args: argparse.Namespace) -> None:
        from netlab.autoresearch.sweep import (
            SweepConfig,
            print_results,
            run_cross_sweep,
        )

        config = SweepConfig(
            output_dir=args.output_dir,
            failure_iterations=args.iterations,
            timeout_s=args.timeout,
            seed=args.seed,
        )
        entries = run_cross_sweep(config)
        print_results(entries)

    ap_auto_xsweep.set_defaults(func=_cmd_autoresearch_cross_sweep)

    args = ap.parse_args(argv)
    if args.verbose:
        set_global_log_level(logging.DEBUG)
    args.func(args)


if __name__ == "__main__":
    main()
