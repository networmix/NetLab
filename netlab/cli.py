#!/usr/bin/env python3
"""
netlab CLI

Subcommands:
- run: Build TopoGen masters, seed per-scenario copies for provided seeds, run
  ngraph inspect+run, and emit simple TSV summaries.
- build: Build TopoGen masters only.

Notes:
- No 'report' step (ngraph report is not used/available).
- TopoGen build logic is integrated here (ported from the old shell script).
"""

from __future__ import annotations

import argparse
import concurrent.futures
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, NoReturn, Optional, Tuple


def die(msg: str, code: int = 1) -> NoReturn:
    print(f"❌ {msg}", file=sys.stderr)
    sys.exit(code)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _detect_topogen_invoke() -> List[str]:
    if shutil.which("topogen"):
        return ["topogen"]
    if shutil.which("python3"):
        return ["python3", "-m", "topogen"]
    if shutil.which("python"):
        return ["python", "-m", "topogen"]
    die(
        "Neither 'topogen' nor Python found on PATH. Activate your venv or install TopoGen."
    )


def _detect_ngraph_invoke() -> List[str]:
    if shutil.which("ngraph"):
        return ["ngraph"]
    if shutil.which("python3"):
        return ["python3", "-m", "ngraph"]
    if shutil.which("python"):
        return ["python", "-m", "ngraph"]
    die(
        "Neither 'ngraph' nor Python found on PATH. Activate your venv or install ngraph."
    )


def find_master_yaml_files(masters_path: Path) -> List[Path]:
    if masters_path.is_file():
        if masters_path.suffix in (".yml", ".yaml"):
            return [masters_path]
        die(f"Not a YAML file: {masters_path}")
    if masters_path.is_dir():
        return sorted(
            [
                p
                for p in masters_path.iterdir()
                if p.is_file() and p.suffix in (".yml", ".yaml")
            ]
        )
    die(f"Path not found: {masters_path}")


_SCENARIO_FILE_CANDIDATES = ("{stem}_scenario.yml", "{stem}_scenario.yaml")
_SCENARIO_SEED_LINE = re.compile(r"^(\s*seed\s*:\s*)\d+\s*$")


def _load_base_scenario(master_stem: str, master_scenarios_root: Path) -> Path:
    base_dir = master_scenarios_root / master_stem
    if not base_dir.exists():
        die(f"Base scenario directory not found: {base_dir}")
    for pat in _SCENARIO_FILE_CANDIDATES:
        p = base_dir / pat.format(stem=master_stem)
        if p.exists():
            return p
    die(f"No base scenario YAML found under: {base_dir}")


def _rewrite_all_seed_lines(yaml_text: str, new_seed: int) -> str:
    lines = yaml_text.splitlines()
    for i, line in enumerate(lines):
        m = _SCENARIO_SEED_LINE.match(line)
        if m:
            prefix = m.group(1)
            lines[i] = f"{prefix}{new_seed}"
    return "\n".join(lines) + ("\n" if yaml_text.endswith("\n") else "")


def create_seeded_scenarios(
    master_stem: str, master_scenarios_root: Path, seeds: List[int]
) -> List[Path]:
    base_yaml = _load_base_scenario(master_stem, master_scenarios_root)
    base_text = base_yaml.read_text(encoding="utf-8")
    created: List[Path] = []
    for s in seeds:
        dest_dir = master_scenarios_root / f"{master_stem}__seed{s}"
        ensure_dir(dest_dir)
        dest_yaml = dest_dir / f"{master_stem}__seed{s}_scenario.yml"
        mutated = _rewrite_all_seed_lines(base_text, s)
        dest_yaml.write_text(mutated, encoding="utf-8")
        created.append(dest_yaml)
    return created


def _run_to_log(cmd: List[str], cwd: Path, log_path: Path) -> int:
    ensure_dir(log_path.parent)
    with log_path.open("w", encoding="utf-8") as fh:
        try:
            proc = subprocess.Popen(
                cmd, cwd=str(cwd), stdout=fh, stderr=subprocess.STDOUT
            )
            return proc.wait()
        except Exception as e:
            try:
                fh.write(f"\n❌ netlab: failed to invoke: {' '.join(cmd)}\n{e}\n")
            except Exception:
                pass
            return 1


def _scenario_io_paths(scn_yaml: Path) -> Tuple[Path, str, Path]:
    scn_yaml_abs = scn_yaml.resolve()
    scn_dir = scn_yaml_abs.parent
    scn_name = scn_yaml_abs.name
    scn_stem = scn_name[: scn_name.rfind(".")]
    results_json = scn_dir / f"{scn_stem}.results.json"
    return scn_dir, scn_stem, results_json


def _has_cached(results_json: Path) -> bool:
    return results_json.exists()


def _inspect_run_one(
    ngraph_invoke: List[str], scn_yaml: Path, force: bool
) -> Tuple[Path, str, str]:
    scn_dir, scn_stem, results_json = _scenario_io_paths(scn_yaml)
    scn_yaml_abs = scn_yaml.resolve()
    log_ins = scn_dir / f"{scn_stem}.inspect.log"
    log_run = scn_dir / f"{scn_stem}.run.log"

    if not force and _has_cached(results_json):
        return scn_yaml, "⏭️ cached", "⏭️ cached"

    ec_ins = _run_to_log(
        ngraph_invoke + ["inspect", "-o", str(scn_dir), str(scn_yaml_abs)],
        scn_dir,
        log_ins,
    )
    if ec_ins != 0:
        return scn_yaml, "❌", "⏭️ skipped"

    ec_run = _run_to_log(
        ngraph_invoke
        + ["run", "-o", str(scn_dir), "-r", str(results_json), str(scn_yaml_abs)],
        scn_dir,
        log_run,
    )
    if ec_run != 0:
        return scn_yaml, "✅", "❌"
    return scn_yaml, "✅", "✅"


def _topogen_generate_build(
    topogen_invoke: List[str],
    cfg_abs: Path,
    workdir: Path,
    force: bool,
) -> Tuple[int, int]:
    graph_work = workdir / f"{cfg_abs.stem}_integrated_graph.json"
    log_gen = workdir / "generate.log"
    log_build = workdir / "build.log"

    # Generate stage
    gen_ec = 100
    run_generate = force or not graph_work.exists()
    if run_generate:
        gen_ec = _run_to_log(
            topogen_invoke + ["generate", str(cfg_abs), "-o", str(workdir)],
            Path.cwd(),
            log_gen,
        )
    else:
        gen_ec = 100
        log_gen.write_text(
            f"⏭️  Skipping generate: found existing {graph_work.name}\n",
            encoding="utf-8",
        )

    # Build stage
    scenario_out = workdir / f"{cfg_abs.stem}_scenario.yml"
    build_ec = _run_to_log(
        topogen_invoke + ["build", str(cfg_abs), "-o", str(scenario_out)],
        Path.cwd(),
        log_build,
    )
    return gen_ec, build_ec


def _build_masters(
    masters: List[Path],
    scenarios_dir: Path,
    build_jobs: int,
    build_timeout: int,
    force: bool,
    force_build: bool,
    topogen_invoke: Optional[List[str]] = None,
) -> List[str]:
    topogen_cmd = topogen_invoke or _detect_topogen_invoke()
    futures: List[Tuple[str, Path, concurrent.futures.Future[Tuple[int, int]]]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, build_jobs)) as ex:
        for master_yaml in masters:
            master_stem = master_yaml.stem
            workdir = scenarios_dir / master_stem / master_stem
            ensure_dir(workdir)
            cfg_abs = master_yaml.resolve()
            run_force_build = bool(force or force_build)
            fut = ex.submit(
                _topogen_generate_build,
                topogen_cmd,
                cfg_abs,
                workdir,
                run_force_build,
            )
            futures.append((master_stem, workdir, fut))

        build_errors: List[str] = []
        for master_stem, workdir, fut in futures:
            try:
                _gen_ec, build_ec = fut.result(
                    timeout=build_timeout
                    if build_timeout and build_timeout > 0
                    else None
                )  # type: ignore[arg-type]
            except concurrent.futures.TimeoutError:
                build_errors.append(f"{master_stem} (timeout)")
                continue
            if build_ec != 0:
                build_errors.append(f"{master_stem} (see logs in {workdir})")
            else:
                print(f"✅ Build completed: {master_stem}")
    return build_errors


def _cmd_build(args: argparse.Namespace) -> None:
    masters_dir: Path = args.configs
    scenarios_dir: Path = args.scenarios_dir

    masters = find_master_yaml_files(masters_dir)
    if not masters:
        die(f"No YAMLs under {masters_dir}")

    ensure_dir(scenarios_dir)
    topogen_invoke = _detect_topogen_invoke()

    print("\n===== Build Plan =====")
    print(f"Masters dir:    {masters_dir}")
    print(f"Masters found:   {len(masters)}")
    print(f"Scenarios dir:   {scenarios_dir}")
    print(f"Build jobs:      {max(1, args.build_jobs)}")
    if args.build_timeout and args.build_timeout > 0:
        print(f"Build timeout:   {args.build_timeout}s per master")
    else:
        print("Build timeout:   disabled")
    print("Behavior:")
    print(" - Build once per master (parallel across masters)")
    print("======================\n")

    build_errors = _build_masters(
        masters=masters,
        scenarios_dir=scenarios_dir,
        build_jobs=args.build_jobs,
        build_timeout=args.build_timeout,
        force=args.force,
        force_build=args.force_build,
        topogen_invoke=topogen_invoke,
    )

    if build_errors:
        die("One or more builds failed: " + "; ".join(build_errors))


def _cmd_run(args: argparse.Namespace) -> None:
    masters_dir: Path = args.configs
    scenarios_dir: Path = args.scenarios_dir
    seeds: List[int] = args.seeds

    if not seeds:
        die("No seeds specified")

    masters = find_master_yaml_files(masters_dir)
    if not masters:
        die(f"No YAMLs under {masters_dir}")

    ensure_dir(scenarios_dir)
    topogen_invoke = _detect_topogen_invoke()
    ngraph_invoke = _detect_ngraph_invoke()

    print("\n===== Experiment Plan =====")
    print(f"Masters dir:    {masters_dir}")
    print(f"Masters found:   {len(masters)}")
    print(f"Seeds:           {seeds}  (count: {len(seeds)})")
    print(f"Scenarios dir:   {scenarios_dir}")
    print(f"Build jobs:      {max(1, args.build_jobs)}")
    print(f"Run jobs:        {max(1, args.run_jobs)} (per master)")
    if args.build_timeout and args.build_timeout > 0:
        print(f"Build timeout:   {args.build_timeout}s per master")
    else:
        print("Build timeout:   disabled")
    print("Behavior:")
    print(" - Build once per master (parallel across masters)")
    print(" - Create per-seed scenario copies by replacing 'seed: <n>' lines")
    print(" - Run per-seed scenarios in parallel per master")
    print("==========================\n")

    # Build once per master
    build_errors = _build_masters(
        masters=masters,
        scenarios_dir=scenarios_dir,
        build_jobs=args.build_jobs,
        build_timeout=args.build_timeout,
        force=args.force,
        force_build=args.force_build,
        topogen_invoke=topogen_invoke,
    )

    if build_errors:
        die("One or more builds failed: " + "; ".join(build_errors))

    # Seed scenarios
    master_contexts: List[Dict[str, object]] = []
    for master_yaml in masters:
        master_stem = master_yaml.stem
        master_root = scenarios_dir / master_stem
        created = create_seeded_scenarios(master_stem, master_root, seeds)
        master_contexts.append({"stem": master_stem, "scenarios": created})
        print(f"📝 Seeded scenarios created for {master_stem}: {len(created)}")

    # Run inspect+run per master
    run_summaries_dir = scenarios_dir / "_run_summaries"
    ensure_dir(run_summaries_dir)
    ngraph_start_wall = time.time()
    ngraph_start = time.perf_counter()

    for ctx in master_contexts:
        stem = ctx["stem"]  # type: ignore[index]
        scenarios: List[Path] = ctx["scenarios"]  # type: ignore[assignment]
        print(
            f"🧪 Running scenarios (inspect+run) for master: {stem} (jobs={max(1, args.run_jobs)})"
        )
        summary_tsv = run_summaries_dir / f"{stem}.tsv"
        summary_tsv.write_text(
            "ScenarioDir\tScenario\tInspect\tRun\n", encoding="utf-8"
        )

        futures2: List[concurrent.futures.Future[Tuple[Path, str, str]]] = []
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=max(1, args.run_jobs)
        ) as ex:
            for scn in scenarios:
                fut = ex.submit(
                    _inspect_run_one,
                    ngraph_invoke,
                    scn,
                    bool(args.force or args.force_run),
                )
                futures2.append(fut)
            for fut in futures2:
                scn_path, ins_status, run_status = fut.result()
                with summary_tsv.open("a", encoding="utf-8") as fh:
                    fh.write(
                        f"{scn_path.parent}\t{scn_path.name}\t{ins_status}\t{run_status}\n"
                    )
        print(f"📦 Run finished: {stem}")

    # Overall timing
    ngraph_end_wall = time.time()
    ngraph_elapsed = time.perf_counter() - ngraph_start
    start_iso = datetime.fromtimestamp(ngraph_start_wall, tz=timezone.utc).isoformat()
    end_iso = datetime.fromtimestamp(ngraph_end_wall, tz=timezone.utc).isoformat()
    timing_tsv = run_summaries_dir / "_overall_ngraph_time.tsv"
    timing_tsv.write_text(
        "Scope\tStartUTC\tEndUTC\tElapsedSec\n"
        f"ngraph\t{start_iso}\t{end_iso}\t{ngraph_elapsed:.3f}\n",
        encoding="utf-8",
    )
    print(f"⏱️ Overall ngraph run time: {ngraph_elapsed:.3f}s")


def main() -> None:
    ap = argparse.ArgumentParser(prog="netlab", description="NetLab CLI")
    sub = ap.add_subparsers(dest="command", required=True)

    # build subcommand
    ap_build = sub.add_parser(
        "build",
        help="Build TopoGen masters only",
        description="Build TopoGen masters only",
    )
    ap_build.add_argument(
        "configs",
        nargs="?",
        default=Path("topogen_configs"),
        type=Path,
        help="Path to TopoGen master YAMLs (directory)",
    )
    ap_build.add_argument(
        "--scenarios-dir", default=Path("scenarios"), type=Path, help="Output root"
    )
    ap_build.add_argument(
        "--force", action="store_true", help="Force both generate and build"
    )
    ap_build.add_argument(
        "--force-build", action="store_true", help="Force TopoGen build stage"
    )
    ap_build.add_argument(
        "--build-jobs", type=int, default=max(1, (os.cpu_count() or 4) // 2)
    )
    ap_build.add_argument("--build-timeout", type=int, default=0)
    ap_build.set_defaults(func=_cmd_build)

    # run subcommand
    ap_run = sub.add_parser(
        "run",
        help="Build, seed by --seeds, and run ngraph inspect+run",
        description="Build TopoGen masters, expand scenarios by --seeds, then run ngraph",
    )
    ap_run.add_argument(
        "configs",
        nargs="?",
        default=Path("topogen_configs"),
        type=Path,
        help="Path to TopoGen master YAMLs (directory)",
    )
    ap_run.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        required=True,
        help="One or more integer seeds",
    )
    ap_run.add_argument(
        "--scenarios-dir", default=Path("scenarios"), type=Path, help="Output root"
    )
    ap_run.add_argument(
        "--force", action="store_true", help="Force both build and ngraph run"
    )
    ap_run.add_argument(
        "--force-build", action="store_true", help="Force TopoGen build only"
    )
    ap_run.add_argument("--force-run", action="store_true", help="Force ngraph run")
    ap_run.add_argument(
        "--build-jobs", type=int, default=max(1, (os.cpu_count() or 4) // 2)
    )
    ap_run.add_argument("--run-jobs", type=int, default=1)
    ap_run.add_argument("--build-timeout", type=int, default=0)
    ap_run.set_defaults(func=_cmd_run)

    # metrics subcommand
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
        help="Print summary tables from CSVs (project.csv and project_baseline_normalized.csv)",
    )
    ap_metrics.add_argument(
        "--summary-plots",
        action="store_true",
        help="When used with --summary, also render cross-seed figures from CSVs",
    )

    def _cmd_metrics(args: argparse.Namespace) -> None:
        from .metrics_cmd import print_summary_from_csv, run_metrics

        if bool(args.summary):
            print_summary_from_csv(args.scenarios_root, plots=bool(args.summary_plots))
            return
        run_metrics(
            root=args.scenarios_root,
            only=args.only,
            no_plots=bool(args.no_plots),
            enable_maxflow=bool(args.enable_maxflow),
        )

    ap_metrics.set_defaults(func=_cmd_metrics)

    # test subcommand (paired t-tests between two scenarios)
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
        from metrics.summary import _build_insights

        root: Path = args.scenarios_root
        out_root = root.parent / f"{root.name}_metrics"
        insights = _build_insights(out_root)
        if not insights:
            print("(no project insights available)")
            return
        a = args.scenario_a
        b = args.scenario_b
        alpha = float(args.alpha)
        # Filter to pairs matching (a,b) or (b,a)
        matches = [
            r
            for r in insights
            if (r.get("scen_a") == a and r.get("scen_b") == b)
            or (r.get("scen_a") == b and r.get("scen_b") == a)
        ]
        if not matches:
            print(f"No common-seed paired results found for {a} vs {b}.")
            return
        # Print compact table
        print(f"Paired t-tests for {a} vs {b} (alpha={alpha}):")
        print("metric  n  mean_diff  [95% CI]  t  p  p_adj  det")
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

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
