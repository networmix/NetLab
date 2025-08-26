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
import hashlib
import logging
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, NoReturn, Optional, Tuple

import yaml

from metrics.aggregate import write_json_atomic

from .log_config import configure_from_env, set_global_log_level


def die(msg: str, code: int = 1) -> NoReturn:
    print(f"❌ {msg}", file=sys.stderr)
    sys.exit(code)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _create_run_provenance(
    masters: List[Path],
    seeds: List[int],
    scenarios_dir: Path,
    configs_root: Optional[Path] = None,
    build_jobs: Optional[int] = None,
    run_jobs: Optional[int] = None,
    build_timeout: Optional[int] = None,
    force: Optional[bool] = None,
    force_run: Optional[bool] = None,
    topogen_invoke: Optional[List[str]] = None,
    ngraph_invoke: Optional[List[str]] = None,
) -> Dict[str, object]:
    """Create comprehensive provenance information for a netlab run."""
    cwd = Path.cwd()
    provenance = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "seeds": sorted(seeds),
        "topogen_configs": {},
        "scenarios_dir": os.path.relpath(scenarios_dir, start=cwd),
    }
    if configs_root is not None:
        provenance["configs_root"] = os.path.relpath(configs_root, start=cwd)
    if build_jobs is not None:
        provenance["build_jobs"] = int(build_jobs)
    if run_jobs is not None:
        provenance["run_jobs"] = int(run_jobs)
    if build_timeout is not None:
        provenance["build_timeout"] = int(build_timeout)
    if force is not None:
        provenance["force"] = bool(force)
    if force_run is not None:
        provenance["force_run"] = bool(force_run)
    if topogen_invoke is not None:
        provenance["topogen_invoke"] = list(topogen_invoke)
    if ngraph_invoke is not None:
        provenance["ngraph_invoke"] = list(ngraph_invoke)

    # Get git commit if available
    try:
        commit = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode("utf-8")
            .strip()
        )
        provenance["git_commit"] = commit
    except Exception as e:
        logging.warning("Failed to retrieve git commit: %s", e)
        provenance["git_commit_error"] = str(e)

    # Add topogen config file information with hashes
    for master_yaml in masters:
        try:
            config_content = master_yaml.read_bytes()
            config_hash = hashlib.sha256(config_content).hexdigest()
            rel_path = os.path.relpath(master_yaml, start=cwd)
            provenance["topogen_configs"][master_yaml.name] = {
                "path": rel_path,
                "sha256": config_hash,
                "size_bytes": len(config_content),
            }
        except Exception as e:
            logging.warning("Failed to hash config %s: %s", master_yaml, e)
            rel_path = os.path.relpath(master_yaml, start=cwd)
            provenance["topogen_configs"][master_yaml.name] = {
                "path": rel_path,
                "hash_error": str(e),
            }

    # Record seeds planned per master
    if masters and seeds:
        seeds_per_master: Dict[str, List[int]] = {
            m.stem: sorted(seeds) for m in masters
        }
        provenance["seeds_per_master"] = seeds_per_master

    return provenance


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


# New approach: generate integrated graph once per master, then for each seed,
# override TopoGen config output.scenario_seed and build a per-seed scenario.


def _topogen_generate_only(
    topogen_invoke: List[str], cfg_abs: Path, workdir: Path, force: bool
) -> int:
    graph_work = workdir / f"{cfg_abs.stem}_integrated_graph.json"
    log_gen = workdir / "generate.log"

    run_generate = force or not graph_work.exists()
    if run_generate:
        logging.info(
            "Topogen generate: config=%s -> graph=%s (log=%s)",
            str(cfg_abs),
            str(graph_work),
            str(log_gen),
        )
        return _run_to_log(
            topogen_invoke + ["generate", str(cfg_abs), "-o", str(workdir)],
            Path.cwd(),
            log_gen,
        )
    else:
        log_gen.write_text(
            f"⏭️  Skipping generate: found existing {graph_work.name}\n",
            encoding="utf-8",
        )
        logging.info(
            "Topogen generate: skip (exists) config=%s graph=%s",
            str(cfg_abs),
            str(graph_work),
        )
        return 0


def _write_seed_overridden_config(master_cfg: Path, seed_cfg: Path, seed: int) -> None:
    raw = master_cfg.read_text(encoding="utf-8")
    data = yaml.safe_load(raw)
    if not isinstance(data, dict):
        die(f"Invalid TopoGen config (expected mapping): {master_cfg}")
    output = data.get("output")
    if not isinstance(output, dict):
        output = {}
    output["scenario_seed"] = int(seed)
    data["output"] = output
    # Preserve other content; dump without flow style
    seed_cfg.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def _build_seed_scenario(
    topogen_invoke: List[str],
    master_cfg: Path,
    workdir: Path,
    seed_dir: Path,
    seed: int,
) -> Tuple[Path, int]:
    stem = master_cfg.stem
    ensure_dir(seed_dir)

    # Ensure integrated graph is available under seed_dir with expected name
    graph_src = workdir / f"{stem}_integrated_graph.json"
    graph_dst = seed_dir / f"{stem}_integrated_graph.json"
    if not graph_src.exists():
        return seed_dir, 1
    try:
        shutil.copy2(graph_src, graph_dst)
    except Exception:
        # Surface the error by returning non-zero; do not skip silently
        return seed_dir, 1

    # Write a per-seed config that keeps the original stem (prefix) for artefacts
    seed_cfg = seed_dir / f"{stem}.yml"
    _write_seed_overridden_config(master_cfg, seed_cfg, seed)

    # Build per-seed scenario YAML into seed_dir, capturing logs
    seed_yaml = seed_dir / f"{stem}__seed{seed}_scenario.yml"
    log_build = seed_dir / "build.log"
    logging.info(
        "Topogen build: stem=%s seed=%s cfg=%s -> out=%s (log=%s)",
        stem,
        seed,
        str(seed_cfg),
        str(seed_yaml),
        str(log_build),
    )
    ec = _run_to_log(
        topogen_invoke + ["build", str(seed_cfg), "-o", str(seed_yaml)],
        Path.cwd(),
        log_build,
    )
    return seed_yaml, ec


def _run_to_log(cmd: List[str], cwd: Path, log_path: Path) -> int:
    ensure_dir(log_path.parent)
    with log_path.open("w", encoding="utf-8") as fh:
        try:
            logging.debug(
                "exec: %s | cwd=%s | log=%s", " ".join(cmd), str(cwd), str(log_path)
            )
            proc = subprocess.Popen(
                cmd, cwd=str(cwd), stdout=fh, stderr=subprocess.STDOUT
            )
            rc = proc.wait()
            logging.debug("exit: rc=%s cmd=%s", rc, " ".join(cmd))
            return rc
        except Exception as e:
            try:
                fh.write(f"\n❌ netlab: failed to invoke: {' '.join(cmd)}\n{e}\n")
            except Exception as write_err:
                # Fall back to stderr if log write fails
                print(
                    f"❌ netlab: failed to invoke and log: {' '.join(cmd)}\n{e}\nlog error: {write_err}",
                    file=sys.stderr,
                )
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

    logging.info(
        "ngraph inspect: %s (log=%s)",
        str(scn_yaml_abs),
        str(log_ins),
    )
    ec_ins = _run_to_log(
        ngraph_invoke + ["inspect", "-o", str(scn_dir), str(scn_yaml_abs)],
        scn_dir,
        log_ins,
    )
    if ec_ins != 0:
        return scn_yaml, "❌", "⏭️ skipped"

    logging.info(
        "ngraph run: %s -> results=%s (log=%s)",
        str(scn_yaml_abs),
        str(results_json),
        str(log_run),
    )
    ec_run = _run_to_log(
        ngraph_invoke
        + ["run", "-o", str(scn_dir), "-r", str(results_json), str(scn_yaml_abs)],
        scn_dir,
        log_run,
    )
    if ec_run != 0:
        return scn_yaml, "✅", "❌"
    return scn_yaml, "✅", "✅"


def _generate_masters(
    masters: List[Path],
    scenarios_dir: Path,
    build_jobs: int,
    build_timeout: int,
    force: bool,
    topogen_invoke: Optional[List[str]] = None,
) -> List[str]:
    topogen_cmd = topogen_invoke or _detect_topogen_invoke()
    futures: List[Tuple[str, Path, concurrent.futures.Future[int]]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, build_jobs)) as ex:
        for master_yaml in masters:
            master_stem = master_yaml.stem
            workdir = scenarios_dir / master_stem / master_stem
            ensure_dir(workdir)
            cfg_abs = master_yaml.resolve()
            fut = ex.submit(
                _topogen_generate_only,
                topogen_cmd,
                cfg_abs,
                workdir,
                bool(force),
            )
            futures.append((master_stem, workdir, fut))

        build_errors: List[str] = []
        for master_stem, workdir, fut in futures:
            try:
                gen_ec = fut.result(
                    timeout=build_timeout
                    if build_timeout and build_timeout > 0
                    else None
                )  # type: ignore[arg-type]
            except concurrent.futures.TimeoutError:
                build_errors.append(f"{master_stem} (timeout during generate)")
                continue
            if gen_ec != 0:
                build_errors.append(
                    f"{master_stem} (generate failed, see logs in {workdir})"
                )
            else:
                print(f"✅ Integrated graph ready: {master_stem}")
    return build_errors

    # NOTE: old _build_masters removed; generation is handled by _generate_masters,
    # and per-seed builds are executed later per seed.


def _cmd_build(args: argparse.Namespace) -> None:
    masters_dir: Path = args.configs
    scenarios_dir: Path = args.scenarios_dir

    masters = find_master_yaml_files(masters_dir)
    if not masters:
        die(f"No YAMLs under {masters_dir}")

    ensure_dir(scenarios_dir)
    topogen_invoke = _detect_topogen_invoke()

    print("\n===== Generate Plan =====")
    print(f"Masters dir:    {masters_dir}")
    print(f"Masters found:   {len(masters)}")
    print(f"Scenarios dir:   {scenarios_dir}")
    print(f"Build jobs:      {max(1, args.build_jobs)}")
    if args.build_timeout and args.build_timeout > 0:
        print(f"Build timeout:   {args.build_timeout}s per master")
    else:
        print("Build timeout:   disabled")
    print("Behavior:")
    print(" - Generate integrated graph once per master (parallel across masters)")
    print("======================\n")

    build_errors = _generate_masters(
        masters=masters,
        scenarios_dir=scenarios_dir,
        build_jobs=args.build_jobs,
        build_timeout=args.build_timeout,
        force=args.force,
        topogen_invoke=topogen_invoke,
    )

    if build_errors:
        die("One or more builds failed: " + "; ".join(build_errors))

    # Create and save comprehensive provenance information for build
    build_provenance = _create_run_provenance(
        masters=masters,
        seeds=[],
        scenarios_dir=scenarios_dir,
        configs_root=masters_dir,
        build_jobs=args.build_jobs,
        run_jobs=None,
        build_timeout=args.build_timeout,
        force=args.force,
        force_run=None,
        topogen_invoke=topogen_invoke,
        ngraph_invoke=None,
    )
    build_provenance["command"] = "build"
    provenance_path = scenarios_dir / "_build_provenance.json"
    write_json_atomic(provenance_path, build_provenance)
    print(f"📋 Build provenance saved to: {provenance_path}")
    print("✅ All builds completed successfully")


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
    print(" - Generate integrated graph once per master")
    print(
        " - For each seed: override TopoGen output.scenario_seed and build per-seed scenario"
    )
    print(" - Run per-seed scenarios in parallel per master")
    print("==========================\n")

    # Generate integrated graph once per master
    build_errors = _generate_masters(
        masters=masters,
        scenarios_dir=scenarios_dir,
        build_jobs=args.build_jobs,
        build_timeout=args.build_timeout,
        force=args.force,
        topogen_invoke=topogen_invoke,
    )

    if build_errors:
        die("One or more builds failed: " + "; ".join(build_errors))

    # Build per-seed scenarios via TopoGen (config-level seed override)
    master_contexts: List[Dict[str, object]] = []
    for master_yaml in masters:
        master_stem = master_yaml.stem
        master_root = scenarios_dir / master_stem
        workdir = master_root / master_stem
        ensure_dir(workdir)

        created: List[Path] = []
        # Optional parallelization across seeds per master
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=max(1, args.run_jobs)
        ) as ex:
            futs: List[concurrent.futures.Future[Tuple[Path, int]]] = []
            for s in seeds:
                seed_dir = master_root / f"{master_stem}__seed{s}"
                futs.append(
                    ex.submit(
                        _build_seed_scenario,
                        topogen_invoke,
                        master_yaml.resolve(),
                        workdir,
                        seed_dir,
                        int(s),
                    )
                )
            for fut in futs:
                seed_yaml, ec = fut.result()
                if ec != 0:
                    die(f"TopoGen build failed for scenario: {seed_yaml}")
                created.append(seed_yaml)

        master_contexts.append({"stem": master_stem, "scenarios": created})
        print(f"📝 Per-seed scenarios built for {master_stem}: {len(created)}")

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

    # Create and save comprehensive provenance information
    provenance = _create_run_provenance(
        masters=masters,
        seeds=seeds,
        scenarios_dir=scenarios_dir,
        configs_root=masters_dir,
        build_jobs=args.build_jobs,
        run_jobs=args.run_jobs,
        build_timeout=args.build_timeout,
        force=args.force,
        force_run=args.force_run,
        topogen_invoke=topogen_invoke,
        ngraph_invoke=ngraph_invoke,
    )
    provenance["command"] = "run"

    # List results JSON files with hashes (relative to scenarios_dir)
    results_files: Dict[str, Dict[str, object]] = {}
    for ctx in master_contexts:
        scenarios_list: List[Path] = ctx.get("scenarios", [])  # type: ignore[assignment]
        for scn in scenarios_list:
            try:
                _, _, results_json = _scenario_io_paths(scn)
                rel_path = os.path.relpath(results_json, start=Path.cwd())
                try:
                    content = results_json.read_bytes()
                    results_files[rel_path] = {
                        "path": rel_path,
                        "sha256": hashlib.sha256(content).hexdigest(),
                        "size_bytes": len(content),
                    }
                except Exception as e:
                    results_files[rel_path] = {
                        "path": rel_path,
                        "hash_error": str(e),
                    }
            except Exception as outer:
                # Associate error with unknown path (should be rare)
                results_files.setdefault("<unknown>", {"errors": []})  # type: ignore[index]
                try:
                    # Append error detail
                    errs = results_files["<unknown>"]["errors"]  # type: ignore[index]
                    if isinstance(errs, list):
                        errs.append(str(outer))
                except Exception:
                    results_files["<unknown>"] = {"errors": [str(outer)]}
    if results_files:
        provenance["results_files"] = results_files

    provenance_path = scenarios_dir / "provenance.json"
    write_json_atomic(provenance_path, provenance)
    print(f"📋 Run provenance saved to: {provenance_path}")


def main() -> None:
    # Initialize logging for NetLab; level can be overridden via NETLAB_LOG_LEVEL
    configure_from_env()
    ap = argparse.ArgumentParser(prog="netlab", description="NetLab CLI")
    ap.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging (overrides NETLAB_LOG_LEVEL)",
    )
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
        help="Print summary tables from CSVs and render cross-seed figures",
    )

    def _cmd_metrics(args: argparse.Namespace) -> None:
        from .metrics_cmd import print_summary_from_csv, run_metrics

        if bool(args.summary):
            # Summary mode: always render cross-seed figures
            print_summary_from_csv(args.scenarios_root, plots=True, quiet=False)
            return
        # Full analysis mode: compute everything and, unless disabled, also render cross-seed figures
        run_metrics(
            root=args.scenarios_root,
            only=args.only,
            no_plots=bool(args.no_plots),
            enable_maxflow=bool(args.enable_maxflow),
        )
        if not bool(args.no_plots):
            # Generate cross-seed figures from the freshly written CSVs
            print_summary_from_csv(args.scenarios_root, plots=True, quiet=True)

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
    # Override log level if -v provided
    if bool(getattr(args, "verbose", False)):
        set_global_log_level(logging.DEBUG)
    args.func(args)


if __name__ == "__main__":
    main()
