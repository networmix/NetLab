"""Run NetGraph experiments with scenario merging, cached results, and provenance."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from .artifacts import write_json_atomic, write_text_atomic
from .scenario import ScenarioMerger
from .simulation import invalidate_results, run_simulation

logger = logging.getLogger(__name__)


class ExperimentRunner(ABC):
    """Base experiment runner. Subclasses configure merge sources in ``get_merger``."""

    def __init__(
        self,
        root: Path,
        results_dir: Optional[Path] = None,
        topologies_dir: Optional[Path] = None,
        scenario_filename: str = "scenario.yml",
    ):
        """Use ``root/results`` and ``root/topologies`` unless paths are supplied."""
        self.root = Path(root)
        self.results_dir = Path(results_dir) if results_dir else self.root / "results"
        self.topologies_dir = (
            Path(topologies_dir) if topologies_dir else self.root / "topologies"
        )
        self.scenario_filename = scenario_filename
        self._merger: Optional[ScenarioMerger] = None

    @abstractmethod
    def get_merger(self) -> ScenarioMerger:
        """Configure the experiment's merge sources. Subclasses must implement this.

        Example:
            merger = ScenarioMerger(self.root)
            merger.add_source(self.root / "policies", "failures",
                              filter_fn=is_failure_policy)
            merger.add_source(self.root / "demands", "demands", source_key="demands")
            merger.add_workflow_source(self.root / "workflows")
            return merger
        """
        pass

    @property
    def merger(self) -> ScenarioMerger:
        """Get the configured merger (cached)."""
        if self._merger is None:
            self._merger = self.get_merger()
        return self._merger

    def discover_scenarios(self) -> List[str]:
        """List directories containing the configured scenario filename."""
        if not self.topologies_dir.exists():
            return []
        return sorted(
            d.name
            for d in self.topologies_dir.iterdir()
            if d.is_dir() and (d / self.scenario_filename).exists()
        )

    def run(
        self,
        scenarios: Optional[List[str]] = None,
        seeds: Optional[List[int]] = None,
        force: bool = False,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """Run selected scenarios and return counts by outcome.

        Defaults to all discovered scenarios and seeds 42, 43, 44. ``force`` bypasses
        the result cache; ``dry_run`` writes merged YAML without simulating.
        """
        if scenarios is None:
            scenarios = self.discover_scenarios()
        if seeds is None:
            seeds = [42, 43, 44]

        overall_stats = {
            "scenarios": [],
            "total_ran": 0,
            "total_cached": 0,
            "total_failed": 0,
            "total_prepared": 0,
        }

        for scenario in scenarios:
            stats = self.run_scenario(scenario, seeds, force, dry_run)
            overall_stats["scenarios"].append(stats)
            overall_stats["total_ran"] += stats["ran"]
            overall_stats["total_cached"] += stats["cached"]
            overall_stats["total_failed"] += stats["failed"]
            overall_stats["total_prepared"] += stats["prepared"]

        return overall_stats

    def run_scenario(
        self,
        scenario: str,
        seeds: List[int],
        force: bool = False,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """Run one scenario across seeds and return counts by outcome.

        ``force`` bypasses the result cache; ``dry_run`` only writes merged YAML.
        """
        stats = {
            "scenario": scenario,
            "seeds": [],
            "cached": 0,
            "ran": 0,
            "failed": 0,
            "prepared": 0,
        }

        for seed in seeds:
            result = self._run_seed(scenario, seed, force, dry_run)
            stats["seeds"].append({"seed": seed, **result})
            if result["status"] == "cached":
                stats["cached"] += 1
            elif result["status"] == "success":
                stats["ran"] += 1
            elif result["status"] == "dry_run":
                stats["prepared"] += 1
            else:
                stats["failed"] += 1

        self._write_provenance(scenario, seeds, stats)

        return stats

    def _run_seed(
        self,
        scenario: str,
        seed: int,
        force: bool,
        dry_run: bool,
    ) -> Dict[str, Any]:
        """Run a single (scenario, seed) combination."""
        seed_dir = self.results_dir / scenario / f"{scenario}__seed{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        scenario_file = seed_dir / f"{scenario}__seed{seed}_scenario.yml"
        results_file = seed_dir / f"{scenario}__seed{seed}_scenario.results.json"

        scenario_path = self.topologies_dir / scenario / self.scenario_filename
        try:
            merged = self.merger.merge(scenario_path, seed=seed)
            text = yaml.safe_dump(merged, sort_keys=False)
        except (ValueError, OSError, yaml.YAMLError) as exc:
            invalidate_results(results_file)
            scenario_file.unlink(missing_ok=True)
            return {"status": "invalid", "error": str(exc)}
        if not scenario_file.exists() or scenario_file.read_text() != text:
            invalidate_results(results_file)
        write_text_atomic(scenario_file, text)

        if dry_run:
            logger.info("[dry-run] %s seed=%d -> %s", scenario, seed, scenario_file)
            print(f"  [dry-run] {scenario} seed={seed} -> {scenario_file}")
            return {"status": "dry_run", "scenario_file": str(scenario_file)}

        outcome = run_simulation(scenario_file, results_path=results_file, force=force)
        logger.info("[%s] %s seed=%d", outcome.status, scenario, seed)
        return {
            "status": outcome.status,
            "results_file": str(results_file),
            "error": outcome.error,
        }

    def _write_provenance(
        self,
        scenario: str,
        seeds: List[int],
        stats: Dict[str, Any],
    ) -> None:
        """Write provenance information for a scenario run."""
        provenance_file = self.results_dir / scenario / "provenance.json"
        provenance = {
            "scenario": scenario,
            "seeds": seeds,
            "timestamp": datetime.now().isoformat(),
            "stats": stats,
        }
        write_json_atomic(provenance_file, provenance)

    def get_results_files(
        self,
        scenarios: Optional[List[str]] = None,
    ) -> Dict[str, List[Path]]:
        """Return result paths grouped by scenario; default to all discovered scenarios."""
        if scenarios is None:
            scenarios = self.discover_scenarios()

        results_by_scenario: Dict[str, List[Path]] = {}
        for scenario in scenarios:
            scenario_dir = self.results_dir / scenario
            if not scenario_dir.exists():
                continue
            results_files = []
            for seed_dir in scenario_dir.iterdir():
                if not seed_dir.is_dir():
                    continue
                for results_file in seed_dir.glob("*.results.json"):
                    results_files.append(results_file)
            if results_files:
                results_by_scenario[scenario] = sorted(results_files)

        return results_by_scenario
