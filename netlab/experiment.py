"""
Experiment Runner Module

Provides a base class for running ngraph experiments with scenario merging,
caching, and provenance tracking.
"""

from __future__ import annotations

import json
import logging
import subprocess
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from .scenario import ScenarioMerger

logger = logging.getLogger(__name__)


class ExperimentRunner(ABC):
    """
    Base class for running ngraph experiments.

    Provides:
    - Scenario discovery
    - Scenario merging (via ScenarioMerger)
    - Running ngraph with caching
    - Provenance tracking

    Subclasses must implement:
    - get_merger(): Configure and return a ScenarioMerger
    - Optionally override other methods for custom behavior
    """

    def __init__(
        self,
        root: Path,
        results_dir: Optional[Path] = None,
        topologies_dir: Optional[Path] = None,
        scenario_filename: str = "scenario.yml",
    ):
        """
        Initialize the experiment runner.

        Args:
            root: Root directory of the experiment
            results_dir: Directory for results (default: root / "results")
            topologies_dir: Directory containing topologies (default: root / "topologies")
            scenario_filename: Name of scenario file in each topology dir
        """
        self.root = Path(root)
        self.results_dir = Path(results_dir) if results_dir else self.root / "results"
        self.topologies_dir = (
            Path(topologies_dir) if topologies_dir else self.root / "topologies"
        )
        self.scenario_filename = scenario_filename
        self._merger: Optional[ScenarioMerger] = None

    @abstractmethod
    def get_merger(self) -> ScenarioMerger:
        """
        Configure and return a ScenarioMerger for this experiment.

        Subclasses must implement this to configure merge sources.

        Example:
            def get_merger(self) -> ScenarioMerger:
                merger = ScenarioMerger(self.root)
                merger.add_source(self.root / "policies", "failures",
                                  filter_fn=is_failure_policy)
                merger.add_source(self.root / "demands", "demands",
                                  source_key="demands")
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
        """
        Find all scenario directories.

        Returns:
            List of scenario names (directory names containing scenario files)
        """
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
        """
        Run scenarios with given seeds.

        Args:
            scenarios: List of scenario names to run (default: all discovered)
            seeds: List of random seeds (default: [42, 43, 44])
            force: Re-run even if results exist
            dry_run: Only generate merged scenarios, don't run ngraph

        Returns:
            Dict with overall run statistics
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
        }

        for scenario in scenarios:
            stats = self.run_scenario(scenario, seeds, force, dry_run)
            overall_stats["scenarios"].append(stats)
            overall_stats["total_ran"] += stats["ran"]
            overall_stats["total_cached"] += stats["cached"]
            overall_stats["total_failed"] += stats["failed"]

        return overall_stats

    def run_scenario(
        self,
        scenario: str,
        seeds: List[int],
        force: bool = False,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """
        Run a single scenario with given seeds.

        Args:
            scenario: Name of scenario directory
            seeds: List of random seeds to run
            force: Re-run even if results exist
            dry_run: Only generate merged scenario, don't run ngraph

        Returns:
            Dict with run statistics for this scenario
        """
        stats = {"scenario": scenario, "seeds": [], "cached": 0, "ran": 0, "failed": 0}

        for seed in seeds:
            result = self._run_seed(scenario, seed, force, dry_run)
            stats["seeds"].append({"seed": seed, **result})
            if result["status"] == "cached":
                stats["cached"] += 1
            elif result["status"] == "success":
                stats["ran"] += 1
            else:
                stats["failed"] += 1

        # Write provenance
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

        # Skip if cached
        if results_file.exists() and not force:
            logger.info("[cached] %s seed=%d", scenario, seed)
            print(f"  [cached] {scenario} seed={seed}")
            return {"status": "cached", "results_file": str(results_file)}

        # Merge and write scenario with seed
        scenario_path = self.topologies_dir / scenario / self.scenario_filename
        merged = self.merger.merge(scenario_path, seed=seed)

        with open(scenario_file, "w") as f:
            yaml.safe_dump(merged, f, sort_keys=False, default_flow_style=False)

        if dry_run:
            logger.info("[dry-run] %s seed=%d -> %s", scenario, seed, scenario_file)
            print(f"  [dry-run] {scenario} seed={seed} -> {scenario_file}")
            return {"status": "dry_run", "scenario_file": str(scenario_file)}

        # Validate with ngraph inspect
        logger.info("[inspect] %s seed=%d", scenario, seed)
        print(f"  [inspect] {scenario} seed={seed}")
        result = subprocess.run(
            ["ngraph", "inspect", str(scenario_file)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            logger.error("[FAILED] inspect: %s", result.stderr.strip())
            print(f"  [FAILED] inspect: {result.stderr.strip()}")
            error_log = seed_dir / f"{scenario}__seed{seed}_error.log"
            with open(error_log, "w") as f:
                f.write(f"ngraph inspect failed:\n{result.stderr}")
            return {"status": "inspect_failed", "error": result.stderr}

        # Run ngraph
        logger.info("[running] %s seed=%d", scenario, seed)
        print(f"  [running] {scenario} seed={seed}")
        result = subprocess.run(
            ["ngraph", "run", str(scenario_file), "-r", str(results_file)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            logger.error("[FAILED] run: %s", result.stderr.strip())
            print(f"  [FAILED] run: {result.stderr.strip()}")
            error_log = seed_dir / f"{scenario}__seed{seed}_error.log"
            with open(error_log, "w") as f:
                f.write(f"ngraph run failed:\n{result.stderr}")
            return {"status": "run_failed", "error": result.stderr}

        logger.info("[done] %s seed=%d", scenario, seed)
        print(f"  [done] {scenario} seed={seed}")
        return {"status": "success", "results_file": str(results_file)}

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
        with open(provenance_file, "w") as f:
            json.dump(provenance, f, indent=2)

    def get_results_files(
        self,
        scenarios: Optional[List[str]] = None,
    ) -> Dict[str, List[Path]]:
        """
        Get all results files grouped by scenario.

        Args:
            scenarios: List of scenarios to include (default: all discovered)

        Returns:
            Dict mapping scenario names to lists of results file paths
        """
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
