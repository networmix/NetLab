"""
Scenario Merger Module

Provides utilities for merging multiple YAML configuration files into a complete
ngraph scenario. This enables modular experiment design where topology, policies,
demands, and workflows can be defined in separate files.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)


class MergeSource:
    """Defines a source for merging into a scenario."""

    def __init__(
        self,
        path: Path,
        target_key: str,
        source_key: Optional[str] = None,
        filter_fn: Optional[Callable[[str, Any], bool]] = None,
    ):
        """
        Args:
            path: Directory containing YAML files to merge
            target_key: Key in scenario where merged data goes (e.g., "failures")
            source_key: Key to extract from each YAML file (e.g., "demands").
                       If None, merges all top-level keys that pass filter_fn.
            filter_fn: Optional filter function (key, value) -> bool.
                      Only items passing filter are merged.
        """
        self.path = path
        self.target_key = target_key
        self.source_key = source_key
        self.filter_fn = filter_fn


class ScenarioMerger:
    """
    Merges multiple YAML configuration files into a complete scenario.

    Typical usage:
        merger = ScenarioMerger(experiment_root)
        merger.add_source("policies", policies_dir, "failures", filter_fn=is_policy)
        merger.add_source("demands", demands_dir, "demands", source_key="demands")
        scenario = merger.merge(topology_dir / "scenario.yml", seed=42)
    """

    def __init__(
        self,
        root: Path,
        shared_dir: Optional[Path] = None,
    ):
        """
        Initialize the scenario merger.

        Args:
            root: Root directory of the experiment
            shared_dir: Optional directory with shared configs (e.g., components)
                       Defaults to root.parent / "_shared"
        """
        self.root = Path(root)
        self.shared_dir = (
            Path(shared_dir) if shared_dir else self.root.parent / "_shared"
        )
        self.sources: List[MergeSource] = []
        self.workflow_sources: List[Path] = []

    def add_source(
        self,
        path: Path,
        target_key: str,
        source_key: Optional[str] = None,
        filter_fn: Optional[Callable[[str, Any], bool]] = None,
    ) -> "ScenarioMerger":
        """
        Add a merge source.

        Args:
            path: Directory containing YAML files to merge
            target_key: Key in scenario where merged data goes
            source_key: Key to extract from each YAML file (if None, merges filtered items)
            filter_fn: Optional filter for items when source_key is None

        Returns:
            self for chaining
        """
        self.sources.append(MergeSource(path, target_key, source_key, filter_fn))
        return self

    def add_workflow_source(self, path: Path) -> "ScenarioMerger":
        """
        Add a directory containing workflow definitions.

        Workflows are loaded and used to resolve string workflow references
        in the scenario.

        Args:
            path: Directory containing workflow YAML files

        Returns:
            self for chaining
        """
        self.workflow_sources.append(Path(path))
        return self

    def merge(
        self,
        scenario_path: Path,
        seed: Optional[int] = None,
    ) -> dict:
        """
        Merge all sources into a complete scenario.

        Args:
            scenario_path: Path to the base scenario YAML file
            seed: Optional random seed to inject into the scenario

        Returns:
            Complete merged scenario dictionary
        """
        # Load base scenario
        with open(scenario_path) as f:
            scenario = yaml.safe_load(f) or {}

        # Merge shared components
        self._merge_components(scenario)

        # Merge all configured sources
        for source in self.sources:
            self._merge_source(scenario, source)

        # Resolve workflow references
        self._resolve_workflows(scenario)

        # Inject seed if provided
        if seed is not None:
            scenario["seed"] = seed

        return scenario

    def _merge_components(self, scenario: dict) -> None:
        """Merge components from shared directory."""
        components_path = self.shared_dir / "components.yml"
        if components_path.exists():
            with open(components_path) as f:
                components = yaml.safe_load(f) or {}
            if "components" in components:
                scenario.setdefault("components", {}).update(components["components"])
                logger.debug("Merged components from %s", components_path)

    def _merge_source(self, scenario: dict, source: MergeSource) -> None:
        """Merge a single source into the scenario."""
        if not source.path.exists():
            logger.debug("Source path does not exist: %s", source.path)
            return

        for yaml_file in sorted(source.path.glob("*.yml")):
            with open(yaml_file) as f:
                data = yaml.safe_load(f) or {}

            if source.source_key:
                # Extract specific key from the file
                if source.source_key in data:
                    scenario.setdefault(source.target_key, {}).update(
                        data[source.source_key]
                    )
                    logger.debug(
                        "Merged %s from %s into %s",
                        source.source_key,
                        yaml_file,
                        source.target_key,
                    )
            else:
                # Merge all items that pass the filter
                for key, value in data.items():
                    if source.filter_fn is None or source.filter_fn(key, value):
                        scenario.setdefault(source.target_key, {})[key] = value
                        logger.debug(
                            "Merged %s from %s into %s",
                            key,
                            yaml_file,
                            source.target_key,
                        )

    def _resolve_workflows(self, scenario: dict) -> None:
        """Resolve workflow string references to actual workflow definitions."""
        # Load all workflow definitions
        all_workflows: Dict[str, Any] = {}
        for workflow_dir in self.workflow_sources:
            if not workflow_dir.exists():
                continue
            for workflow_file in sorted(workflow_dir.glob("*.yml")):
                with open(workflow_file) as f:
                    workflow_config = yaml.safe_load(f) or {}
                if "workflows" in workflow_config:
                    all_workflows.update(workflow_config["workflows"])

        # Resolve workflow reference if it's a string
        if "workflow" in scenario and isinstance(scenario["workflow"], str):
            workflow_name = scenario["workflow"]
            if workflow_name in all_workflows:
                scenario["workflow"] = all_workflows[workflow_name]
                logger.debug("Resolved workflow reference: %s", workflow_name)
            else:
                raise ValueError(
                    f"Unknown workflow '{workflow_name}'. "
                    f"Available: {list(all_workflows.keys())}"
                )


def is_failure_policy(key: str, value: Any) -> bool:
    """Filter function to identify failure policy definitions."""
    return isinstance(value, dict) and "modes" in value
