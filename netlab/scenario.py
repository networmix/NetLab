"""Merge topology, hardware, failure policies, demands, and workflows into a scenario."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MergeSource:
    """One YAML directory, destination section, and optional selection rule."""

    path: Path
    target_key: str
    source_key: str | None = None
    filter_fn: Callable[[str, Any], bool] | None = None


def _yaml_files(path: Path) -> list[Path]:
    if not path.is_dir():
        raise FileNotFoundError(f"Configuration directory does not exist: {path}")
    return sorted(
        p for p in path.iterdir() if p.is_file() and p.suffix in {".yml", ".yaml"}
    )


def _load_mapping(path: Path) -> dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return data


class ScenarioMerger:
    """Merge YAML configuration sources into a NetGraph scenario.

    Example:
        merger = ScenarioMerger(experiment_root)
        merger.add_source(policies_dir, "failures", filter_fn=is_failure_policy)
        merger.add_source(demands_dir, "demands", source_key="demands")
        scenario = merger.merge(topology_dir / "scenario.yml", seed=42)
    """

    def __init__(
        self,
        root: Path,
        shared_dir: Optional[Path] = None,
    ):
        """Use shared configs from ``shared_dir``, defaulting to root.parent / "_shared"."""
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
        """Register a YAML directory to merge into ``target_key``; return self.

        Extract ``source_key`` from each file when set. Otherwise, merge top-level
        items accepted by ``filter_fn`` (all items when no filter is supplied).
        """
        self.sources.append(MergeSource(path, target_key, source_key, filter_fn))
        return self

    def add_workflow_source(self, path: Path) -> "ScenarioMerger":
        """Register workflow definitions used to resolve named references; return self."""
        self.workflow_sources.append(Path(path))
        return self

    def merge(
        self,
        scenario_path: Path,
        seed: Optional[int] = None,
    ) -> dict:
        """Merge configured sources into the base scenario and optionally set its seed."""
        scenario = _load_mapping(scenario_path)

        self._merge_components(scenario)

        for source in self.sources:
            self._merge_source(scenario, source)

        self._resolve_workflows(scenario)

        if seed is not None:
            scenario["seed"] = seed

        return scenario

    def _merge_components(self, scenario: dict) -> None:
        """Merge components from shared directory."""
        components_path = self.shared_dir / "components.yml"
        if components_path.exists():
            components = _load_mapping(components_path)
            if "components" in components:
                scenario.setdefault("components", {}).update(components["components"])
                logger.debug("Merged components from %s", components_path)

    def _merge_source(self, scenario: dict, source: MergeSource) -> None:
        """Merge a single source into the scenario."""
        for yaml_file in _yaml_files(source.path):
            data = _load_mapping(yaml_file)

            if source.source_key:
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
        all_workflows: Dict[str, Any] = {}
        for workflow_dir in self.workflow_sources:
            for workflow_file in _yaml_files(workflow_dir):
                workflow_config = _load_mapping(workflow_file)
                if "workflows" in workflow_config:
                    all_workflows.update(workflow_config["workflows"])

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
