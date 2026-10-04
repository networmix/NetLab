"""TopoGen integration through its configuration, graph and scenario APIs."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from topogen import build_integrated_graph, load_from_json, save_to_json
from topogen.config import TopologyConfig
from topogen.context import RunContext
from topogen.scenario import build_scenario
from topogen.validation import validate_scenario_yaml

from .artifacts import (
    atomic_path,
    cache_matches,
    cache_path,
    fingerprint,
    package_versions,
    record_cache,
    sha256_file,
    write_text_atomic,
)


def generate_graph(config_path: Path, output_dir: Path, force: bool = False) -> Path:
    """Generate a corridor graph, reusing only verified inputs and output.

    Relative data paths and ``lib/`` resolve against the caller's working
    directory, matching TopoGen's public API. Batch callers use isolated workers.
    """
    config = TopologyConfig.from_yaml(config_path)
    config.validate()
    graph_path = output_dir / f"{config_path.stem}_integrated_graph.json"
    key = fingerprint(
        {
            "schema": 1,
            "config": config_path.read_text(encoding="utf-8"),
            "data": {
                name: sha256_file(path)
                for name, path in asdict(config.data_sources).items()
            },
            "packages": package_versions(),
        }
    )
    if not force and cache_matches(graph_path, key):
        return graph_path
    cache_path(graph_path).unlink(missing_ok=True)
    context = RunContext(output_dir, config_path.stem)
    output_dir.mkdir(parents=True, exist_ok=True)
    graph = build_integrated_graph(config, context=context)
    with atomic_path(graph_path) as temporary:
        save_to_json(
            graph, temporary, config.projection.target_crs, config.output.formatting
        )
    record_cache(graph_path, key)
    return graph_path


def build_seed_scenario(
    config_path: Path,
    graph_path: Path,
    scenario_path: Path,
    seed: int,
) -> Path:
    """Assemble and validate one seeded scenario without copying graph/config files."""
    config = TopologyConfig.from_yaml(config_path)
    config.output.scenario_seed = seed
    graph, crs = load_from_json(graph_path)
    if crs != config.projection.target_crs:
        raise ValueError(
            f"Graph CRS {crs} differs from configured {config.projection.target_crs}"
        )
    scenario_path.parent.mkdir(parents=True, exist_ok=True)
    text = build_scenario(
        graph, config, context=RunContext(scenario_path.parent, config_path.stem)
    )
    issues = validate_scenario_yaml(
        text,
        integrated_graph_path=graph_path,
        hw_component_map=config.components.hw_component,
        optics_map=config.components.optics,
    )
    if issues:
        raise ValueError("Scenario validation failed:\n" + "\n".join(issues))
    write_text_atomic(scenario_path, text)
    return scenario_path
