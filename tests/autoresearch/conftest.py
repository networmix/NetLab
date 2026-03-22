"""Shared fixtures for autoresearch tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from netlab.autoresearch.hypothesis import HypothesisTemplate

DATA_DIR = Path(__file__).parent / "data"


@pytest.fixture
def square_mesh_results() -> dict:
    """Pre-generated ngraph results from square_mesh.yaml.

    Contains workflow steps: msd_baseline, tm_placement, node_to_node_capacity_matrix.
    Generated via: Scenario.from_yaml(square_mesh.yaml).run().results.to_dict()

    Note: The results use current ngraph output format which differs from the
    format expected by analyze_one_seed() in metrics_cmd.py. Specifically:
    - base_demands uses source/target/volume instead of source_path/sink_path/demand
    - metadata lacks 'baseline: true' flag
    - flow_results[0].failure_id is not "baseline"
    The individual metric functions (compute_alpha_star, compute_bac) also
    expect the older format. This fixture is suitable for testing autoresearch
    objective functions that extract metrics directly from the results dict.
    """
    results_path = DATA_DIR / "square_mesh_results.json"
    with open(results_path) as f:
        return json.load(f)


@pytest.fixture
def sample_template_path() -> Path:
    """Path to the sample hypothesis_template.yml fixture."""
    return DATA_DIR / "hypothesis_template.yml"


@pytest.fixture
def sample_template(sample_template_path: Path) -> HypothesisTemplate:
    """Parsed HypothesisTemplate from the sample fixture."""
    return HypothesisTemplate(sample_template_path)
