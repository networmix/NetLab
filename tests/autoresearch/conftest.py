"""Shared fixtures for autoresearch tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from netlab.autoresearch.hypothesis import HypothesisTemplate

DATA_DIR = Path(__file__).parent / "data"


@pytest.fixture
def square_mesh_results() -> dict:
    """Load the square-mesh simulation results."""
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
