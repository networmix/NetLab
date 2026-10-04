"""Explicit merge sources are required, and both standard YAML suffixes work."""

import pytest

from netlab.scenario import ScenarioMerger


def test_merge_sources_and_named_workflow(tmp_path):
    base = tmp_path / "scenario.yml"
    base.write_text("network: {}\nworkflow: test\n")
    sources = tmp_path / "sources"
    sources.mkdir()
    (sources / "demands.yaml").write_text("demands: {tm: []}\n")
    workflows = tmp_path / "workflows"
    workflows.mkdir()
    (workflows / "steps.yaml").write_text(
        "workflows: {test: [{type: NetworkStats, name: stats}]}\n"
    )
    merger = ScenarioMerger(tmp_path)
    merger.add_source(sources, "demands", source_key="demands")
    merger.add_workflow_source(workflows)
    merged = merger.merge(base, seed=8)
    assert merged["seed"] == 8
    assert merged["demands"] == {"tm": []}
    assert merged["workflow"] == [{"type": "NetworkStats", "name": "stats"}]
    merger.add_source(tmp_path / "typo", "failures")
    with pytest.raises(FileNotFoundError, match="typo"):
        merger.merge(base)
