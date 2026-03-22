"""Tests for netlab.autoresearch.prompt module.

Covers all acceptance criteria from the plan (Step 5).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import pytest

from netlab.autoresearch.experiment_log import LogEntry
from netlab.autoresearch.hypothesis import HypothesisTemplate
from netlab.autoresearch.prompt import (
    ParseError,
    build_hypothesis_prompt,
    build_reflection_prompt,
    parse_hypothesis_response,
    render_memory_section,
)

# ---------------------------------------------------------------------------
# Stubs / mocks for ResearchMemory (F-8 not yet implemented)
# ---------------------------------------------------------------------------


@dataclass
class StubInsight:
    claim: str
    confidence: str
    evidence_for: list[str] = field(default_factory=list)
    evidence_against: list[str] = field(default_factory=list)


@dataclass
class StubDeadEnd:
    params_summary: str
    failure_type: str
    reason: str
    lesson: str


@dataclass
class StubMemory:
    active_insights: list[StubInsight] = field(default_factory=list)
    dead_ends: list[StubDeadEnd] = field(default_factory=list)
    strategy: str = ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).parent / "data"


def _make_entry(
    exp_id: str = "exp_001",
    params: Optional[dict[str, Any]] = None,
    status: str = "success",
    score: Optional[float] = None,
    error_detail: Optional[str] = None,
    seed: int = 42,
) -> LogEntry:
    if params is None:
        params = {"link_capacity": 400, "flow_policy": "SHORTEST_PATHS_ECMP"}
    return LogEntry(
        exp_id=exp_id,
        params=params,
        params_hash="deadbeef",
        status=status,
        metrics={"alpha_star": 0.5} if score is not None else None,
        objective_score=score,
        error_detail=error_detail,
        execution_time_s=1.0,
        seed=seed,
        timestamp="2026-01-01T00:00:00",
    )


# ---------------------------------------------------------------------------
# Acceptance criteria tests
# ---------------------------------------------------------------------------


class TestPromptStructure:
    """Prompt structure: program.md + template + 5 entries + best."""

    def test_prompt_contains_sections_in_order(
        self, sample_template: HypothesisTemplate
    ):
        program_md = "You are a network research assistant."
        entries = [_make_entry(f"exp_{i:03d}", score=i * 0.1) for i in range(1, 6)]
        best = entries[-1]  # exp_005, score=0.5

        history = "\n".join(
            f"  {e.exp_id}: status={e.status}, score={e.objective_score:.4f}"
            for e in entries
        )
        memory_section = ""

        sys_prompt, user_prompt = build_hypothesis_prompt(
            program_md=program_md,
            template=sample_template,
            history=history,
            memory_section=memory_section,
            best=best,
        )

        # System prompt is program.md
        assert sys_prompt == program_md

        # User prompt contains sections in order
        param_idx = user_prompt.index("Parameter Space")
        history_idx = user_prompt.index("Experiment History")
        best_idx = user_prompt.index("Current Best")

        assert param_idx < history_idx < best_idx

        # Parameter space includes types and ranges
        assert "link_capacity (int)" in user_prompt
        assert "range [100.0, 1000.0]" in user_prompt
        assert "flow_policy (enum)" in user_prompt

        # History contains all 5 entries
        for i in range(1, 6):
            assert f"exp_{i:03d}" in user_prompt

        # Best entry with params and score
        assert "exp_005" in user_prompt
        assert "0.5000" in user_prompt


class TestMemorySectionPresent:
    """Memory section present: 2 insights, 3 dead_ends, strategy text."""

    def test_memory_section_renders_all_parts(self):
        memory = StubMemory(
            active_insights=[
                StubInsight(
                    claim="Higher capacity improves throughput",
                    confidence="moderate",
                    evidence_for=["exp_001", "exp_002", "exp_003"],
                    evidence_against=["exp_004"],
                ),
                StubInsight(
                    claim="ECMP outperforms UCMP on square meshes",
                    confidence="tentative",
                    evidence_for=["exp_005", "exp_006"],
                    evidence_against=[],
                ),
            ],
            dead_ends=[
                StubDeadEnd(
                    params_summary="link_capacity=100",
                    failure_type="crash",
                    reason="Insufficient capacity for demands",
                    lesson="Minimum viable capacity is ~200",
                ),
                StubDeadEnd(
                    params_summary="demand_volume=100000",
                    failure_type="infeasible",
                    reason="Demand exceeds network capacity",
                    lesson="Keep demand_volume below 80000",
                ),
                StubDeadEnd(
                    params_summary="seed=999",
                    failure_type="timeout",
                    reason="Solver did not converge",
                    lesson="Avoid extreme seeds with large demands",
                ),
            ],
            strategy="Focus on mid-range capacities (300-600) with ECMP.",
        )

        section = render_memory_section(memory)

        # Contains Verified Insights with both insights as bullets
        assert "Verified Insights" in section
        assert "Higher capacity improves throughput" in section
        assert "ECMP outperforms UCMP on square meshes" in section

        # Contains Known Dead Ends with 3 entries
        assert "Known Dead Ends" in section
        assert "link_capacity=100" in section
        assert "demand_volume=100000" in section
        assert "seed=999" in section

        # Contains strategy
        assert "Your Current Strategy" in section
        assert "Focus on mid-range capacities" in section


class TestMemorySectionAbsent:
    """Memory section absent: empty memory -> no headers injected."""

    def test_empty_memory_returns_empty_string(self):
        memory = StubMemory()
        section = render_memory_section(memory)
        assert section == ""

    def test_empty_memory_no_research_notes_header(
        self, sample_template: HypothesisTemplate
    ):
        """build_hypothesis_prompt with empty memory has no Research Notes section."""
        memory_section = render_memory_section(StubMemory())
        _, user_prompt = build_hypothesis_prompt(
            program_md="test",
            template=sample_template,
            history="No experiments run yet.",
            memory_section=memory_section,
            best=None,
        )
        assert "Your Research Notes" not in user_prompt


class TestAdvisoryFraming:
    """Advisory framing: memory content uses advisory language, not commands."""

    def test_contains_revise_language(self):
        memory = StubMemory(
            active_insights=[
                StubInsight(
                    claim="test insight",
                    confidence="tentative",
                    evidence_for=["exp_001", "exp_002"],
                    evidence_against=[],
                ),
            ],
        )
        section = render_memory_section(memory)
        assert "revise" in section.lower()


class TestErrorDetailInclusion:
    """Error detail inclusion: history entry with error_detail is rendered."""

    def test_error_detail_in_history(self, sample_template: HypothesisTemplate):
        error_entry = _make_entry(
            "exp_001",
            status="crash",
            error_detail="ValueError: bad flow",
        )
        # Simulate what windowed_history would produce (the prompt builder
        # receives pre-rendered history). The entry line must include the error.
        history_line = (
            f"  {error_entry.exp_id}: status={error_entry.status}, "
            f"error={error_entry.error_detail!r}"
        )
        _, user_prompt = build_hypothesis_prompt(
            program_md="test",
            template=sample_template,
            history=history_line,
            memory_section="",
            best=None,
        )
        assert "ValueError: bad flow" in user_prompt


class TestParseFencedYaml:
    """Parse fenced YAML block."""

    def test_extracts_params_from_fenced_block(self):
        response = (
            "I think we should try x=5.\n"
            "```yaml\n"
            "params:\n"
            "  x: 5\n"
            "```\n"
            "This should improve the score."
        )
        result = parse_hypothesis_response(response)
        assert result == {"x": 5}

    def test_extracts_params_with_multiple_values(self):
        response = (
            "reasoning...\n"
            "```yaml\n"
            "params:\n"
            "  link_capacity: 500\n"
            "  flow_policy: SHORTEST_PATHS_ECMP\n"
            "```\n"
        )
        result = parse_hypothesis_response(response)
        assert result == {"link_capacity": 500, "flow_policy": "SHORTEST_PATHS_ECMP"}


class TestParseRawYaml:
    """Parse raw YAML (no fences)."""

    def test_parses_raw_yaml_response(self):
        response = "params:\n  x: 5"
        result = parse_hypothesis_response(response)
        assert result == {"x": 5}

    def test_parses_raw_yaml_without_params_key(self):
        response = "x: 5\ny: 10"
        result = parse_hypothesis_response(response)
        assert result == {"x": 5, "y": 10}


class TestParseFailure:
    """Parse failure: non-YAML response raises ParseError."""

    def test_plain_text_raises_parse_error(self):
        with pytest.raises(ParseError):
            parse_hypothesis_response("I don't know what to try")

    def test_invalid_yaml_in_fence_raises_parse_error(self):
        response = "```yaml\n{invalid: [yaml: broken\n```"
        with pytest.raises(ParseError):
            parse_hypothesis_response(response)

    def test_none_yaml_raises_parse_error(self):
        # yaml.safe_load("just a string") returns a string, not a dict
        with pytest.raises(ParseError):
            parse_hypothesis_response("just a plain string")


class TestCharBudget:
    """Char budget: build_hypothesis_prompt includes history as-is.

    The windowed_history method (tested in F-2) handles truncation.
    This test verifies that build_hypothesis_prompt passes the history
    string through unchanged.
    """

    def test_long_history_included_unchanged(self, sample_template: HypothesisTemplate):
        # Simulate a pre-rendered history string (already truncated by windowed_history)
        long_history = "Summary: Total experiments: 30\n" + "\n".join(
            f"  exp_{i:03d}: status=success, score={i * 0.01:.4f}" for i in range(1, 31)
        )
        _, user_prompt = build_hypothesis_prompt(
            program_md="test",
            template=sample_template,
            history=long_history,
            memory_section="",
            best=_make_entry("exp_030", score=0.30),
        )
        # History is included verbatim
        assert long_history in user_prompt

    def test_truncated_history_included_as_is(
        self, sample_template: HypothesisTemplate
    ):
        """A short (already-truncated) history is included unchanged."""
        short_history = "Summary: Total experiments: 30\nLast 3 experiments:\n  exp_028\n  exp_029\n  exp_030"
        _, user_prompt = build_hypothesis_prompt(
            program_md="test",
            template=sample_template,
            history=short_history,
            memory_section="",
            best=None,
        )
        assert short_history in user_prompt


class TestReflectionPrompt:
    """Reflection prompt: 5 recent results + memory + best."""

    def test_reflection_prompt_contains_all_sections(self):
        recent = [_make_entry(f"exp_{i:03d}", score=i * 0.1) for i in range(1, 6)]
        memory = StubMemory(
            active_insights=[
                StubInsight(
                    claim="test insight",
                    confidence="tentative",
                    evidence_for=["exp_001", "exp_002"],
                    evidence_against=[],
                ),
            ],
            dead_ends=[
                StubDeadEnd(
                    params_summary="cap=100",
                    failure_type="crash",
                    reason="too low",
                    lesson="min is 200",
                ),
            ],
            strategy="Try higher capacities.",
        )
        best = recent[-1]

        sys_prompt, user_prompt = build_reflection_prompt(
            recent_entries=recent,
            memory=memory,
            best=best,
        )

        # System prompt is about reflection
        assert "reflect" in sys_prompt.lower()

        # Contains all 5 experiment summaries
        for i in range(1, 6):
            assert f"exp_{i:03d}" in user_prompt

        # Contains current memory contents
        assert "test insight" in user_prompt
        assert "cap=100" in user_prompt
        assert "Try higher capacities" in user_prompt

        # Contains current best
        assert "exp_005" in user_prompt
        assert "0.5000" in user_prompt

        # Contains the 3 task sections
        assert "INSIGHTS" in user_prompt
        assert "DEAD ENDS" in user_prompt
        assert "STRATEGY" in user_prompt

    def test_reflection_prompt_with_empty_memory(self):
        recent = [_make_entry("exp_001", score=0.5)]
        sys_prompt, user_prompt = build_reflection_prompt(
            recent_entries=recent,
            memory=StubMemory(),
            best=recent[0],
        )
        # Should still have task sections even with empty memory
        assert "INSIGHTS" in user_prompt
        assert "DEAD ENDS" in user_prompt
        assert "STRATEGY" in user_prompt

    def test_reflection_prompt_includes_error_detail(self):
        entry = _make_entry(
            "exp_001", status="crash", error_detail="ValueError: bad flow"
        )
        _, user_prompt = build_reflection_prompt(
            recent_entries=[entry],
            memory=StubMemory(),
            best=None,
        )
        assert "ValueError: bad flow" in user_prompt


class TestEdgeCases:
    """Additional edge case tests."""

    def test_best_is_none(self, sample_template: HypothesisTemplate):
        _, user_prompt = build_hypothesis_prompt(
            program_md="test",
            template=sample_template,
            history="No experiments run yet.",
            memory_section="",
            best=None,
        )
        assert "No successful experiments yet" in user_prompt

    def test_parse_yaml_with_nested_params(self):
        response = "```yaml\nparams:\n  x: 5\n  y: 10\n  z: hello\n```"
        result = parse_hypothesis_response(response)
        assert result == {"x": 5, "y": 10, "z": "hello"}

    def test_parse_yaml_without_params_key_returns_whole_dict(self):
        response = "```yaml\nx: 5\ny: 10\n```"
        result = parse_hypothesis_response(response)
        assert result == {"x": 5, "y": 10}

    def test_memory_section_with_only_strategy(self):
        memory = StubMemory(strategy="Focus on exploration.")
        section = render_memory_section(memory)
        assert "Your Research Notes" in section
        assert "Your Current Strategy" in section
        assert "Focus on exploration" in section
        # Should not have empty insight/dead_end headers
        assert "Verified Insights" not in section
        assert "Known Dead Ends" not in section

    def test_memory_section_with_only_insights(self):
        memory = StubMemory(
            active_insights=[
                StubInsight(
                    claim="test",
                    confidence="strong",
                    evidence_for=[
                        "exp_001",
                        "exp_002",
                        "exp_003",
                        "exp_004",
                        "exp_005",
                    ],
                    evidence_against=["exp_006"],
                ),
            ],
        )
        section = render_memory_section(memory)
        assert "Verified Insights" in section
        assert "[strong]" in section
        assert "5 for, 1 against" in section
