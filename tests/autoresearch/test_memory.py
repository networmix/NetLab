"""Tests for memory module — covers every acceptance-criteria row."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from netlab.autoresearch.experiment_log import ExperimentLog, LogEntry
from netlab.autoresearch.memory import (
    DeadEnd,
    Insight,
    ResearchMemory,
)


def _make_entry(
    exp_id: str = "exp_001",
    params: dict | None = None,
    params_hash: str = "abc123",
    status: str = "success",
    metrics: dict | None = None,
    objective_score: float | None = 0.5,
    seed: int = 42,
) -> LogEntry:
    return LogEntry(
        exp_id=exp_id,
        params=params or {"x": 1},
        params_hash=params_hash,
        status=status,
        metrics=metrics or ({"m": 0.5} if status == "success" else None),
        objective_score=objective_score,
        error_detail=None,
        execution_time_s=1.0,
        seed=seed,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


def _make_log(tmp_path: Path, n: int = 10) -> ExperimentLog:
    """Create an ExperimentLog with n entries (exp_001 .. exp_NNN)."""
    log = ExperimentLog(tmp_path)
    for i in range(1, n + 1):
        log.append(
            _make_entry(
                exp_id=f"exp_{i:03d}",
                params_hash=f"h{i}",
                objective_score=i * 0.1,
            )
        )
    return log


def _make_insight(
    insight_id: str = "ins_001",
    evidence_for: list[str] | None = None,
    evidence_against: list[str] | None = None,
    claim: str = "Test claim",
) -> Insight:
    return Insight(
        id=insight_id,
        created_at_exp=1,
        updated_at_exp=1,
        claim=claim,
        evidence_for=evidence_for or ["exp_001", "exp_002"],
        evidence_against=evidence_against or [],
    )


# ---------- Create insight with 1 exp ----------


class TestInsightMinEvidence:
    def test_rejected_with_1_evidence(self, tmp_path: Path) -> None:
        """add_insight with 1 evidence_for is rejected."""
        log = _make_log(tmp_path)
        mem = ResearchMemory(tmp_path / "memory")

        ins = _make_insight(evidence_for=["exp_001"])
        err = mem.add_insight(ins, log)

        assert err is not None
        assert "minimum 2" in err
        assert len(mem.active_insights) == 0


# ---------- Create insight with 2 exps ----------


class TestInsightAccepted:
    def test_accepted_with_2_evidence(self, tmp_path: Path) -> None:
        """add_insight with 2 evidence_for is accepted; confidence == tentative."""
        log = _make_log(tmp_path)
        mem = ResearchMemory(tmp_path / "memory")

        ins = _make_insight(evidence_for=["exp_001", "exp_002"])
        err = mem.add_insight(ins, log)

        assert err is None
        assert len(mem.active_insights) == 1
        assert mem.active_insights[0].confidence == "tentative"


# ---------- Confidence upgrade ----------


class TestConfidenceUpgrade:
    def test_moderate_at_4_evidence(self, tmp_path: Path) -> None:
        """Adding a 3rd and 4th experiment upgrades confidence to moderate."""
        log = _make_log(tmp_path)
        mem = ResearchMemory(tmp_path / "memory")

        ins = _make_insight(evidence_for=["exp_001", "exp_002"])
        mem.add_insight(ins, log)

        # Add 3rd experiment -> still tentative (2-3 = tentative)
        mem.update_insight(
            "ins_001",
            {"evidence_for": ["exp_001", "exp_002", "exp_003"], "updated_at_exp": 3},
        )
        assert mem.active_insights[0].confidence == "tentative"

        # Add 4th experiment -> moderate
        mem.update_insight(
            "ins_001",
            {
                "evidence_for": ["exp_001", "exp_002", "exp_003", "exp_004"],
                "updated_at_exp": 4,
            },
        )
        assert mem.active_insights[0].confidence == "moderate"


# ---------- Confidence strong ----------


class TestConfidenceStrong:
    def test_strong_at_5_evidence(self, tmp_path: Path) -> None:
        """Adding a 5th experiment upgrades confidence to strong."""
        log = _make_log(tmp_path)
        mem = ResearchMemory(tmp_path / "memory")

        ins = _make_insight(
            evidence_for=["exp_001", "exp_002", "exp_003", "exp_004", "exp_005"]
        )
        mem.add_insight(ins, log)

        assert mem.active_insights[0].confidence == "strong"


# ---------- 20-entry limit ----------


class TestInsightLimit:
    def test_21st_insight_rejected(self, tmp_path: Path) -> None:
        """Attempting to add 21st active insight is rejected."""
        log = _make_log(tmp_path, n=50)
        mem = ResearchMemory(tmp_path / "memory")

        # Add 20 insights
        for i in range(1, 21):
            ins = _make_insight(
                insight_id=f"ins_{i:03d}",
                evidence_for=[f"exp_{2 * i - 1:03d}", f"exp_{2 * i:03d}"],
                claim=f"Claim {i}",
            )
            err = mem.add_insight(ins, log)
            assert err is None, f"Insight {i} should have been accepted: {err}"

        assert len(mem.active_insights) == 20

        # 21st should be rejected
        ins21 = _make_insight(
            insight_id="ins_021",
            evidence_for=["exp_041", "exp_042"],
            claim="Claim 21",
        )
        err = mem.add_insight(ins21, log)
        assert err is not None
        assert "limit" in err.lower()
        assert len(mem.active_insights) == 20


# ---------- Retire + add ----------


class TestRetireAndAdd:
    def test_retire_frees_slot(self, tmp_path: Path) -> None:
        """Retire 1 insight, then add a new one => 20 active, retired has status."""
        log = _make_log(tmp_path, n=50)
        mem = ResearchMemory(tmp_path / "memory")

        for i in range(1, 21):
            ins = _make_insight(
                insight_id=f"ins_{i:03d}",
                evidence_for=[f"exp_{2 * i - 1:03d}", f"exp_{2 * i:03d}"],
                claim=f"Claim {i}",
            )
            mem.add_insight(ins, log)

        assert len(mem.active_insights) == 20

        # Retire one
        mem.retire_insight("ins_001")
        assert len(mem.active_insights) == 19

        # Check retired status
        retired = [ins for ins in mem._insights if ins.id == "ins_001"]
        assert len(retired) == 1
        assert retired[0].status == "retired"

        # Now adding succeeds
        new_ins = _make_insight(
            insight_id="ins_new",
            evidence_for=["exp_041", "exp_042"],
            claim="New claim",
        )
        err = mem.add_insight(new_ins, log)
        assert err is None
        assert len(mem.active_insights) == 20


# ---------- Contradiction flag ----------


class TestContradictionFlag:
    def test_flagged_at_half_ratio(self, tmp_path: Path) -> None:
        """evidence_for: 4, evidence_against: 2 => ratio 0.5 => flagged."""
        log = _make_log(tmp_path)
        mem = ResearchMemory(tmp_path / "memory")

        ins = _make_insight(
            evidence_for=["exp_001", "exp_002", "exp_003", "exp_004"],
            evidence_against=["exp_005", "exp_006"],
        )
        mem.add_insight(ins, log)

        assert mem.active_insights[0].flagged_for_revision is True

    def test_not_flagged_below_half(self, tmp_path: Path) -> None:
        """evidence_for: 4, evidence_against: 1 => ratio 0.25 => not flagged."""
        log = _make_log(tmp_path)
        mem = ResearchMemory(tmp_path / "memory")

        ins = _make_insight(
            evidence_for=["exp_001", "exp_002", "exp_003", "exp_004"],
            evidence_against=["exp_005"],
        )
        mem.add_insight(ins, log)

        assert mem.active_insights[0].flagged_for_revision is False


# ---------- Dead end dedup ----------


class TestDeadEndDedup:
    def test_same_lesson_merges(self, tmp_path: Path) -> None:
        """Two dead ends with same lesson => 1 entry with both exp_ids."""
        mem = ResearchMemory(tmp_path / "memory")

        de1 = DeadEnd(
            id="de_001",
            exp_ids=["exp_001"],
            params_summary="x=1",
            failure_type="crash",
            reason="OOM",
            lesson="X causes OOM",
        )
        de2 = DeadEnd(
            id="de_002",
            exp_ids=["exp_002"],
            params_summary="x=2",
            failure_type="crash",
            reason="OOM again",
            lesson="X causes OOM",
        )

        mem.add_dead_end(de1)
        mem.add_dead_end(de2)

        assert len(mem.dead_ends) == 1
        assert "exp_001" in mem.dead_ends[0].exp_ids
        assert "exp_002" in mem.dead_ends[0].exp_ids


# ---------- Dead end window ----------


class TestDeadEndWindow:
    def test_16_dead_ends_keeps_15(self, tmp_path: Path) -> None:
        """Adding 16 dead ends with different lessons => 15 kept, oldest dropped."""
        mem = ResearchMemory(tmp_path / "memory")

        for i in range(1, 17):
            de = DeadEnd(
                id=f"de_{i:03d}",
                exp_ids=[f"exp_{i:03d}"],
                params_summary=f"x={i}",
                failure_type="crash",
                reason=f"Reason {i}",
                lesson=f"Lesson {i}",
            )
            mem.add_dead_end(de)

        assert len(mem.dead_ends) == 15
        # Oldest (lesson "Lesson 1") should be dropped
        lessons = {de.lesson for de in mem.dead_ends}
        assert "Lesson 1" not in lessons
        assert "Lesson 16" in lessons


# ---------- Strategy line limit ----------


class TestStrategyLineLimit:
    def test_40_lines_truncated_to_30(self, tmp_path: Path) -> None:
        """Writing 40-line strategy => stored <= 30 lines."""
        mem = ResearchMemory(tmp_path / "memory")

        text = "\n".join(f"Line {i}" for i in range(1, 41))
        mem.update_strategy(text)

        stored_lines = mem.strategy.splitlines()
        assert len(stored_lines) <= 30
        assert stored_lines[0] == "Line 1"
        assert stored_lines[-1] == "Line 30"


# ---------- Invalid experiment ID ----------


class TestInvalidExpId:
    def test_nonexistent_exp_rejected(self, tmp_path: Path) -> None:
        """Insight citing exp_999 not in log is rejected."""
        log = _make_log(tmp_path, n=5)
        mem = ResearchMemory(tmp_path / "memory")

        ins = _make_insight(evidence_for=["exp_001", "exp_999"])
        err = mem.add_insight(ins, log)

        assert err is not None
        assert "exp_999" in err
        assert len(mem.active_insights) == 0


# ---------- Reflection parse ----------


class TestReflectionParse:
    def test_valid_reflection(self, tmp_path: Path) -> None:
        """Valid JSON + strategy markdown parsed correctly."""
        log = _make_log(tmp_path)
        mem = ResearchMemory(tmp_path / "memory")

        response = (
            "Here are my reflections:\n\n"
            "```json\n"
            "{\n"
            '  "insights": [\n'
            "    {\n"
            '      "claim": "Higher x improves score",\n'
            '      "evidence_for": ["exp_001", "exp_002"],\n'
            '      "evidence_against": []\n'
            "    }\n"
            "  ],\n"
            '  "dead_ends": [\n'
            "    {\n"
            '      "params_summary": "x=100",\n'
            '      "failure_type": "crash",\n'
            '      "reason": "OOM",\n'
            '      "lesson": "Keep x below 50"\n'
            "    }\n"
            "  ],\n"
            '  "retire_insights": []\n'
            "}\n"
            "```\n\n"
            "STRATEGY:\n"
            "Focus on exploring x in range 10-50.\n"
            "Avoid values above 50.\n"
        )

        err = mem.parse_reflection_output(response, log)

        assert err is None
        assert len(mem.active_insights) == 1
        assert mem.active_insights[0].claim == "Higher x improves score"
        assert mem.active_insights[0].confidence == "tentative"
        assert len(mem.dead_ends) == 1
        assert mem.dead_ends[0].lesson == "Keep x below 50"
        assert "exploring x" in mem.strategy

    def test_malformed_output_returns_error(self, tmp_path: Path) -> None:
        """Malformed reflection output returns error string."""
        log = _make_log(tmp_path)
        mem = ResearchMemory(tmp_path / "memory")

        response = "This is just some text with no JSON block at all."
        err = mem.parse_reflection_output(response, log)

        assert err is not None
        assert "parse failed" in err.lower()

    def test_invalid_json_returns_error(self, tmp_path: Path) -> None:
        """Invalid JSON inside fenced block returns error."""
        log = _make_log(tmp_path)
        mem = ResearchMemory(tmp_path / "memory")

        response = "```json\n{broken json\n```\n"
        err = mem.parse_reflection_output(response, log)

        assert err is not None
        assert "invalid JSON" in err

    def test_reflection_invalid_exp_id(self, tmp_path: Path) -> None:
        """Reflection citing nonexistent exp_id produces error for that insight."""
        log = _make_log(tmp_path, n=5)
        mem = ResearchMemory(tmp_path / "memory")

        response = (
            "```json\n"
            "{\n"
            '  "insights": [\n'
            "    {\n"
            '      "claim": "Some claim",\n'
            '      "evidence_for": ["exp_001", "exp_999"],\n'
            '      "evidence_against": []\n'
            "    }\n"
            "  ],\n"
            '  "dead_ends": [],\n'
            '  "retire_insights": []\n'
            "}\n"
            "```\n"
        )

        err = mem.parse_reflection_output(response, log)
        assert err is not None
        assert "exp_999" in err
        assert len(mem.active_insights) == 0


# ---------- Load empty memory ----------


class TestLoadEmpty:
    def test_empty_dir(self, tmp_path: Path) -> None:
        """Empty memory dir => all collections empty, no errors."""
        mem_dir = tmp_path / "memory"
        mem_dir.mkdir()

        mem = ResearchMemory(mem_dir)
        mem.load()

        assert mem.active_insights == []
        assert mem.dead_ends == []
        assert mem.strategy == ""

    def test_nonexistent_dir(self, tmp_path: Path) -> None:
        """Non-existent memory dir => created on load, all empty."""
        mem_dir = tmp_path / "memory_new"

        mem = ResearchMemory(mem_dir)
        mem.load()

        assert mem_dir.exists()
        assert mem.active_insights == []
        assert mem.dead_ends == []
        assert mem.strategy == ""


# ---------- Round-trip ----------


class TestRoundTrip:
    def test_save_and_reload(self, tmp_path: Path) -> None:
        """Save 3 insights + 5 dead_ends + strategy, reload => all match."""
        log = _make_log(tmp_path, n=20)
        mem_dir = tmp_path / "memory"
        mem = ResearchMemory(mem_dir)

        # Add 3 insights
        for i in range(1, 4):
            ins = _make_insight(
                insight_id=f"ins_{i:03d}",
                evidence_for=[f"exp_{2 * i - 1:03d}", f"exp_{2 * i:03d}"],
                claim=f"Claim {i}",
            )
            err = mem.add_insight(ins, log)
            assert err is None

        # Add 5 dead ends
        for i in range(1, 6):
            de = DeadEnd(
                id=f"de_{i:03d}",
                exp_ids=[f"exp_{i:03d}"],
                params_summary=f"x={i}",
                failure_type="crash",
                reason=f"Reason {i}",
                lesson=f"Lesson {i}",
            )
            mem.add_dead_end(de)

        # Set strategy
        mem.update_strategy("Focus on parameter x.\nAvoid y > 10.")

        # Save
        mem.save()

        # Reload into fresh instance
        mem2 = ResearchMemory(mem_dir)
        mem2.load()

        # Verify insights
        assert len(mem2.active_insights) == 3
        for i, ins in enumerate(mem2.active_insights, start=1):
            assert ins.id == f"ins_{i:03d}"
            assert ins.claim == f"Claim {i}"
            assert ins.confidence == "tentative"
            assert ins.status == "active"

        # Verify dead ends
        assert len(mem2.dead_ends) == 5
        for i, de in enumerate(mem2.dead_ends, start=1):
            assert de.lesson == f"Lesson {i}"

        # Verify strategy
        assert mem2.strategy == "Focus on parameter x.\nAvoid y > 10."


# ---------- Protocol compliance ----------


class TestProtocolCompliance:
    """Verify ResearchMemory satisfies the ResearchMemoryLike protocol from prompt.py."""

    def test_satisfies_protocol(self, tmp_path: Path) -> None:
        from netlab.autoresearch.prompt import (
            DeadEndLike,
            InsightLike,
            ResearchMemoryLike,
        )

        log = _make_log(tmp_path)
        mem_dir = tmp_path / "memory"
        mem = ResearchMemory(mem_dir)

        # ResearchMemory satisfies ResearchMemoryLike
        assert isinstance(mem, ResearchMemoryLike)

        # Add an insight and check protocol
        ins = _make_insight()
        mem.add_insight(ins, log)
        assert isinstance(mem.active_insights[0], InsightLike)

        # Add a dead end and check protocol
        de = DeadEnd(
            id="de_001",
            exp_ids=["exp_001"],
            params_summary="x=1",
            failure_type="crash",
            reason="OOM",
            lesson="Avoid x=1",
        )
        mem.add_dead_end(de)
        assert isinstance(mem.dead_ends[0], DeadEndLike)

    def test_render_memory_section(self, tmp_path: Path) -> None:
        """render_memory_section works with a real ResearchMemory."""
        from netlab.autoresearch.prompt import render_memory_section

        log = _make_log(tmp_path)
        mem_dir = tmp_path / "memory"
        mem = ResearchMemory(mem_dir)

        # Empty memory => empty string
        assert render_memory_section(mem) == ""

        # Add data
        ins = _make_insight(claim="Higher x is better")
        mem.add_insight(ins, log)
        mem.update_strategy("Try x > 5.")

        result = render_memory_section(mem)
        assert "Higher x is better" in result
        assert "Try x > 5." in result
        assert "Research Notes" in result
