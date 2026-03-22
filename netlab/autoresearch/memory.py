"""Research memory: persistent insights, dead ends, and strategy for autoresearch."""

from __future__ import annotations

import json
import logging
import re
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from netlab.autoresearch.experiment_log import ExperimentLog

logger = logging.getLogger(__name__)

MAX_ACTIVE_INSIGHTS = 20
MAX_DEAD_ENDS = 15
MAX_STRATEGY_LINES = 30
MIN_EVIDENCE_FOR = 2

VALID_CONFIDENCE_LEVELS = ("tentative", "moderate", "strong")
VALID_STATUSES = ("active", "retired", "contradicted")
VALID_FAILURE_TYPES = ("crash", "timeout", "infeasible", "poor_score", "other")


def _compute_confidence(n_evidence_for: int) -> str:
    """Derive confidence level from evidence count."""
    if n_evidence_for >= 5:
        return "strong"
    if n_evidence_for >= 4:
        return "moderate"
    return "tentative"


def _should_flag(evidence_for: list[str], evidence_against: list[str]) -> bool:
    """True when contradiction ratio >= 0.5."""
    if not evidence_for:
        return False
    return len(evidence_against) / len(evidence_for) >= 0.5


@dataclass
class Insight:
    id: str
    created_at_exp: int
    updated_at_exp: int
    claim: str
    evidence_for: list[str] = field(default_factory=list)
    evidence_against: list[str] = field(default_factory=list)
    confidence: str = "tentative"
    status: str = "active"
    flagged_for_revision: bool = False


@dataclass
class DeadEnd:
    id: str
    exp_ids: list[str] = field(default_factory=list)
    params_summary: str = ""
    failure_type: str = "other"
    reason: str = ""
    lesson: str = ""


def _insight_to_dict(ins: Insight) -> dict[str, Any]:
    return asdict(ins)


def _dict_to_insight(d: dict[str, Any]) -> Insight:
    return Insight(
        id=d["id"],
        created_at_exp=d["created_at_exp"],
        updated_at_exp=d["updated_at_exp"],
        claim=d["claim"],
        evidence_for=d.get("evidence_for", []),
        evidence_against=d.get("evidence_against", []),
        confidence=d.get("confidence", "tentative"),
        status=d.get("status", "active"),
        flagged_for_revision=d.get("flagged_for_revision", False),
    )


def _dead_end_to_dict(de: DeadEnd) -> dict[str, Any]:
    return asdict(de)


def _dict_to_dead_end(d: dict[str, Any]) -> DeadEnd:
    return DeadEnd(
        id=d["id"],
        exp_ids=d.get("exp_ids", []),
        params_summary=d.get("params_summary", ""),
        failure_type=d.get("failure_type", "other"),
        reason=d.get("reason", ""),
        lesson=d.get("lesson", ""),
    )


class ResearchMemory:
    """Persistent research memory backed by JSONL and markdown files.

    Storage layout inside memory_dir:
        insights.jsonl   -- one Insight per line
        dead_ends.jsonl  -- one DeadEnd per line
        strategy.md      -- plain text, <= 30 lines
    """

    def __init__(self, memory_dir: Path) -> None:
        self._dir = Path(memory_dir)
        self._insights: list[Insight] = []
        self._dead_ends: list[DeadEnd] = []
        self._strategy: str = ""

    # ---- Persistence ----

    def load(self) -> None:
        """Load all memory files from disk. Missing files are treated as empty."""
        self._dir.mkdir(parents=True, exist_ok=True)

        self._insights = self._load_jsonl(
            self._dir / "insights.jsonl", _dict_to_insight
        )
        self._dead_ends = self._load_jsonl(
            self._dir / "dead_ends.jsonl", _dict_to_dead_end
        )

        strategy_path = self._dir / "strategy.md"
        if strategy_path.exists():
            self._strategy = strategy_path.read_text(encoding="utf-8")
        else:
            self._strategy = ""

    def save(self) -> None:
        """Write all memory to disk."""
        self._dir.mkdir(parents=True, exist_ok=True)

        self._save_jsonl(self._dir / "insights.jsonl", self._insights, _insight_to_dict)
        self._save_jsonl(
            self._dir / "dead_ends.jsonl", self._dead_ends, _dead_end_to_dict
        )

        strategy_path = self._dir / "strategy.md"
        strategy_path.write_text(self._strategy, encoding="utf-8")

    @staticmethod
    def _load_jsonl(path: Path, converter) -> list:
        if not path.exists():
            return []
        items = []
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                items.append(converter(json.loads(stripped)))
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                logger.warning("Corrupt line %d in %s (skipped): %s", i + 1, path, exc)
        return items

    @staticmethod
    def _save_jsonl(path: Path, items: list, converter) -> None:
        lines = [json.dumps(converter(item), separators=(",", ":")) for item in items]
        path.write_text("\n".join(lines) + "\n" if lines else "", encoding="utf-8")

    # ---- Insights ----

    @property
    def active_insights(self) -> list[Insight]:
        return [ins for ins in self._insights if ins.status == "active"]

    def add_insight(self, insight: Insight, log: ExperimentLog) -> Optional[str]:
        """Add an insight. Returns error message string if rejected, None on success.

        Rejection reasons:
        - Fewer than 2 evidence_for entries
        - Active insight limit (20) exceeded
        - evidence_for or evidence_against contain exp IDs not in the log
        """
        if len(insight.evidence_for) < MIN_EVIDENCE_FOR:
            return (
                f"Insight rejected: minimum {MIN_EVIDENCE_FOR} evidence_for required, "
                f"got {len(insight.evidence_for)}"
            )

        if len(self.active_insights) >= MAX_ACTIVE_INSIGHTS:
            return f"Insight rejected: active insight limit of {MAX_ACTIVE_INSIGHTS} reached"

        # Validate experiment IDs
        entries = log.load()
        valid_ids = {e.exp_id for e in entries}
        all_cited = set(insight.evidence_for) | set(insight.evidence_against)
        invalid = all_cited - valid_ids
        if invalid:
            return (
                f"Insight rejected: experiment IDs not found in log: {sorted(invalid)}"
            )

        # Set confidence from evidence count
        insight.confidence = _compute_confidence(len(insight.evidence_for))
        insight.flagged_for_revision = _should_flag(
            insight.evidence_for, insight.evidence_against
        )

        self._insights.append(insight)
        return None

    def update_insight(self, insight_id: str, updates: dict) -> None:
        """Update fields on an existing insight.

        After update, recalculates confidence and flagged_for_revision.
        """
        for ins in self._insights:
            if ins.id == insight_id:
                for key, value in updates.items():
                    if hasattr(ins, key):
                        setattr(ins, key, value)
                # Recalculate derived fields
                ins.confidence = _compute_confidence(len(ins.evidence_for))
                ins.flagged_for_revision = _should_flag(
                    ins.evidence_for, ins.evidence_against
                )
                return
        logger.warning("Insight %s not found for update", insight_id)

    def retire_insight(self, insight_id: str) -> None:
        """Set an insight's status to 'retired'."""
        for ins in self._insights:
            if ins.id == insight_id:
                ins.status = "retired"
                return
        logger.warning("Insight %s not found for retirement", insight_id)

    # ---- Dead Ends ----

    @property
    def dead_ends(self) -> list[DeadEnd]:
        return list(self._dead_ends)

    def add_dead_end(self, dead_end: DeadEnd) -> None:
        """Add a dead end. Deduplicates by lesson; enforces sliding window of 15."""
        # Deduplicate: merge exp_ids into existing entry with same lesson
        for existing in self._dead_ends:
            if existing.lesson == dead_end.lesson:
                for eid in dead_end.exp_ids:
                    if eid not in existing.exp_ids:
                        existing.exp_ids.append(eid)
                return

        self._dead_ends.append(dead_end)

        # Enforce sliding window: drop oldest if over limit
        if len(self._dead_ends) > MAX_DEAD_ENDS:
            self._dead_ends = self._dead_ends[-MAX_DEAD_ENDS:]

    # ---- Strategy ----

    @property
    def strategy(self) -> str:
        return self._strategy

    def update_strategy(self, text: str) -> None:
        """Overwrite strategy. Truncates to 30 lines."""
        lines = text.splitlines()
        if len(lines) > MAX_STRATEGY_LINES:
            lines = lines[:MAX_STRATEGY_LINES]
        self._strategy = "\n".join(lines)

    # ---- Reflection parsing ----

    def parse_reflection_output(
        self, response: str, log: ExperimentLog
    ) -> Optional[str]:
        """Parse LLM reflection output, validate, and apply updates.

        Expected format:
            ```json
            {
              "insights": [...],
              "dead_ends": [...],
              "retire_insights": [...]
            }
            ```

            STRATEGY:
            Free-text strategy content...

        Returns error string on failure, None on success.
        """
        # Extract JSON block
        json_match = re.search(r"```json\s*\n(.*?)```", response, re.DOTALL)
        if not json_match:
            return "Reflection parse failed: no ```json``` block found"

        json_text = json_match.group(1).strip()
        try:
            data = json.loads(json_text)
        except json.JSONDecodeError as exc:
            return f"Reflection parse failed: invalid JSON: {exc}"

        if not isinstance(data, dict):
            return f"Reflection parse failed: expected JSON object, got {type(data).__name__}"

        errors: list[str] = []

        # Process retirements first
        for insight_id in data.get("retire_insights", []):
            self.retire_insight(insight_id)

        # Process insights
        exp_counter = max((ins.created_at_exp for ins in self._insights), default=0)

        for ins_data in data.get("insights", []):
            if not isinstance(ins_data, dict):
                errors.append(f"Skipped non-dict insight: {ins_data!r}")
                continue

            claim = ins_data.get("claim", "")
            evidence_for = ins_data.get("evidence_for", [])
            evidence_against = ins_data.get("evidence_against", [])

            if not claim:
                errors.append("Skipped insight with empty claim")
                continue

            exp_counter += 1
            insight = Insight(
                id=f"insight_{uuid.uuid4().hex[:8]}",
                created_at_exp=exp_counter,
                updated_at_exp=exp_counter,
                claim=claim,
                evidence_for=evidence_for,
                evidence_against=evidence_against,
            )

            err = self.add_insight(insight, log)
            if err:
                errors.append(err)

        # Process dead ends
        for de_data in data.get("dead_ends", []):
            if not isinstance(de_data, dict):
                errors.append(f"Skipped non-dict dead_end: {de_data!r}")
                continue

            # Extract exp_ids from the dead_end data, defaulting to empty
            exp_ids = de_data.get("exp_ids", [])

            dead_end = DeadEnd(
                id=f"deadend_{uuid.uuid4().hex[:8]}",
                exp_ids=exp_ids,
                params_summary=de_data.get("params_summary", ""),
                failure_type=de_data.get("failure_type", "other"),
                reason=de_data.get("reason", ""),
                lesson=de_data.get("lesson", ""),
            )
            self.add_dead_end(dead_end)

        # Extract strategy (everything after "STRATEGY:" outside JSON block)
        strategy_match = re.search(r"STRATEGY:\s*\n(.*)", response, re.DOTALL)
        if strategy_match:
            strategy_text = strategy_match.group(1).strip()
            self.update_strategy(strategy_text)

        if errors:
            return "; ".join(errors)
        return None
