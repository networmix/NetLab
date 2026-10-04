"""Store experiment records in JSONL and format history for research prompts."""

from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

from netlab.artifacts import write_text_atomic

logger = logging.getLogger(__name__)

VALID_STATUSES = frozenset(
    {
        "success",
        "cached",
        "parse_error",
        "invalid_hypothesis",
        "generation_error",
        "crash",
        "timeout_no_result",
        "validation_error",
        "backend_error",
        "infeasible",
        "circuit_breaker",
    }
)


@dataclass
class LogEntry:
    exp_id: str  # "exp_001"
    params: dict[str, Any]
    params_hash: str
    status: str  # one of VALID_STATUSES
    metrics: Optional[dict[str, float]]  # None if status != "success"
    objective_score: Optional[float]
    error_detail: Optional[str]
    execution_time_s: Optional[float]
    seed: int
    timestamp: str  # ISO 8601


class ExperimentLog:
    """Store experiment records with atomic writes; warn and skip invalid lines."""

    def __init__(self, project_dir: Path) -> None:
        self.project_dir = Path(project_dir)
        self._log_path = self.project_dir / "experiment_log.jsonl"
        self._results_dir = self.project_dir / "results"

    def append(self, entry: LogEntry) -> None:
        """Append an entry by writing a temporary file and replacing the log."""
        if entry.status not in VALID_STATUSES:
            raise ValueError(f"Unknown experiment status: {entry.status}")
        self._append_record(asdict(entry))

    def _append_record(self, record: dict) -> None:
        lines = (
            self._log_path.read_text().splitlines() if self._log_path.exists() else []
        )
        if lines:
            try:
                json.loads(lines[-1])
            except json.JSONDecodeError:
                lines.pop()
        lines.append(json.dumps(record, separators=(",", ":")))
        write_text_atomic(self._log_path, "\n".join(lines) + "\n")

    def load(self) -> list[LogEntry]:
        """Read entries, skipping metadata and warning about invalid JSON lines."""
        if not self._log_path.exists():
            return []

        text = self._log_path.read_text(encoding="utf-8")
        lines = text.splitlines()
        entries: list[LogEntry] = []

        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                d = json.loads(stripped)
                if "_type" in d:
                    continue
                entries.append(LogEntry(**d))
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                if i == len(lines) - 1:
                    logger.warning(
                        "Discarding corrupt trailing line in experiment log: %s", exc
                    )
                else:
                    # Non-trailing corrupt line: still warn but discard
                    logger.warning(
                        "Corrupt line %d in experiment log (discarded): %s", i + 1, exc
                    )

        return entries

    def next_experiment_id(self) -> str:
        """Allocate monotonically across logs and interrupted experiment directories."""
        names = [entry.exp_id for entry in self.load()]
        if self._results_dir.exists():
            names.extend(p.name for p in self._results_dir.iterdir() if p.is_dir())
        pattern = re.compile(r"^exp_(\d+)$")
        numbers = [
            int(match.group(1)) for name in names if (match := pattern.fullmatch(name))
        ]
        return f"exp_{max(numbers, default=0) + 1:03d}"

    def best_entry(self) -> Optional[LogEntry]:
        """Return the highest finite score among success/cached entries, or None."""
        entries = self.load()
        scoreable = [
            e
            for e in entries
            if e.status in {"success", "cached"}
            and e.objective_score is not None
            and math.isfinite(e.objective_score)
        ]
        if not scoreable:
            return None

        return max(scoreable, key=lambda e: e.objective_score)  # type: ignore[arg-type]

    def windowed_history(
        self, last_n: int = 10, top_n: int = 5, max_chars: int = 16000
    ) -> str:
        """Summarize the best and recent experiments within a character budget."""
        entries = self.load()
        if not entries:
            return "No experiments run yet."

        scoreable = [
            e
            for e in entries
            if e.status in {"success", "cached"}
            and e.objective_score is not None
            and math.isfinite(e.objective_score)
        ]
        summary_parts = [f"Total experiments: {len(entries)}"]
        if scoreable:
            scores: list[float] = [
                e.objective_score for e in scoreable if e.objective_score is not None
            ]
            summary_parts.append(f"Scoreable: {len(scoreable)}")
            summary_parts.append(f"Min score: {min(scores):.4f}")
            summary_parts.append(f"Max score: {max(scores):.4f}")
            summary_parts.append(f"Mean score: {sum(scores) / len(scores):.4f}")

        summary_line = "Summary: " + ", ".join(summary_parts)

        top_entries = sorted(
            scoreable,
            key=lambda e: e.objective_score,  # type: ignore[arg-type]
            reverse=True,
        )[:top_n]

        recent_entries = entries[-last_n:]

        sections: list[str] = []
        sections.append(summary_line)
        sections.append("")

        if top_entries:
            sections.append(f"Top {min(top_n, len(top_entries))} experiments by score:")
            for e in top_entries:
                sections.append(_format_entry_brief(e))
            sections.append("")

        sections.append(f"Last {min(last_n, len(recent_entries))} experiments:")
        for e in recent_entries:
            sections.append(_format_entry_brief(e))

        result = "\n".join(sections)

        if len(result) <= max_chars:
            return result

        return _truncated_history(summary_line, top_entries, entries, last_n, max_chars)

    def config_hash(self) -> Optional[str]:
        """Hash stored in log metadata. None if no metadata line found."""
        if not self._log_path.exists():
            return None
        text = self._log_path.read_text(encoding="utf-8")
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            try:
                d = json.loads(stripped)
                if d.get("_type") == "metadata" and "config_hash" in d:
                    return d["config_hash"]
            except (json.JSONDecodeError, KeyError):
                continue
        return None

    def write_config_hash(self, config_hash: str) -> None:
        """Write a metadata line with the config hash."""
        self._append_record({"_type": "metadata", "config_hash": config_hash})

    def consecutive_failures(self) -> int:
        """Count of consecutive non-success entries from the tail."""
        entries = self.load()
        count = 0
        for entry in reversed(entries):
            if entry.status not in {"success", "cached"}:
                count += 1
            else:
                break
        return count


def _format_entry_brief(entry: LogEntry) -> str:
    """Format a log entry as a single concise line."""
    parts = [f"  {entry.exp_id}: status={entry.status}"]
    if entry.objective_score is not None:
        parts.append(f"score={entry.objective_score:.4f}")
    if entry.params:
        params_str = ", ".join(f"{k}={v}" for k, v in sorted(entry.params.items()))
        parts.append("params={" + params_str + "}")
    if entry.error_detail:
        detail = entry.error_detail
        if len(detail) > 100:
            detail = detail[:97] + "..."
        parts.append(f"error={detail!r}")
    if entry.execution_time_s is not None:
        parts.append(f"time={entry.execution_time_s:.1f}s")
    return ", ".join(parts)


def _truncated_history(
    summary_line: str,
    top_entries: list[LogEntry],
    all_entries: list[LogEntry],
    last_n: int,
    max_chars: int,
) -> str:
    """Reduce top and recent entries to fit the character budget.

    Keep the summary and best entry when the budget allows.
    """
    max_recent = min(last_n, len(all_entries))
    max_top = len(top_entries)

    # Try reducing top count first, then recent count
    for t in range(max_top, -1, -1):
        for n in range(max_recent, 0, -1):
            sections: list[str] = [summary_line, ""]
            current_top = top_entries[:t]

            if current_top:
                sections.append(f"Top {len(current_top)} experiments by score:")
                for e in current_top:
                    sections.append(_format_entry_brief(e))
                sections.append("")

            recent = all_entries[-n:]
            sections.append(f"Last {len(recent)} experiments:")
            for e in recent:
                sections.append(_format_entry_brief(e))

            result = "\n".join(sections)
            if len(result) <= max_chars:
                return result

    # Absolute minimum: summary + best only
    sections = [summary_line]
    if top_entries:
        sections.append("")
        sections.append("Best: " + _format_entry_brief(top_entries[0]))
    return "\n".join(sections)[:max_chars]
