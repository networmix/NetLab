"""Experiment log: JSONL-backed append-only log for autoresearch experiments."""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

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
        "timeout_partial",
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


def _entry_to_dict(entry: LogEntry) -> dict[str, Any]:
    return asdict(entry)


def _dict_to_entry(d: dict[str, Any]) -> LogEntry:
    return LogEntry(
        exp_id=d["exp_id"],
        params=d["params"],
        params_hash=d["params_hash"],
        status=d["status"],
        metrics=d.get("metrics"),
        objective_score=d.get("objective_score"),
        error_detail=d.get("error_detail"),
        execution_time_s=d.get("execution_time_s"),
        seed=d["seed"],
        timestamp=d["timestamp"],
    )


class ExperimentLog:
    """JSONL-backed experiment log with atomic appends and corrupt-tail recovery."""

    def __init__(self, project_dir: Path, direction: str = "maximize") -> None:
        self.project_dir = Path(project_dir)
        if direction not in ("maximize", "minimize"):
            raise ValueError(
                f"direction must be 'maximize' or 'minimize', got {direction!r}"
            )
        self.direction = direction
        self._log_path = self.project_dir / "experiment_log.jsonl"
        self._results_dir = self.project_dir / "results"

    def append(self, entry: LogEntry) -> None:
        """Atomic append: write to .tmp file, then rename over the original.

        We read existing content, append the new line, write to tmp, rename.
        This ensures the file is never left in a partial-write state.
        """
        existing = b""
        if self._log_path.exists():
            existing = self._log_path.read_bytes()

        line = json.dumps(_entry_to_dict(entry), separators=(",", ":")) + "\n"
        new_content = existing + line.encode("utf-8")

        self._log_path.parent.mkdir(parents=True, exist_ok=True)

        # Write to temp file in the same directory (same filesystem), then rename.
        fd, tmp_path = tempfile.mkstemp(
            dir=self._log_path.parent, prefix=".experiment_log_", suffix=".tmp"
        )
        try:
            os.write(fd, new_content)
            os.fsync(fd)
            os.close(fd)
            os.rename(tmp_path, self._log_path)
        except BaseException:
            os.close(fd) if not _fd_closed(fd) else None
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise

    def load(self) -> list[LogEntry]:
        """Read all entries. Discard corrupt trailing line with a warning.

        Metadata lines (those with a ``_type`` field) are silently skipped.
        """
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
                # Skip metadata lines (e.g. config_hash records)
                if "_type" in d:
                    continue
                entries.append(_dict_to_entry(d))
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
        """Derive from max(existing exp_NNN dirs in results/) + 1."""
        if not self._results_dir.exists():
            return "exp_001"

        pattern = re.compile(r"^exp_(\d+)$")
        max_num = 0
        for entry in self._results_dir.iterdir():
            if entry.is_dir():
                m = pattern.match(entry.name)
                if m:
                    max_num = max(max_num, int(m.group(1)))

        return f"exp_{max_num + 1:03d}"

    def best_entry(self) -> Optional[LogEntry]:
        """Re-derive from all entries. None if no scoreable entries."""
        entries = self.load()
        scoreable = [e for e in entries if e.objective_score is not None]
        if not scoreable:
            return None

        if self.direction == "maximize":
            return max(scoreable, key=lambda e: e.objective_score)  # type: ignore[arg-type]
        else:
            return min(scoreable, key=lambda e: e.objective_score)  # type: ignore[arg-type]

    def windowed_history(
        self, last_n: int = 10, top_n: int = 5, max_chars: int = 16000
    ) -> str:
        """Render history section for prompt. Truncates to char budget."""
        entries = self.load()
        if not entries:
            return "No experiments run yet."

        # Summary stats from all scoreable entries
        scoreable = [e for e in entries if e.objective_score is not None]
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

        # Top N by score
        top_entries = sorted(
            scoreable,
            key=lambda e: e.objective_score,  # type: ignore[arg-type]
            reverse=(self.direction == "maximize"),
        )[:top_n]

        # Last N entries
        recent_entries = entries[-last_n:]

        # Build sections
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

        # Truncate to budget if needed
        if len(result) <= max_chars:
            return result

        # Progressive truncation: reduce recent entries until we fit
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
        meta = {"_type": "metadata", "config_hash": config_hash}
        line = json.dumps(meta, separators=(",", ":")) + "\n"

        existing = b""
        if self._log_path.exists():
            existing = self._log_path.read_bytes()

        new_content = existing + line.encode("utf-8")
        self._log_path.parent.mkdir(parents=True, exist_ok=True)

        fd, tmp_path = tempfile.mkstemp(
            dir=self._log_path.parent, prefix=".experiment_log_", suffix=".tmp"
        )
        try:
            os.write(fd, new_content)
            os.fsync(fd)
            os.close(fd)
            os.rename(tmp_path, self._log_path)
        except BaseException:
            os.close(fd) if not _fd_closed(fd) else None
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise

    def consecutive_failures(self) -> int:
        """Count of consecutive non-success entries from the tail."""
        entries = self.load()
        count = 0
        for entry in reversed(entries):
            if entry.status != "success":
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
        # Truncate long error details
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
    """Build a truncated history that fits within max_chars.

    Strategy: progressively reduce top entries and recent entries counts
    until the result fits. Always tries to keep the best entry and at
    least 3 recent entries.
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


def _fd_closed(fd: int) -> bool:
    """Check if a file descriptor is already closed."""
    try:
        os.fstat(fd)
        return False
    except OSError:
        return True
