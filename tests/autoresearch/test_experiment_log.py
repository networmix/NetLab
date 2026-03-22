"""Tests for experiment_log module — covers every acceptance-criteria row."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import pytest

from netlab.autoresearch.experiment_log import (
    ExperimentLog,
    LogEntry,
)


def _make_entry(
    exp_id: str = "exp_001",
    params: dict | None = None,
    params_hash: str = "abc123",
    status: str = "success",
    metrics: dict | None = None,
    objective_score: float | None = 0.5,
    error_detail: str | None = None,
    execution_time_s: float | None = 1.0,
    seed: int = 42,
    timestamp: str | None = None,
) -> LogEntry:
    return LogEntry(
        exp_id=exp_id,
        params=params or {"x": 1},
        params_hash=params_hash,
        status=status,
        metrics=metrics or ({"m": 0.5} if status == "success" else None),
        objective_score=objective_score,
        error_detail=error_detail,
        execution_time_s=execution_time_s,
        seed=seed,
        timestamp=timestamp or datetime.now(timezone.utc).isoformat(),
    )


# ---------- Append + read ----------


class TestAppendAndRead:
    def test_append_and_read_five_entries(self, tmp_path: Path) -> None:
        """Append 5 entries, read — len == 5, field-by-field match."""
        log = ExperimentLog(tmp_path)
        written: list[LogEntry] = []
        for i in range(1, 6):
            entry = _make_entry(
                exp_id=f"exp_{i:03d}",
                params={"x": i},
                params_hash=f"hash_{i}",
                objective_score=i * 0.1,
                seed=i,
            )
            log.append(entry)
            written.append(entry)

        loaded = log.load()
        assert len(loaded) == 5

        for w, r in zip(written, loaded, strict=False):
            assert r.exp_id == w.exp_id
            assert r.params == w.params
            assert r.params_hash == w.params_hash
            assert r.status == w.status
            assert r.metrics == w.metrics
            assert r.objective_score == w.objective_score
            assert r.error_detail == w.error_detail
            assert r.execution_time_s == w.execution_time_s
            assert r.seed == w.seed
            assert r.timestamp == w.timestamp


# ---------- Atomic write ----------


class TestAtomicWrite:
    def test_no_tmp_file_persists(self, tmp_path: Path) -> None:
        """After append, no .tmp file remains."""
        log = ExperimentLog(tmp_path)
        log.append(_make_entry())

        files = list(tmp_path.iterdir())
        tmp_files = [f for f in files if f.suffix == ".tmp"]
        assert tmp_files == [], f"Temp files should not persist: {tmp_files}"
        assert (tmp_path / "experiment_log.jsonl").exists()


# ---------- Corrupt-tail recovery ----------


class TestCorruptTailRecovery:
    def test_corrupt_tail_discarded_with_warning(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Write 5 entries, append garbage, read => 5 entries + warning."""
        log = ExperimentLog(tmp_path)
        for i in range(1, 6):
            log.append(_make_entry(exp_id=f"exp_{i:03d}"))

        # Append garbage to the file
        log_path = tmp_path / "experiment_log.jsonl"
        with open(log_path, "ab") as f:
            f.write(b"this is 30 bytes of garbage!!!")

        with caplog.at_level(logging.WARNING):
            entries = log.load()

        assert len(entries) == 5
        assert any("corrupt" in r.message.lower() for r in caplog.records)


# ---------- Counter derivation ----------


class TestCounterDerivation:
    def test_counter_from_existing_dirs(self, tmp_path: Path) -> None:
        """Create exp_001..exp_010 dirs => next_experiment_id == 'exp_011'."""
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        for i in range(1, 11):
            (results_dir / f"exp_{i:03d}").mkdir()

        log = ExperimentLog(tmp_path)
        assert log.next_experiment_id() == "exp_011"

    def test_counter_from_empty(self, tmp_path: Path) -> None:
        """No dirs in results/ => next_experiment_id == 'exp_001'."""
        log = ExperimentLog(tmp_path)
        assert log.next_experiment_id() == "exp_001"

    def test_counter_ignores_non_exp_dirs(self, tmp_path: Path) -> None:
        """Non-matching directories are ignored."""
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        (results_dir / "exp_005").mkdir()
        (results_dir / "other_dir").mkdir()
        (results_dir / "exp_notanumber").mkdir()

        log = ExperimentLog(tmp_path)
        assert log.next_experiment_id() == "exp_006"


# ---------- Best derivation ----------


class TestBestDerivation:
    def test_best_entry_maximize(self, tmp_path: Path) -> None:
        """10 entries with known scores; best_entry picks score=0.9."""
        log = ExperimentLog(tmp_path, direction="maximize")
        scores = [0.1, 0.5, 0.9, 0.3, 0.2, 0.7, 0.4, 0.6, 0.8, 0.05]
        for i, score in enumerate(scores, start=1):
            log.append(
                _make_entry(
                    exp_id=f"exp_{i:03d}",
                    objective_score=score,
                    params_hash=f"h{i}",
                )
            )

        best = log.best_entry()
        assert best is not None
        assert best.exp_id == "exp_003"
        assert best.objective_score == 0.9

    def test_best_entry_minimize(self, tmp_path: Path) -> None:
        """direction='minimize', scores [0.9, 0.1, 0.5] => best is 0.1."""
        log = ExperimentLog(tmp_path, direction="minimize")
        scores = [0.9, 0.1, 0.5]
        for i, score in enumerate(scores, start=1):
            log.append(
                _make_entry(
                    exp_id=f"exp_{i:03d}",
                    objective_score=score,
                    params_hash=f"h{i}",
                )
            )

        best = log.best_entry()
        assert best is not None
        assert best.exp_id == "exp_002"
        assert best.objective_score == 0.1

    def test_best_entry_no_scoreable(self, tmp_path: Path) -> None:
        """All entries have None score => best_entry returns None."""
        log = ExperimentLog(tmp_path)
        log.append(_make_entry(status="crash", objective_score=None, metrics=None))
        assert log.best_entry() is None


# ---------- History windowing ----------


class TestHistoryWindowing:
    def test_windowed_history_25_entries(self, tmp_path: Path) -> None:
        """25 entries: result has last 10, top 5, summary with stats."""
        log = ExperimentLog(tmp_path)
        for i in range(1, 26):
            log.append(
                _make_entry(
                    exp_id=f"exp_{i:03d}",
                    objective_score=i * 0.04,  # 0.04..1.0
                    params={"x": i},
                    params_hash=f"h{i}",
                )
            )

        result = log.windowed_history(last_n=10, top_n=5)

        # Should contain last 10 entries (exp_016..exp_025)
        for i in range(16, 26):
            assert f"exp_{i:03d}" in result, f"Missing recent entry exp_{i:03d}"

        # Should contain top 5 by score (exp_025, exp_024, exp_023, exp_022, exp_021)
        for i in range(21, 26):
            assert f"exp_{i:03d}" in result, f"Missing top entry exp_{i:03d}"

        # Should contain summary stats
        assert "Total experiments: 25" in result
        assert "Min score:" in result
        assert "Max score:" in result
        assert "Mean score:" in result

    def test_char_budget(self, tmp_path: Path) -> None:
        """25 entries, budget=500 chars: fits in budget, has recent + best."""
        log = ExperimentLog(tmp_path)
        for i in range(1, 26):
            log.append(
                _make_entry(
                    exp_id=f"exp_{i:03d}",
                    objective_score=i * 0.04,
                    params={"x": i},
                    params_hash=f"h{i}",
                )
            )

        result = log.windowed_history(last_n=10, top_n=5, max_chars=500)
        assert len(result) <= 500

        # Must contain best entry (exp_025, score=1.0)
        assert "exp_025" in result

        # Must contain at least 3 recent entries
        recent_count = sum(1 for i in range(16, 26) if f"exp_{i:03d}" in result)
        # Or at least 3 entries from the tail
        tail_count = sum(1 for i in range(23, 26) if f"exp_{i:03d}" in result)
        assert recent_count >= 3 or tail_count >= 3, (
            f"Expected at least 3 recent entries, found {recent_count} recent, "
            f"{tail_count} from tail"
        )

    def test_empty_log(self, tmp_path: Path) -> None:
        """Empty log => descriptive message, no crash."""
        log = ExperimentLog(tmp_path)
        result = log.windowed_history()
        assert "No experiments" in result


# ---------- Config hash detection ----------


class TestConfigHash:
    def test_config_hash_stored_and_retrieved(self, tmp_path: Path) -> None:
        """Write config hash, retrieve it, compare to a different hash."""
        log = ExperimentLog(tmp_path)
        log.write_config_hash("abc123")

        stored = log.config_hash()
        assert stored == "abc123"

        # Simulate config change detection
        current_hash = "def456"
        config_changed = stored != current_hash
        assert config_changed is True

    def test_config_hash_none_when_empty(self, tmp_path: Path) -> None:
        """No entries => config_hash returns None."""
        log = ExperimentLog(tmp_path)
        assert log.config_hash() is None

    def test_config_hash_none_without_metadata(self, tmp_path: Path) -> None:
        """Entries exist but no metadata line => config_hash returns None."""
        log = ExperimentLog(tmp_path)
        log.append(_make_entry())
        assert log.config_hash() is None

    def test_config_hash_same_means_no_change(self, tmp_path: Path) -> None:
        """When stored hash matches current, no change detected."""
        log = ExperimentLog(tmp_path)
        log.write_config_hash("abc123")

        stored = log.config_hash()
        current_hash = "abc123"
        assert stored == current_hash


# ---------- Empty log ----------


class TestEmptyLog:
    def test_read_nonexistent_file(self, tmp_path: Path) -> None:
        """Read from nonexistent file => empty list, no error."""
        log = ExperimentLog(tmp_path)
        entries = log.load()
        assert len(entries) == 0
        assert entries == []


# ---------- Consecutive failures ----------


class TestConsecutiveFailures:
    def test_all_success(self, tmp_path: Path) -> None:
        log = ExperimentLog(tmp_path)
        for i in range(3):
            log.append(_make_entry(exp_id=f"exp_{i:03d}", status="success"))
        assert log.consecutive_failures() == 0

    def test_trailing_failures(self, tmp_path: Path) -> None:
        log = ExperimentLog(tmp_path)
        log.append(_make_entry(exp_id="exp_001", status="success"))
        log.append(
            _make_entry(
                exp_id="exp_002",
                status="parse_error",
                objective_score=None,
                metrics=None,
            )
        )
        log.append(
            _make_entry(
                exp_id="exp_003",
                status="crash",
                objective_score=None,
                metrics=None,
            )
        )
        assert log.consecutive_failures() == 2

    def test_all_failures(self, tmp_path: Path) -> None:
        log = ExperimentLog(tmp_path)
        for i in range(5):
            log.append(
                _make_entry(
                    exp_id=f"exp_{i:03d}",
                    status="parse_error",
                    objective_score=None,
                    metrics=None,
                )
            )
        assert log.consecutive_failures() == 5

    def test_empty_log(self, tmp_path: Path) -> None:
        log = ExperimentLog(tmp_path)
        assert log.consecutive_failures() == 0

    def test_failure_then_success_resets(self, tmp_path: Path) -> None:
        log = ExperimentLog(tmp_path)
        log.append(
            _make_entry(
                exp_id="exp_001",
                status="crash",
                objective_score=None,
                metrics=None,
            )
        )
        log.append(_make_entry(exp_id="exp_002", status="success"))
        assert log.consecutive_failures() == 0


# ---------- Direction validation ----------


class TestDirectionValidation:
    def test_invalid_direction_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="direction"):
            ExperimentLog(tmp_path, direction="invalid")


# ---------- Error detail in history ----------


class TestErrorDetailInHistory:
    def test_error_detail_appears_in_history(self, tmp_path: Path) -> None:
        """Error detail from entry appears in windowed history output."""
        log = ExperimentLog(tmp_path)
        log.append(
            _make_entry(
                exp_id="exp_001",
                status="crash",
                error_detail="ValueError: bad flow",
                objective_score=None,
                metrics=None,
            )
        )
        result = log.windowed_history()
        assert "ValueError: bad flow" in result


# ---------- Metadata line coexists with entries ----------


class TestMetadataCoexistence:
    def test_metadata_does_not_break_load(self, tmp_path: Path) -> None:
        """Metadata lines are silently skipped by load()."""
        log = ExperimentLog(tmp_path)
        log.write_config_hash("hash1")
        log.append(_make_entry(exp_id="exp_001"))
        log.append(_make_entry(exp_id="exp_002"))

        entries = log.load()
        assert len(entries) == 2
        assert entries[0].exp_id == "exp_001"
        assert entries[1].exp_id == "exp_002"
