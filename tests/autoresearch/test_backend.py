"""Tests for netlab.autoresearch.backend — LLM backend abstractions."""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest.mock import patch

import pytest

from netlab.autoresearch.backend import (
    ClaudeCLIBackend,
    CodexCLIBackend,
    LLMBackend,
    MockBackend,
    OpenAICompatibleBackend,
)

# ---------------------------------------------------------------------------
# Helpers for mock HTTP server
# ---------------------------------------------------------------------------


def _make_handler(responses: list[tuple[int, dict | str, dict[str, str] | None]]):
    """Create a handler class that returns responses in sequence.

    Each item: (status_code, body_dict_or_str, extra_headers_or_None).
    """
    call_count = {"n": 0}
    received_requests: list[dict] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length)
            received_requests.append(
                {
                    "path": self.path,
                    "body": json.loads(raw) if raw else {},
                    "headers": dict(self.headers),
                }
            )

            idx = min(call_count["n"], len(responses) - 1)
            call_count["n"] += 1
            status, body, extra_headers = responses[idx]

            if isinstance(body, dict):
                body_bytes = json.dumps(body).encode()
            else:
                body_bytes = body.encode()

            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            if extra_headers:
                for k, v in extra_headers.items():
                    self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body_bytes)

        def log_message(self, format, *args):
            pass  # suppress stderr output during tests

    return Handler, call_count, received_requests


def _make_delayed_handler(delay_seconds: float):
    """Create a handler that sleeps before responding."""

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            time.sleep(delay_seconds)
            body = json.dumps(
                {"choices": [{"message": {"content": "delayed"}}]}
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):
            pass

    return Handler


def _start_server(handler_class) -> tuple[HTTPServer, int, threading.Thread]:
    server = HTTPServer(("127.0.0.1", 0), handler_class)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, port, thread


# ---------------------------------------------------------------------------
# MockBackend tests
# ---------------------------------------------------------------------------


class TestMockBackend:
    def test_returns_scripted_responses(self):
        """Mock returns scripted: 3 scripted responses returned in order."""
        backend = MockBackend(["alpha", "beta", "gamma"])
        assert backend.generate("p1") == "alpha"
        assert backend.generate("p2", system="s2") == "beta"
        assert backend.generate("p3") == "gamma"

    def test_records_calls(self):
        """Mock records (prompt, system) tuples."""
        backend = MockBackend(["a", "b"])
        backend.generate("hello", system="sys")
        backend.generate("world")
        assert backend.calls == [("hello", "sys"), ("world", "")]

    def test_exhausted_raises_stopiteration(self):
        """Mock exhausted: call beyond scripted list raises StopIteration."""
        backend = MockBackend(["only_one"])
        backend.generate("p1")
        with pytest.raises(StopIteration):
            backend.generate("p2")

    def test_is_llm_backend_subclass(self):
        """MockBackend is a proper LLMBackend subclass."""
        assert issubclass(MockBackend, LLMBackend)
        assert isinstance(MockBackend([]), LLMBackend)


# ---------------------------------------------------------------------------
# ClaudeCLIBackend tests
# ---------------------------------------------------------------------------


class TestClaudeCLIBackend:
    def test_subprocess_args_format(self):
        """Claude CLI format: subprocess args contain expected flags and values."""
        backend = ClaudeCLIBackend(model="opus")

        with patch("netlab.autoresearch.backend.subprocess.run") as mock_run:
            mock_run.return_value.returncode = 0
            mock_run.return_value.stdout = "response text"
            mock_run.return_value.stderr = ""

            result = backend.generate("my prompt", system="my system instructions")

        assert result == "response text"
        args = mock_run.call_args[0][0]
        assert "claude" in args
        assert "--print" in args
        assert "-p" in args
        # Prompt text follows -p
        p_idx = args.index("-p")
        assert args[p_idx + 1] == "my prompt"
        # System prompt
        assert "--system-prompt" in args
        sp_idx = args.index("--system-prompt")
        assert args[sp_idx + 1] == "my system instructions"

    def test_subprocess_no_system(self):
        """Claude CLI omits --system-prompt when system is empty."""
        backend = ClaudeCLIBackend()

        with patch("netlab.autoresearch.backend.subprocess.run") as mock_run:
            mock_run.return_value.returncode = 0
            mock_run.return_value.stdout = "ok"
            mock_run.return_value.stderr = ""

            backend.generate("prompt only")

        args = mock_run.call_args[0][0]
        assert "--system-prompt" not in args

    def test_subprocess_failure_raises(self):
        """Claude CLI raises RuntimeError on non-zero exit code."""
        backend = ClaudeCLIBackend()

        with patch("netlab.autoresearch.backend.subprocess.run") as mock_run:
            mock_run.return_value.returncode = 1
            mock_run.return_value.stdout = ""
            mock_run.return_value.stderr = "some error"

            with pytest.raises(RuntimeError, match="claude CLI failed"):
                backend.generate("bad prompt")

    def test_is_llm_backend_subclass(self):
        assert issubclass(ClaudeCLIBackend, LLMBackend)


# ---------------------------------------------------------------------------
# CodexCLIBackend tests
# ---------------------------------------------------------------------------


class TestCodexCLIBackend:
    def test_subprocess_args_format(self, tmp_path):
        """Codex CLI format: subprocess args contain expected flags and values."""
        backend = CodexCLIBackend(model="o4-mini")

        with (
            patch("netlab.autoresearch.backend.subprocess.run") as mock_run,
            patch(
                "netlab.autoresearch.backend.Path.read_text",
                return_value="response text",
            ),
            patch("netlab.autoresearch.backend.Path.unlink"),
        ):
            mock_run.return_value.returncode = 0
            mock_run.return_value.stdout = ""
            mock_run.return_value.stderr = ""

            result = backend.generate("my prompt", system="my system instructions")

        assert result == "response text"
        args = mock_run.call_args[0][0]
        assert "codex" in args
        assert "exec" in args
        assert "--ephemeral" in args
        assert "--sandbox" in args
        assert "read-only" in args
        assert "--skip-git-repo-check" in args
        assert "-m" in args
        m_idx = args.index("-m")
        assert args[m_idx + 1] == "o4-mini"
        assert "-o" in args
        # Prompt should include system + prompt
        prompt_arg = args[-1]
        assert "my system instructions" in prompt_arg
        assert "my prompt" in prompt_arg

    def test_subprocess_no_system(self):
        """Codex CLI: prompt without system instructions does not prepend system text."""
        backend = CodexCLIBackend()

        with (
            patch("netlab.autoresearch.backend.subprocess.run") as mock_run,
            patch("netlab.autoresearch.backend.Path.read_text", return_value="ok"),
            patch("netlab.autoresearch.backend.Path.unlink"),
        ):
            mock_run.return_value.returncode = 0
            mock_run.return_value.stdout = ""
            mock_run.return_value.stderr = ""

            backend.generate("prompt only")

        args = mock_run.call_args[0][0]
        prompt_arg = args[-1]
        assert prompt_arg == "prompt only"

    def test_subprocess_failure_raises(self):
        """Codex CLI raises RuntimeError on non-zero exit code."""
        backend = CodexCLIBackend()

        with (
            patch("netlab.autoresearch.backend.subprocess.run") as mock_run,
            patch("netlab.autoresearch.backend.Path.unlink"),
        ):
            mock_run.return_value.returncode = 1
            mock_run.return_value.stdout = ""
            mock_run.return_value.stderr = "some error"

            with pytest.raises(RuntimeError, match="codex CLI failed"):
                backend.generate("bad prompt")

    def test_is_llm_backend_subclass(self):
        assert issubclass(CodexCLIBackend, LLMBackend)

    def test_default_model(self):
        """Default model is empty (uses codex default)."""
        backend = CodexCLIBackend()
        assert backend.model == ""


# ---------------------------------------------------------------------------
# OpenAICompatibleBackend tests
# ---------------------------------------------------------------------------


class TestOpenAICompatibleBackend:
    def test_request_format(self):
        """OpenAI request format: POST to /v1/chat/completions with correct body."""
        handler_cls, call_count, received = _make_handler(
            [
                (200, {"choices": [{"message": {"content": "hi"}}]}, None),
            ]
        )
        server, port, _ = _start_server(handler_cls)
        try:
            backend = OpenAICompatibleBackend(
                base_url=f"http://127.0.0.1:{port}",
                model="test-model",
                api_key="sk-test",
            )
            result = backend.generate("hello", system="be helpful")

            assert result == "hi"
            assert len(received) == 1
            req = received[0]
            assert req["path"] == "/v1/chat/completions"
            body = req["body"]
            assert body["model"] == "test-model"
            messages = body["messages"]
            assert messages[0] == {"role": "system", "content": "be helpful"}
            assert messages[1] == {"role": "user", "content": "hello"}
        finally:
            server.shutdown()

    def test_parses_response(self):
        """OpenAI parses response: extracts content from choices."""
        handler_cls, _, _ = _make_handler(
            [
                (200, {"choices": [{"message": {"content": "hello"}}]}, None),
            ]
        )
        server, port, _ = _start_server(handler_cls)
        try:
            backend = OpenAICompatibleBackend(
                base_url=f"http://127.0.0.1:{port}",
                model="gpt-4",
            )
            assert backend.generate("test") == "hello"
        finally:
            server.shutdown()

    def test_retry_on_429(self):
        """Retry on 429: retries with Retry-After header, then succeeds."""
        handler_cls, call_count, _ = _make_handler(
            [
                (429, {"error": "rate limited"}, {"Retry-After": "0.1"}),
                (200, {"choices": [{"message": {"content": "ok"}}]}, None),
            ]
        )
        server, port, _ = _start_server(handler_cls)
        try:
            backend = OpenAICompatibleBackend(
                base_url=f"http://127.0.0.1:{port}",
                model="m",
            )
            backend.base_delay = 0.1  # fast retries for test
            result = backend.generate("test")
            assert result == "ok"
            assert call_count["n"] == 2
        finally:
            server.shutdown()

    def test_retry_on_500(self):
        """Retry on 500: retries once, then succeeds."""
        handler_cls, call_count, _ = _make_handler(
            [
                (500, {"error": "internal"}, None),
                (200, {"choices": [{"message": {"content": "recovered"}}]}, None),
            ]
        )
        server, port, _ = _start_server(handler_cls)
        try:
            backend = OpenAICompatibleBackend(
                base_url=f"http://127.0.0.1:{port}",
                model="m",
            )
            backend.base_delay = 0.1
            result = backend.generate("test")
            assert result == "recovered"
            assert call_count["n"] == 2
        finally:
            server.shutdown()

    def test_no_retry_on_400(self):
        """No retry on 400: raises immediately with HTTP status in message."""
        handler_cls, call_count, _ = _make_handler(
            [
                (400, {"error": "bad request"}, None),
            ]
        )
        server, port, _ = _start_server(handler_cls)
        try:
            backend = OpenAICompatibleBackend(
                base_url=f"http://127.0.0.1:{port}",
                model="m",
            )
            backend.base_delay = 0.1
            with pytest.raises(RuntimeError, match="400"):
                backend.generate("test")
            # Only 1 request made (no retry)
            assert call_count["n"] == 1
        finally:
            server.shutdown()

    def test_connection_refused(self):
        """Connection refused: retries 3 times with backoff, total wall time >= 3s."""
        backend = OpenAICompatibleBackend(
            base_url="http://127.0.0.1:1",  # nothing listening
            model="m",
        )
        # base_delay=1.0: delays are 1.0, 2.0, 4.0 = 7.0 total
        # But we need total >= 3s and to stay under the 30s pytest timeout.
        # Use base_delay=1.0 with max_retries=3 (4 attempts).
        # Delays: 1.0 + 2.0 = 3.0 at minimum (3 sleeps before 4th attempt fails).
        # Actually with max_retries=3: attempts 0,1,2,3. Sleeps after 0,1,2 = 1+2+4=7s. Too long.
        # Use base_delay=0.5: 0.5 + 1.0 + 2.0 = 3.5s >= 3s, under 30s timeout.
        backend.base_delay = 0.5
        backend.max_retries = 3

        start = time.monotonic()
        with pytest.raises(RuntimeError, match="connection"):
            backend.generate("test")
        elapsed = time.monotonic() - start

        assert elapsed >= 3.0, f"Expected >= 3s of backoff, got {elapsed:.1f}s"

    def test_timeout(self):
        """Timeout: mock server delays 10s, timeout=2s, raises within 5s."""
        handler_cls = _make_delayed_handler(10.0)
        server, port, _ = _start_server(handler_cls)
        try:
            backend = OpenAICompatibleBackend(
                base_url=f"http://127.0.0.1:{port}",
                model="m",
                timeout=2,
            )

            start = time.monotonic()
            with pytest.raises(RuntimeError, match="(?i)timeout"):
                backend.generate("test")
            elapsed = time.monotonic() - start

            assert elapsed < 5.0, f"Expected < 5s, got {elapsed:.1f}s"
        finally:
            server.shutdown()

    def test_no_system_message_when_empty(self):
        """When system is empty, messages list has only user role."""
        handler_cls, _, received = _make_handler(
            [
                (200, {"choices": [{"message": {"content": "ok"}}]}, None),
            ]
        )
        server, port, _ = _start_server(handler_cls)
        try:
            backend = OpenAICompatibleBackend(
                base_url=f"http://127.0.0.1:{port}",
                model="m",
            )
            backend.generate("hello")
            messages = received[0]["body"]["messages"]
            assert len(messages) == 1
            assert messages[0]["role"] == "user"
        finally:
            server.shutdown()

    def test_is_llm_backend_subclass(self):
        assert issubclass(OpenAICompatibleBackend, LLMBackend)
