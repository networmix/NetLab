"""LLM backend abstractions for autoresearch.

Provides:
- LLMBackend: ABC for single-shot text generation
- MockBackend: scripted responses for testing
- ClaudeCLIBackend: subprocess wrapper around `claude --print`
- OpenAICompatibleBackend: HTTP client for OpenAI-compatible APIs
"""

from __future__ import annotations

import json
import subprocess
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from http.client import HTTPException


class LLMBackend(ABC):
    """Abstract base class for LLM backends."""

    @abstractmethod
    def generate(self, prompt: str, system: str = "") -> str:
        """Single-shot text generation. Returns response text. Raises on failure."""


class MockBackend(LLMBackend):
    """Backend that returns scripted responses in order. For testing."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self._index = 0
        self.calls: list[tuple[str, str]] = []

    def generate(self, prompt: str, system: str = "") -> str:
        self.calls.append((prompt, system))
        if self._index >= len(self._responses):
            raise StopIteration("MockBackend: no more scripted responses")
        response = self._responses[self._index]
        self._index += 1
        return response


class ClaudeCLIBackend(LLMBackend):
    """Backend that calls the `claude` CLI tool via subprocess."""

    def __init__(self, model: str = "opus") -> None:
        self.model = model

    def generate(self, prompt: str, system: str = "") -> str:
        cmd = ["claude", "--print", "-p", prompt]
        if system:
            cmd.extend(["--system-prompt", system])
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"claude CLI failed (exit {result.returncode}): {result.stderr}"
            )
        return result.stdout


class OpenAICompatibleBackend(LLMBackend):
    """Backend that calls an OpenAI-compatible HTTP API.

    Uses urllib.request (no third-party deps). Implements retry with
    exponential backoff for 429 and 5xx errors.
    """

    # Retry parameters — overridable in tests to keep wall time short.
    max_retries: int = 3
    base_delay: float = 2.0  # seconds

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "",
        timeout: int = 120,
    ) -> None:
        # Strip trailing slash for consistent URL construction
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout

    def generate(self, prompt: str, system: str = "") -> str:
        url = f"{self.base_url}/v1/chat/completions"
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        body = json.dumps({"model": self.model, "messages": messages}).encode()

        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        last_exc: BaseException | None = None
        for attempt in range(self.max_retries + 1):
            req = urllib.request.Request(url, data=body, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    data = json.loads(resp.read().decode())
                    return data["choices"][0]["message"]["content"]
            except urllib.error.HTTPError as exc:
                status = exc.code
                if status == 429:
                    # Use Retry-After header when present
                    retry_after = (
                        exc.headers.get("Retry-After") if exc.headers else None
                    )
                    if retry_after is not None:
                        delay = float(retry_after)
                    else:
                        delay = self.base_delay * (2**attempt)
                    last_exc = exc
                    if attempt < self.max_retries:
                        time.sleep(delay)
                        continue
                    raise RuntimeError(
                        f"HTTP {status} after {self.max_retries + 1} attempts"
                    ) from exc
                elif status >= 500:
                    delay = self.base_delay * (2**attempt)
                    last_exc = exc
                    if attempt < self.max_retries:
                        time.sleep(delay)
                        continue
                    raise RuntimeError(
                        f"HTTP {status} after {self.max_retries + 1} attempts"
                    ) from exc
                else:
                    # 4xx (except 429): fail immediately
                    raise RuntimeError(f"HTTP {status}: {exc.reason}") from exc
            except urllib.error.URLError as exc:
                # Connection refused, DNS failure, etc.
                last_exc = exc
                if attempt < self.max_retries:
                    delay = self.base_delay * (2**attempt)
                    time.sleep(delay)
                    continue
                raise RuntimeError(
                    f"connection error after {self.max_retries + 1} attempts: {exc.reason}"
                ) from exc
            except (TimeoutError, OSError, HTTPException) as exc:
                # Socket timeout or low-level OS error
                last_exc = exc
                msg = str(exc).lower()
                if "timed out" in msg or isinstance(exc, TimeoutError):
                    raise RuntimeError(f"timeout after {self.timeout}s: {exc}") from exc
                # Other OS errors: retry
                if attempt < self.max_retries:
                    delay = self.base_delay * (2**attempt)
                    time.sleep(delay)
                    continue
                raise RuntimeError(
                    f"connection error after {self.max_retries + 1} attempts: {exc}"
                ) from exc

        # Should not reach here, but just in case
        raise RuntimeError(
            f"Unexpected retry exhaustion: {last_exc}"
        )  # pragma: no cover
