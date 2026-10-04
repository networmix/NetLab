"""Local task queue: Python callables, isolated workers, enforceable deadlines."""

from __future__ import annotations

import math
import multiprocessing
from concurrent.futures import Future
from types import TracebackType
from typing import Any, Callable, TypeVar, cast

from pebble import ProcessPool

R = TypeVar("R")


class TaskQueue:
    """Own a bounded process pool; failures propagate through standard futures.

    Use as a context manager. Functions and arguments must be picklable. A timeout
    starts when the worker starts a task, not while it waits in the queue.
    On exceptional exit, cancel queued/running work and join all workers.
    """

    def __init__(self, workers: int = 1) -> None:
        if type(workers) is not int or workers < 1:
            raise ValueError("workers must be positive")
        self._pool = ProcessPool(
            max_workers=workers, context=cast(Any, multiprocessing.get_context("spawn"))
        )

    def __enter__(self) -> TaskQueue:
        return self

    def submit(
        self,
        function: Callable[..., R],
        *args: Any,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> Future[R]:
        if timeout is not None and (not math.isfinite(timeout) or timeout <= 0):
            raise ValueError("timeout must be positive or None")
        return cast(
            Future[R],
            self._pool.schedule(function, args=args, kwargs=kwargs, timeout=timeout),
        )

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if exc_type is None:
            self._pool.close()
        else:
            self._pool.stop()
        self._pool.join()
