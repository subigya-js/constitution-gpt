"""Concurrency and deadline controls for blocking RAG work."""

from __future__ import annotations

import asyncio
import os
from functools import lru_cache
from typing import Any, Callable

from starlette.concurrency import run_in_threadpool


class CapacityExceededError(Exception):
    """Raised when no RAG execution slot becomes available in time."""


class RagExecutionTimeoutError(Exception):
    """Raised when a RAG request exceeds its response deadline."""


def positive_number(name: str, default: float) -> float:
    raw_value = os.getenv(name, str(default)).strip()
    try:
        value = float(raw_value)
    except ValueError as error:
        raise RuntimeError(f"{name} must be a positive number.") from error
    if value <= 0:
        raise RuntimeError(f"{name} must be a positive number.")
    return value


def positive_integer(name: str, default: int) -> int:
    value = positive_number(name, default)
    if not value.is_integer():
        raise RuntimeError(f"{name} must be a positive integer.")
    return int(value)


class RagExecutionLimiter:
    def __init__(self, capacity: int) -> None:
        self._semaphore = asyncio.Semaphore(capacity)

    async def run(self, function: Callable[..., Any], *args: Any) -> Any:
        queue_timeout = positive_number("RAG_QUEUE_TIMEOUT_SECONDS", 1)
        execution_timeout = positive_number("RAG_REQUEST_TIMEOUT_SECONDS", 90)

        try:
            await asyncio.wait_for(self._semaphore.acquire(), timeout=queue_timeout)
        except TimeoutError as error:
            raise CapacityExceededError from error

        task = asyncio.create_task(run_in_threadpool(function, *args))
        release_immediately = True
        try:
            return await asyncio.wait_for(
                asyncio.shield(task),
                timeout=execution_timeout,
            )
        except TimeoutError as error:
            # A Python worker thread cannot be forcefully cancelled. Keep its slot
            # occupied until it really exits, even though the client receives 504.
            release_immediately = False
            task.add_done_callback(self._release_finished_task)
            raise RagExecutionTimeoutError from error
        except asyncio.CancelledError:
            release_immediately = False
            task.add_done_callback(self._release_finished_task)
            raise
        finally:
            if release_immediately:
                self._semaphore.release()

    def _release_finished_task(self, task: asyncio.Task[Any]) -> None:
        self._semaphore.release()
        if not task.cancelled():
            task.exception()


@lru_cache(maxsize=1)
def get_rag_execution_limiter() -> RagExecutionLimiter:
    capacity = positive_integer("MAX_CONCURRENT_RAG_REQUESTS", 3)
    return RagExecutionLimiter(capacity)
