"""Validated environment configuration shared by RAG entry points."""

from __future__ import annotations

import os


def _positive_integer(name: str, default: int) -> int:
    raw_value = os.getenv(name, str(default)).strip()
    try:
        value = int(raw_value)
    except ValueError as error:
        raise RuntimeError(f"{name} must be a positive integer.") from error
    if value <= 0:
        raise RuntimeError(f"{name} must be a positive integer.")
    return value


def _nonnegative_integer(name: str, default: int) -> int:
    raw_value = os.getenv(name, str(default)).strip()
    try:
        value = int(raw_value)
    except ValueError as error:
        raise RuntimeError(f"{name} must be a nonnegative integer.") from error
    if value < 0:
        raise RuntimeError(f"{name} must be a nonnegative integer.")
    return value


def openai_client_options() -> dict[str, int]:
    return {
        "timeout": _positive_integer("OPENAI_REQUEST_TIMEOUT_SECONDS", 45),
        "max_retries": _nonnegative_integer("OPENAI_MAX_RETRIES", 2),
    }
