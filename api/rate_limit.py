"""Distributed, fail-closed rate limiting for expensive API operations."""

from __future__ import annotations

import hashlib
import logging
import os
import secrets
from dataclasses import dataclass
from functools import lru_cache

from redis.asyncio import Redis
from redis.exceptions import RedisError


logger = logging.getLogger("constitution_gpt.rate_limit")

SLIDING_WINDOW_SCRIPT = """
local key = KEYS[1]
local limit = tonumber(ARGV[1])
local window_ms = tonumber(ARGV[2])
local member = ARGV[3]
local now = redis.call('TIME')
local now_ms = (now[1] * 1000) + math.floor(now[2] / 1000)
local window_start = now_ms - window_ms

redis.call('ZREMRANGEBYSCORE', key, 0, window_start)
local count = redis.call('ZCARD', key)

if count >= limit then
    local oldest = redis.call('ZRANGE', key, 0, 0, 'WITHSCORES')
    local retry_after_ms = window_ms
    if oldest[2] then
        retry_after_ms = math.max(1, tonumber(oldest[2]) + window_ms - now_ms)
    end
    redis.call('PEXPIRE', key, window_ms)
    return {0, 0, retry_after_ms}
end

redis.call('ZADD', key, now_ms, member)
redis.call('PEXPIRE', key, window_ms)
return {1, limit - count - 1, 0}
"""


@dataclass(frozen=True)
class RateLimitResult:
    allowed: bool
    remaining: int
    retry_after_seconds: int


def _positive_integer(name: str, default: int) -> int:
    raw_value = os.getenv(name, str(default)).strip()
    try:
        value = int(raw_value)
    except ValueError as error:
        raise RuntimeError(f"{name} must be a positive integer.") from error
    if value <= 0:
        raise RuntimeError(f"{name} must be a positive integer.")
    return value


@lru_cache(maxsize=1)
def get_redis_client() -> Redis:
    redis_url = os.getenv("REDIS_URL", "").strip()
    if not redis_url:
        raise RuntimeError("REDIS_URL is required for server-side rate limiting.")
    return Redis.from_url(
        redis_url,
        decode_responses=False,
        socket_connect_timeout=2,
        socket_timeout=2,
        health_check_interval=30,
    )


def client_rate_limit_key(client_host: str | None) -> str:
    identifier = client_host or "unknown"
    digest = hashlib.sha256(identifier.encode("utf-8")).hexdigest()
    return f"rate_limit:chat:{digest}"


async def check_chat_rate_limit(client_host: str | None) -> RateLimitResult:
    limit = _positive_integer("RATE_LIMIT_REQUESTS", 10)
    window_seconds = _positive_integer("RATE_LIMIT_WINDOW_SECONDS", 60)
    member = secrets.token_urlsafe(16)

    try:
        result = await get_redis_client().eval(
            SLIDING_WINDOW_SCRIPT,
            1,
            client_rate_limit_key(client_host),
            limit,
            window_seconds * 1000,
            member,
        )
    except (RedisError, OSError, TimeoutError) as error:
        logger.exception("Redis rate limiter is unavailable")
        raise RuntimeError("Rate limiter unavailable") from error

    allowed, remaining, retry_after_ms = (int(value) for value in result)
    retry_after_seconds = max(1, (retry_after_ms + 999) // 1000)
    return RateLimitResult(bool(allowed), remaining, retry_after_seconds)
