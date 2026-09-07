"""PostgreSQL persistence for completed chatbot interactions."""

from __future__ import annotations

import os
from uuid import UUID, uuid4

from psycopg_pool import AsyncConnectionPool


CREATE_INTERACTIONS_TABLE = """
CREATE TABLE IF NOT EXISTS chatbot_interactions (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    request_id UUID NOT NULL UNIQUE,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT chatbot_interactions_question_not_blank
        CHECK (length(btrim(question)) > 0),
    CONSTRAINT chatbot_interactions_answer_not_blank
        CHECK (length(btrim(answer)) > 0)
)
"""

CREATE_CREATED_AT_INDEX = """
CREATE INDEX IF NOT EXISTS chatbot_interactions_created_at_idx
ON chatbot_interactions (created_at DESC)
"""

INSERT_INTERACTION = """
INSERT INTO chatbot_interactions (request_id, question, answer)
VALUES (%s, %s, %s)
"""


def _positive_integer(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None or not raw_value.strip():
        return default

    try:
        value = int(raw_value)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be a positive integer") from exc

    if value <= 0:
        raise RuntimeError(f"{name} must be a positive integer")
    return value


def get_database_url() -> str:
    """Select a server-side PostgreSQL URL without exposing its value."""
    database_url = os.getenv("DATABASE_URL", "").strip()
    if database_url:
        return database_url

    # Render's private hostname works only from another Render service. Local
    # development therefore prefers the external URL, while Render prefers its
    # faster private network URL.
    running_on_render = os.getenv("RENDER", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }
    candidate_names = (
        ("INTERNAL_DB_URL", "EXTERNAL_DB_URL", "EXTERNAL_DB")
        if running_on_render
        else ("EXTERNAL_DB_URL", "EXTERNAL_DB", "INTERNAL_DB_URL")
    )
    for name in candidate_names:
        value = os.getenv(name, "").strip()
        if value:
            return value

    raise RuntimeError(
        "A PostgreSQL connection URL is required. Set DATABASE_URL, "
        "INTERNAL_DB_URL, or EXTERNAL_DB_URL."
    )


class ChatRepository:
    """Own the connection pool and persistence operations for chat records."""

    def __init__(self, database_url: str | None = None) -> None:
        min_size = _positive_integer("DB_POOL_MIN_SIZE", 1)
        max_size = _positive_integer("DB_POOL_MAX_SIZE", 5)
        if min_size > max_size:
            raise RuntimeError("DB_POOL_MIN_SIZE cannot exceed DB_POOL_MAX_SIZE")

        self._pool = AsyncConnectionPool(
            conninfo=database_url or get_database_url(),
            min_size=min_size,
            max_size=max_size,
            timeout=float(_positive_integer("DB_POOL_TIMEOUT_SECONDS", 10)),
            open=False,
        )

    async def start(self) -> None:
        """Open the pool and create the table before accepting traffic."""
        await self._pool.open()
        await self._pool.wait()
        async with self._pool.connection() as connection:
            await connection.execute(CREATE_INTERACTIONS_TABLE)
            await connection.execute(CREATE_CREATED_AT_INDEX)

    async def close(self) -> None:
        await self._pool.close()

    async def save_interaction(self, question: str, answer: str) -> UUID:
        """Atomically save one completed question/answer pair."""
        request_id = uuid4()
        async with self._pool.connection() as connection:
            await connection.execute(
                INSERT_INTERACTION,
                (request_id, question, answer),
            )
        return request_id

    async def check_connection(self) -> None:
        async with self._pool.connection() as connection:
            await connection.execute("SELECT 1")
