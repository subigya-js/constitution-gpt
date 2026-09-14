"""PostgreSQL persistence for completed chatbot interactions."""

from __future__ import annotations

import os
from typing import Any
from uuid import UUID, uuid4

from psycopg_pool import AsyncConnectionPool
from psycopg.types.json import Jsonb


CREATE_CONVERSATIONS_TABLE = """
CREATE TABLE IF NOT EXISTS chatbot_conversations (
    id UUID PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
)
"""


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

ADD_CONVERSATION_ID = """
ALTER TABLE chatbot_interactions
ADD COLUMN IF NOT EXISTS conversation_id UUID
REFERENCES chatbot_conversations(id) ON DELETE CASCADE
"""

ADD_RESOLVED_QUESTION = """
ALTER TABLE chatbot_interactions
ADD COLUMN IF NOT EXISTS resolved_question TEXT
"""

ADD_MODE = """
ALTER TABLE chatbot_interactions
ADD COLUMN IF NOT EXISTS mode TEXT NOT NULL DEFAULT 'constitutional'
"""

ADD_SOURCES = """
ALTER TABLE chatbot_interactions
ADD COLUMN IF NOT EXISTS sources JSONB NOT NULL DEFAULT '[]'::jsonb
"""

CREATE_CONVERSATION_INDEX = """
CREATE INDEX IF NOT EXISTS chatbot_interactions_conversation_idx
ON chatbot_interactions (conversation_id, created_at DESC)
"""

INSERT_INTERACTION = """
INSERT INTO chatbot_interactions (
    request_id,
    conversation_id,
    question,
    resolved_question,
    answer,
    mode,
    sources
)
VALUES (%s, %s, %s, %s, %s, %s, %s)
"""

UPSERT_CONVERSATION = """
INSERT INTO chatbot_conversations (id)
VALUES (%s)
ON CONFLICT (id) DO UPDATE SET updated_at = CURRENT_TIMESTAMP
"""

SELECT_RECENT_INTERACTIONS = """
SELECT question, answer
FROM chatbot_interactions
WHERE conversation_id = %s
ORDER BY created_at DESC, id DESC
LIMIT %s
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
            await connection.execute(CREATE_CONVERSATIONS_TABLE)
            await connection.execute(CREATE_INTERACTIONS_TABLE)
            await connection.execute(ADD_CONVERSATION_ID)
            await connection.execute(ADD_RESOLVED_QUESTION)
            await connection.execute(ADD_MODE)
            await connection.execute(ADD_SOURCES)
            await connection.execute(CREATE_CREATED_AT_INDEX)
            await connection.execute(CREATE_CONVERSATION_INDEX)

    async def close(self) -> None:
        await self._pool.close()

    async def ensure_conversation(self, conversation_id: UUID | None = None) -> UUID:
        """Create a conversation or refresh the timestamp of an existing one."""

        resolved_id = conversation_id or uuid4()
        async with self._pool.connection() as connection:
            await connection.execute(UPSERT_CONVERSATION, (resolved_id,))
        return resolved_id

    async def get_recent_messages(
        self,
        conversation_id: UUID,
        limit: int = 5,
    ) -> list[dict[str, str]]:
        """Return recent exchanges in chronological message order."""

        if limit <= 0 or limit > 20:
            raise ValueError("limit must be between 1 and 20")

        async with self._pool.connection() as connection:
            cursor = await connection.execute(
                SELECT_RECENT_INTERACTIONS,
                (conversation_id, limit),
            )
            rows = await cursor.fetchall()

        messages: list[dict[str, str]] = []
        for question, answer in reversed(rows):
            messages.append({"role": "user", "content": question})
            messages.append({"role": "assistant", "content": answer})
        return messages

    async def save_interaction(
        self,
        question: str,
        answer: str,
        *,
        conversation_id: UUID | None = None,
        resolved_question: str | None = None,
        mode: str = "constitutional",
        sources: list[dict[str, Any]] | None = None,
    ) -> UUID:
        """Atomically save one completed question/answer pair."""
        request_id = uuid4()
        resolved_conversation_id = conversation_id or uuid4()
        async with self._pool.connection() as connection:
            await connection.execute(
                UPSERT_CONVERSATION,
                (resolved_conversation_id,),
            )
            await connection.execute(
                INSERT_INTERACTION,
                (
                    request_id,
                    resolved_conversation_id,
                    question,
                    resolved_question or question,
                    answer,
                    mode,
                    Jsonb(sources or []),
                ),
            )
        return request_id

    async def check_connection(self) -> None:
        async with self._pool.connection() as connection:
            await connection.execute("SELECT 1")
