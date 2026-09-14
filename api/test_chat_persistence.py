import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import UUID

from api.main import QueryRequest, chat
from rag.research_assistant import AssistantResult, ResearchSource


class _SuccessfulLimiter:
    async def run(self, function, *args):
        return AssistantResult(
            answer="Stored constitutional answer",
            mode="legal_research",
            resolved_question="How is the Prime Minister appointed in Nepal?",
            sources=[
                ResearchSource(title="Official source", url="https://example.gov.np")
            ],
        )


class ChatPersistenceTests(unittest.IsolatedAsyncioTestCase):
    async def test_successful_answer_is_saved_before_it_is_returned(self):
        conversation_id = UUID("f8d53b9b-dbae-4477-b7a4-bf7c30c2b411")
        repository = SimpleNamespace(
            ensure_conversation=AsyncMock(return_value=conversation_id),
            get_recent_messages=AsyncMock(return_value=[]),
            save_interaction=AsyncMock(),
        )
        http_request = SimpleNamespace(
            app=SimpleNamespace(
                state=SimpleNamespace(chat_repository=repository),
            ),
        )

        with patch(
            "api.main.get_rag_execution_limiter",
            return_value=_SuccessfulLimiter(),
        ):
            response = await chat(
                QueryRequest(question="How is the Prime Minister appointed?"),
                http_request,
            )

        repository.ensure_conversation.assert_awaited_once_with(None)
        repository.get_recent_messages.assert_awaited_once_with(conversation_id)
        repository.save_interaction.assert_awaited_once_with(
            "How is the Prime Minister appointed?",
            "Stored constitutional answer",
            conversation_id=conversation_id,
            resolved_question="How is the Prime Minister appointed in Nepal?",
            mode="legal_research",
            sources=[
                {
                    "title": "Official source",
                    "url": "https://example.gov.np",
                    "source_type": "web",
                }
            ],
        )
        self.assertEqual(response.answer, "Stored constitutional answer")
        self.assertEqual(response.conversation_id, conversation_id)
        self.assertEqual(response.sources[0].title, "Official source")


if __name__ == "__main__":
    unittest.main()
