import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from api.main import QueryRequest, chat


class _SuccessfulLimiter:
    async def run(self, function, *args):
        return "Stored constitutional answer"


class ChatPersistenceTests(unittest.IsolatedAsyncioTestCase):
    async def test_successful_answer_is_saved_before_it_is_returned(self):
        repository = SimpleNamespace(save_interaction=AsyncMock())
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

        repository.save_interaction.assert_awaited_once_with(
            "How is the Prime Minister appointed?",
            "Stored constitutional answer",
        )
        self.assertEqual(response.answer, "Stored constitutional answer")


if __name__ == "__main__":
    unittest.main()
