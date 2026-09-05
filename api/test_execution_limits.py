import asyncio
import threading
import unittest
from unittest.mock import patch

from api.execution_limits import (
    CapacityExceededError,
    RagExecutionLimiter,
    RagExecutionTimeoutError,
    positive_integer,
)


class ExecutionLimitTests(unittest.IsolatedAsyncioTestCase):
    def test_rejects_fractional_integer_configuration(self):
        with patch.dict("os.environ", {"TEST_CAPACITY": "1.5"}):
            with self.assertRaisesRegex(RuntimeError, "positive integer"):
                positive_integer("TEST_CAPACITY", 3)

    async def test_rejects_when_all_slots_are_occupied(self):
        blocker = threading.Event()
        limiter = RagExecutionLimiter(1)

        def blocked_work():
            blocker.wait()

        with patch.dict(
            "os.environ",
            {
                "RAG_QUEUE_TIMEOUT_SECONDS": "0.01",
                "RAG_REQUEST_TIMEOUT_SECONDS": "1",
            },
        ):
            first = asyncio.create_task(limiter.run(blocked_work))
            await asyncio.sleep(0.02)
            with self.assertRaises(CapacityExceededError):
                await limiter.run(lambda: None)
            blocker.set()
            await first

    async def test_reports_execution_timeout(self):
        limiter = RagExecutionLimiter(1)

        with patch.dict(
            "os.environ",
            {
                "RAG_QUEUE_TIMEOUT_SECONDS": "1",
                "RAG_REQUEST_TIMEOUT_SECONDS": "0.01",
            },
        ):
            with self.assertRaises(RagExecutionTimeoutError):
                await limiter.run(lambda: __import__("time").sleep(0.05))


if __name__ == "__main__":
    unittest.main()
