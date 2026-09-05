import os
import unittest
from unittest.mock import AsyncMock, patch

from api.rate_limit import (
    RateLimitResult,
    check_chat_rate_limit,
    client_rate_limit_key,
)


class RateLimitTests(unittest.IsolatedAsyncioTestCase):
    def test_client_key_does_not_store_the_raw_address(self):
        key = client_rate_limit_key("203.0.113.9")

        self.assertTrue(key.startswith("rate_limit:chat:"))
        self.assertNotIn("203.0.113.9", key)

    async def test_allowed_result_exposes_remaining_capacity(self):
        redis = AsyncMock()
        redis.eval.return_value = [1, 7, 0]

        with patch("api.rate_limit.get_redis_client", return_value=redis):
            result = await check_chat_rate_limit("203.0.113.9")

        self.assertEqual(result, RateLimitResult(True, 7, 1))

    async def test_rejected_result_rounds_retry_after_up(self):
        redis = AsyncMock()
        redis.eval.return_value = [0, 0, 1501]

        with patch("api.rate_limit.get_redis_client", return_value=redis):
            result = await check_chat_rate_limit("203.0.113.9")

        self.assertEqual(result, RateLimitResult(False, 0, 2))

    async def test_invalid_limit_configuration_fails_before_redis(self):
        with patch.dict(os.environ, {"RATE_LIMIT_REQUESTS": "0"}):
            with self.assertRaisesRegex(RuntimeError, "positive integer"):
                await check_chat_rate_limit("203.0.113.9")


if __name__ == "__main__":
    unittest.main()
