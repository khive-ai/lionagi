import asyncio
import unittest
from typing import Any
from unittest.mock import AsyncMock, patch

from lionagi.ln import alcall


async def mock_func(x: int, add: int = 0) -> int:
    await asyncio.sleep(0.1)
    return x + add


async def mock_func_with_error(x: int) -> int:
    await asyncio.sleep(0.1)
    if x == 3:
        raise ValueError("mock error")
    return x


async def mock_handler(e: Exception) -> str:
    return f"handled: {str(e)}"


class TestLCallFunction(unittest.IsolatedAsyncioTestCase):
    async def test_lcall_basic(self):
        inputs = [1, 2, 3]
        results = await alcall(inputs, mock_func, add=1)
        self.assertEqual(results, [2, 3, 4])

    async def test_lcall_with_retries(self):
        inputs = [1, 2, 3]
        results = await alcall(
            inputs, mock_func_with_error, retry_attempts=1, retry_default=0
        )
        self.assertEqual(results, [1, 2, 0])

    async def test_lcall_with_timeout(self):
        inputs = [1, 2, 3]
        # With timeout of 0.05s and mock_func sleeping for 0.1s, all should timeout
        results = await alcall(
            inputs,
            mock_func,
            retry_timeout=0.05,
            retry_default="timeout",
            retry_attempts=0,
        )
        self.assertEqual(results, ["timeout", "timeout", "timeout"])

    async def test_lcall_with_max_concurrent(self):
        inputs = [1, 2, 3]
        results = await alcall(inputs, mock_func, max_concurrent=1)
        self.assertEqual(results, [1, 2, 3])

    async def test_lcall_with_throttle_period(self):
        inputs = [1, 2, 3]
        results = await alcall(inputs, mock_func, throttle_period=0.2)
        self.assertEqual(results, [1, 2, 3])

    # test_lcall_with_timing removed - retry_timing parameter no longer exists

    async def test_lcall_with_dropna(self):
        async def func(x: int) -> Any:
            return None if x == 2 else x

        inputs = [1, 2, 3]
        results = await alcall(inputs, func, output_dropna=True)
        self.assertEqual(results, [1, 3])

    async def test_lcall_with_backoff_factor(self):
        inputs = [1, 2, 3]
        with patch("anyio.sleep", new_callable=AsyncMock) as mock_sleep:
            await alcall(
                inputs,
                mock_func_with_error,
                retry_attempts=2,
                retry_initial_delay=0.1,
                retry_backoff=2,
                retry_default=0,
            )
            mock_sleep.assert_any_call(0.1)
            mock_sleep.assert_any_call(0.2)


if __name__ == "__main__":
    unittest.main()
