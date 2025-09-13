#!/usr/bin/env python3
"""Example demonstrating AsyncExecutor jitter and backoff retry features."""

import asyncio
import time

from lionagi.ln.concurrency.executor import AsyncExecutor


class TransientNetworkError(Exception):
    """Custom exception for network-related transient failures."""

    pass


class PermanentError(Exception):
    """Permanent error that should not trigger retries."""

    pass


async def example_jitter_and_backoff():
    """Demonstrate exponential backoff with jitter for resilient retries."""
    print("🔄 Testing exponential backoff with jitter...")

    attempts = []

    async def flaky_network_call(url_id):
        """Simulate a flaky network call that sometimes fails."""
        attempts.append((url_id, time.perf_counter()))

        # Fail first few attempts for demonstration
        url_attempts = [a for a in attempts if a[0] == url_id]
        if len(url_attempts) <= 2:
            raise TransientNetworkError(f"Network timeout for {url_id}")

        return f"Success: fetched data from {url_id}"

    async with AsyncExecutor(
        max_concurrent=2,
        retry_attempts=3,  # Up to 3 retries
        retry_delay=0.1,  # Start with 100ms
        retry_max_delay=1.0,  # Cap at 1 second
        retry_jitter=0.3,  # 30% random jitter
        retry_on=(TransientNetworkError,),  # Only retry network errors
    ) as executor:
        urls = ["api/users", "api/posts", "api/comments"]
        results = await executor(flaky_network_call, urls)

        print("✅ All requests succeeded after retries!")
        for result in results:
            print(f"  {result}")

    # Analyze retry timing
    print("\n📊 Retry timing analysis:")
    for url in urls:
        url_attempts = [
            (t, idx) for idx, (u, t) in enumerate(attempts) if u == url
        ]
        if len(url_attempts) > 1:
            delays = [
                url_attempts[i][0] - url_attempts[i - 1][0]
                for i in range(1, len(url_attempts))
            ]
            print(
                f"  {url}: {len(url_attempts)} attempts, delays: {[f'{d:.2f}s' for d in delays]}"
            )


async def example_selective_retry():
    """Demonstrate selective retry based on exception type."""
    print("\n🎯 Testing selective retry behavior...")

    permanent_attempts = 0
    transient_attempts = 0

    async def mixed_failures(error_type):
        """Function that can fail with different error types."""
        nonlocal permanent_attempts, transient_attempts

        if error_type == "permanent":
            permanent_attempts += 1
            raise PermanentError("Database schema error")
        else:
            transient_attempts += 1
            if transient_attempts < 3:  # Succeed on 3rd attempt
                raise TransientNetworkError("Connection reset")
            return f"Success after {transient_attempts} attempts"

    async with AsyncExecutor(
        retry_attempts=3,
        retry_delay=0.05,
        retry_jitter=0.0,  # No jitter for clearer demonstration
        retry_on=(TransientNetworkError,),  # Only retry network errors
    ) as executor:
        try:
            # This should fail immediately (no retries)
            await executor(mixed_failures, ["permanent"])
        except PermanentError:
            print(
                f"  ❌ Permanent error: {permanent_attempts} attempt (no retries)"
            )

        # Reset transient counter
        transient_attempts = 0

        # This should succeed after retries
        result = await executor(mixed_failures, ["transient"])
        print(
            f"  ✅ Transient error: {transient_attempts} attempts (with retries)"
        )
        print(f"     Result: {result[0]}")


async def example_custom_backoff():
    """Demonstrate custom backoff configuration."""
    print("\n⚙️  Testing custom backoff configuration...")

    timestamps = []

    async def timing_sensitive_task(task_id):
        """Task that records precise timing for backoff analysis."""
        timestamps.append(time.perf_counter())
        if len(timestamps) <= 3:  # Fail first 3 attempts
            raise TransientNetworkError(f"Retry #{len(timestamps)}")
        return f"Task {task_id} completed"

    # Configuration with aggressive backoff but jitter for distributed systems
    async with AsyncExecutor(
        retry_attempts=3,
        retry_delay=0.2,  # 200ms initial delay
        retry_max_delay=2.0,  # Cap at 2 seconds
        retry_jitter=0.5,  # 50% jitter to avoid thundering herd
        retry_on=(TransientNetworkError,),
    ) as executor:
        result = await executor(timing_sensitive_task, [1])
        print(f"  ✅ {result[0]}")

    # Calculate actual delays with jitter
    if len(timestamps) >= 4:
        delays = [
            timestamps[i] - timestamps[i - 1]
            for i in range(1, len(timestamps))
        ]
        expected = [0.2, 0.4, 0.8]  # Base exponential backoff
        print(f"  📈 Expected delays: {expected}")
        print(
            f"  🎲 Actual delays (with jitter): {[f'{d:.2f}s' for d in delays]}"
        )


async def main():
    """Run all examples."""
    print("🚀 AsyncExecutor Jitter & Backoff Examples\n")

    await example_jitter_and_backoff()
    await example_selective_retry()
    await example_custom_backoff()

    print("\n🎯 Key Features Demonstrated:")
    print("  • Exponential backoff with configurable base and max delays")
    print("  • Jitter to prevent thundering herd problems")
    print("  • Selective retry based on exception type")
    print("  • Resilient error handling for distributed systems")


if __name__ == "__main__":
    asyncio.run(main())
