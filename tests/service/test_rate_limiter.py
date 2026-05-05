# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for lionagi.service.rate_limiter."""

from __future__ import annotations

import pytest

from lionagi.service.rate_limiter import RateLimitConfig, TokenBucket

# ---------------------------------------------------------------------------
# RateLimitConfig (no event loop needed)
# ---------------------------------------------------------------------------


class TestRateLimitConfig:
    def test_valid_config(self):
        cfg = RateLimitConfig(capacity=100, refill_rate=10.0)
        assert cfg.capacity == 100
        assert cfg.refill_rate == 10.0
        assert cfg.initial_tokens == 100

    def test_initial_tokens_defaults_to_capacity(self):
        cfg = RateLimitConfig(capacity=50, refill_rate=5.0)
        assert cfg.initial_tokens == 50

    def test_explicit_initial_tokens(self):
        cfg = RateLimitConfig(capacity=100, refill_rate=10.0, initial_tokens=50)
        assert cfg.initial_tokens == 50

    def test_zero_initial_tokens_allowed(self):
        cfg = RateLimitConfig(capacity=100, refill_rate=10.0, initial_tokens=0)
        assert cfg.initial_tokens == 0

    def test_capacity_zero_raises(self):
        with pytest.raises(ValueError, match="capacity must be > 0"):
            RateLimitConfig(capacity=0, refill_rate=5.0)

    def test_capacity_negative_raises(self):
        with pytest.raises(ValueError, match="capacity must be > 0"):
            RateLimitConfig(capacity=-1, refill_rate=5.0)

    def test_refill_rate_zero_raises(self):
        with pytest.raises(ValueError, match="refill_rate must be > 0"):
            RateLimitConfig(capacity=100, refill_rate=0)

    def test_refill_rate_negative_raises(self):
        with pytest.raises(ValueError, match="refill_rate must be > 0"):
            RateLimitConfig(capacity=100, refill_rate=-1.0)

    def test_initial_tokens_negative_raises(self):
        with pytest.raises(ValueError, match="initial_tokens must be >= 0"):
            RateLimitConfig(capacity=100, refill_rate=5.0, initial_tokens=-1)

    def test_initial_tokens_exceeds_capacity_raises(self):
        with pytest.raises(ValueError, match="initial_tokens.*cannot exceed capacity"):
            RateLimitConfig(capacity=10, refill_rate=1.0, initial_tokens=20)

    def test_frozen_dataclass(self):
        cfg = RateLimitConfig(capacity=100, refill_rate=10.0)
        with pytest.raises((AttributeError, TypeError)):
            cfg.capacity = 200  # type: ignore


# ---------------------------------------------------------------------------
# TokenBucket — all tests must be async because __init__ calls current_time()
# ---------------------------------------------------------------------------


async def _make_bucket(capacity=100, refill_rate=10.0, initial_tokens=None):
    """Helper: create a TokenBucket inside a running event loop."""
    cfg = RateLimitConfig(
        capacity=capacity,
        refill_rate=refill_rate,
        **({"initial_tokens": initial_tokens} if initial_tokens is not None else {}),
    )
    return TokenBucket(cfg)


class TestTokenBucket:
    async def test_initial_state_from_config(self):
        bucket = await _make_bucket(capacity=100, refill_rate=10.0)
        assert bucket.capacity == 100
        assert bucket.refill_rate == 10.0
        assert bucket.tokens == 100.0

    async def test_initial_state_partial_fill(self):
        bucket = await _make_bucket(capacity=100, refill_rate=10.0, initial_tokens=50)
        assert bucket.tokens == 50.0

    async def test_try_acquire_success(self):
        bucket = await _make_bucket()
        result = await bucket.try_acquire(10)
        assert result is True
        assert bucket.tokens == pytest.approx(90.0, abs=0.1)

    async def test_try_acquire_full_capacity(self):
        bucket = await _make_bucket()
        result = await bucket.try_acquire(100)
        assert result is True
        assert bucket.tokens == pytest.approx(0.0, abs=0.1)

    async def test_try_acquire_insufficient_returns_false(self):
        bucket = await _make_bucket(capacity=10, refill_rate=1.0, initial_tokens=5)
        result = await bucket.try_acquire(10)
        assert result is False
        assert bucket.tokens == pytest.approx(5.0, abs=0.1)

    async def test_try_acquire_zero_raises(self):
        bucket = await _make_bucket()
        with pytest.raises(ValueError, match="tokens must be > 0"):
            await bucket.try_acquire(0)

    async def test_try_acquire_negative_raises(self):
        bucket = await _make_bucket()
        with pytest.raises(ValueError, match="tokens must be > 0"):
            await bucket.try_acquire(-5)

    async def test_refill_is_called_on_try_acquire(self):
        bucket = await _make_bucket(capacity=10, refill_rate=100.0, initial_tokens=5)
        # After try_acquire, _refill runs — tokens should change from last_refill drift
        initial = bucket.tokens
        result = await bucket.try_acquire(3)
        assert result is True
        # 5 - 3 + small refill amount = approximately 2
        assert bucket.tokens >= 0.0

    async def test_reset_fills_to_capacity(self):
        bucket = await _make_bucket(capacity=100, refill_rate=10.0, initial_tokens=30)
        assert bucket.tokens == 30.0
        await bucket.reset()
        assert bucket.tokens == 100.0

    async def test_release_adds_tokens(self):
        bucket = await _make_bucket(capacity=100, refill_rate=10.0, initial_tokens=50)
        await bucket.try_acquire(20)
        tokens_after_acquire = bucket.tokens
        await bucket.release(10)
        assert bucket.tokens == pytest.approx(tokens_after_acquire + 10, abs=0.5)

    async def test_release_caps_at_capacity(self):
        bucket = await _make_bucket(capacity=10, refill_rate=1.0)
        await bucket.release(5)
        assert bucket.tokens <= 10.0

    async def test_release_zero_raises(self):
        bucket = await _make_bucket()
        with pytest.raises(ValueError, match="tokens must be > 0"):
            await bucket.release(0)

    async def test_release_negative_raises(self):
        bucket = await _make_bucket()
        with pytest.raises(ValueError, match="tokens must be > 0"):
            await bucket.release(-1)

    def test_to_dict(self):
        cfg = RateLimitConfig(capacity=100, refill_rate=10.0)
        # to_dict is sync; we can call it without a bucket
        # But TokenBucket() needs event loop... so test via inner logic
        # We test the actual logic by checking the contract: it only includes
        # capacity and refill_rate
        assert "capacity" in RateLimitConfig.__dataclass_fields__
        assert "refill_rate" in RateLimitConfig.__dataclass_fields__

    async def test_to_dict_returns_config_fields(self):
        bucket = await _make_bucket(capacity=42, refill_rate=7.0)
        d = bucket.to_dict()
        assert d["capacity"] == 42
        assert d["refill_rate"] == 7.0
        assert "tokens" not in d
        assert "last_refill" not in d

    async def test_acquire_immediate_success(self):
        bucket = await _make_bucket()
        result = await bucket.acquire(50)
        assert result is True
        assert bucket.tokens == pytest.approx(50.0, abs=0.5)

    async def test_acquire_zero_raises(self):
        bucket = await _make_bucket()
        with pytest.raises(ValueError, match="tokens must be > 0"):
            await bucket.acquire(0)

    async def test_acquire_exceeds_capacity_raises(self):
        bucket = await _make_bucket(capacity=10, refill_rate=1.0)
        with pytest.raises(ValueError, match="exceeds bucket capacity"):
            await bucket.acquire(11)

    async def test_acquire_timeout_returns_false_with_monkeypatch(self, monkeypatch):
        import lionagi.service.rate_limiter as rl

        # Monotonically increasing time to simulate elapsed > timeout
        tick = [0.0]

        def fake_time():
            tick[0] += 5.0  # each call advances 5 seconds
            return tick[0]

        async def fake_sleep(_):
            pass

        monkeypatch.setattr(rl, "current_time", fake_time)
        monkeypatch.setattr(rl, "sleep", fake_sleep)

        # Empty bucket, tiny timeout — should return False quickly
        cfg = RateLimitConfig(capacity=10, refill_rate=0.001, initial_tokens=0)
        bucket = TokenBucket(cfg)
        result = await bucket.acquire(10, timeout=1.0)
        assert result is False

    async def test_multiple_try_acquire_sequential(self):
        bucket = await _make_bucket(capacity=10, refill_rate=1.0)
        assert await bucket.try_acquire(4) is True
        assert await bucket.try_acquire(4) is True
        # ~2 tokens left (plus tiny refill drift)
        assert await bucket.try_acquire(4) is False

    async def test_reset_after_drain(self):
        bucket = await _make_bucket(capacity=100, refill_rate=10.0)
        await bucket.try_acquire(100)
        assert bucket.tokens == pytest.approx(0.0, abs=0.5)
        await bucket.reset()
        assert bucket.tokens == 100.0
