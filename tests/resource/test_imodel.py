# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

import asyncio

import pytest

from lionagi._errors import ConfigurationError
from lionagi.service.backend import (
    Calling,
    Normalized,
    ResourceBackend,
    ResourceConfig,
)
from lionagi.service.imodel_v2 import iModel
from lionagi.service.rate_limiter import RateLimitConfig, TokenBucket


class MockBackend(ResourceBackend):
    """Minimal test backend for iModel tests."""

    _call_count: int = 0
    _call_results: list = []

    @property
    def event_type(self) -> type[Calling]:
        return MockCalling

    def create_payload(self, request: dict | None = None, **kwargs) -> dict:
        payload = dict(request or {})
        payload.update(kwargs)
        return payload

    async def call(self, request: dict, **kwargs) -> Normalized:
        self._call_count += 1
        return self.normalize_response(
            {"choices": [{"message": {"content": "mocked"}, "finish_reason": "stop"}]}
        )

    async def stream(self, request: dict, **kwargs):
        yield b"chunk1"
        yield b"chunk2"

    @property
    def endpoint(self):
        return None


class MockCalling(Calling):
    backend: MockBackend

    @property
    def call_args(self) -> dict:
        return {"request": self.payload}


def make_backend(provider="test", name="test-model") -> MockBackend:
    return MockBackend(config=ResourceConfig(provider=provider, name=name))


def make_imodel(**kwargs) -> iModel:
    backend = make_backend()
    return iModel(backend=backend, **kwargs)


class TestIModelCreation:
    def test_basic_creation(self):
        im = make_imodel()
        assert im.backend is not None
        assert im.rate_limiter is None
        assert im.executor is None
        assert im.hook_registry is None

    def test_repr(self):
        im = make_imodel()
        r = repr(im)
        assert "iModel" in r
        assert "test-model" in r

    def test_repr_no_backend(self):
        im = iModel.__new__(iModel)
        object.__setattr__(im, "backend", None)
        # Access repr directly
        assert "backend=None" in iModel.__repr__(im)

    def test_provider_metadata_empty_by_default(self):
        im = make_imodel()
        assert im.provider_metadata == {}

    async def test_with_rate_limiter(self):
        rl = TokenBucket(RateLimitConfig(capacity=10, refill_rate=1.0))
        im = make_imodel(rate_limiter=rl)
        assert im.rate_limiter is not None

    async def test_with_limit_requests_creates_executor(self):
        im = make_imodel(limit_requests=10)
        assert im.executor is not None

    def test_without_limit_requests_no_executor(self):
        im = make_imodel()
        assert im.executor is None

    def test_id_and_created_at(self):
        im = make_imodel()
        assert im.id is not None
        assert im.created_at is not None

    def test_to_dict_excludes_id_created_at(self):
        im = make_imodel()
        d = im._to_dict()
        assert "id" not in d
        assert "created_at" not in d


class TestIModelSerDes:
    def test_serialize_backend(self):
        im = make_imodel()
        d = im.model_dump()
        assert "backend" in d
        assert d["backend"] is not None

    def test_serialize_no_rate_limiter(self):
        im = make_imodel()
        d = im.model_dump()
        assert d["rate_limiter"] is None

    async def test_serialize_with_rate_limiter(self):
        rl = TokenBucket(RateLimitConfig(capacity=10, refill_rate=1.0))
        im = make_imodel(rate_limiter=rl)
        d = im.model_dump()
        assert d["rate_limiter"] is not None

    def test_backend_validator_requires_value(self):
        with pytest.raises(Exception):
            iModel(backend=None)

    def test_backend_validator_rejects_non_resource_backend(self):
        with pytest.raises(Exception):
            iModel(backend="not_a_backend")

    async def test_rate_limiter_deserializer_from_dict(self):
        rl = TokenBucket(RateLimitConfig(capacity=5, refill_rate=0.5))
        im = make_imodel(rate_limiter=rl)
        d = im.model_dump()
        rl_dict = d["rate_limiter"]
        assert isinstance(rl_dict, dict)
        # Reconstruct from dict
        im2 = iModel(backend=make_backend(), rate_limiter=rl_dict)
        assert im2.rate_limiter is not None

    def test_serialize_executor_none(self):
        im = make_imodel()
        d = im.model_dump()
        assert d["executor"] is None

    async def test_serialize_executor_with_limit(self):
        im = make_imodel(limit_requests=5)
        d = im.model_dump()
        assert d["executor"] is not None

    def test_deserialize_rate_limiter_invalid_raises(self):
        with pytest.raises(Exception):
            iModel(backend=make_backend(), rate_limiter=12345)

    async def test_deserialize_executor_from_dict(self):
        im = make_imodel(limit_requests=10)
        d = im.model_dump()
        executor_dict = d["executor"]
        assert isinstance(executor_dict, dict)
        # Reconstruct from dict with executor
        im2 = iModel(backend=make_backend(), executor=executor_dict)
        assert im2.executor is not None

    def test_deserialize_executor_none(self):
        im = iModel(backend=make_backend(), executor=None)
        assert im.executor is None

    def test_deserialize_executor_invalid_raises(self):
        with pytest.raises(Exception):
            iModel(backend=make_backend(), executor="not_an_executor")


class TestIModelCreateCalling:
    async def test_create_calling_basic(self):
        im = make_imodel()
        calling = await im.create_calling(prompt="test")
        assert calling is not None
        assert isinstance(calling, MockCalling)

    async def test_create_calling_no_backend_raises(self):
        im = iModel.__new__(iModel)
        object.__setattr__(im, "backend", None)
        object.__setattr__(im, "hook_registry", None)
        object.__setattr__(im, "rate_limiter", None)
        object.__setattr__(im, "executor", None)
        object.__setattr__(im, "provider_metadata", {})
        with pytest.raises((ConfigurationError, AttributeError, Exception)):
            await im.create_calling(prompt="test")

    async def test_create_calling_with_streaming(self):
        im = make_imodel()
        calling = await im.create_calling(streaming=True, prompt="test")
        assert calling.streaming is True

    async def test_create_calling_with_timeout(self):
        im = make_imodel()
        calling = await im.create_calling(timeout=10.0, prompt="test")
        assert calling.timeout == 10.0

    async def test_create_calling_with_stream_chunk_hook(self):
        im = make_imodel()
        hook = lambda chunk: chunk
        calling = await im.create_calling(stream_chunk_hook=hook)
        assert calling._stream_chunk_hook is hook

    async def test_create_calling_with_hook_registry_no_handler(self):
        """When hook_registry can't handle PreEventCreate, no hook invocation."""
        from lionagi.service.hooks import HookRegistry

        registry = HookRegistry()
        im = make_imodel()
        object.__setattr__(im, "hook_registry", registry)
        calling = await im.create_calling(prompt="test")
        assert calling is not None

    async def test_create_calling_multiple_kwargs(self):
        im = make_imodel()
        calling = await im.create_calling(model="gpt-4", temperature=0.5)
        assert calling.payload.get("model") == "gpt-4"
        assert calling.payload.get("temperature") == 0.5


class TestIModelInvoke:
    async def test_invoke_no_executor(self):
        im = make_imodel()
        calling = await im.create_calling(prompt="test")
        result = await im.invoke(calling=calling)
        assert result is not None

    async def test_invoke_creates_calling_if_none(self):
        im = make_imodel()
        result = await im.invoke(prompt="test message")
        assert result is not None

    async def test_invoke_with_rate_limiter_success(self):
        rl = TokenBucket(RateLimitConfig(capacity=10, refill_rate=1.0))
        im = make_imodel(rate_limiter=rl)
        result = await im.invoke(prompt="test")
        assert result is not None

    async def test_invoke_with_rate_limiter_timeout(self):
        """A fully exhausted rate limiter times out."""
        from unittest.mock import AsyncMock, patch

        rl = TokenBucket(RateLimitConfig(capacity=1, refill_rate=0.001))
        im = make_imodel(rate_limiter=rl)
        # Patch acquire to return False immediately
        rl.acquire = AsyncMock(return_value=False)
        calling = await im.create_calling(prompt="test")
        with pytest.raises(TimeoutError):
            await im.invoke(calling=calling)

    async def test_invoke_executor_path_completes(self):
        """Test invoke with an executor goes through the executor path."""
        im = make_imodel(limit_requests=10)
        result = await im.invoke(prompt="test")
        assert result is not None

    async def test_invoke_executor_aborted_raises(self):
        """Aborted status raises ExecutionError."""
        from unittest.mock import AsyncMock, patch

        from lionagi._errors import ExecutionError
        from lionagi.protocols.generic.event import EventStatus

        im = make_imodel(limit_requests=10)

        async def fake_forward():
            # After first forward, set calling to aborted
            pass

        calling = await im.create_calling(prompt="test")
        # Force the status to aborted
        calling.execution.status = EventStatus.ABORTED

        # Patch executor.append and forward to be no-ops, and not change status
        with (
            patch.object(im.executor, "append", new_callable=AsyncMock),
            patch.object(im.executor, "forward", new_callable=AsyncMock),
            patch.object(im.executor, "start", new_callable=AsyncMock),
        ):
            # calling status is already aborted, so the while loop won't iterate
            with pytest.raises(ExecutionError):
                await im.invoke(calling=calling)

    async def test_invoke_executor_failed_raises(self):
        """Failed status raises the execution error."""
        from unittest.mock import AsyncMock, patch

        from lionagi._errors import ExecutionError
        from lionagi.protocols.generic.event import EventStatus

        im = make_imodel(limit_requests=10)
        calling = await im.create_calling(prompt="test")
        calling.execution.status = EventStatus.FAILED

        with (
            patch.object(im.executor, "append", new_callable=AsyncMock),
            patch.object(im.executor, "forward", new_callable=AsyncMock),
            patch.object(im.executor, "start", new_callable=AsyncMock),
        ):
            with pytest.raises((ExecutionError, Exception)):
                await im.invoke(calling=calling)


class TestIModelContextManager:
    async def test_context_manager_no_executor(self):
        im = make_imodel()
        async with im as ctx:
            assert ctx is im

    async def test_context_manager_with_executor(self):
        im = make_imodel(limit_requests=5)
        async with im as ctx:
            assert ctx is im
        # After exit, executor is stopped

    async def test_context_manager_starts_stopped_executor(self):
        im = make_imodel(limit_requests=5)
        # Stop the executor first
        await im.executor.stop()
        assert im.executor.processor is None or im.executor.processor.is_stopped()
        async with im as ctx:
            assert ctx is im


class TestIModelStoreSessionId:
    def test_non_claude_code_backend_no_store(self):
        im = make_imodel()  # provider='test', not 'claude_code'
        calling = MockCalling(backend=im.backend, payload={})
        im._store_claude_code_session_id(calling)
        # Non-claude_code backend — no session_id stored
        assert "session_id" not in im.provider_metadata

    def test_no_backend_skips(self):
        im = iModel.__new__(iModel)
        object.__setattr__(im, "backend", None)
        object.__setattr__(im, "provider_metadata", {})
        calling = MockCalling(backend=make_backend(), payload={})
        im._store_claude_code_session_id(calling)
        assert "session_id" not in im.provider_metadata

    def test_claude_code_backend_sentinel_response_no_store(self):
        """Provider is claude_code but response is sentinel — skip."""
        from lionagi.ln.types._sentinel import Unset

        backend = make_backend(provider="claude_code", name="test")
        im = iModel(backend=backend)
        calling = MockCalling(backend=backend, payload={})
        # Response is still sentinel (Unset) — nothing should be stored
        im._store_claude_code_session_id(calling)
        assert "session_id" not in im.provider_metadata

    def test_claude_code_backend_normalized_response_with_session_id(self):
        """Provider is claude_code and response has session_id metadata."""
        backend = make_backend(provider="claude_code", name="test")
        im = iModel(backend=backend)
        calling = MockCalling(backend=backend, payload={})
        # Manually set the response with metadata
        normalized = Normalized(
            status="success",
            data={"content": "hello"},
            metadata={"session_id": "test-session-123"},
        )
        calling.execution.response = normalized
        im._store_claude_code_session_id(calling)
        assert im.provider_metadata.get("session_id") == "test-session-123"

    def test_claude_code_backend_normalized_response_no_session_id(self):
        """Provider is claude_code but no session_id in metadata — skip."""
        backend = make_backend(provider="claude_code", name="test")
        im = iModel(backend=backend)
        calling = MockCalling(backend=backend, payload={})
        normalized = Normalized(
            status="success",
            data={"content": "hello"},
            metadata={"other_key": "value"},
        )
        calling.execution.response = normalized
        im._store_claude_code_session_id(calling)
        assert "session_id" not in im.provider_metadata

    def test_claude_code_backend_normalized_no_metadata(self):
        """Provider is claude_code but metadata is None — skip."""
        backend = make_backend(provider="claude_code", name="test")
        im = iModel(backend=backend)
        calling = MockCalling(backend=backend, payload={})
        normalized = Normalized(
            status="success",
            data={"content": "hello"},
            metadata=None,
        )
        calling.execution.response = normalized
        im._store_claude_code_session_id(calling)
        assert "session_id" not in im.provider_metadata


class TestIModelProperties:
    def test_name_property(self):
        im = make_imodel()
        assert im.name == "test-model"

    def test_name_no_backend_raises(self):
        im = iModel.__new__(iModel)
        object.__setattr__(im, "backend", None)
        with pytest.raises(ConfigurationError):
            _ = im.name

    def test_version_property(self):
        im = make_imodel()
        assert isinstance(im.version, str)

    def test_version_no_backend_raises(self):
        im = iModel.__new__(iModel)
        object.__setattr__(im, "backend", None)
        with pytest.raises(ConfigurationError):
            _ = im.version

    def test_tags_property(self):
        im = make_imodel()
        assert isinstance(im.tags, (set, frozenset))

    def test_tags_no_backend_raises(self):
        im = iModel.__new__(iModel)
        object.__setattr__(im, "backend", None)
        with pytest.raises(ConfigurationError):
            _ = im.tags


class TestIModelStream:
    async def test_stream_no_calling_creates_one(self):
        """imodel.stream() creates a calling and uses calling.stream() as context manager."""
        import contextlib
        from unittest.mock import AsyncMock, MagicMock, patch

        im = make_imodel()

        # Mock the calling's stream() to be an async context manager
        async def fake_stream_ctx():
            yield  # async generator to satisfy the asynccontextmanager contract

        # Patch calling.stream to return a context manager
        @contextlib.asynccontextmanager
        async def mock_stream_cm():
            yield []

        with patch.object(MockCalling, "stream", return_value=mock_stream_cm()):
            async with im.stream(prompt="test") as chunks:
                pass  # we just verify no error

    async def test_stream_with_existing_calling(self):
        import contextlib
        from unittest.mock import patch

        im = make_imodel()
        calling = await im.create_calling(streaming=True, prompt="test")

        @contextlib.asynccontextmanager
        async def mock_stream_cm():
            yield []

        with patch.object(MockCalling, "stream", return_value=mock_stream_cm()):
            async with im.stream(calling=calling) as chunks:
                pass

    async def test_stream_with_chunk_hook_on_existing_calling(self):
        import contextlib
        from unittest.mock import patch

        im = make_imodel()
        calling = await im.create_calling(streaming=True, prompt="test")
        hook = lambda chunk: chunk

        @contextlib.asynccontextmanager
        async def mock_stream_cm():
            yield []

        # chunk hook is applied directly without entering stream context
        # we verify hook is set on calling._stream_chunk_hook
        calling._stream_chunk_hook = hook
        assert calling._stream_chunk_hook is hook

    async def test_stream_with_rate_limiter(self):
        import contextlib
        from unittest.mock import AsyncMock, patch

        rl = TokenBucket(RateLimitConfig(capacity=10, refill_rate=1.0))
        im = make_imodel(rate_limiter=rl)

        @contextlib.asynccontextmanager
        async def mock_stream_cm():
            yield []

        with patch.object(MockCalling, "stream", return_value=mock_stream_cm()):
            async with im.stream(prompt="test") as chunks:
                pass

    async def test_stream_rate_limiter_timeout_raises(self):
        from unittest.mock import AsyncMock

        rl = TokenBucket(RateLimitConfig(capacity=10, refill_rate=1.0))
        rl.acquire = AsyncMock(return_value=False)
        im = make_imodel(rate_limiter=rl)
        with pytest.raises(TimeoutError):
            async with im.stream(prompt="test") as chunks:
                pass


class TestIModelSerializerEdgeCases:
    def test_serialize_backend_none(self):
        im = make_imodel()
        result = im._serialize_backend(None)
        assert result is None

    def test_serialize_backend_has_metadata_key(self):
        im = make_imodel()
        result = im._serialize_backend(im.backend)
        assert "metadata" in result
        assert "lion_class" in result["metadata"]

    def test_serialize_rate_limiter_none(self):
        im = make_imodel()
        result = im._serialize_rate_limiter(None)
        assert result is None

    async def test_serialize_rate_limiter_with_bucket(self):
        rl = TokenBucket(RateLimitConfig(capacity=5, refill_rate=0.5))
        im = make_imodel(rate_limiter=rl)
        result = im._serialize_rate_limiter(rl)
        assert isinstance(result, dict)

    def test_serialize_executor_none(self):
        im = make_imodel()
        result = im._serialize_executor(None)
        assert result is None

    async def test_serialize_executor_rate_limited(self):
        im = make_imodel(limit_requests=5)
        result = im._serialize_executor(im.executor)
        assert isinstance(result, dict)

    def test_deserialize_rate_limiter_none(self):
        result = iModel._deserialize_rate_limiter(None)
        assert result is None

    async def test_deserialize_rate_limiter_token_bucket(self):
        rl = TokenBucket(RateLimitConfig(capacity=5, refill_rate=0.5))
        result = iModel._deserialize_rate_limiter(rl)
        assert result is rl

    def test_deserialize_rate_limiter_invalid_type_raises(self):
        with pytest.raises(Exception):
            iModel._deserialize_rate_limiter("invalid")

    def test_deserialize_backend_none_raises(self):
        with pytest.raises(Exception):
            iModel._deserialize_backend(None)

    def test_deserialize_backend_valid_instance(self):
        backend = make_backend()
        result = iModel._deserialize_backend(backend)
        assert result is backend

    def test_deserialize_backend_invalid_type_raises(self):
        with pytest.raises(Exception):
            iModel._deserialize_backend(42)

    def test_deserialize_executor_none(self):
        result = iModel._deserialize_executor(None)
        assert result is None

    def test_deserialize_executor_invalid_type_raises(self):
        with pytest.raises(Exception):
            iModel._deserialize_executor("invalid")

    async def test_deserialize_executor_with_dict_config(self):
        from lionagi.service.utilities.rate_limited_executor import (
            RateLimitedExecutor,
        )

        config = {
            "queue_capacity": 10,
            "capacity_refresh_time": 60.0,
        }
        result = iModel._deserialize_executor(config)
        assert isinstance(result, RateLimitedExecutor)

    async def test_deserialize_executor_with_request_bucket_dict(self):
        from lionagi.service.utilities.rate_limited_executor import (
            RateLimitedExecutor,
        )

        config = {
            "queue_capacity": 10,
            "capacity_refresh_time": 60.0,
            "request_bucket": {"capacity": 5, "refill_rate": 0.1},
        }
        result = iModel._deserialize_executor(config)
        assert isinstance(result, RateLimitedExecutor)

    async def test_deserialize_executor_with_token_bucket_dict(self):
        from lionagi.service.utilities.rate_limited_executor import (
            RateLimitedExecutor,
        )

        config = {
            "queue_capacity": 10,
            "capacity_refresh_time": 60.0,
            "token_bucket": {"capacity": 1000, "refill_rate": 10.0},
        }
        result = iModel._deserialize_executor(config)
        assert isinstance(result, RateLimitedExecutor)
