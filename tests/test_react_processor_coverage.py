"""Coverage tests for uncovered lines in:
- lionagi/operations/ReAct/ReAct.py
- lionagi/protocols/generic/processor.py (Executor class + streaming path)
"""

from __future__ import annotations

import asyncio
from typing import ClassVar
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from lionagi.operations.ReAct.ReAct import ReAct_v1, handle_field_models
from lionagi.operations.ReAct.utils import Analysis, ReActAnalysis
from lionagi.protocols.generic.event import Event, EventStatus
from lionagi.protocols.generic.processor import Executor, Processor

# ---------------------------------------------------------------------------
# Minimal concrete helpers
# ---------------------------------------------------------------------------


class _OkEvent(Event):
    async def _invoke(self):
        return "ok"


class _StreamEvent(Event):
    streaming: bool = True

    async def _stream(self):
        for chunk in ["a", "b", "c"]:
            yield chunk


class _FailEvent(Event):
    async def _invoke(self):
        raise ValueError("intentional failure")


class _Proc(Processor):
    event_type: ClassVar[type[Event]] = _OkEvent


def _proc(**kw) -> _Proc:
    defaults = dict(queue_capacity=10, capacity_refresh_time=0.01, concurrency_limit=2)
    defaults.update(kw)
    return _Proc(**defaults)


class _MyExecutor(Executor):
    processor_type: ClassVar[type[Processor]] = _Proc


# ---------------------------------------------------------------------------
# Helpers for building mocked Branch for ReAct tests
# ---------------------------------------------------------------------------


def _make_branch_mock():
    """Return a MagicMock branch satisfying ReAct's interface."""
    from uuid import uuid4

    branch = MagicMock()
    branch.user = "tester"
    branch.id = uuid4()
    branch.chat_model = MagicMock()
    # msgs.last_response.response is used in the except clause of ReActStream
    branch.msgs = MagicMock()
    branch.msgs.last_response = MagicMock()
    branch.msgs.last_response.response = "fallback_response"
    return branch


def _make_react_analysis(extension_needed: bool = False) -> ReActAnalysis:
    return ReActAnalysis(
        analysis="test reasoning",
        planned_actions=[],
        extension_needed=extension_needed,
    )


def _make_analysis(answer: str = "final answer") -> Analysis:
    return Analysis(answer=answer)


# ===========================================================================
# Processor – streaming path
# ===========================================================================


class TestProcessorStreamingPath:
    @pytest.mark.asyncio
    async def test_process_streaming_event_completes(self):
        """Streaming path (event.streaming=True) in process() should be hit."""

        class _StreamProc(Processor):
            event_type: ClassVar[type[Event]] = _StreamEvent

        proc = _StreamProc(
            queue_capacity=5, capacity_refresh_time=0.01, concurrency_limit=1
        )
        event = _StreamEvent()
        await proc.enqueue(event)
        await asyncio.wait_for(proc.process(), timeout=3.0)
        assert event.status == EventStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_process_streaming_with_semaphore(self):
        """Streaming + semaphore path."""

        class _StreamProc2(Processor):
            event_type: ClassVar[type[Event]] = _StreamEvent

        proc = _StreamProc2(
            queue_capacity=5, capacity_refresh_time=0.01, concurrency_limit=2
        )
        events = [_StreamEvent() for _ in range(2)]
        for e in events:
            await proc.enqueue(e)
        await asyncio.wait_for(proc.process(), timeout=3.0)
        for e in events:
            assert e.status == EventStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_process_streaming_no_semaphore(self):
        """Streaming with concurrency_limit=0 (no semaphore)."""

        class _StreamProc3(Processor):
            event_type: ClassVar[type[Event]] = _StreamEvent

        proc = _StreamProc3(
            queue_capacity=5, capacity_refresh_time=0.01, concurrency_limit=0
        )
        event = _StreamEvent()
        await proc.enqueue(event)
        await asyncio.wait_for(proc.process(), timeout=3.0)
        assert event.status == EventStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_process_non_streaming_no_semaphore(self):
        """Non-streaming with concurrency_limit=0 hits _invoke_safe without semaphore."""
        proc = _Proc(
            queue_capacity=5, capacity_refresh_time=0.01, concurrency_limit=0
        )
        event = _OkEvent()
        await proc.enqueue(event)
        await asyncio.wait_for(proc.process(), timeout=3.0)
        assert event.status == EventStatus.COMPLETED


# ===========================================================================
# Executor class
# ===========================================================================


class TestExecutorInit:
    def test_default_init(self):
        ex = _MyExecutor()
        assert ex.processor is None
        assert ex.processor_config == {}
        assert len(ex.pending) == 0
        assert ex.pile is not None

    def test_with_config(self):
        cfg = dict(queue_capacity=5, capacity_refresh_time=0.05, concurrency_limit=1)
        ex = _MyExecutor(processor_config=cfg)
        assert ex.processor_config == cfg

    def test_strict_event_type_false(self):
        ex = _MyExecutor(strict_event_type=False)
        assert ex.strict_event_type is False

    def test_event_type_property(self):
        ex = _MyExecutor()
        assert ex.event_type is _OkEvent


class TestExecutorAppend:
    @pytest.mark.asyncio
    async def test_append_adds_to_pile_and_pending(self):
        ex = _MyExecutor()
        event = _OkEvent()
        await ex.append(event)
        assert event in ex.pile
        assert len(ex.pending) == 1

    @pytest.mark.asyncio
    async def test_append_multiple_events(self):
        ex = _MyExecutor()
        events = [_OkEvent() for _ in range(3)]
        for e in events:
            await ex.append(e)
        assert len(ex.pile) == 3
        assert len(ex.pending) == 3


class TestExecutorStartStop:
    @pytest.mark.asyncio
    async def test_start_creates_processor(self):
        cfg = dict(queue_capacity=5, capacity_refresh_time=0.05, concurrency_limit=1)
        ex = _MyExecutor(processor_config=cfg)
        assert ex.processor is None
        await ex.start()
        assert ex.processor is not None
        assert isinstance(ex.processor, _Proc)

    @pytest.mark.asyncio
    async def test_start_twice_does_not_create_new_processor(self):
        cfg = dict(queue_capacity=5, capacity_refresh_time=0.05, concurrency_limit=1)
        ex = _MyExecutor(processor_config=cfg)
        await ex.start()
        first = ex.processor
        await ex.start()
        assert ex.processor is first

    @pytest.mark.asyncio
    async def test_stop_stops_processor(self):
        cfg = dict(queue_capacity=5, capacity_refresh_time=0.05, concurrency_limit=1)
        ex = _MyExecutor(processor_config=cfg)
        await ex.start()
        await ex.stop()
        assert ex.processor.is_stopped()

    @pytest.mark.asyncio
    async def test_stop_without_processor_is_noop(self):
        ex = _MyExecutor()
        assert ex.processor is None
        await ex.stop()  # should not raise


class TestExecutorForward:
    @pytest.mark.asyncio
    async def test_forward_processes_pending_events(self):
        cfg = dict(queue_capacity=10, capacity_refresh_time=0.01, concurrency_limit=2)
        ex = _MyExecutor(processor_config=cfg)
        await ex.start()

        events = [_OkEvent() for _ in range(3)]
        for e in events:
            await ex.append(e)

        await asyncio.wait_for(ex.forward(), timeout=3.0)

        for e in events:
            assert e.status == EventStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_forward_drains_pending(self):
        cfg = dict(queue_capacity=10, capacity_refresh_time=0.01, concurrency_limit=2)
        ex = _MyExecutor(processor_config=cfg)
        await ex.start()

        event = _OkEvent()
        await ex.append(event)
        await asyncio.wait_for(ex.forward(), timeout=3.0)
        assert len(ex.pending) == 0


class TestExecutorEventProperties:
    @pytest.mark.asyncio
    async def test_completed_events(self):
        cfg = dict(queue_capacity=10, capacity_refresh_time=0.01, concurrency_limit=2)
        ex = _MyExecutor(processor_config=cfg)
        await ex.start()

        event = _OkEvent()
        await ex.append(event)
        await asyncio.wait_for(ex.forward(), timeout=3.0)

        completed = ex.completed_events
        assert len(completed) == 1

    @pytest.mark.asyncio
    async def test_failed_events(self):
        cfg = dict(queue_capacity=10, capacity_refresh_time=0.01, concurrency_limit=2)

        class _FailProc(Processor):
            event_type: ClassVar[type[Event]] = _FailEvent

        class _FailExecutor(Executor):
            processor_type: ClassVar[type[Processor]] = _FailProc

        ex = _FailExecutor(processor_config=cfg)
        await ex.start()

        event = _FailEvent()
        await ex.append(event)
        await asyncio.wait_for(ex.forward(), timeout=3.0)

        failed = ex.failed_events
        assert len(failed) == 1

    @pytest.mark.asyncio
    async def test_pending_events_before_process(self):
        ex = _MyExecutor()
        event = _OkEvent()
        await ex.append(event)
        # Not yet started — event stays PENDING
        pending = ex.pending_events
        assert len(pending) == 1

    @pytest.mark.asyncio
    async def test_cancelled_events_property(self):
        ex = _MyExecutor()
        event = _OkEvent()
        await ex.append(event)
        # Manually set status to CANCELLED
        event.status = EventStatus.CANCELLED
        cancelled = ex.cancelled_events
        assert len(cancelled) == 1

    @pytest.mark.asyncio
    async def test_skipped_events_property(self):
        ex = _MyExecutor()
        event = _OkEvent()
        await ex.append(event)
        event.status = EventStatus.SKIPPED
        skipped = ex.skipped_events
        assert len(skipped) == 1


class TestExecutorStatusCounts:
    @pytest.mark.asyncio
    async def test_status_counts_after_complete(self):
        cfg = dict(queue_capacity=10, capacity_refresh_time=0.01, concurrency_limit=2)
        ex = _MyExecutor(processor_config=cfg)
        await ex.start()

        for _ in range(2):
            await ex.append(_OkEvent())
        await asyncio.wait_for(ex.forward(), timeout=3.0)

        counts = ex.status_counts()
        assert counts.get("completed", 0) == 2

    def test_status_counts_empty(self):
        ex = _MyExecutor()
        assert ex.status_counts() == {}


class TestExecutorCleanup:
    @pytest.mark.asyncio
    async def test_cleanup_completed_removes_events(self):
        cfg = dict(queue_capacity=10, capacity_refresh_time=0.01, concurrency_limit=2)
        ex = _MyExecutor(processor_config=cfg)
        await ex.start()

        events = [_OkEvent() for _ in range(3)]
        for e in events:
            await ex.append(e)
        await asyncio.wait_for(ex.forward(), timeout=3.0)

        removed = ex.cleanup_completed()
        assert removed == 3
        assert len(ex.pile) == 0

    def test_cleanup_completed_empty_pile(self):
        ex = _MyExecutor()
        assert ex.cleanup_completed() == 0


class TestExecutorInspectState:
    def test_inspect_state_no_processor(self):
        ex = _MyExecutor()
        state = ex.inspect_state()
        assert state["total_events"] == 0
        assert state["processor_running"] is False
        assert state["processor_stopped"] is True

    @pytest.mark.asyncio
    async def test_inspect_state_with_processor(self):
        cfg = dict(queue_capacity=5, capacity_refresh_time=0.05, concurrency_limit=1)
        ex = _MyExecutor(processor_config=cfg)
        await ex.start()
        state = ex.inspect_state()
        assert state["processor_running"] is False  # not executing yet
        assert state["processor_stopped"] is False


class TestExecutorContains:
    @pytest.mark.asyncio
    async def test_contains_appended_event(self):
        ex = _MyExecutor()
        event = _OkEvent()
        await ex.append(event)
        assert event in ex

    @pytest.mark.asyncio
    async def test_not_contains_unrelated_event(self):
        ex = _MyExecutor()
        event = _OkEvent()
        assert event not in ex


# ===========================================================================
# handle_field_models() in ReAct.py
# ===========================================================================


class TestHandleFieldModels:
    def test_no_intermediate_returns_empty(self):
        result = handle_field_models(None, None)
        assert result == []

    def test_with_field_models_only(self):
        from lionagi.models.field_model import FieldModel

        fm = FieldModel(name="test_field")
        result = handle_field_models([fm], None)
        assert len(result) == 1
        assert result[0] is fm

    def test_intermediate_options_single_model(self):
        from pydantic import BaseModel

        class MyOutput(BaseModel):
            value: str = ""

        fms = handle_field_models(None, MyOutput)
        # Should have one FieldModel for intermediate_response_options
        assert len(fms) == 1
        assert fms[0].name == "intermediate_response_options"

    def test_intermediate_options_list_of_models(self):
        from pydantic import BaseModel

        class ModelA(BaseModel):
            a: str = ""

        class ModelB(BaseModel):
            b: int = 0

        fms = handle_field_models(None, [ModelA, ModelB])
        assert len(fms) == 1
        assert fms[0].name == "intermediate_response_options"

    def test_intermediate_options_listable(self):
        from pydantic import BaseModel

        class MyOut(BaseModel):
            v: str = ""

        fms = handle_field_models(None, MyOut, intermediate_listable=True)
        # Listable wraps the type in list
        assert len(fms) == 1

    def test_intermediate_options_nullable(self):
        from pydantic import BaseModel

        class MyOut(BaseModel):
            v: str = ""

        fms = handle_field_models(None, MyOut, intermediate_nullable=True)
        assert len(fms) == 1

    def test_intermediate_options_with_existing_field_models(self):
        from pydantic import BaseModel

        from lionagi.models.field_model import FieldModel

        class Extra(BaseModel):
            x: int = 0

        fm = FieldModel(name="existing")
        fms = handle_field_models([fm], Extra)
        assert len(fms) == 2


# ===========================================================================
# ReAct_v1 – verbose_analysis and return_analysis paths
# ===========================================================================


def _make_chat_param(branch):
    """Build a minimal ChatParam using the real branch."""
    from lionagi.operations.types import ChatParam

    return ChatParam(
        guidance=None,
        context=None,
        sender=branch.user,
        recipient=branch.id,
        response_format=None,
        progression=None,
        tool_schemas=[],
        images=[],
        image_detail="auto",
        plain_content="",
        include_token_usage_to_model=False,
        imodel=branch.chat_model,
        imodel_kw={},
    )


def _make_parse_param():
    from lionagi.ln.fuzzy import FuzzyMatchKeysParams
    from lionagi.operations.parse.parse import get_default_call
    from lionagi.operations.types import ParseParam

    return ParseParam(
        response_format=ReActAnalysis,
        fuzzy_match_params=FuzzyMatchKeysParams(),
        handle_validation="return_value",
        alcall_params=get_default_call(),
        imodel=None,
        imodel_kw={},
    )


class TestReActV1ReturnAnalysis:
    """Test return_analysis=True path in ReAct_v1."""

    @pytest.mark.asyncio
    async def test_return_analysis_returns_list(self):
        branch = _make_branch_mock()
        chat_param = _make_chat_param(branch)
        parse_param = _make_parse_param()

        analysis_obj = _make_react_analysis(extension_needed=False)
        final_obj = _make_analysis("done")

        call_count = 0

        async def mock_operate(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return analysis_obj if call_count == 1 else final_obj

        with patch(
            "lionagi.operations.operate.operate.operate",
            new=AsyncMock(side_effect=mock_operate),
        ):
            result = await asyncio.wait_for(
                ReAct_v1(
                    branch=branch,
                    instruction="What is 2+2?",
                    chat_param=chat_param,
                    parse_param=parse_param,
                    max_extensions=0,
                    extension_allowed=False,
                    return_analysis=True,
                ),
                timeout=5.0,
            )

        assert isinstance(result, list)
        assert len(result) >= 1


class TestReActV1VerboseAnalysis:
    """Test verbose_analysis=True path (prints and collects analyses)."""

    @pytest.mark.asyncio
    async def test_verbose_analysis_runs_without_error(self):
        branch = _make_branch_mock()
        chat_param = _make_chat_param(branch)
        parse_param = _make_parse_param()

        analysis_obj = _make_react_analysis(extension_needed=False)
        final_obj = _make_analysis("verbose answer")

        call_count = 0

        async def mock_operate(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return analysis_obj if call_count == 1 else final_obj

        with patch(
            "lionagi.operations.operate.operate.operate",
            new=AsyncMock(side_effect=mock_operate),
        ), patch("lionagi.libs.schema.as_readable.as_readable", return_value=""):
            result = await asyncio.wait_for(
                ReAct_v1(
                    branch=branch,
                    instruction="Test verbose",
                    chat_param=chat_param,
                    parse_param=parse_param,
                    max_extensions=0,
                    extension_allowed=False,
                    verbose_analysis=True,
                ),
                timeout=5.0,
            )

        # verbose path returns the final answer
        assert result is not None


class TestReActV1FinalResultWithoutAnswer:
    """Test path where final result has no .answer attribute."""

    @pytest.mark.asyncio
    async def test_final_result_no_answer_attribute(self):
        branch = _make_branch_mock()
        chat_param = _make_chat_param(branch)

        analysis_obj = _make_react_analysis(extension_needed=False)
        # Return a plain string instead of Analysis object
        plain_result = "plain string result"

        call_count = 0

        async def mock_operate(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return analysis_obj if call_count == 1 else plain_result

        with patch(
            "lionagi.operations.operate.operate.operate",
            new=AsyncMock(side_effect=mock_operate),
        ):
            result = await asyncio.wait_for(
                ReAct_v1(
                    branch=branch,
                    instruction="Return raw",
                    chat_param=chat_param,
                    max_extensions=0,
                    extension_allowed=False,
                ),
                timeout=5.0,
            )

        assert result == plain_result


# ===========================================================================
# ReActStream edge cases
# ===========================================================================


class TestReActStreamMaxExtensionsClamp:
    """max_extensions > 100 is clamped to 100."""

    @pytest.mark.asyncio
    async def test_max_extensions_over_100_clamped(self):
        from lionagi.operations.ReAct.ReAct import ReActStream

        branch = _make_branch_mock()
        chat_param = _make_chat_param(branch)

        analysis_obj = _make_react_analysis(extension_needed=False)
        final_obj = _make_analysis("ok")

        call_count = 0

        async def mock_operate(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return analysis_obj if call_count == 1 else final_obj

        results = []
        with patch(
            "lionagi.operations.operate.operate.operate",
            new=AsyncMock(side_effect=mock_operate),
        ):
            async for item in ReActStream(
                branch=branch,
                instruction="test",
                chat_param=chat_param,
                max_extensions=150,
                extension_allowed=False,  # skip extension loop
            ):
                results.append(item)

        assert len(results) >= 1


class TestReActStreamContinueAfterFailedResponse:
    """continue_after_failed_response=True skips the ValueError."""

    @pytest.mark.asyncio
    async def test_continue_after_failed_response(self):
        from lionagi.operations.ReAct.ReAct import ReActStream

        branch = _make_branch_mock()
        chat_param = _make_chat_param(branch)

        # First call: initial analysis with extension_needed=True
        analysis_obj = _make_react_analysis(extension_needed=True)
        # Second call: all-None dict (failed response)
        failed_dict = {"analysis": None, "extension_needed": None}
        final_obj = _make_analysis("recovered")

        call_count = 0

        async def mock_operate(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return analysis_obj
            elif call_count == 2:
                return failed_dict
            else:
                return final_obj

        results = []
        with patch(
            "lionagi.operations.operate.operate.operate",
            new=AsyncMock(side_effect=mock_operate),
        ):
            async for item in ReActStream(
                branch=branch,
                instruction="test recovery",
                chat_param=chat_param,
                max_extensions=1,
                extension_allowed=True,
                continue_after_failed_response=True,
            ):
                results.append(item)

        assert len(results) >= 1


class TestReActStreamExceptPath:
    """Test the except Exception path when final operate raises."""

    @pytest.mark.asyncio
    async def test_except_path_returns_last_response(self):
        from lionagi.operations.ReAct.ReAct import ReActStream

        branch = _make_branch_mock()
        branch.msgs.last_response.response = "fallback"
        chat_param = _make_chat_param(branch)

        analysis_obj = _make_react_analysis(extension_needed=False)
        call_count = 0

        async def mock_operate(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return analysis_obj
            raise RuntimeError("final operate failed")

        results = []
        with patch(
            "lionagi.operations.operate.operate.operate",
            new=AsyncMock(side_effect=mock_operate),
        ):
            async for item in ReActStream(
                branch=branch,
                instruction="trigger except",
                chat_param=chat_param,
                max_extensions=0,
                extension_allowed=False,
            ):
                results.append(item)

        assert len(results) >= 1
        # Last result should be the fallback
        last = results[-1]
        assert last == "fallback"


class TestReActStreamBetweenRounds:
    """Test between_rounds callback path."""

    @pytest.mark.asyncio
    async def test_between_rounds_with_injection(self):
        from lionagi.operations.ReAct.ReAct import ReActStream

        branch = _make_branch_mock()
        chat_param = _make_chat_param(branch)

        # First call returns analysis needing extension
        ext_analysis = _make_react_analysis(extension_needed=True)
        # After injection, return no-extension analysis
        no_ext_analysis = _make_react_analysis(extension_needed=False)
        final_obj = _make_analysis("between rounds answer")

        call_count = 0

        async def mock_operate(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return ext_analysis
            elif call_count == 2:
                return no_ext_analysis
            else:
                return final_obj

        injection_called = []

        async def between_rounds(b, round_num):
            injection_called.append(round_num)
            return "injected instruction"

        results = []
        with patch(
            "lionagi.operations.operate.operate.operate",
            new=AsyncMock(side_effect=mock_operate),
        ):
            async for item in ReActStream(
                branch=branch,
                instruction="test between rounds",
                chat_param=chat_param,
                max_extensions=1,
                extension_allowed=True,
                between_rounds=between_rounds,
            ):
                results.append(item)

        assert len(injection_called) >= 1
        assert len(results) >= 2


class TestReActStreamBetweenRoundsNoInjection:
    """between_rounds returns None → extension goes through normal path."""

    @pytest.mark.asyncio
    async def test_between_rounds_no_injection(self):
        from lionagi.operations.ReAct.ReAct import ReActStream

        branch = _make_branch_mock()
        chat_param = _make_chat_param(branch)

        ext_analysis = _make_react_analysis(extension_needed=True)
        no_ext_analysis = _make_react_analysis(extension_needed=False)
        final_obj = _make_analysis("normal extension answer")

        call_count = 0

        async def mock_operate(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return ext_analysis
            elif call_count == 2:
                return no_ext_analysis
            else:
                return final_obj

        async def between_rounds(b, round_num):
            return None  # no injection → normal extension path

        results = []
        with patch(
            "lionagi.operations.operate.operate.operate",
            new=AsyncMock(side_effect=mock_operate),
        ):
            async for item in ReActStream(
                branch=branch,
                instruction="test no injection",
                chat_param=chat_param,
                max_extensions=1,
                extension_allowed=True,
                between_rounds=between_rounds,
            ):
                results.append(item)

        assert len(results) >= 2


class TestReActStreamReasoningEffort:
    """reasoning_effort path in prepare_analysis_kwargs."""

    @pytest.mark.asyncio
    async def test_reasoning_effort_low(self):
        from lionagi.operations.ReAct.ReAct import ReActStream

        branch = _make_branch_mock()
        chat_param = _make_chat_param(branch)

        ext_analysis = _make_react_analysis(extension_needed=True)
        no_ext_analysis = _make_react_analysis(extension_needed=False)
        final_obj = _make_analysis("effort answer")

        call_count = 0

        async def mock_operate(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return ext_analysis
            elif call_count == 2:
                return no_ext_analysis
            else:
                return final_obj

        results = []
        with patch(
            "lionagi.operations.operate.operate.operate",
            new=AsyncMock(side_effect=mock_operate),
        ):
            async for item in ReActStream(
                branch=branch,
                instruction="test effort",
                chat_param=chat_param,
                max_extensions=1,
                extension_allowed=True,
                reasoning_effort="low",
            ):
                results.append(item)

        assert len(results) >= 2


class TestReActWrapper:
    """Test the ReAct() legacy wrapper function."""

    @pytest.mark.asyncio
    async def test_react_with_verbose_kwarg(self):
        """verbose kwarg is popped and mapped to verbose_analysis."""
        branch = _make_branch_mock()

        analysis_obj = _make_react_analysis(extension_needed=False)
        final_obj = _make_analysis("wrapper answer")

        call_count = 0

        async def mock_operate(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return analysis_obj if call_count == 1 else final_obj

        from lionagi.operations.ReAct.ReAct import ReAct

        with patch(
            "lionagi.operations.operate.operate.operate",
            new=AsyncMock(side_effect=mock_operate),
        ), patch("lionagi.libs.schema.as_readable.as_readable", return_value=""):
            result = await asyncio.wait_for(
                ReAct(
                    branch=branch,
                    instruct={"instruction": "test verbose kwarg"},
                    extension_allowed=False,
                    verbose=True,  # legacy kwarg
                ),
                timeout=5.0,
            )

        assert result is not None

    @pytest.mark.asyncio
    async def test_react_with_instruct_object(self):
        """ReAct accepts an Instruct object."""
        from lionagi.operations.fields import Instruct

        branch = _make_branch_mock()
        analysis_obj = _make_react_analysis(extension_needed=False)
        final_obj = _make_analysis("instruct answer")

        call_count = 0

        async def mock_operate(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return analysis_obj if call_count == 1 else final_obj

        from lionagi.operations.ReAct.ReAct import ReAct

        with patch(
            "lionagi.operations.operate.operate.operate",
            new=AsyncMock(side_effect=mock_operate),
        ):
            result = await asyncio.wait_for(
                ReAct(
                    branch=branch,
                    instruct=Instruct(instruction="test instruct object"),
                    extension_allowed=False,
                ),
                timeout=5.0,
            )

        assert result is not None

    @pytest.mark.asyncio
    async def test_react_with_tools_and_tool_schemas(self):
        """tools and tool_schemas paths build ActionParam."""

        def multiply(a: int, b: int) -> int:
            return a * b

        branch = _make_branch_mock()
        analysis_obj = _make_react_analysis(extension_needed=False)
        final_obj = _make_analysis("tools answer")

        call_count = 0

        async def mock_operate(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return analysis_obj if call_count == 1 else final_obj

        from lionagi.operations.ReAct.ReAct import ReAct

        with patch(
            "lionagi.operations.operate.operate.operate",
            new=AsyncMock(side_effect=mock_operate),
        ):
            result = await asyncio.wait_for(
                ReAct(
                    branch=branch,
                    instruct={"instruction": "multiply something"},
                    tools=[multiply],
                    extension_allowed=False,
                ),
                timeout=5.0,
            )

        assert result is not None

    @pytest.mark.asyncio
    async def test_react_return_analysis(self):
        """return_analysis=True returns a list of analyses."""
        branch = _make_branch_mock()
        analysis_obj = _make_react_analysis(extension_needed=False)
        final_obj = _make_analysis("analysis list answer")

        call_count = 0

        async def mock_operate(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return analysis_obj if call_count == 1 else final_obj

        from lionagi.operations.ReAct.ReAct import ReAct

        with patch(
            "lionagi.operations.operate.operate.operate",
            new=AsyncMock(side_effect=mock_operate),
        ):
            result = await asyncio.wait_for(
                ReAct(
                    branch=branch,
                    instruct={"instruction": "return list"},
                    extension_allowed=False,
                    return_analysis=True,
                ),
                timeout=5.0,
            )

        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_react_with_response_format(self):
        """response_format is passed into resp_ctx."""
        from pydantic import BaseModel

        class CustomAnswer(BaseModel):
            answer: str = ""

        branch = _make_branch_mock()
        analysis_obj = _make_react_analysis(extension_needed=False)
        final_obj = CustomAnswer(answer="custom")

        call_count = 0

        async def mock_operate(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return analysis_obj if call_count == 1 else final_obj

        from lionagi.operations.ReAct.ReAct import ReAct

        with patch(
            "lionagi.operations.operate.operate.operate",
            new=AsyncMock(side_effect=mock_operate),
        ):
            result = await asyncio.wait_for(
                ReAct(
                    branch=branch,
                    instruct={"instruction": "custom format"},
                    extension_allowed=False,
                    response_format=CustomAnswer,
                ),
                timeout=5.0,
            )

        assert result is not None
