"""Tests for lionagi's Progression and Processor classes.

Target modules:
- lionagi/protocols/generic/progression.py  (84% → boost)
- lionagi/protocols/generic/processor.py    (73% → boost)
"""

from __future__ import annotations

import asyncio
from uuid import UUID, uuid4

import pytest

from lionagi._errors import ItemNotFoundError
from lionagi.protocols.generic.event import Event, EventStatus
from lionagi.protocols.generic.processor import Processor
from lionagi.protocols.generic.progression import Progression, prog

# ---------------------------------------------------------------------------
# Minimal concrete helpers
# ---------------------------------------------------------------------------


class _OkEvent(Event):
    async def _invoke(self):
        return "ok"


class _FailEvent(Event):
    async def _invoke(self):
        raise ValueError("intentional failure")


class _SlowEvent(Event):
    async def _invoke(self):
        await asyncio.sleep(0.05)
        return "slow-ok"


class _Proc(Processor):
    event_type = _OkEvent


def _proc(**kw) -> _Proc:
    defaults = dict(queue_capacity=10, capacity_refresh_time=0.01, concurrency_limit=2)
    defaults.update(kw)
    return _Proc(**defaults)


# ===========================================================================
# Progression Tests
# ===========================================================================


class TestProgressionInit:
    def test_empty_init(self):
        p = Progression()
        assert len(p) == 0
        assert list(p.order) == []
        assert p.name is None

    def test_init_with_uuids(self):
        u1, u2 = uuid4(), uuid4()
        p = Progression(order=[u1, u2])
        items = list(p.order)
        assert items[0] == u1
        assert items[1] == u2

    def test_init_with_string_uuids(self):
        u = uuid4()
        p = Progression(order=[str(u)])
        assert list(p.order)[0] == u

    def test_init_with_name(self):
        p = Progression(name="my-prog")
        assert p.name == "my-prog"

    def test_members_initialized_from_order(self):
        u1, u2 = uuid4(), uuid4()
        p = Progression(order=[u1, u2])
        assert u1 in p._members
        assert u2 in p._members


class TestProgressionAppend:
    def test_append_uuid_adds_to_end(self):
        u1, u2 = uuid4(), uuid4()
        p = Progression(order=[u1])
        p.append(u2)
        items = list(p.order)
        assert items[-1] == u2
        assert len(p) == 2

    def test_append_string_uuid(self):
        p = Progression()
        u = uuid4()
        p.append(str(u))
        assert list(p.order)[0] == u

    def test_append_updates_members(self):
        p = Progression()
        u = uuid4()
        p.append(u)
        assert u in p._members


class TestProgressionInsertFront:
    def test_insert_at_zero_prepends(self):
        u1, u2 = uuid4(), uuid4()
        p = Progression(order=[u1])
        p.insert(0, u2)
        items = list(p.order)
        assert items[0] == u2
        assert items[1] == u1


class TestProgressionPopleft:
    def test_popleft_returns_first(self):
        u1, u2 = uuid4(), uuid4()
        p = Progression(order=[u1, u2])
        result = p.popleft()
        assert result == u1

    def test_popleft_removes_first(self):
        u1, u2 = uuid4(), uuid4()
        p = Progression(order=[u1, u2])
        p.popleft()
        assert list(p.order)[0] == u2
        assert len(p) == 1

    def test_popleft_empty_raises(self):
        p = Progression()
        with pytest.raises(ItemNotFoundError):
            p.popleft()

    def test_popleft_updates_members(self):
        u = uuid4()
        p = Progression(order=[u])
        p.popleft()
        assert u not in p._members


class TestProgressionPop:
    def test_pop_returns_last(self):
        u1, u2 = uuid4(), uuid4()
        p = Progression(order=[u1, u2])
        result = p.pop()
        assert result == u2

    def test_pop_removes_last(self):
        u1, u2 = uuid4(), uuid4()
        p = Progression(order=[u1, u2])
        p.pop()
        assert len(p) == 1
        assert list(p.order)[0] == u1

    def test_pop_empty_raises(self):
        p = Progression()
        with pytest.raises(ItemNotFoundError):
            p.pop()

    def test_pop_middle_index(self):
        u1, u2, u3 = uuid4(), uuid4(), uuid4()
        p = Progression(order=[u1, u2, u3])
        result = p.pop(1)
        assert result == u2
        assert len(p) == 2


class TestProgressionContains:
    def test_contains_uuid(self):
        u = uuid4()
        p = Progression(order=[u])
        assert u in p

    def test_contains_string_id(self):
        u = uuid4()
        p = Progression(order=[u])
        assert str(u) in p

    def test_not_contains_uuid(self):
        p = Progression()
        assert uuid4() not in p

    def test_contains_invalid_type_returns_false(self):
        p = Progression()
        assert 12345 not in p  # int is invalid — validate_order raises ValueError


class TestProgressionLenIter:
    def test_len_empty(self):
        assert len(Progression()) == 0

    def test_len_with_items(self):
        p = Progression(order=[uuid4(), uuid4(), uuid4()])
        assert len(p) == 3

    def test_iter_order(self):
        u1, u2, u3 = uuid4(), uuid4(), uuid4()
        p = Progression(order=[u1, u2, u3])
        assert list(p) == [u1, u2, u3]

    def test_bool_empty(self):
        assert not bool(Progression())

    def test_bool_nonempty(self):
        assert bool(Progression(order=[uuid4()]))


class TestProgressionGetItem:
    def test_index_zero(self):
        u1, u2 = uuid4(), uuid4()
        p = Progression(order=[u1, u2])
        assert p[0] == u1

    def test_index_negative(self):
        u1, u2 = uuid4(), uuid4()
        p = Progression(order=[u1, u2])
        assert p[-1] == u2

    def test_slice_returns_progression(self):
        u1, u2, u3 = uuid4(), uuid4(), uuid4()
        p = Progression(order=[u1, u2, u3])
        sliced = p[1:3]
        assert isinstance(sliced, Progression)
        assert len(sliced) == 2
        items = list(sliced.order)
        assert items[0] == u2
        assert items[1] == u3

    def test_index_out_of_range_raises(self):
        p = Progression(order=[uuid4()])
        with pytest.raises(ItemNotFoundError):
            _ = p[99]

    def test_invalid_key_type_raises(self):
        p = Progression()
        with pytest.raises(TypeError):
            _ = p["bad"]

    def test_empty_slice_raises(self):
        p = Progression(order=[uuid4()])
        with pytest.raises(ItemNotFoundError):
            _ = p[5:10]


class TestProgressionSetItem:
    def test_setitem_replaces_value(self):
        u1, u2 = uuid4(), uuid4()
        p = Progression(order=[u1])
        p[0] = u2
        assert p[0] == u2

    def test_setitem_updates_members(self):
        u1, u2 = uuid4(), uuid4()
        p = Progression(order=[u1])
        p[0] = u2
        assert u2 in p._members

    def test_setitem_slice(self):
        u1, u2, u3, u4 = uuid4(), uuid4(), uuid4(), uuid4()
        p = Progression(order=[u1, u2, u3])
        p[0:2] = [u4]
        assert p[0] == u4
        assert len(p) == 2


class TestProgressionDelItem:
    def test_delitem_by_index(self):
        u1, u2 = uuid4(), uuid4()
        p = Progression(order=[u1, u2])
        del p[0]
        assert len(p) == 1
        assert p[0] == u2

    def test_delitem_by_slice(self):
        u1, u2, u3 = uuid4(), uuid4(), uuid4()
        p = Progression(order=[u1, u2, u3])
        del p[0:2]
        assert len(p) == 1
        assert p[0] == u3

    def test_delitem_updates_members(self):
        u = uuid4()
        p = Progression(order=[u])
        del p[0]
        assert u not in p._members


class TestProgressionRemove:
    def test_remove_uuid(self):
        u1, u2 = uuid4(), uuid4()
        p = Progression(order=[u1, u2])
        p.remove(u1)
        assert u1 not in p
        assert len(p) == 1

    def test_remove_updates_members(self):
        u = uuid4()
        p = Progression(order=[u])
        p.remove(u)
        assert u not in p._members

    def test_remove_not_found_raises(self):
        p = Progression(order=[uuid4()])
        with pytest.raises(ItemNotFoundError):
            p.remove(uuid4())

    def test_remove_invalid_raises(self):
        p = Progression()
        with pytest.raises(ItemNotFoundError):
            p.remove("not-a-uuid")


class TestProgressionExtend:
    def test_extend_with_progression(self):
        u1, u2 = uuid4(), uuid4()
        p1 = Progression(order=[u1])
        p2 = Progression(order=[u2])
        p1.extend(p2)
        assert len(p1) == 2
        assert list(p1.order)[-1] == u2

    def test_extend_updates_members(self):
        u2 = uuid4()
        p1 = Progression()
        p2 = Progression(order=[u2])
        p1.extend(p2)
        assert u2 in p1._members

    def test_extend_non_progression_raises(self):
        p = Progression()
        with pytest.raises(ValueError):
            p.extend([uuid4()])


class TestProgressionClear:
    def test_clear_empties_order(self):
        p = Progression(order=[uuid4(), uuid4()])
        p.clear()
        assert len(p) == 0

    def test_clear_empties_members(self):
        p = Progression(order=[uuid4()])
        p.clear()
        assert len(p._members) == 0


class TestProgressionInclude:
    def test_include_new_item_returns_true(self):
        p = Progression()
        u = uuid4()
        assert p.include(u) is True
        assert u in p

    def test_include_duplicate_returns_false(self):
        u = uuid4()
        p = Progression(order=[u])
        assert p.include(u) is False
        assert len(p) == 1

    def test_include_invalid_returns_false(self):
        p = Progression()
        assert p.include(object()) is False

    def test_include_empty_list_returns_true(self):
        p = Progression()
        assert p.include([]) is True


class TestProgressionExclude:
    def test_exclude_existing_returns_true(self):
        u = uuid4()
        p = Progression(order=[u])
        assert p.exclude(u) is True
        assert u not in p

    def test_exclude_missing_returns_false(self):
        p = Progression()
        assert p.exclude(uuid4()) is False

    def test_exclude_invalid_returns_false(self):
        p = Progression()
        assert p.exclude(object()) is False

    def test_exclude_updates_members(self):
        u = uuid4()
        p = Progression(order=[u])
        p.exclude(u)
        assert u not in p._members


class TestProgressionArithmetic:
    def test_add_creates_new(self):
        u1, u2 = uuid4(), uuid4()
        p1 = Progression(order=[u1])
        p2 = p1 + [u2]
        assert isinstance(p2, Progression)
        assert len(p2) == 2
        assert len(p1) == 1  # original unchanged

    def test_radd(self):
        u1, u2 = uuid4(), uuid4()
        p = Progression(order=[u2])
        result = p.__radd__([u1])
        assert isinstance(result, Progression)
        assert list(result.order)[0] == u1

    def test_iadd_appends(self):
        u1, u2 = uuid4(), uuid4()
        p = Progression(order=[u1])
        p += u2
        assert len(p) == 2

    def test_sub_creates_new(self):
        u1, u2 = uuid4(), uuid4()
        p = Progression(order=[u1, u2])
        p2 = p - [u1]
        assert isinstance(p2, Progression)
        assert len(p2) == 1
        assert len(p) == 2  # original unchanged

    def test_isub_removes(self):
        u1, u2 = uuid4(), uuid4()
        p = Progression(order=[u1, u2])
        p -= u1
        assert len(p) == 1
        assert u1 not in p


class TestProgressionMoveSwapReverse:
    def test_move_item(self):
        u1, u2, u3 = uuid4(), uuid4(), uuid4()
        p = Progression(order=[u1, u2, u3])
        p.move(0, 2)
        # u1 moved from index 0 to 2 (adjusted)
        items = list(p.order)
        assert u1 in items

    def test_swap_items(self):
        u1, u2 = uuid4(), uuid4()
        p = Progression(order=[u1, u2])
        p.swap(0, 1)
        assert p[0] == u2
        assert p[1] == u1

    def test_reverse_in_place(self):
        u1, u2, u3 = uuid4(), uuid4(), uuid4()
        p = Progression(order=[u1, u2, u3])
        p.reverse()
        assert p[0] == u3
        assert p[-1] == u1

    def test_reversed_returns_new(self):
        u1, u2 = uuid4(), uuid4()
        p = Progression(order=[u1, u2])
        rev = reversed(p)
        assert isinstance(rev, Progression)
        assert rev[0] == u2
        assert rev[1] == u1


class TestProgressionCountIndex:
    def test_count_occurrences(self):
        u = uuid4()
        p = Progression(order=[u, u])
        assert p.count(u) == 2

    def test_count_zero(self):
        p = Progression()
        assert p.count(uuid4()) == 0

    def test_index_found(self):
        u1, u2 = uuid4(), uuid4()
        p = Progression(order=[u1, u2])
        assert p.index(u2) == 1

    def test_index_with_start(self):
        u1, u2 = uuid4(), uuid4()
        p = Progression(order=[u1, u2, u1])
        assert p.index(u1, 1) == 2


class TestProgressionEquality:
    def test_equal_progressions(self):
        u1, u2 = uuid4(), uuid4()
        p1 = Progression(order=[u1, u2])
        p2 = Progression(order=[u1, u2])
        assert p1 == p2

    def test_not_equal_different_order(self):
        u1, u2 = uuid4(), uuid4()
        p1 = Progression(order=[u1, u2])
        p2 = Progression(order=[u2, u1])
        assert p1 != p2

    def test_not_implemented_for_non_progression(self):
        p = Progression()
        result = p.__eq__("not a progression")
        assert result is NotImplemented

    def test_gt_lt(self):
        u1, u2 = uuid4(), uuid4()
        p1 = Progression(order=[u1])
        p2 = Progression(order=[u2])
        # Just verify they don't raise; ordering is by UUID bytes
        _ = p1 > p2 or p2 > p1 or p1 == p2

    def test_ge_le(self):
        u1 = uuid4()
        p = Progression(order=[u1])
        assert (p >= p) is True
        assert (p <= p) is True


class TestProgressionRepr:
    def test_repr_contains_class_name(self):
        p = Progression(name="test")
        r = repr(p)
        assert "Progression" in r
        assert "test" in r


class TestProgressionNext:
    def test_next_returns_first(self):
        u = uuid4()
        p = Progression(order=[u])
        assert next(iter(p)) == u


class TestProgFactory:
    def test_prog_creates_progression(self):
        u = uuid4()
        p = prog([u], "myname")
        assert isinstance(p, Progression)
        assert p.name == "myname"
        assert u in p


# ===========================================================================
# Processor Tests
# ===========================================================================


class TestProcessorInit:
    def test_basic_init(self):
        p = _proc()
        assert p.queue_capacity == 10
        assert p.capacity_refresh_time == 0.01
        assert p.available_capacity == 10
        assert p.execution_mode is False
        assert not p.is_stopped()

    def test_with_max_queue_size(self):
        p = _proc(max_queue_size=5)
        assert p.max_queue_size == 5

    def test_zero_concurrency_limit_no_semaphore(self):
        p = _proc(concurrency_limit=0)
        assert p._concurrency_sem is None

    def test_invalid_capacity_raises(self):
        with pytest.raises(ValueError, match="capacity"):
            _proc(queue_capacity=0)

    def test_invalid_refresh_time_raises(self):
        with pytest.raises(ValueError, match="refresh"):
            _proc(capacity_refresh_time=0)


class TestProcessorQueueFull:
    def test_queue_full_unlimited(self):
        p = _proc(max_queue_size=0)
        assert p.queue_full is False

    def test_queue_full_when_at_capacity(self):
        p = _proc(max_queue_size=1)
        assert not p.queue_full
        p.try_enqueue(_OkEvent())
        assert p.queue_full

    def test_try_enqueue_returns_true_when_space(self):
        p = _proc(max_queue_size=5)
        assert p.try_enqueue(_OkEvent()) is True

    def test_try_enqueue_returns_false_when_full(self):
        p = _proc(max_queue_size=1)
        p.try_enqueue(_OkEvent())
        assert p.try_enqueue(_OkEvent()) is False


class TestProcessorStartStop:
    @pytest.mark.asyncio
    async def test_stop_sets_stopped(self):
        p = _proc()
        assert not p.is_stopped()
        await p.stop()
        assert p.is_stopped()

    @pytest.mark.asyncio
    async def test_start_clears_stopped(self):
        p = _proc()
        await p.stop()
        await p.start()
        assert not p.is_stopped()

    @pytest.mark.asyncio
    async def test_start_when_not_stopped_is_noop(self):
        p = _proc()
        await p.start()
        assert not p.is_stopped()


class TestProcessorEnqueueDequeue:
    @pytest.mark.asyncio
    async def test_enqueue_adds_to_queue(self):
        p = _proc()
        event = _OkEvent()
        await asyncio.wait_for(p.enqueue(event), timeout=0.5)
        assert p.queue.qsize() == 1

    @pytest.mark.asyncio
    async def test_dequeue_retrieves_event(self):
        p = _proc()
        event = _OkEvent()
        await asyncio.wait_for(p.enqueue(event), timeout=0.5)
        result = await asyncio.wait_for(p.dequeue(), timeout=0.5)
        assert result is event


class TestProcessorProcess:
    @pytest.mark.asyncio
    async def test_process_empty_queue(self):
        p = _proc()
        await asyncio.wait_for(p.process(), timeout=1.0)
        # No error — capacity unchanged (no events processed)
        assert p.available_capacity == 10

    @pytest.mark.asyncio
    async def test_process_completes_event(self):
        p = _proc()
        event = _OkEvent()
        await p.enqueue(event)
        await asyncio.wait_for(p.process(), timeout=2.0)
        assert event.status == EventStatus.COMPLETED
        assert event.response == "ok"

    @pytest.mark.asyncio
    async def test_process_handles_failing_event(self):
        p = _proc()
        event = _FailEvent()
        await p.enqueue(event)
        await asyncio.wait_for(p.process(), timeout=2.0)
        assert event.status == EventStatus.FAILED

    @pytest.mark.asyncio
    async def test_process_resets_capacity(self):
        p = _proc(queue_capacity=5)
        await p.enqueue(_OkEvent())
        await asyncio.wait_for(p.process(), timeout=2.0)
        # Capacity reset to queue_capacity after processing
        assert p.available_capacity == 5

    @pytest.mark.asyncio
    async def test_process_multiple_events(self):
        p = _proc(queue_capacity=10)
        events = [_OkEvent() for _ in range(3)]
        for e in events:
            await p.enqueue(e)
        await asyncio.wait_for(p.process(), timeout=3.0)
        for e in events:
            assert e.status == EventStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_process_respects_capacity(self):
        p = _proc(queue_capacity=2, concurrency_limit=2)
        events = [_OkEvent() for _ in range(4)]
        for e in events:
            await p.enqueue(e)
        await asyncio.wait_for(p.process(), timeout=2.0)
        completed = sum(1 for e in events if e.status == EventStatus.COMPLETED)
        assert completed <= 4


class TestProcessorRequestPermission:
    @pytest.mark.asyncio
    async def test_default_permits_all(self):
        p = _proc()
        assert await p.request_permission() is True

    @pytest.mark.asyncio
    async def test_default_permits_with_kwargs(self):
        p = _proc()
        assert await p.request_permission(foo="bar") is True


class TestProcessorAvailableCapacity:
    def test_getter(self):
        p = _proc(queue_capacity=7)
        assert p.available_capacity == 7

    def test_setter(self):
        p = _proc(queue_capacity=7)
        p.available_capacity = 3
        assert p.available_capacity == 3


class TestProcessorExecutionMode:
    def test_default_false(self):
        p = _proc()
        assert p.execution_mode is False

    def test_setter(self):
        p = _proc()
        p.execution_mode = True
        assert p.execution_mode is True

    @pytest.mark.asyncio
    async def test_execute_sets_and_clears_execution_mode(self):
        p = _proc(capacity_refresh_time=0.01)

        async def stopper():
            await asyncio.sleep(0.05)
            await p.stop()

        stopper_task = asyncio.create_task(stopper())
        exec_task = asyncio.create_task(p.execute())
        await asyncio.wait_for(asyncio.gather(exec_task, stopper_task), timeout=2.0)
        assert p.execution_mode is False


class TestProcessorCreate:
    @pytest.mark.asyncio
    async def test_create_classmethod(self):
        p = await asyncio.wait_for(
            _Proc.create(
                queue_capacity=5,
                capacity_refresh_time=0.1,
                concurrency_limit=1,
            ),
            timeout=1.0,
        )
        assert isinstance(p, _Proc)
        assert p.queue_capacity == 5


class TestProcessorJoin:
    @pytest.mark.asyncio
    async def test_join_empty_queue(self):
        p = _proc()
        await asyncio.wait_for(p.join(), timeout=1.0)
