from lionagi.ln.concurrency import is_coro_func


def test_is_coro_func_basic():
    async def coro():
        pass

    def func():
        pass

    assert is_coro_func(coro) is True
    assert is_coro_func(func) is False
    # cached
    assert is_coro_func(coro) is True
    assert is_coro_func(func) is False
