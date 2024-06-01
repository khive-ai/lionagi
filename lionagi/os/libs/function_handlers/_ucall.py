import asyncio
from typing import Any, Callable
from ._util import is_coroutine_func, custom_error_handler, force_async


async def ucall(
    func: Callable,
    *args,
    error_map: dict[type, Callable] = None,
    **kwargs,
) -> Any:
    """
    A unified call handler that executes a function asynchronously with error
    handling.

    This function checks if the given function is a coroutine. If not, it
    forces it to run asynchronously. It then executes the function, ensuring
    the proper handling of event loops. If an error occurs, it applies custom
    error handling based on the provided error map.

    Args:
        func (Callable): The function to be executed.
        *args: Positional arguments to pass to the function.
        error_map (dict[type, Callable], optional): A dictionary mapping
            exception types to error handling functions. Defaults to None.
        **kwargs: Additional keyword arguments to pass to the function.

    Returns:
        Any: The result of the function call.

    Raises:
        Exception: Propagates any exception raised during the function
            execution.
    """
    try:
        if not is_coroutine_func(func):
            func = force_async(func)

        # Checking for a running event loop
        try:
            loop = asyncio.get_running_loop()
            if loop.is_running():
                return await func(*args, **kwargs)
            else:
                return await asyncio.run(func(*args, **kwargs))

        except RuntimeError:  # No running event loop
            loop = asyncio.new_event_loop()
            result = loop.run_until_complete(func(*args, **kwargs))
            loop.close()
            return result

    except Exception as e:
        if error_map:
            custom_error_handler(e, error_map)
        raise e
