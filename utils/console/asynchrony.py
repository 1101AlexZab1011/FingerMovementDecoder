import asyncio
import functools
import os
from asyncio import Task
from typing import Union, Callable, Optional, Coroutine


def looped(func):
    """
    Decorator for running asyncio coroutine functions within a new event loop.

    This decorator is designed to be used with asynchronous coroutine functions. It creates a new asyncio event loop,
    runs the decorated function within that loop using `run_until_complete`, and ensures proper cleanup in case of
    KeyboardInterrupt by handling it.

    Args:
        func (callable): The asynchronous coroutine function to be decorated.

    Returns:
        callable: A wrapped function that can be executed within a new event loop.

    Example:
        To decorate an asynchronous coroutine function 'async_function' for running within a new event loop:

        >>> @looped
        >>> async def async_function():
        >>>     # Your asynchronous code here
        >>>     await asyncio.sleep(1)

        When 'async_function' is called, it will be executed within a new event loop.

    Notes:
        - The decorator uses `asyncio.new_event_loop()` to create a new event loop.
        - It handles KeyboardInterrupt by stopping the event loop and closing it.
        - In case of a KeyboardInterrupt, the program will exit immediately.
        - Decorated functions should be asynchronous coroutines.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(func(*args, **kwargs))
            loop.close()
        except KeyboardInterrupt:
            loop.stop()
            loop.close()
            print('\rProgram was stopped due to an KeyboardInterrupt', True)
            result = None
            os._exit(0)

        return result

    return wrapper

