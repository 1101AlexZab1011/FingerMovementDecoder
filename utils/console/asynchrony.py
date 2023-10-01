import asyncio
import functools
import os
from asyncio import Task
from typing import Union, Callable, Optional, Coroutine


def looped(func):
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

