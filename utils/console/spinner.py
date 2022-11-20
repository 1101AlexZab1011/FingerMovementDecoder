import asyncio
import functools
import itertools
import sys
import time
from typing import Optional, Callable, Union, List, Tuple

from utils.console.asynchrony import looped
from utils.console.colored import clean_styles


class Spinner(object):
    def __init__(
        self,
        delay: Optional[float] = 0.1,
        states: Optional[Tuple[str, ...]] = ('/', '-', '\\', '|')
    ):
        self.delay = delay
        self.__states = states
        self.__current_state = 0

    def __call__(self):
        sys.stdout.write(self.__states[self.__current_state])
        sys.stdout.flush()
        time.sleep(0.1)
        sys.stdout.write('\b')
        self.__current_state = (self.__current_state + 1) % len(self.__states)

    def __iter__(self):
        return iter(self.__states)

    def __next__(self):
        self.__current_state = (self.__current_state + 1) % len(self.__states)
        return self.__states[self.__current_state - 1]


async def async_spinner(
        chars: Optional[List[str]] = None,
        prefix: Optional[str] = '',
        postfix: Optional[str] = '',
        delay: Optional[Union[float, int]] = .1
):
    if chars is None:
        chars_to_use = ['|', '/', '-', '\\']
    else:
        chars_to_use = chars
    write, flush = sys.stdout.write, sys.stdout.flush
    for char in itertools.cycle(chars_to_use):
        status = f'{prefix}{char}{postfix}'
        actual_prefix = clean_styles(prefix) if prefix else ''
        actual_postfix = clean_styles(postfix) if postfix else ''
        actual_char = clean_styles(char) if char else ''
        # print(len(actual_char), actual_char, char.encode('ascii'))
        space = len(actual_prefix) + len(actual_postfix) + len(actual_char)
        write(status)
        flush()
        write('\x08' * space)
        try:
            await asyncio.sleep(delay)
        except asyncio.CancelledError:
            break
    write("\033[K")


def spinned(
        chars: Optional[List[str]] = None,
        prefix: Optional[Union[str, Callable]] = '',
        postfix: Optional[Union[str, Callable]] = '',
        delay: Optional[Union[float, int]] = .1
):
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            if isinstance(prefix, Callable):
                prefix_to_use = prefix(*args, **kwargs)
            elif isinstance(prefix, str):
                prefix_to_use = prefix
            else:
                prefix_to_use = str(prefix)
            if isinstance(postfix, Callable):
                postfix_to_use = postfix(*args, **kwargs)
            elif isinstance(postfix, str):
                postfix_to_use = postfix
            else:
                postfix_to_use = str(postfix)
            spinner = asyncio.ensure_future(
                async_spinner(
                    chars,
                    prefix_to_use,
                    postfix_to_use,
                    delay
                )
            )
            result = await asyncio.gather(asyncio.to_thread(func, *args, **kwargs))
            spinner.cancel()
            return result[0]

        return wrapper

    return decorator


def spinner(
        chars: Optional[List[str]] = None,
        prefix: Optional[Union[str, Callable]] = '',
        postfix: Optional[Union[str, Callable]] = '',
        delay: Optional[Union[float, int]] = .1
):
    def wrapper(func):
        return looped(spinned(chars, prefix, postfix, delay)(func))

    return wrapper
