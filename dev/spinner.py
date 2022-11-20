import asyncio
import functools
import itertools
import os
import pickle
import sys
import time
from typing import Optional, Callable, Union, Any, List, Tuple

from utils.console.asynchrony import looped
from utils.console.colored import ColoredText, clean_styles, alarm


class Spinner(object):
    def __init__(self, delay: Optional[float] = 0.1, states: Optional[Tuple[str, ...]] = ('/', '-', '\\', '|')):
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
            spinner = asyncio.ensure_future(async_spinner(chars, prefix_to_use, postfix_to_use, delay))
            result = await asyncio.gather(asyncio.to_thread(func, *args, **kwargs))
            spinner.cancel()
            return result

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

string = "|/-\\|/-\\"
styles = [
    ColoredText().color("r").bright(),
    ColoredText().color("y").bright(),
    ColoredText().color("g").bright(),
    ColoredText().color("c").bright(),
    ColoredText().color("b").bright(),
    ColoredText().color("v").bright(),
    ColoredText().color("grey"),
    ColoredText().color("grey").bright(),
]

@spinner(
    chars=[style(s) for s, style in zip(string, styles)],
    prefix='Something very slow happening... '
)
# colored spinner
def slow_process(delay: Optional[int] = 10):
    time.sleep(delay)

if __name__ == '__main__':
    slow_process()


# @spinned(prefix='spinned slow function: ')
# async def slow_function_42(path: str):
#     await asyncio.sleep(2)
#     return 42


# async def supervisor():  # <7>
#     spinner = asyncio.ensure_future(async_spinner(prefix='thinking!'))  # <8>
#     print('spinner object:', spinner)  # <9>
#     result = await slow_function()  # <10>
#     spinner.cancel()  # <11>
#     return result

# async def supervisor():  # <7>
#     spinner = asyncio.ensure_future(async_spinner(prefix='thinking!'))  # <8>
#     print('spinner object:', spinner)  # <9>
#     # task = asyncio.create_task(read_pkl(path))
#     result = await asyncio.gather(asyncio.to_thread(read, path))  # <10>
#     spinner.cancel()  # <11>
#     return result

# @spinner(prefix='Reading file... ', postfix=lambda path: f' {path.split("/")[-3]}')
# def read(path: str) -> Any:
#     return pickle.load(
#         open(
#             path,
#             'rb'
#         )
#     )


# path = './Source/Subjects/Az_Mar_05/Info/ML_Subject05_P1_tsss_mc_trans_info.pkl'
#
#
# def main():
#     result = read(path)
#     print('Answer:', result)
#
#
# if __name__ == '__main__':
#     main()

# def main():
#     loop = asyncio.get_event_loop()  # <12>
#     result = loop.run_until_complete(supervisor())  # <13>
#     loop.close()
#     print('Answer:', result)

# def f1():
#     time.sleep(2)
#     for i in range(30):
#         time.sleep(.1)
#         print('f1')
#
#
# def f2():
#     time.sleep(2)
#     for i in range(20):
#         time.sleep(.1)
#         print('f2')
#
#
# def f3():
#     time.sleep(3)
#     for i in range(10):
#         time.sleep(.1)
#         print('f3')
#
#
# async def s():
#     r = await asyncio.gather(
#         asyncio.to_thread(f1),
#         asyncio.to_thread(f2),
#         asyncio.to_thread(f3),
#     )
#     return r


# def main():
#     loop = asyncio.get_event_loop()  # <12>
#     result = loop.run_until_complete(s())  # <13>
#     loop.close()
#     print('Answer:', result)
