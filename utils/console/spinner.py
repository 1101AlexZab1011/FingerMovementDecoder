import asyncio
import functools
import itertools
import sys
from typing import Optional, Callable, Union, List
from utils.console.asynchrony import looped
from utils.console.colored import clean_styles


async def async_spinner(
        chars: Optional[List[str]] = None,
        prefix: Optional[str] = '',
        postfix: Optional[str] = '',
        delay: Optional[Union[float, int]] = .1
):
    """
    Asynchronously display a spinning animation in the console.

    The `async_spinner` function displays a spinning animation in the console using a set of characters. This is often
    used to provide visual feedback during asynchronous tasks.

    Args:
        chars (Optional[List[str]]): A list of characters to use for the spinning animation. Defaults to `['|', '/', '-', '\\']`.
        prefix (Optional[str]): A string to be displayed before the spinning animation. Defaults to an empty string ('').
        postfix (Optional[str]): A string to be displayed after the spinning animation. Defaults to an empty string ('').
        delay (Optional[Union[float, int]]): The time delay (in seconds) between each animation frame. Defaults to 0.1 seconds.

    Example:
        To display a spinning animation with default settings:

        >>> await async_spinner()

    Notes:
        - You can customize the animation frames and time delay by providing the `chars` and `delay` arguments.
        - The animation continues until the task that is awaiting `async_spinner()` is canceled using `asyncio.CancelledError`.

    Warning:
        - Using `async_spinner()` in Jupyter Notebook may not display the animation correctly. It's better suited for console applications.

    """
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

    """
    Decorator for displaying a spinning animation during the execution of an asynchronous function.

    The `spinned` decorator is used to wrap an asynchronous function and display a spinning animation in the console
    while the function is running. This can provide visual feedback to the user during long-running tasks.

    Args:
        chars (Optional[List[str]]): A list of characters to use for the spinning animation. Defaults to `['|', '/', '-', '\\']`.
        prefix (Optional[Union[str, Callable]]): A string or callable function to be displayed before the spinning animation.
            Defaults to an empty string ('').
        postfix (Optional[Union[str, Callable]]): A string or callable function to be displayed after the spinning animation.
            Defaults to an empty string ('').
        delay (Optional[Union[float, int]]): The time delay (in seconds) between each animation frame. Defaults to 0.1 seconds.

    Returns:
        Callable: A wrapped asynchronous function with the spinning animation.

    Example:
        To decorate an asynchronous function 'async_task' with a spinning animation:

        >>> @spinned(chars=['|', '/', '-', '\\'], prefix='Processing: ')
        >>> async def async_task():
        >>>     await asyncio.sleep(3)  # Simulate a long-running task

    Notes:
        - The spinning animation continues until the decorated function completes.
        - You can customize the animation frames, delay, prefix, and postfix as needed.
        - The `chars` argument allows you to specify a list of characters for the spinning animation.
        - The `prefix` and `postfix` arguments can be either strings or callable functions that return strings.
        - The decorated function should be asynchronous.

    Warning:
        - Using `spinned` in Jupyter Notebook may not display the animation correctly. It's better suited for console applications.

    """
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
    """
    Decorator for displaying a spinning animation during the execution of a synchronous or asynchronous function.

    The `spinner` decorator combines the functionality of the `spinned` and `looped` decorators to create a spinning
    animation that can be applied to both synchronous and asynchronous functions. It displays the animation in the console
    while the wrapped function is running.

    Args:
        chars (Optional[List[str]]): A list of characters to use for the spinning animation. Defaults to `['|', '/', '-', '\\']`.
        prefix (Optional[Union[str, Callable]]): A string or callable function to be displayed before the spinning animation.
            Defaults to an empty string ('').
        postfix (Optional[Union[str, Callable]]): A string or callable function to be displayed after the spinning animation.
            Defaults to an empty string ('').
        delay (Optional[Union[float, int]]): The time delay (in seconds) between each animation frame. Defaults to 0.1 seconds.

    Returns:
        Callable: A wrapped function (synchronous or asynchronous) with the spinning animation.

    Example:
        To decorate a synchronous function 'sync_task' with a spinning animation:

        >>> @spinner(chars=['|', '/', '-', '\\'], prefix='Processing: ')
        >>> def sync_task():
        >>>     time.sleep(3)  # Simulate a long-running task

        To decorate an asynchronous function 'async_task' with the same spinning animation:

        >>> @spinner(chars=['|', '/', '-', '\\'], prefix='Processing: ')
        >>> async def async_task():
        >>>     await asyncio.sleep(3)  # Simulate a long-running task

    Notes:
        - The spinning animation continues until the wrapped function completes.
        - You can customize the animation frames, delay, prefix, and postfix as needed.
        - The `chars` argument allows you to specify a list of characters for the spinning animation.
        - The `prefix` and `postfix` arguments can be either strings or callable functions that return strings.
        - The wrapped function can be either synchronous or asynchronous.

    Warning:
        - Using `spinner` in Jupyter Notebook may not display the animation correctly. It's better suited for console applications.

    """
    def wrapper(func):
        return looped(spinned(chars, prefix, postfix, delay)(func))

    return wrapper
