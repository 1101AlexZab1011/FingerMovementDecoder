import asyncio
import functools
import os
from asyncio import Task
from typing import Union, Callable, Optional, Coroutine

from utils.console.colored import alarm
from utils.structures import Deploy


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
            alarm('\rProgram was stopped due to an KeyboardInterrupt', True)
            result = None
            os._exit(0)

        return result

    return wrapper


async def combine_thread(*args: Union[Deploy, Callable]):
    return await asyncio.gather(
        *(
            asyncio.to_thread(arg)
            for arg in args
        )
    )


def to_thread(*args: Union[Deploy, Callable]):
    return (
        asyncio.to_thread(arg)
        for arg in args
    )


def closed_async(
        *args: Union[Deploy, Callable],
        close_event_loop: Optional[bool] = False
):
    loop = None
    result = None
    try:
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
        future = combine_thread(*args)
        result = loop.run_until_complete(future)
        if close_event_loop:
            loop.close()
    except KeyboardInterrupt:
        loop.stop()
        loop.close()
    return result


class Handler(object):
    def __init__(
        self,
        list_of_tasks: list[Union[Callable, Deploy]],
        n_task_simultaneously: Optional[int] = 5
    ):
        self._tasks = list_of_tasks
        self._batch_size = n_task_simultaneously

    def __call__(
            self,
            unfinished: Optional[set[Union[Task, Coroutine]]] = None,
            finished: Optional[set[Union[Task, Coroutine]]] = None
    ):
        if unfinished is None and finished is None:
            out = self.tasks[:self.n_tasks]
            self._tasks = self._tasks[self.n_tasks:]
            return out
        if unfinished is not None:
            unfinished = list(unfinished)
            if self._tasks:
                n = self.n_tasks - len(unfinished)
                out = unfinished + list(to_thread(*self.tasks[:n]))
                self._tasks = self._tasks[n:]
                return out
            else:
                return unfinished

    @property
    def n_tasks(self):
        return self._batch_size

    @n_tasks.setter
    def n_tasks(self, value):
        raise AttributeError('Can not change number of tasks per run after initialization')

    @property
    def tasks(self):
        return self._tasks

    @tasks.setter
    def tasks(self, value):
        raise AttributeError('Can not change tasks after initialization')

    def add_task(self, task: Union[Callable, Deploy]):
        self._tasks.append(task)


def async_generator(
        *args: Optional[Union[Deploy, Callable]],
        handler: Optional[Handler] = None,
        close_event_loop: Optional[bool] = False
):
    if not args and handler is not None:
        args = handler()
    elif args is None and handler is None:
        raise ValueError('Initial tasks undefined')
    tasks = to_thread(*args)
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            loop = asyncio.new_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
    finished, unfinished = loop.run_until_complete(
        asyncio.wait(list(tasks), return_when=asyncio.FIRST_COMPLETED)
    )
    yield finished
    while len(unfinished) != 0:
        if handler is not None:
            if len(unfinished) < handler.n_tasks:
                unfinished = handler(unfinished, finished)
        finished, unfinished = loop.run_until_complete(
            asyncio.wait(list(unfinished), return_when=asyncio.FIRST_COMPLETED)
        )
        yield finished
    if close_event_loop:
        loop.close()
