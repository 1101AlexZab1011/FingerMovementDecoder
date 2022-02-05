import asyncio
import functools
import os
import random
import sys
import time
from asyncio import Task
from typing import Optional, Union, Any, Callable, Coroutine

from utils.console import Deploy
from utils.console.asynchrony import async_generator, closed_async, Handler
from utils.console.colored import ColoredText, alarm, warn
# from progress_bar import ProgressBar, Progress
# from utils.console.progress_bar import AboveProgressBarTextWrapper


# def f1():
#     bar = ProgressBar()
#     start_time = time.time()
#     n_iters = 100
#     time.sleep(0.01)
#     n_progress = bar.add_progress(
#         Progress(n_iters, prefix='Progress 1: ',
#                  fill=ColoredText().color('y').bright()('|'),
#                  ending=ColoredText().color('y')('|')
#                  ),
#         return_index=True
#     )
#     for i in range(n_iters):
#         time.sleep(random.uniform(0.01, .5))
#         bar(n_progress, random.randint(1, 10))
#         # warn(f'{i}: f1')
#         if bar[n_progress].done() == 1.:
#             # bar.delete_progress(n_progress)
#             return time.time() - start_time
#
#
# def f2():
#     bar = ProgressBar()
#     start_time = time.time()
#     n_iters = 100
#     time.sleep(0.02)
#     n_progress = bar.add_progress(
#         Progress(n_iters, prefix='Progress 2: ',
#                  fill=ColoredText().color('r').bright()('|'),
#                  ending=ColoredText().color('r')('|')
#                  ),
#         return_index=True
#     )
#     for i in range(n_iters):
#         time.sleep(random.uniform(0.01, .5))
#         bar(n_progress, random.randint(1, 10))
#         # warn(f'{i}: f2')
#         if bar[n_progress].done() == 1.:
#             # bar.delete_progress(n_progress)
#             return time.time() - start_time
#
#
# def f3():
#     bar = ProgressBar()
#     start_time = time.time()
#     n_iters = 100
#     time.sleep(0.03)
#     n_progress = bar.add_progress(
#         Progress(n_iters, prefix='Progress 3: ',
#                  fill=ColoredText().color('b').bright()('|'),
#                  ending=ColoredText().color('b')('|')
#                  ),
#         return_index=True
#     )
#     for i in range(n_iters):
#         time.sleep(random.uniform(0.01, .5))
#         bar(n_progress, random.randint(1, 10))
#         # warn(f'{i}: f3')
#         if bar[n_progress].done() == 1.:
#             # bar.delete_progress(n_progress)
#             return time.time() - start_time
#
#
# def f4():
#     bar = ProgressBar()
#     start_time = time.time()
#     n_iters = 100
#     time.sleep(0.04)
#     n_progress = bar.add_progress(
#         Progress(n_iters, prefix='Progress 4: ',
#                  fill=ColoredText().color('c').bright()('|'),
#                  ending=ColoredText().color('c')('|')
#                  ),
#         return_index=True
#     )
#     for i in range(n_iters):
#         time.sleep(random.uniform(0.01, .5))
#         bar(n_progress, random.randint(1, 10))
#         # warn(f'{i}: f4')
#         if bar[n_progress].done() == 1.:
#             # bar.delete_progress(n_progress)
#             return time.time() - start_time
#
#
# def f5():
#     bar = ProgressBar()
#     start_time = time.time()
#     n_iters = 100
#     time.sleep(0.05)
#     n_progress = bar.add_progress(
#         Progress(n_iters, prefix='Progress 5: ',
#                  fill=ColoredText().color('v').bright()('|'),
#                  ending=ColoredText().color('v')('|')
#                  ),
#         return_index=True
#     )
#     for i in range(n_iters):
#         time.sleep(random.uniform(0.01, .5))
#         bar(n_progress, random.randint(1, 10))
#         # warn(f'{i}: f4')
#         if bar[n_progress].done() == 1.:
#             # bar.delete_progress(n_progress)
#             return time.time() - start_time


def plug(secs: Optional[int] = 3, msg: Optional[str] = 'Message'):
    for i in range(10):
        time.sleep(secs)
        if msg:
            print(f'{msg} {i}')


def slow_function(secs: float, *, msg: Optional[str] = '') -> float:
    time.sleep(secs // 2)
    print(msg)
    time.sleep(secs // 2)
    return random.uniform(.5, 1.5)


def slow_for_loop(n_iters: Optional[int] = 100, timelag: Optional[Union[int, float]] = .1,
                  result: Optional[Any] = None):
    for i in range(n_iters):
        print(f'{result}, step {i}')
        time.sleep(timelag)
    return f'\tdone {result}'



if __name__ == '__main__':

    list_of_tasks = [
        Deploy(slow_for_loop, random.randint(1, 10), 1, f'task {i}')
        for i in range(20)
    ]
    # print('__________Sequence of Tasks____________')
    # start_time = time.time()
    # for task in list_of_tasks:
    #     task()
    # print(f'Runtime: {time.time() - start_time}')
    # print('______________________\n')
    print('__________Closed Async____________')
    start_time = time.time()
    print(closed_async(*list_of_tasks))
    # print(f'Runtime: {time.time() - start_time}')
    # print('______________________\n')
    # print('__________Opened Async____________')
    # start_time = time.time()
    # for tasks in async_generator(*list_of_tasks):
    #     for task in list(tasks):
    #         print(task.result())
    #         n = random.randint(1, 5)
    #         time.sleep(n)
    #         print(f'Some processing of {task.result()} that took {n} seconds')
    # print(f'Runtime: {time.time() - start_time}')
    # print('______________________\n')
    # print('__________Opened Async with batches____________')
    # start_time = time.time()
    # handler = Handler(list_of_tasks, 3)
    # for tasks in async_generator(
    #         handler=handler
    # ):
    #     for task in list(tasks):
    #         print(task.result())
    #         n = random.randint(1, 5)
    #         time.sleep(n)
    #         print(f'Some processing of {task.result()} that took {n} seconds')
    #         # original_stdout = sys.stdout
    #         # bar = ProgressBar()
    #         # sys.stdout = AboveProgressBarTextWrapper(ProgressBar())
    #         # closed_async(f1, f2)
    #         # sys.stdout = original_stdout
    # print(f'Runtime: {time.time() - start_time}')
    # print('______________________\n')

# if __name__ == '__main__':
    # pass
    # original_stdout = sys.stdout
    # bar = ProgressBar()
    # sys.stdout = AboveProgressBarTextWrapper(ProgressBar())
    # closed_async(f1, f2, f3,
    #               Deploy(plug, 1, msg='Unexpected message')
    #               )
    # sys.stdout = original_stdout
