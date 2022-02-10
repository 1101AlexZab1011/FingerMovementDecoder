import sys
import os
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 


import asyncio
import os
import random
import re
import sys
import time
from asyncio import Task, all_tasks
from dataclasses import dataclass
from typing import Optional, Callable, Union, NoReturn, Any, Coroutine
from utils.console import delete_previous_line, edit_previous_line, add_line_above
from utils.console.asynchrony import closed_async, Handler, to_thread, async_generator
from utils.console.colored import warn, ColoredText, alarm, clean_styles, success
from utils.console.progress_bar import ProgressBar, Progress, AboveProgressBarTextWrapper, Process, Step, \
    Spinner, run_spinner, SpinnerRunner
from utils.data_management import dict2str
from utils.structures import Deploy


def f1():
    bar = ProgressBar()
    start_time = time.time()
    n_iters = 100
    time.sleep(random.uniform(0.01, .1))
    n_progress = bar.add_progress(
        Progress(n_iters, prefix='Progress 1: ',
                 fill=ColoredText().color('y').bright()('|'),
                 ending=ColoredText().color('y')('|')
                 ),
        return_index=True
    )
    for i in range(n_iters):
        time.sleep(random.uniform(0.01, .5))
        bar(n_progress, random.randint(1, 10))
        # warn(f'{i}: f1')
        if bar[n_progress].done() == 1.:
            # bar.delete_progress(n_progress)
            return time.time() - start_time


def f2():
    bar = ProgressBar()
    start_time = time.time()
    n_iters = 100
    time.sleep(random.uniform(0.01, .1))
    n_progress = bar.add_progress(
        Progress(n_iters, prefix='Progress 2: ',
                 fill=ColoredText().color('r').bright()('|'),
                 ending=ColoredText().color('r')('|')
                 ),
        return_index=True
    )
    for i in range(n_iters):
        time.sleep(random.uniform(0.01, .5))
        bar(n_progress, random.randint(1, 10))
        # warn(f'{i}: f2')
        if bar[n_progress].done() == 1.:
            # bar.delete_progress(n_progress)
            return time.time() - start_time


def f3():
    bar = ProgressBar()
    start_time = time.time()
    n_iters = 100
    time.sleep(random.uniform(0.01, .1))
    n_progress = bar.add_progress(
        Progress(n_iters, prefix='Progress 3: ',
                 fill=ColoredText().color('b').bright()('|'),
                 ending=ColoredText().color('b')('|')
                 ),
        return_index=True
    )
    for i in range(n_iters):
        time.sleep(random.uniform(0.01, .5))
        bar(n_progress, random.randint(1, 10))
        # warn(f'{i}: f3')
        if bar[n_progress].done() == 1.:
            # bar.delete_progress(n_progress)
            return time.time() - start_time


def f4():
    bar = ProgressBar()
    start_time = time.time()
    n_iters = 100
    time.sleep(random.uniform(0.01, .1))
    n_progress = bar.add_progress(
        Progress(n_iters, prefix='Progress 4: ',
                 fill=ColoredText().color('c').bright()('|'),
                 ending=ColoredText().color('c')('|')
                 ),
        return_index=True
    )
    for i in range(n_iters):
        time.sleep(random.uniform(0.01, .5))
        bar(n_progress, random.randint(1, 10))
        # warn(f'{i}: f4')
        if bar[n_progress].done() == 1.:
            # bar.delete_progress(n_progress)
            return time.time() - start_time


def f5():
    bar = ProgressBar()
    start_time = time.time()
    n_iters = 100
    time.sleep(random.uniform(0.01, .1))
    n_progress = bar.add_progress(
        Progress(n_iters, prefix='Progress 5: ',
                 fill=ColoredText().color('v').bright()('|'),
                 ending=ColoredText().color('v')('|')
                 ),
        return_index=True
    )
    for i in range(n_iters):
        time.sleep(random.uniform(0.01, .5))
        bar(n_progress, random.randint(1, 10))
        # warn(f'{i}: f4')
        if bar[n_progress].done() == 1.:
            # bar.delete_progress(n_progress)
            return time.time() - start_time


def spinner_progress(timelag, pre, post, del_=False):
    def progress(start_time):
        time.sleep(timelag)
        return time.time() - start_time

    bar = ProgressBar()
    start_time = time.time()
    time.sleep(random.uniform(0.01, .1))
    result = bar.run_with_spinner(
        Deploy(progress, start_time),
        Spinner(prefix=pre, report_message=post),
        delete_final=del_
    )
    return result


def plug(secs: Optional[int] = 3, msg: Optional[str] = 'Unexpected message'):
    for i in range(10):
        time.sleep(secs)
        if msg:
            print(f'{msg} {i}')
    return "Spammer Done"


async def s():
    bar = ProgressBar()
    return await asyncio.gather(
        asyncio.to_thread(f1),
        asyncio.to_thread(f2),
        asyncio.to_thread(f3),
        asyncio.to_thread(f4),
        asyncio.to_thread(f5),
        asyncio.to_thread(plug, 1),
        asyncio.to_thread(
            spinner_progress,
            random.uniform(4., 5.5),
            'Spinned Progress 1: ',
            'Spinned Progress 1: DONE',
            # del_=True
        ),
        asyncio.to_thread(
            spinner_progress,
            random.uniform(4., 5.5),
            'Spinned Progress 2: ',
            'Spinned Progress 2: DONE',
            # del_=True
        ),
        asyncio.to_thread(spinner_progress, random.uniform(4., 5.5), 'Spinned Progress 3: ', 'Spinned Progress 3: DONE')
    )


def main():
    loop = None
    b = None
    try:
        loop = asyncio.get_event_loop()
        future = s()
        b = loop.run_until_complete(future)
        loop.close()
    except KeyboardInterrupt as e:
        loop.stop()
        loop.close()
        ProgressBar().interrupt(str(e))
    print(b)
    print('MAIN DONE')


def slow_function(secs: float, *, msg: Optional[str] = '') -> float:
    start = time.time()
    # time.sleep(secs // 2)
    # print(msg)
    # time.sleep(secs // 2)
    time.sleep(secs)
    print(msg)
    return time.time() - start


def proc_with_progress(bar, name):
    start_time = time.time()
    n_iters = 100
    time.sleep(random.uniform(0.01, .1))
    n_progress = bar.add_progress(
        Progress(n_iters, prefix=f'{name}: '),
        return_index=True
    )
    for i in range(n_iters):
        time.sleep(random.uniform(0.01, .5))
        bar(n_progress, random.randint(1, 10))
        # warn(f'{i}: f4')
        if bar[n_progress].done() == 1.:
            bar.delete_progress(n_progress)
            return time.time() - start_time


def suffix_gen(progress: Progress):
    spinner_content = ['|', '/', '-', '\\']
    if progress.iteration != progress.total:
        step_report_message = progress.step_report_message if progress.step_report_message else ''
        return f'{100 * (progress.iteration / float(progress.total)): .{progress.decimals}f}% ' \
               f'{step_report_message} {spinner_content[progress.iteration % len(spinner_content)]}'
    else:
        return progress.report_message


if __name__ == '__main1__':
    bar = ProgressBar()
    Process(
        *[Step(slow_function, random.uniform(.05, .1), cost=random.uniform(.1, .2), report_message=f'Step {j} done') for
          j in range(100)],
        progress=Progress(
            prefix=f'Progress: ',
            suffix=suffix_gen,
            fill=ColoredText().color('v').bright()('|'),
            ending=ColoredText().color('v')('ᐳ'),
            progress_report_message=ColoredText().color('g')('All is done')
        ),
        bar=bar
    )()

if __name__ == '__main__':
    bar = ProgressBar()
    processes = [Process(
        *[Step(slow_function, random.uniform(.1, .5),
               cost=random.uniform(.1, .5), report_message=ColoredText().color('normal').style('b')(f'Step {j} done'))
          for j in range(10)],

        progress=Progress(
            prefix=f'Progress {i}: ',
            suffix=suffix_gen,
            fill=ColoredText().color(['r', 'y', 'b', 'v', 'c', 'g'][i % 6]).bright()('|'),
            ending=ColoredText().color(['r', 'y', 'b', 'v', 'c', 'g'][i % 6])('|'),
            report_message=ColoredText().color('g')('All is done')
        ),
        delete_final=True,
        # performance='generator'
        bar=ProgressBar()
    ) for i in range(25)]
    spinners = [
        Deploy(
            run_spinner,
            Deploy(slow_function, random.uniform(1.5, 5.5)),
            Spinner(prefix=f'Spinner {i}:', report_message=f'Spinner {i}: Done'),
            delete_final=True,
            bar=bar
        )
        for i in range(25)
    ]

    # for prog in rest:
    #     for tasks in async_generator(*curr):
    #         for task in list(tasks):
    #             print(task.result())
    #             curr.append(prog)

    # bar = ProgressBar()
    # res = closed_async(
    #     *spinners
    # )
    # for i, tasks in enumerate(async_generator(*spinners)):
    #     for task in list(tasks):
    #         print(f'{i}: Done')

    # processes = [
    #     Deploy(proc_with_progress, bar, f'P{i}')
    #     for i in range(100)
    # ]
    start = time.time()
    bar = ProgressBar()
    all_tasks = processes + spinners
    # all_tasks = spinners
    random.shuffle(all_tasks)
    # spr = SpinnerRunner(
    #     Spinner(prefix=f'Overall spinner process:', suffix='0.00%', report_message=f'Overall spinner process: Done'),
    #     bar,
    #     delete_final=True
    # )
    # all_tasks = [
    #                 spr,
    #                 # lambda: time.sleep(.2),
    #                 # *[lambda: time.sleep(.2) for _ in range(15)]
    #             ] + all_tasks
    n_progress = bar.add_progress(
        Progress(50, prefix=f'Overall Progress: ',
                 space=ColoredText().color('grey')('⋅'),
                 ending=ColoredText().color('grey').bright().highlight()(' '),
                 fill=ColoredText().color('grey').highlight()(' '),
                 edges=(ColoredText().color('normal').style('b')('⎭'), ColoredText().color('normal').style('b')('⎩')),
                 report_message=ColoredText().color('g')('All is done')
                 ),
        return_index=True
    )
    handler = Handler(all_tasks, 5)
    try:
        for i, tasks in enumerate(async_generator(handler=handler)):
            # spr.update_spinner_msg(prefix=f'Overall spinner process:',
            #                        suffix=f'{(i+1)/(len(all_tasks)-1)*100 : .2f}%, {(i+1)} of {len(all_tasks)-1}')
            # if (i+1)/(len(all_tasks)-1) == 1 and not spr.done:
            #     spr.done = True
            #     print('Overall spinner process: Done')
            bar(n_progress)
            print(f'Done: {(i + 1)}')
            for task in list(tasks):

                s = 0
                for el in bar._taken_lines:
                    if el != -1:
                        s += 1
                # print(f'{i}: {s}, {bar._taken_lines}, {len(bar._taken_lines)}')
                pass
    except KeyboardInterrupt:
        os._exit(0)
    bar.release_console()
    print(time.time() - start)
