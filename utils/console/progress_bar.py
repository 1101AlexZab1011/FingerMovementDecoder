import os
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Callable, Union, NoReturn, Any
from utils.console import delete_previous_line, edit_previous_line, add_line_above
from utils.console.asynchrony import async_generator
from utils.console.colored import warn, alarm
from utils.structures import Deploy


def print_progress_bar(
        iteration: int,
        total: int,
        prefix: Optional[str] = '',
        suffix: Optional[str] = '',
        decimals: Optional[int] = 1,
        length: Optional[int] = 100,
        fill: Optional[str] = '█',
        edges: Optional[tuple[str, str]] = ('|', '|'),
        space: Optional[str] = '-',
        ending: Optional[Union[str, Callable]] = '',
        print_end: Optional[str] = '\r',
        show_last: Optional[bool] = True
):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    end = None
    if iteration == total:
        end = ''
    elif isinstance(ending, str):
        end = ending
    elif isinstance(ending, Callable):
        end = ending(iteration, total)
    bar = fill * filled_length + end + space * (length - filled_length - 1)
    print(f'\r{prefix} {edges[0]}{bar}{edges[1]} {percent}% {suffix}', end=print_end)
    if iteration == total:
        print()
        if not show_last:
            delete_previous_line()


class AbstractProgress(ABC):
    @abstractmethod
    def __call__(self, *args, **kwargs) -> str:
        pass

    @abstractmethod
    def __next__(self):
        pass

    @abstractmethod
    def __iter__(self):
        pass

    @abstractmethod
    def done(self) -> float:
        pass


@dataclass
class Progress(AbstractProgress):
    total: Optional[int] = None
    iteration: Optional[int] = 0
    prefix: Optional[Union[str, Callable]] = ''
    suffix: Optional[Union[str, Callable]] = None
    decimals: Optional[int] = 1
    length: Optional[int] = None
    fill: Optional[str] = '█'
    edges: Optional[tuple[str, str]] = ('|', '|')
    space: Optional[str] = '-'
    ending: Optional[Union[str, Callable]] = None
    step_report_message: Optional[str] = None
    report_message: Optional[str] = None

    def __post_init__(self):
        if self.ending is None:
            self.ending = self.space
        if self.suffix is None:
            self.suffix = lambda proc:\
                f'{100 * (proc.iteration / float(proc.total)): .{proc.decimals}f}%'
        if self.length is None:
            try:
                self.length = os.get_terminal_size().columns // 2
            except OSError:
                self.length = 100

    def __call__(self, move: Optional[Union[bool, int]] = True) -> str:
        if self.total is not None:
            if move and self.iteration != self.total:
                if isinstance(move, int):
                    if self.iteration + move >= self.total:
                        self.iteration = self.total
                    else:
                        self.iteration = self.iteration + move
                elif isinstance(move, bool) and move:
                    self.iteration += 1
                else:
                    raise ValueError(
                        'The parameter "move" must be either '
                        f'boolean or integer, but {type(move)} is given'
                    )
        else:
            warn('Progress can not run, because number of iterations is unknown')

        filled_length = int(self.length * self.iteration // self.total)
        end = None
        if self.iteration == self.total:
            end = ''
        elif isinstance(self.ending, str):
            end = self.ending
        elif isinstance(self.ending, Callable):
            end = self.ending(self.iteration, self.total)
        if isinstance(self.prefix, str):
            prefix = self.prefix
        elif isinstance(self.prefix, Callable):
            prefix = self.prefix(self)
        else:
            raise TypeError(
                'Prefix must be either string '
                f'or callable, {type(self.prefix)} is given'
            )
        if self.iteration == self.total and self.report_message:
            suffix = self.report_message
        elif isinstance(self.suffix, str):
            suffix = self.suffix
        elif isinstance(self.suffix, Callable):
            suffix = self.suffix(self)
        else:
            raise TypeError(
                'Suffix must be either string '
                f'or callable, {type(self.suffix)} is given'
            )
        bar = self.fill * filled_length + end + self.space * (self.length - filled_length - 1)
        return f'\r{prefix} {self.edges[0]}{bar}{self.edges[1]} {suffix}'

    def __next__(self):
        return self()

    def __iter__(self):
        while self.iteration != self.total:
            yield self()
        return self.total

    def done(self):
        return self.iteration / float(self.total)


@dataclass
class Spinner(AbstractProgress):
    prefix: Optional[str] = ''
    suffix: Optional[str] = ''
    report_message: Optional[str] = ''
    chars: Optional[list[str]] = None
    delay: Optional[float] = .1

    def __post_init__(self):
        self.iteration = 0
        self.total = 1
        self.__current_state = 0
        if self.chars is None:
            self.chars = ['|', '/', '-', '\\']

    def __call__(self, move: Optional[Union[bool, int]] = True) -> str:
        if move and self.iteration != self.total:
            if isinstance(move, bool):
                self.iteration += 0
            elif isinstance(move, int):
                if self.iteration + move >= self.total:
                    self.iteration = self.total
                else:
                    self.iteration = self.iteration + move
            else:
                raise ValueError(
                    'The parameter "move" must be either '
                    f'boolean or integer, but {type(move)} is given'
                )

        return f'\r{self.prefix} {next(self)} {self.suffix}'

    def __next__(self):
        out = self.chars[self.__current_state % len(self.chars)]
        self.__current_state += 1
        return out

    def __iter__(self):
        return iter(self.chars)

    def done(self):
        return self.iteration


class ProgressBar(object):
    _instance = None

    def __new__(cls):
        if not isinstance(cls._instance, cls):
            cls._instance = object.__new__(cls)
        return cls._instance

    def __init__(self):
        self._progresses: list[AbstractProgress] = []
        self._taken_lines: dict[int, int] = {}
        self._running_progresses: list[bool] = []
        self.__printing = False
        self.__changing = False
        self.__initial_stdout = sys.stdout
        if not isinstance(sys.stdout, AboveProgressBarTextWrapper):
            sys.stdout = AboveProgressBarTextWrapper(self)

    def __getitem__(self, i):
        return self._progresses[i]

    def __call__(self, progress_index: int, change: Optional[Union[bool, int]] = True):

        self.__wait_changes()
        self.__changing = True
        if isinstance(self.progresses[progress_index], Spinner) and change is True:
            self.__spin(progress_index)
        progress_content = self.progresses[progress_index](change)
        if self.progresses[progress_index].done() == 1.:
            self._running_progresses[progress_index] = False

        self.__protected_print(
            edit_previous_line,
            progress_content,
            self._taken_lines[progress_index]
        )
        if not any(self.running_progresses):
            self.release_console()
        self.__changing = False

    def __protected_print(self, printer: Callable, *args, **kwargs) -> NoReturn:
        if self.__printing is True:
            while self.__printing:
                time.sleep(.01)
        # FIXME: Small timelag to prevent a bug when filling in multiple progress at the same time.
        #  Each printing takes 10 more milliseconds
        time.sleep(.01)
        self.__printing = True
        printer(*args, **kwargs)
        self.__printing = False

    def __wait_changes(self):
        # FIXME:
        #  dirty code:
        #  waiting some small but meaningful amount of time (10 ms)
        #  to prevent two processes from being added at the same time
        #  the same dirty code in add_progress() and delete_progress()
        if self.__changing:
            while self.__changing:
                time.sleep(.01)

    def __spin(self, progress_index, delete_final: Optional[bool] = False):
        spinner = self._progresses[progress_index]
        if isinstance(spinner, Spinner):
            i = 0
            while not spinner.done():
                start = time.time()
                self.__wait_changes()
                self.__changing = True
                i += 1
                progress_content = spinner()
                self.__protected_print(
                    edit_previous_line,
                    progress_content,
                    self._taken_lines[progress_index]
                )
                self.__changing = False
                time.sleep(max([0, spinner.delay - (time.time() - start)]))
            self._running_progresses[progress_index] = False
            if delete_final:
                self.delete_progress(progress_index)
            else:
                self.__wait_changes()
                self.__changing = True
                self.__protected_print(
                    edit_previous_line,
                    spinner.report_message,
                    self._taken_lines[progress_index]
                )
                self.__changing = False
        else:
            self.interrupt(
                'This is not a spinner: line '
                f'{self._taken_lines[progress_index]}, running: {type(spinner)}'
            )

    def run_with_spinner(
        self,
        func: Union[Deploy, Callable],
        spinner: Spinner,
        *,
        delete_final: Optional[bool] = False
    ):
        self.__wait_changes()
        self.__changing = True
        self._progresses.append(spinner)
        self._running_progresses.append(True)
        spinner_index = len(self._progresses) - 1
        self.__take_line(spinner_index)
        progress_content = self._progresses[-1](False)
        self.__protected_print(print, progress_content)
        self.__changing = False
        out = None
        for i, result in enumerate(async_generator(
                func, Deploy(self.__spin, spinner_index, delete_final=delete_final))
        ):
            if i == 0:
                out = result
                spinner(1)
        return list(out)[0].result()

    @property
    def running_progresses(self):
        return self._running_progresses

    @running_progresses.setter
    def running_progresses(self, value):
        raise AttributeError('Can not set state of processes directly')

    @property
    def progresses(self):
        return self._progresses

    @progresses.setter
    def progresses(self, value):
        raise AttributeError('Can not set new processes directly')

    @property
    def printing(self):
        return self.__printing

    @printing.setter
    def printing(self, value):
        raise AttributeError('Can not set printing status to the progress bar')

    def __take_line(self, progress_index):
        self._taken_lines.update({progress_index: 0})
        for progress_index in self._taken_lines:
            self._taken_lines[progress_index] += 1

    def add_progress(
            self,
            progress: AbstractProgress,
            return_index: Optional[bool] = False
    ):
        self.__wait_changes()
        self.__changing = True
        self._progresses.append(progress)
        self._running_progresses.append(True)
        progress_index = len(self._progresses) - 1
        if not isinstance(sys.stdout, AboveProgressBarTextWrapper):
            self.__initial_stdout = sys.stdout
            sys.stdout = AboveProgressBarTextWrapper(self)

        self.__take_line(len(self._progresses) - 1)
        progress_content = self._progresses[-1](False)
        self.__protected_print(print, progress_content)
        self.__changing = False
        if return_index:
            return progress_index
        return self

    def delete_progress(self, progress_index: int) -> NoReturn:
        self.__wait_changes()
        self.__changing = True
        line = self._taken_lines[progress_index]
        del self._taken_lines[progress_index]
        for i in self._taken_lines:
            if i < progress_index:
                self._taken_lines[i] -= 1
        self.__protected_print(delete_previous_line, line)
        self._running_progresses[progress_index] = False
        self.__changing = False

    def interrupt(self, why: Optional[str] = None):
        self.__protected_print(
            alarm,
            f'\n\rProgram was stopped due to KeyboardInterrupt, exiting...\n{why}',
            True
        )
        os._exit(0)

    def print(self, message: str):
        self.__wait_changes()
        self.__changing = True
        if message not in ['\n', '', ' ', '\t']:
            messages = message.split('\n')
            for message in messages:
                taken_lines = 0
                for i in self._taken_lines:
                    if self._taken_lines[i] != -1:
                        taken_lines += 1

                self.__protected_print(add_line_above, message, taken_lines)
        self.__changing = False

    def release_console(self):
        sys.stdout = self.__initial_stdout


class SpinnerRunner(object):
    def __init__(self, spinner: Spinner, bar: ProgressBar, delete_final: Optional[bool] = False):
        self.spinner = spinner
        self.bar = bar
        self.done = False
        self.__delete_final = delete_final

    def __call__(self, *args, **kwargs):
        run_spinner(self.__loop, self.spinner, self.__delete_final, bar=self.bar)

    def __loop(self):
        while not self.done:
            time.sleep(.1)

    def update_spinner_msg(self, prefix: Optional[str] = None, suffix: Optional[str] = None):
        if prefix is not None:
            self.spinner.prefix = prefix
        if suffix is not None:
            self.spinner.suffix = suffix


class AboveProgressBarTextWrapper(object):
    _instance = None

    def __new__(cls, bar):
        if not isinstance(cls._instance, cls):
            cls._instance = object.__new__(cls)
        return cls._instance

    def __init__(self, bar):
        self.terminal = sys.stdout
        self.bar = bar

    def write(self, message):
        if any(self.bar.running_progresses):
            if message:
                if self.bar.printing:
                    self.terminal.write(message)
                else:
                    self.bar.print(message)
        else:
            self.terminal.write(message)

    def flush(self):
        self.terminal.flush()


class Step(Deploy):
    def __init__(
            self,
            func: Callable,
            *args: Any,
            cost: Optional[Union[int, float]] = 1,
            report_message: Optional[str] = None,
            **kwargs: Any):
        super().__init__(func, *args, **kwargs)
        self.cost = cost
        self.report_message = report_message


class Process(object):
    def __init__(
            self,
            *args: Step,
            progress: Optional[Progress] = Progress(),
            bar: Optional[ProgressBar] = ProgressBar(),
            performance: Optional[str] = 'sequence',
            delete_final: Optional[bool] = False
    ):
        self.steps = args
        self.progress = progress
        self.bar = bar
        self.costs = [step.cost for step in self.steps]
        self.performance = performance
        self.delete_final = delete_final

    def __getitem__(self, index):
        return Process(*self.steps[index:], progress=self.progress)

    def __call__(self):
        progress_index = self.bar.add_progress(self.progress, return_index=True)
        if self.performance == 'generator':
            def generate():
                for i, step in enumerate(self.steps):
                    self.progress.step_report_message = step.report_message
                    out = step()
                    if i == len(self.steps) - 1 and self.delete_final:
                        self.bar.delete_progress(progress_index)
                        yield out
                    else:
                        self.bar(progress_index, self.costs[i])
                        yield out

            return generate()
        elif self.performance == 'sequence':
            out = list()
            for i, step in enumerate(self.steps):
                self.progress.step_report_message = step.report_message
                res = step()
                self.bar(progress_index, self.costs[i])
                out.append(res)
            if self.delete_final:
                self.bar.delete_progress(progress_index)
            return out
        elif self.performance == 'pipeline':
            res = None
            for i, step in enumerate(self.steps):
                self.progress.step_report_message = step.report_message
                if res is None:
                    res = step()
                else:
                    res = step(res)
                self.bar(progress_index, self.costs[i])
            if self.delete_final:
                self.bar.delete_progress(progress_index)
            return res
        else:
            raise ValueError(
                'Process performance must be either '
                '"generator", "sequence" or "pipeline"'
            )

    @property
    def costs(self):
        return self._costs

    @costs.setter
    def costs(self, costs: list[Union[int, float]]):
        self._costs = [int(cost / float(sum(costs)) * 100) for cost in costs]
        self.progress.total = sum(self.costs)


def run_spinner(
        func: Union[Callable, Deploy],
        spinner: Spinner,
        delete_final: Optional[bool] = False,
        bar: Optional[ProgressBar] = ProgressBar()
):
    return bar.run_with_spinner(
        func,
        spinner,
        delete_final=delete_final
    )
