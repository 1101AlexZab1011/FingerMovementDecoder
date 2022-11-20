import os
import sys
from typing import Optional


class Silence:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def delete_previous_line(line: Optional[int] = 1, *, return_str: Optional[bool] = False):
    out = f'\033[{line}F\033[M\033[A'
    for i in range(line - 1):
        out += '\n'
    if return_str:
        return out
    else:
        print(out)


def erase_previous_line(line: Optional[int] = 1, *, return_str: Optional[bool] = False):
    out = f'\033[{line}F\033[K'
    for i in range(line - 1):
        out += '\n'
    if return_str:
        return out
    else:
        print(out)


def edit_previous_line(text: str, line: Optional[int] = 1, *, return_str: Optional[bool] = False):
    # out = f'\033[{line}F\033[K\033[a{text}'
    out = f'\033[{line}F\033[K{text}'
    for i in range(line - 1):
        out += '\n'
    if return_str:
        return out
    else:
        print(out)


def add_line_above(
    text: Optional[str] = '',
    line: Optional[int] = 1,
    *,
    return_str: Optional[bool] = False
):
    out = f'\033[{line}F\033[L{text}'
    for i in range(line):
        out += '\n'
    if return_str:
        return out
    else:
        print(out)
