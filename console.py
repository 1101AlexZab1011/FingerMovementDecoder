import sys
from typing import Optional


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
    out = f'\033[{line}F\033[K\033[a{text}'
    for i in range(line-1):
        out += '\n'
    if return_str:
        return out
    else:
        print(out)


def add_line_above(text: Optional[str] = '', line: Optional[int] = 1, *, return_str: Optional[bool] = False):
    out = f'\033[{line}F\033[L{text}'
    for i in range(line):
        out += '\n'
    if return_str:
        return out
    else:
        print(out)


if __name__ == '__main__':
    original_stdout = sys.stdout
    # sys.stdout = Logger()
    for i in range(10):
        print(f'Message {i + 1}')
    # erase_previous_line(2)
    edit_previous_line('Edited', 1)
    # add_line_above('Additional Line', 2)
    # print()
    # delete_previous_line()
