import os
import sys
from typing import Optional


class Silence:
    def __enter__(self):
        """
        Redirect standard output to /dev/null (silence output).
        """
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Restore the original standard output.

        Args:
            exc_type: Exception type.
            exc_val: Exception value.
            exc_tb: Exception traceback.
        """
        sys.stdout.close()
        sys.stdout = self._original_stdout


def delete_previous_line(line: Optional[int] = 1, *, return_str: Optional[bool] = False):
    """
    Delete the previous line(s) in the terminal.

    Args:
        line (int, optional): Number of previous lines to delete. Default is 1.
        return_str (bool, optional): If True, returns the ANSI escape sequences as a string instead of printing them.

    Returns:
        None or str: If return_str is True, returns the ANSI escape sequences as a string; otherwise, prints them.
    """
    out = f'\033[{line}F\033[M\033[A'
    for i in range(line - 1):
        out += '\n'
    if return_str:
        return out
    else:
        print(out)


def erase_previous_line(line: Optional[int] = 1, *, return_str: Optional[bool] = False):
    """
    Erase the content of the previous line(s) in the terminal.

    Args:
        line (int, optional): Number of previous lines to erase. Default is 1.
        return_str (bool, optional): If True, returns the ANSI escape sequences as a string instead of printing them.

    Returns:
        None or str: If return_str is True, returns the ANSI escape sequences as a string; otherwise, prints them.
    """
    out = f'\033[{line}F\033[K'
    for i in range(line - 1):
        out += '\n'
    if return_str:
        return out
    else:
        print(out)


def edit_previous_line(text: str, line: Optional[int] = 1, *, return_str: Optional[bool] = False):
    """
    Replace the content of the previous line(s) in the terminal with the specified text.

    Args:
        text (str): The text to replace the previous line(s) with.
        line (int, optional): Number of previous lines to edit. Default is 1.
        return_str (bool, optional): If True, returns the ANSI escape sequences as a string instead of printing them.

    Returns:
        None or str: If return_str is True, returns the ANSI escape sequences as a string; otherwise, prints them.

    """
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
    """
    Add a new line above the current line in the terminal with the specified text.

    Args:
        text (str, optional): The text to add in the new line. Default is an empty string.
        line (int, optional): Number of lines to add above the current line. Default is 1.
        return_str (bool, optional): If True, returns the ANSI escape sequences as a string instead of printing them.

    Returns:
        None or str: If return_str is True, returns the ANSI escape sequences as a string; otherwise, prints them.
    """
    out = f'\033[{line}F\033[L{text}'
    for i in range(line):
        out += '\n'
    if return_str:
        return out
    else:
        print(out)
