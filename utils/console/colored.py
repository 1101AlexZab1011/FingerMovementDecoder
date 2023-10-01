import re
from typing import Optional, NoReturn


class ColoredText(object):
    """
    A utility class for generating colored and styled text for terminal output.

    This class allows you to easily format text with different colors, styles, and highlights for terminal output.
    It uses ANSI escape codes to apply text formatting.

    Usage:
    You can create an instance of this class and use its methods to format text. Example:
    ```
    colored_text = ColoredText()
    colored_text.color('red').highlight().bright().add().style('bold')
    formatted_text = colored_text('This is a colored and styled text')
    print(formatted_text)
    ```

    Attributes:
        styles (list): A list of ANSI escape codes representing the current text styles.
        text (str): The text string to be formatted.

    Methods:
        color(color_name='normal'): Set the text color. Available color names include 'black', 'red', 'green', 'yellow',
                                     'blue', 'violet', 'cyan', 'grey', 'white', and 'normal'.
        highlight(): Apply text highlighting.
        bright(): Make the text brighter.
        add(): Add a new style to the text.
        style(style_name): Apply a specific text style, such as 'bold', 'italic', 'underline', or 'reverse'.

    Note:
        - ANSI escape codes for text formatting may not be supported in all terminal environments.
        - The 'normal' color resets the text formatting to default.
    """
    def __init__(self):
        self.styles: list = [30, ]
        self.__current_style: int = 0
        self._text = ''

    def __call__(self, text: str) -> str:
        code = ';'.join([str(style) for style in self.styles])
        self._text = text
        return f'\033[{code}m{text}\033[0m'

    @property
    def text(self):
        return self._text

    @text.setter
    def text(self, value):
        raise AttributeError('Text can not be set directly')

    def color(self, color_name: Optional[str] = 'normal'):
        """
        Set the text color.

        Args:
            color_name (str): The name of the color to apply. Default is 'normal'.

        Returns:
            ColoredText: The current instance of ColoredText with the specified color style applied.
        """
        self.styles[self.__current_style] = 30
        self.styles[self.__current_style] += {
            'black': 0,
            'red': 1,
            'r': 1,
            'green': 2,
            'g': 2,
            'yellow': 3,
            'y': 3,
            'blue': 4,
            'b': 4,
            'violet': 5,
            'v': 5,
            'cyan': 6,
            'c': 6,
            'grey': 7,
            'white': 7,
            'normal': 8,
        }[color_name]
        return self

    def highlight(self):
        """
        Apply text highlighting.

        Returns:
            ColoredText: The current instance of ColoredText with text highlighting applied.
        """
        self.styles[self.__current_style] += 10
        return self

    def bright(self):
        """
        Make the text brighter.

        Returns:
            ColoredText: The current instance of ColoredText with brighter text style applied.
        """
        self.styles[self.__current_style] += 60
        return self

    def add(self):
        """
        Add a new style to the text.

        Returns:
            ColoredText: A new instance of ColoredText with an additional text style.
        """
        self.styles.append(30)
        self.__current_style += 1
        return self

    def style(self, style_name: str):
        """
        Apply a specific text style.

        Args:
            style_name (str): The name of the text style to apply, such as 'bold', 'italic', 'underline', or 'reverse'.

        Returns:
            ColoredText: The current instance of ColoredText with the specified text style applied.
        """
        self.styles = [{
            'bold': 1,
            'b': 1,
            'italic': 3,
            'i': 3,
            'underline': 4,
            'u': 4,
            'reverse': 7,
            'r': 7
        }[style_name]] + self.styles
        self.__current_style += 1
        return self


def clean_styles(styled_text: str) -> str:
    """
    Remove ANSI escape codes used for text styling and formatting from a styled text.

    This function is used to clean a styled text string by removing ANSI escape codes that are used for text
    styling and formatting. ANSI escape codes are typically added to text to change its color, style, and other
    visual attributes in terminal output.

    Args:
        styled_text (str): The styled text string containing ANSI escape codes.

    Returns:
        str: The cleaned text string with ANSI escape codes removed.

    Example:
    ```python
    styled_text = "\x1b[31mThis is \x1b[1mbold and \x1b[4munderlined\x1b[0m text."
    cleaned_text = clean_styles(styled_text)
    print(cleaned_text)
    # Output: "This is bold and underlined text."
    ```
    """
    def find_sublist(sublist, in_list):
        sublist_length = len(sublist)
        for i in range(len(in_list) - sublist_length):
            if sublist == in_list[i:i + sublist_length]:
                return i, i + sublist_length
        return None

    def remove_sublist_from_list(in_list, sublist):
        indices = find_sublist(sublist, in_list)
        if indices is not None:
            return in_list[0:indices[0]] + in_list[indices[1]:]
        else:
            return in_list

    pure_text = str(styled_text.encode('ascii'))
    found_styles = re.findall(r'\\x1b\[[\d*;]*m', pure_text)
    clean_text = [char for char in pure_text]
    for style in found_styles:
        style = [char for char in style]
        clean_text = remove_sublist_from_list(clean_text, style)
    out = ''.join(clean_text[2:-1])
    return out if out != '\\\\' else '\\'


def bold(msg: str, **kwargs) -> NoReturn:
    """
    Print a message in bold text.

    This function prints the given message in bold text by using ANSI escape codes for text styling.

    Args:
        msg (str): The message to be printed in bold.
        **kwargs: Additional keyword arguments for the print function (e.g., end='\n').

    Returns:
        None
    """
    print(ColoredText().color().style('b')(msg), **kwargs)


def warn(
    msg: str,
    in_bold: Optional[bool] = False,
    bright: Optional[bool] = True,
    **kwargs
) -> NoReturn:
    """
    Print a warning message in yellow text.

    This function prints the given message in yellow text, which is commonly used for warning messages.
    You can specify whether the text should be in bold and whether it should be bright.

    Args:
        msg (str): The warning message to be printed.
        in_bold (bool, optional): Set to True to print the message in bold. Default is False.
        bright (bool, optional): Set to True to make the text brighter. Default is True.
        **kwargs: Additional keyword arguments for the print function (e.g., end='\n').

    Returns:
        None
    """
    yellow = ColoredText().color('y')
    if bright:
        yellow.bright()
    if in_bold:
        bold(yellow(msg), **kwargs)
    else:
        print(yellow(msg), **kwargs)


def alarm(
    msg: str,
    in_bold: Optional[bool] = False,
    bright: Optional[bool] = True,
    **kwargs
) -> NoReturn:
    """
    Print an alarm message in red text.

    This function prints the given message in red text, which is commonly used for alarm or error messages.
    You can specify whether the text should be in bold and whether it should be bright.

    Args:
        msg (str): The alarm message to be printed.
        in_bold (bool, optional): Set to True to print the message in bold. Default is False.
        bright (bool, optional): Set to True to make the text brighter. Default is True.
        **kwargs: Additional keyword arguments for the print function (e.g., end='\n').

    Returns:
        None
    """
    red = ColoredText().color('r')
    if bright:
        red.bright()
    if in_bold:
        bold(red(msg), **kwargs)
    else:
        print(red(msg), **kwargs)


def success(
    msg: str,
    in_bold: Optional[bool] = False,
    bright: Optional[bool] = True,
    **kwargs
) -> NoReturn:
    """
    Print a success message in green text.

    This function prints the given message in green text, which is commonly used for success or confirmation messages.
    You can specify whether the text should be in bold and whether it should be bright.

    Args:
        msg (str): The success message to be printed.
        in_bold (bool, optional): Set to True to print the message in bold. Default is False.
        bright (bool, optional): Set to True to make the text brighter. Default is True.
        **kwargs: Additional keyword arguments for the print function (e.g., end='\n').

    Returns:
        None
    """
    green = ColoredText().color('g')
    if bright:
        green.bright()
    if in_bold:
        bold(green(msg), **kwargs)
    else:
        print(green(msg), **kwargs)


if __name__ == '__main__':
    styles = [
        ColoredText().color("r").bright(),
        ColoredText().color("y").bright(),
        ColoredText().color("g").bright(),
        ColoredText().color("c").bright(),
        ColoredText().color("b").bright(),
        ColoredText().color("b").highlight(),
        ColoredText().color("v").bright(),
        ColoredText().color("grey"),
        ColoredText().color("grey").bright(),
        ColoredText().color("r"),
        ColoredText().color("y"),
        ColoredText().color("c")
    ]
    text = 'hello world'

    # ctext = ''.join([
    #     style(char) for style, char in zip(
    #         styles,
    #         text
    #     )
    # ])
    # ctext = '\x1b[97m\\\x1b[0m'
    ctext = ColoredText().color("r").bright()('\\')
    print(ctext)
    print(f'length: {len(text)}')
    print(f'actual length: {len(ctext)}')
    print(f'actual string: {ctext.encode("ascii")}')
    s = clean_styles(ctext)
    print(f'magic length: {len(clean_styles(ctext))}')
    print(s)
