import re
from typing import Optional, NoReturn


class ColoredText(object):
    def __init__(self):
        self.styles: list = [30, ]
        self.__current_style: int = 0
        self._text = ''

    def __call__(self, text):
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
        self.styles[self.__current_style] += 10
        return self

    def bright(self):
        self.styles[self.__current_style] += 60
        return self

    def add(self):
        self.styles.append(30)
        self.__current_style += 1
        return self

    def style(self, style_name: str):
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


def clean_styles(styled_text):
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
    print(ColoredText().color().style('b')(msg), **kwargs)


def warn(
    msg: str,
    in_bold: Optional[bool] = False,
    bright: Optional[bool] = True,
    **kwargs
) -> NoReturn:
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
