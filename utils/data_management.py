from typing import Optional, Union


def dict2str(
    dictionary: dict,
    *,
    space: Optional[int] = 1,
    tabulation: Optional[int] = 2,
    first_space=True
) -> str:

    def tab(tabulation: int) -> str:
        ' ' * tabulation

    string = f'{tab(tabulation) if first_space else ""}{{\n'
    tabulation += space
    for key, value in zip(dictionary.keys(), dictionary.values()):
        if not isinstance(key, str):
            key = f'{key}'
        if '\n' in key:
            key = key.replace('\n', '')
        string += f'{tab(tabulation)}{key}: '
        if not isinstance(value, dict):
            string += f'{value},\n'
        else:
            string += dict2str(value, space=space, tabulation=tabulation + space, first_space=False)
    string += f'{tab(tabulation - space)}}}\n'
    return string


def convert_base(
    num: Union[int, str],
    to_base: Optional[int] = 10,
    from_base: Optional[int] = 10
) -> str:
    if from_base < 2 or from_base > 36:
        raise ValueError('from_base must be >= 2 and <= 36, or 0')
    if to_base < 2 or to_base > 36:
        raise ValueError('to_base must be >= 2 and <= 36, or 0')
    if isinstance(num, str):
        n = int(num, from_base)
    else:
        n = int(str(num), from_base)
    alphabet = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    if n < to_base:
        return alphabet[n]
    else:
        return convert_base(n // to_base, to_base) + alphabet[n % to_base]
