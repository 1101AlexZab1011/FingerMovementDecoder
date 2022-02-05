from functools import wraps
import hashlib
from typing import Optional, Callable, Any, Union, Dict
import os


def check_path(path: str) -> bool:
    if not os.path.isdir(path):
        os.mkdir(path)
        return False
    else:
        return True


def get_dir_size(start_path: str = './') -> float:
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)

    return total_size


def read_or_write(
        file_ext,
        reader: Optional[Callable] = None,
        writer: Optional[Callable] = None,
        path: Optional[Union[str, Callable]] = './rw_storage',
        file_prefix: Optional[Union[str, Callable]] = ''
):
    def encrypt_callable(func: Callable, args: tuple, kwargs: dict[str: Any], prefix: str):
        args_str = ", ".join([str(arg) for arg in args])
        kwargs_str = ', '.join([f'{key}={value}' for key, value in kwargs.items()])
        params_str = args_str + ', ' + kwargs_str if not not args_str else kwargs_str
        return prefix + str(hashlib.md5(
            bytes(
                f'{func.__name__}({params_str})',
                'utf-8'
            )
        ).hexdigest())

    def write(func: Callable, args: tuple, kwargs: Dict[str, Any], path: str, prefix: str) -> None:
        result = func(*args, **kwargs)
        if writer is not None:
            check_path(path)
            writer(
                os.path.join(
                    path,
                    f'{encrypt_callable(func, args, kwargs, prefix)}.{file_ext}'
                ),
                result
            )
        return result

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:

            if isinstance(path, Callable):
                place = path(*args, **kwargs)
            elif isinstance(path, str):
                place = path
            else:
                raise ValueError(f'Path have to be string or callable, but {type(path)} is given')

            check_path(path)

            if isinstance(file_prefix, Callable):
                prefix = file_prefix(*args, **kwargs)
            elif isinstance(file_prefix, str):
                prefix = file_prefix
            else:
                raise ValueError(f'Prefix to file have to be string or callable, but {type(path)} is given')

            if reader is not None and os.path.exists(place):
                code = encrypt_callable(func, args, kwargs, prefix)
                for address, _, files in os.walk(place):
                    for file in files:
                        if code in file:
                            return reader(os.path.join(address, file))
            return write(func, args, kwargs, place, prefix)

        return wrapper

    return decorator