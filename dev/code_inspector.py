# import dataclasses
# import re
# import sys
# import os
# current_dir = os.path.dirname(os.path.abspath('./'))
# if not current_dir in sys.path:
#     sys.path.append(current_dir)
# from cross_runs_TF_planes import CrossRunsTFScorer
# import inspect
# import numpy as np
# import time
# from utils.console.colored import ColoredText
# import utils.console.colored as ucc
# from utils.data_management import dict2str
# from numpy import ones, arange
# import random as r
# import itertools as it
# from typing import *
# import string



# def inspect_original_code(func: Callable) -> tuple[str, list[str], list[str], list[str]]:
#     """Analyzes the code of the given function and finds all libraries and entities imported by it

#     Args:
#         func (Callable): function to analyze

#     Returns:
#         tuple[str, list[str], list[str], list[str], list[object]]: code of the given function, 
#             list of imports in python syntax, 
#             list of used packages, 
#             list of names of used unknown entities, 
#             list of used unknown entities
#     """
    
    
#     def protect_source_module(source_module: object, member: object) -> object:
#         """Internal function of `inspect_original_code` function, checks if output of `inspect.getmodule` is correct

#         Args:
#             source_module (object): module object to check
#             member (object): any entity of the given module

#         Raises:
#             ModuleNotFoundError: given object is not a module of its name is incorrect and correct module wasn't found

#         Returns:
#             object: necessary module
#         """
        
#         if source_module is None:
#             try:
#                 return sys.modules[re.findall( r'\'.*\'',str(type(member)))[0][1:-1].split('.')[0]]
#             except KeyError:
#                 raise ModuleNotFoundError(f'Can not determine module of {member.__name__}')
#         elif '._' in source_module.__name__:
#             try:
#                 return sys.modules[source_module.__name__.split('._')[0]]
#             except KeyError:
#                 raise ModuleNotFoundError(f'Can not determine correct name of {source_module.__name__}')
#         else:
#             return source_module
            
    
#     def get_module_type(member: object) -> str:
#         """Internal function of `inspect_original_code` function, checks the module type of the object included in it

#         Args:
#             member (object): any entity to check its module type

#         Returns:
#             str: type of the found module: `__main__` if the given entity belongs to current python-file,
#                 `built-in` or `python` if the given entity belongs to python internal libraries,
#                 `pip` if the given entity belongs to any python package included to `site-packages`,
#                 `custom` if the given entity belongs to unknown package
#         """
        
#         module = protect_source_module(inspect.getmodule(member), member)
#         if module.__spec__ is None:
#             return '__main__'
#         elif module.__spec__.origin == 'built-in':
#             return 'built-in'
#         elif 'site-packages' in module.__spec__.origin:
#             return 'pip'
#         elif '/lib/' in module.__spec__.origin or '\\lib\\' in module.__spec__.origin:
#             return 'python'
#         else:
#             return 'custom'
    
#     def import_member(member: object, source_module: object, imports: list[str], packages: list[str], locals: list[str], entities: list[object]) -> NoReturn:
#         """Internal function of `inspect_original_code` function, generates import string for the given entity, finds all entities it uses

#         Args:
#             member (object): entity to import
#             source_module (object): module the given entity belongs to
#             imports (list[str]): list of imports for sibling entities
#             packages (list[str]): list of packages used in the mother entity
#             locals (list[str]): list of names of external entities used in the mother entity
#             entities (list[object]): list of external entities used in the mother entity

#         Returns:
#             NoReturn
#         """
        
#         member_name = member.__name__
#         source_module_name = source_module.__name__ if get_module_type(source_module) not in ['__main__', 'custom'] else 'locals'
#         raw_module_name = source_module_name.split('.')[0]
#         if source_module_name == 'locals' and member_name not in locals:
#             locals.append(member_name)
#             entities.append(member)
#         elif source_module_name != 'locals' and raw_module_name not in packages:
#             packages.append(raw_module_name)
        
#         same_source = [i for i, import_ in enumerate(imports) if f'from {source_module_name}' in import_]
#         if same_source:
#             imports[same_source[0]] += f', {member_name}'
#         else:
#             imports.append(f'from {source_module_name} import {member_name}')
    
#     def import_module(member: object, member_name: str, imports: list[str], packages: list[str]) -> NoReturn:
#         """Internal function of `inspect_original_code` function, generates import string for the given module

#         Args:
#             member (object): module object to import
#             member_name (str): name of this module as it used in the source code of the mother entity
#             imports (list[str]): list of imports for sibling entities
#             packages (list[str]): list of packages used in the mother entity

#         Returns:
#             NoReturn
#         """
        
#         module_type = get_module_type(member)
#         module_name = member.__name__ if module_type not in ['custom', '__main__'] else 'locals'
#         raw_module_name = module_name.split('.')[0]
        
#         if module_name != 'locals' and module_type == 'pip' and raw_module_name not in packages:
#             packages.append(raw_module_name)
        
#         if module_name == member_name:
#             imports.append(f'import {module_name}')
#         else:
#             imports.append(f'import {module_name} as {member_name}')
    
#     imports = list()
#     packages = list()
#     locals = list()
#     entities = list()
    
#     for member_name, member in inspect.getclosurevars(func).globals.items():
#         source_module = protect_source_module(inspect.getmodule(member), member)
        
#         if inspect.ismodule(member):
#             import_module(member, member_name, imports, packages)
#         else:
#             import_member(member, source_module, imports, packages, locals, entities)


#     return inspect.getsource(func), imports, packages, locals, entities


# def combine_original_code(source_code: str, imports: list[str]) -> str:
#     """Generates a string of code with all the necessary imports

#     Args:
#         source_code (str): code supposed to run
#         imports (list[str]): list of imports used in this code

#     Returns:
#         str: code with all necessary entities imported
#     """
    
#     line_break = '\n'
#     return f'{line_break.join(imports)}{line_break*2}{source_code}'



import dataclasses
import re
import sys
import os
current_dir = os.path.dirname(os.path.abspath('./'))
if not current_dir in sys.path:
    sys.path.append(current_dir)
from cross_runs_TF_planes import CrossRunsTFScorer
import console as csl
import utils.data_management as dm
import inspect
import numpy as np
import numpy as nmp
from scipy import exp, tanh
import time
from utils.console.colored import ColoredText
import utils.console.colored as ucc
from utils.data_management import dict2str
from numpy import ones, arange
import random as r
import itertools as it
from typing import *
import scipy as sp
from scipy.signal import bessel
from numpy import cos
from utils.console.progress_bar import ProgressBar
from utils.structures import Linked



def b(a):
    return a+10

def f1() -> int:
    print(1)
    # Linked('name')
    # e = exp(2)
    # tanh(e)
    # bessel(10)
    # cos(2)
    # r.randint(1, 2)
    # sp.sin(2)
    # np.ones(100)
    # dm.dict2str(dict(a=1, b=2))
    cs = CrossRunsTFScorer()
    # text = ucc.ColoredText().style('r')('text')

def f2(a: int = 2):
    cos(2)
    ProgressBar()
    os.path.join('.', 'Source')
    arange(100)
    ones(100)
    np.zeros(22)
    x = nmp.linespace(0, 100, 1000)
    print('ok')
    sp.sin(2)
    np.ones(100)
    time.time()
    r.randint(1, 2)
    it.combinations([num for num in range(100)])
    return b(a**2)


def protect_source_module(source_module: object, member: object) -> object:
        
        if source_module is None:
            try:
                return sys.modules[re.findall( r'\'.*\'',str(type(member)))[0][1:-1].split('.')[0]]
            except KeyError:
                raise ModuleNotFoundError(f'Can not determine module of {member.__name__}')
        elif '._' in source_module.__name__:
            try:
                return sys.modules[source_module.__name__.split('._')[0]]
            except KeyError:
                raise ModuleNotFoundError(f'Can not determine correct name of {source_module.__name__}')
        else:
            return source_module
            

def get_module_type(member: object) -> str:
    
    module = protect_source_module(inspect.getmodule(member), member)
    if module.__spec__ is None:
        return '__main__'
    elif module.__spec__.origin == 'built-in':
        return 'built-in'
    elif 'site-packages' in module.__spec__.origin:
        return 'pip'
    elif '/lib/' in module.__spec__.origin or '\\lib\\' in module.__spec__.origin:
        return 'python'
    else:
        return 'custom'


def import_member(member: object, source_module: object, imports: list[str], packages: list[str], locals: list[str], entities: list[object]) -> NoReturn:
        member_name = member.__name__
        source_module_name = source_module.__name__ if get_module_type(source_module) not in ['__main__', 'custom'] else 'locals'
        raw_module_name = source_module_name.split('.')[0]
        if source_module_name == 'locals' and member_name not in locals:
            locals.append(member_name)
            entities.append(member)
        elif source_module_name != 'locals' and raw_module_name not in packages:
            packages.append(raw_module_name)
        
        same_source = [i for i, import_ in enumerate(imports) if f'from {source_module_name}' in import_]
        if same_source:
            imports[same_source[0]] += f', {member_name}'
        else:
            imports.append(f'from {source_module_name} import {member_name}')


def import_module(member: object, member_name: str, imports: list[str], packages: list[str]) -> NoReturn:
    
    module_type = get_module_type(member)
    module_name = member.__name__ if module_type not in ['custom', '__main__'] else 'locals'
    raw_module_name = module_name.split('.')[0]
    
    if module_name != 'locals' and module_type == 'pip' and raw_module_name not in packages:
        packages.append(raw_module_name)
    
    if module_name == member_name:
        imports.append(f'import {module_name}')
    else:
        imports.append(f'import {module_name} as {member_name}')


def inspect_function_code(func: Callable) -> tuple[str, list[str], list[str], list[str]]:
    
    imports = list()
    packages = list()
    locals = list()
    entities = list()
    
    for member_name, member in inspect.getclosurevars(func).globals.items():
        source_module = protect_source_module(inspect.getmodule(member), member)
        
        if inspect.ismodule(member):
            import_module(member, member_name, imports, packages)
        else:
            import_member(member, source_module, imports, packages, locals, entities)


    return inspect.getsource(func), imports, packages, locals, entities


def combine_original_code(source_code: str, imports: list[str]) -> str:
    
    line_break = '\n'
    return f'{line_break.join(imports)}{line_break*2}{source_code}'

def find_all_internal_functions(entities: list[object]) -> list[object]:
    callables = list()
    for entity in entities:
        if inspect.isfunction(entity):
            callables.append(entity)
        elif inspect.isclass(entity):
            for _, member in inspect.getmembers(entity):
                if inspect.isfunction(member):
                    callables.append(member)
                elif inspect.isdatadescriptor(member):
                    for _, property in inspect.getmembers(member):
                        if inspect.isfunction(property):
                            callables.append(property)
    return callables


def concatenate_code(*args: str, n_lines: Optional[int] = 2) -> str:
    line_break = '\n'*n_lines
    return line_break.join(args)


def concatenate_members(*args: list[Any]) -> list[Any]:
    
    def uniq(lst):
        last = object()
        for item in lst:
            if item == last:
                continue
            yield item
            last = item
    
    all_members = list()
    
    for arg in args:
        all_members += arg
    
    return list(uniq(all_members))


def concatenate_imports(*args: list[str]) -> list[str]:
    raw_imports = concatenate_members(*[
        list(filter(lambda item: 'from' not in item, arg))
        for arg in args
    ])
    
    complex_imports = [
        list(filter(lambda item: 'from' in item, arg))
        for arg in args
    ]
    
    complex_imports = concatenate_members(*[
        list(map(lambda item: list(item[5:].split(' import ')), arg))
        for arg in complex_imports
    ])
    
    complex_imports = list(map(
        lambda item: [item[0], item[1].split(', ')],
        complex_imports
    ))
    
    for i, ci1 in enumerate(complex_imports):
        package1, import1 = ci1
        if import1 is None:
            continue
        for j, ci2 in enumerate(complex_imports):
            package2, import2 = ci2
            if package1 == package2 and i != j:
                complex_imports[i][1] = concatenate_members(import1, import2)
                complex_imports[j][1] = None
    
    complex_imports = list(filter(lambda item: bool(item[1]), complex_imports))
    complex_imports = list(map(lambda item: f'from {item[0]} import {", ".join(item[1])}', complex_imports))
    return concatenate_members(raw_imports, complex_imports)


def full_inspection(source_code: str, imports: list[str], packages: list[str], locals: list[str], entities: list[object]):
    for entity in find_all_internal_functions(entities):
        new_source_code, new_imports, new_packages, new_locals, new_entities = inspect_function_code(entity)
        
        if new_entities:
            new_source_code, new_imports, new_packages, new_locals, new_entities = full_inspection(new_source_code, new_imports, new_packages, new_locals, new_entities)
        
        source_code = concatenate_code(source_code, new_source_code)
        imports = concatenate_imports(imports, new_imports)
        packages = concatenate_members(packages, new_packages)
        locals = concatenate_members(locals, new_locals)
        entities = concatenate_members(entities, new_entities)
    
    return source_code, imports, packages, locals, entities



scode, imports, packages, locals, entities = inspect_function_code(f1)

new_source_code, new_imports, new_packages, new_locals, new_entities = full_inspection('', [], [], [], entities)
print(new_source_code)
# for entity in entities:
#     print(entity)
    
# if __name__ == "__main__":
#     scode, imports, packages, localh(e)
    
#     for package_name in packages:
#         print(package_name)

#     print('#'*20)
#     for local in locals:
#         print(local)
    
    
#     print('#'*20)
    
            # print(inspect.getsource(entity))
    
    # print('#'*20)
    # code = combine_original_code(scode, imports)
    # print(code)
