from typing import Optional, Union, Any,\
    Callable, List, Iterable, Dict, Tuple,\
    NoReturn
from collections import UserDict, UserList, namedtuple
from utils.console.colored import alarm, ColoredText


class Linked(object):
    def __init__(
            self,
            name: Optional[str] = 'Linked',
            parent: Optional[Any] = None,
            meta: Optional[dict] = None,
            options: Optional[Iterable] = None,
            child_options_generator: Optional[Union[Callable, List[Callable]]] = None,
            children: Optional[Union[Any, List[Any]]] = None
    ):
        self.name = name
        self.__siblings = None
        self.parent = parent
        if not isinstance(children, list) and issubclass(type(children), Linked):
            self.children = [children]
        elif isinstance(children, list) or children is None:
            self.children = children
        else:
            raise AttributeError(
                'Children must be either subclass of '
                f'Linked or list of it {issubclass(type(children), Linked)}'
            )

        if self.children is not None:
            if issubclass(type(self.children), Linked):
                self.children._parent = self
            elif isinstance(self.children, list):
                for child in self.children:
                    child._parent = self

        if options is None or isinstance(options, Iterable):
            self._options = options
        else:
            raise AttributeError('Options must be an iterable object')
        self.child_options_generator = child_options_generator
        self.__selected_option = None
        self.meta = dict() if not isinstance(meta, dict) else meta

    def __call__(
        self,
        options: Optional[List[Any]] = None,
        option_index_to_choose: Optional[int] = 0
    ):

        if options is not None and options != [] and options != ():
            self._options = options
            self.__selected_option = self.options[option_index_to_choose]

        if self.children is not None and self.child_options_generator is not None:

            if issubclass(
                type(self.children),
                Linked
            ) and isinstance(
                self.child_options_generator,
                Callable
            ):
                self.children(self.child_options_generator(self))

            elif isinstance(self.children, list) and isinstance(
                self.child_options_generator,
                Callable
            ):

                for child in self.children:
                    child(self.child_options_generator(self))

            elif isinstance(self.child_options_generator, list):

                if isinstance(self.children, list):

                    if len(self.children) > len(self.child_options_generator):
                        gens = self.child_options_generator + \
                            [
                                self.child_options_generator[-1]
                                for _ in range(
                                    len(
                                        self.children
                                    ) - len(
                                        self.child_options_generator
                                    )
                                )
                            ]
                        children = self.children

                    else:
                        gens = self.child_options_generator
                        children = self.children

                elif isinstance(self.children, Linked):
                    gens = self.child_options_generator
                    children = [self.children]

                else:
                    raise ValueError(f'Children can not belong to this type: {type(self.children)}')

                for child, gen in zip(children, gens):
                    child(gen(self))

    def __iter__(self):
        return iter([self] + self.inverse_dependencies())

    def __str__(self):
        return f'{self.name}'

    def __getitem__(self, i):
        chain = [self] + list(self.inverse_dependencies())
        return chain[i]

    @property
    def parent(self):
        return self._parent

    @parent.setter
    def parent(self, parent):
        if parent is not None and not issubclass(type(parent), Linked):
            raise AttributeError(
                'Parent of Linked must be Linked '
                f'or subclass of it, but it is: {type(parent)}'
            )
        self._parent = parent
        if self._parent is not None:
            self._parent.add_children(self)

    @property
    def children(self):
        return self._children

    @children.setter
    def children(self, children):
        valid_children = issubclass(type(children), Linked)

        if isinstance(children, list):
            valid_children = True
            for child in children:

                if not issubclass(type(child), Linked):
                    valid_children = False
                    break

        if children is not None and not valid_children:
            raise AttributeError(
                f'Children of Linked must be list of Linked or Linked or subclass of it, '
                f'but it is: {type(children)}'
            )

        self._children = children
        self.__introduce_children()

    @property
    def siblings(self):
        return self.__siblings

    @siblings.setter
    def siblings(self, value):
        raise AttributeError('Siblings of Linked cannot be set')

    @property
    def selected_option(self):
        return self.__selected_option

    @selected_option.setter
    def selected_option(self, value):
        raise AttributeError('Options can be selected only via \'select\' method')

    @property
    def options(self):
        return self._options

    @options.setter
    def options(self, options: Iterable):
        if self.parent is None:
            self._options = options
        else:
            raise AttributeError(
                'Can not set options to Linked with dependencies. '
                f'This Linked depends on: {self.dependencies()}'
            )

    def __introduce_children(self):
        if isinstance(self.children, list):
            for child in self.children:
                child.__siblings = [sib for sib in self.children if sib != child]

    def add_children(self, children):
        if not isinstance(self.children, list):
            self.children = [self.children] if self.children is not None else []
        if issubclass(type(children), Linked):
            self.children.append(children)
        elif isinstance(children, list):
            for child in children:
                if not issubclass(child, Linked):
                    raise AttributeError(
                        f'All the children must be Linked, but {type(child)} found '
                        f'in the list of given children'
                    )
            self.children += children
        else:
            raise AttributeError(f'All the children must be Linked, but {type(children)} is given')
        self.__introduce_children()

    def remove(self):
        self.parent.children = [child for child in self.parent.children if child != self]
        for dep in self.inverse_dependencies():
            del dep
        del self

    def select(
        self,
        option,
        index: Optional[bool] = False,
        child_option_index_to_choose: Optional[int] = 0
    ):
        if index:
            self.__selected_option = self.options[option]
        else:
            self.__selected_option = option
        self(option_index_to_choose=child_option_index_to_choose)

    def inverse_dependencies(self):
        if self.children is None:
            return []
        elif isinstance(self.children, list) and self.children:
            invdep = [child for child in self.children]
            for child in invdep:
                # Add only children which are not added already
                invdep += [
                    dep for dep in
                    child.inverse_dependencies()
                    if dep not in invdep
                ]
            return invdep
        elif issubclass(type(self.children), Linked):
            return [self.children]

    def dependencies(self):
        deps = list()
        dep = self.parent
        while dep is not None:
            deps.append(dep)
            dep = dep.parent
        return tuple(deps)


class Deploy(object):
    def __init__(self, func: Callable, *args: Any, **kwargs: Any):
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def __call__(self, *args, **kwargs):
        given_kwargs = self.kwargs.copy()
        given_kwargs.update(kwargs)
        try:
            return self.func(*args, *self.args, **given_kwargs)
        except Exception as e:
            alarm(
                ColoredText().color('r').style('b')('Exception: ') + f'{e}'
            )


class Pipeline(object):
    def __init__(self, *args: Union[Callable, Deploy]):
        self._run_flow = args

    def __call__(
        self,
        *args,
        kwargs: Optional[Union[Dict[str, Any], Tuple[Dict[str, Any]]]] = None
    ):
        if kwargs is None:
            kwargs = dict()
        if isinstance(kwargs, dict):
            kwargs = [kwargs for _ in range(len(self.run_flow))]
        elif isinstance(kwargs, list) or isinstance(kwargs, tuple):
            if len(kwargs) != len(self.run_flow):
                raise ValueError(
                    'Number of the given tuples of keyword arguments is '
                    'different from length of callables in the runflow:\n'
                    f'kwargs: {len(kwargs)}, runflow: {len(self.run_flow)}'
                )
        else:
            raise ValueError(
                'The "kwargs" argument must be '
                f'either a tuple or a dictionary, {type(kwargs)} is given'
            )

        out = self.run_flow[0](*args, **kwargs[0])
        for step, kws in zip(self._run_flow[1:], kwargs[1:]):
            out = step(out, **kws)
        return out

    def __getitem__(self, i):
        return self.run_flow[i]

    def __iter__(self):
        return iter(self.run_flow)

    @property
    def run_flow(self):
        return self._run_flow

    @run_flow.setter
    def run_flow(self, steps):
        if not isinstance(steps, Iterable)\
            and not isinstance(steps, Callable)\
                and not isinstance(steps, Deploy):
            raise AttributeError('The run_flow must be a container for callable')
        elif isinstance(steps, Iterable) and any([not isinstance(el, Callable) for el in steps]):
            raise AttributeError('All the run_flow elements must be callable')
        elif isinstance(steps, Callable) or isinstance(steps, Deploy):
            self._run_flow = [steps]
        else:
            self._run_flow = steps

    def append(self, *steps):
        if any([not isinstance(el, Callable) for el in steps]):
            raise AttributeError('All elements to append must be callable')
        else:
            self.run_flow += steps


class NumberedDict(UserDict):
    def __getitem__(self, key):
        if isinstance(key, int) and key not in self.data:
            key = list(self.data.keys())[key]

        return super().__getitem__(key)

    def __add__(self, item):

        if issubclass(type(item), (list, tuple, UserList)):

            for val in item:
                self.append(val)

        elif issubclass(type(item), (dict, UserDict)):

            for key, val in item.items():
                self.data[key] = val

        return self

    def keys(self):
        return self.data.keys()

    def values(self):
        return self.data.values()

    def append(self, *args, **kwargs):
        if args and kwargs:
            raise TypeError(
                'append() takes either positional or '
                'keyword arguments but both were given'
            )
        elif args:
            if len(args) == 1:
                self.data[len(self.data)] = args[0]
            elif len(args) == 2:
                self.data[args[0]] = args[1]
            else:
                raise TypeError(
                    'append() takes 1 or 2 positional '
                    f'arguments but {len(args)} were given'
                )
        elif kwargs:
            self.data.update(kwargs)
        else:
            raise TypeError('append() missing any arguments')


class Expandable(object):
    def __init__(self):
        self.__data = dict()
        self.__field_properties = namedtuple('FieldPropertes', 'readonly writable')(list(), list())

    def __setattr__(self, name: str, value: Union[Any, tuple[Any, str]]):

        if hasattr(self, '_Expandable__field_properties'):

            value, mode = self.__check_item(name, value)

            if mode == 'writable':

                if name in self.__field_properties.readonly:
                    self.__field_properties.readonly.remove(name)

                if name not in self.__field_properties.writable:
                    self.__field_properties.writable.append(name)

            elif mode == 'readonly':

                if name in self.__field_properties.writable:
                    self.__field_properties.writable.remove(name)

                if name not in self.__field_properties.readonly:
                    self.__field_properties.readonly.append(name)

            self.__data[name] = value

        if self.__is_allowed_name(name):
            self.__dict__[name] = value

    def __getitem__(self, key: Any) -> Any:
        return self.__data[key]

    def __setitem__(self, key: Any, item: Any) -> NoReturn:

        self.__setattr__(key, item)

    def __contains__(self, item: Any) -> bool:
        return item in self.__data

    def __iter__(self):
        return iter(self.__data)

    @staticmethod
    def __is_allowed_name(name: Any) -> bool:

        if not isinstance(name, str):
            name = str(name)

        for i, char in enumerate(name):

            if i == 0 and not char.isalpha() and not char == '_':
                return False
            elif not char.isalpha() and not char.isdigit() and not char == '_':
                return False

        return True

    def __check_item(self, name: Any, value: Any) -> tuple[Any, str]:
        deployed = False

        if isinstance(value, tuple):
            value, mode = value

            if mode in self.__field_properties._fields:
                deployed = True
            else:
                value = (value, mode)

        if not deployed:

            if name in self.__field_properties.readonly:
                raise AttributeError(
                    'Impossible to set a new value '
                    f'for a read-only field "{name}"'
                )

            mode = 'writable'

        return value, mode

    def keys(self, mode: Optional[str] = 'all') -> list[Any]:

        if mode == 'all':
            return list(self.__data.keys())
        elif mode == 'writable':
            return self.__field_properties.writable
        elif mode == 'readonly':
            return self.__field_properties.readonly

    def values(self, mode: Optional[str] = 'all') -> list[Any]:

        if mode == 'all':
            return list(self.__data.values())
        elif mode == 'writable':
            return [self.__data[key] for key in self.__field_properties.writable]
        elif mode == 'readonly':
            return [self.__data[key] for key in self.__field_properties.readonly]

    def items(self, mode: Optional[str] = 'all') -> list[tuple[Any, Any]]:

        return list(zip(self.keys(mode), self.values(mode)))

    def is_writable(self, key: Any) -> bool:
        return key in self.__field_properties.writable

    def is_readonly(self, key: Any) -> bool:
        return key in self.__field_properties.readonly
