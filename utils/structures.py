from typing import Optional, Union, Any,\
    Callable, List, Iterable, Dict, Tuple,\
    NoReturn, Hashable
from collections import UserDict, UserList, namedtuple
from utils.console.colored import alarm, ColoredText


class Linked(object):
    """A linked object that can form interacting chains

    Args:
        name (:obj:`str`, optional): Name of this Linked object.
            Used as a string representation. Defaults to 'Linked'.
        parent (:obj:`Any`, optional): Linked instanse playing role of
            a \"parent\" to the current Linked. The current Linked supposed to be depending
            on its parent and observing its changes. Defaults to None.
        meta (:obj:`dict`, optional): Any meta-information that can be related to this Linked.
            Defaults to None.
        options (:obj:`Iterable`, optional): Content of Linked object. Defaults to None.
        child_options_generator (:obj:`Callable` or :obj:`list` of :obj:`Callable`, optional):
            Function or list of functions to generate new options for children of
            the current Linked. Each children options generator must take current Linked as
            a single argument and return list of options. If list of functions is given,
            enerates different oprions for different children according to an order of given
            generators and an order of children. If number of children is bigger than number
            of options generators, then applying last generator in sequence for every child
            who does not have one. If number of options generators is bigger than number of
            children, then the extra ones will be ignored. Defaults to None.
        children (:obj:`Any` or :obj:`list` of :obj:`Any`, optional):
            Child or list of children of the current Linked. Defaults to None.

    Raises:
        AttributeError:
            * If children are not Linked or subclass of it
            * If the given options are not iterable

    Note:
        Children are Linked objects, depending to the current one by a way defined in
        options generators. Parent is an Linked object which this object depends on.

    """
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
        """Takes new options, produces new options for all children,
            calls every children.

        Args:
            options (:obj:`Any`, optional): New content for this Linked. Defaults to None.
            option_index_to_choose (Optional[int], optional): Which one of options to select.
                By default, selects the first one. Defaults to 0.

        Raises:
            ValueError: If children are not Linked or subclass of it
        """
        # chnames =  [child.name for child in self.children] if self.children is not None else None
        # print(self.name, chnames)
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
        """Iterates a current Linked and all its descendants.

        Note:
            Descendants are all chilren and all children's children.
        """
        return iter([self] + self.inverse_dependencies())

    def __str__(self):
        return f'{self.name}'

    def __getitem__(self, i):
        """Returns ith  element of list of
            the current :class:`Linked` and all its descendants.
        """
        chain = [self] + list(self.inverse_dependencies())
        return chain[i]

    @property
    def parent(self):
        """:class:`Linked`: Parent of a current :class:`Linked`.

        Raises:
            AttributeError: If parent is not Linked or subclass of it.
        """
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
        """:class:`Linked` or :obj:`list` of :class:`Linked`:
            Children of a current :class:`Linked`.

        Raises:
            AttributeError: If children are not Linked or subclasses of it.
        """
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
        """:class:`Linked` or :obj:`list` of :class:`Linked`:
            Siblings of a current :class:`Linked`.

        Note:
            Siblings are all :class:`Linked` that have a common parent
        """
        return self.__siblings

    @siblings.setter
    def siblings(self, value):
        raise AttributeError('Siblings of Linked cannot be set')

    @property
    def selected_option(self):
        """:obj:`Any`: A currently selected option
        """
        return self.__selected_option

    @selected_option.setter
    def selected_option(self, value):
        raise AttributeError('Options can be selected only via \'select\' method')

    @property
    def options(self):
        """:obj:`list` of :obj:`Any`: Options of a current :class:`Linked`.
        """
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
        """Adds :class:`Linked` to the list of children of the current :class:`Linked`.

        Args:
            children (:class:`Linked` or :obj:`list` of :class:`Linked`): Child
                or the list of children to add

        Raises:
            AttributeError:
                * If the existing children are not Linked or subclasses of it.
                * If a given children are not Linked or subclasses of it.
        """

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
        """Removes the current :class:`Linked` and all its children
        """

        self.parent.children = [child for child in self.parent.children if child != self]
        for dep in self.inverse_dependencies():
            del dep
        del self

    def select(
        self,
        option: Union[int, Any],
        index: Optional[bool] = False,
    ):
        """Selects an option

        Args:
            option (:obj:`int` or :obj:`Any`): option to choose or an index of option to choose
            index (Optional[bool], optional): If False, to set option derectly,
                otherwise to set option under the given index. Defaults to False.
        """
        if index:
            self.__selected_option = self.options[option]
        else:
            self.__selected_option = option

        self()

    def inverse_dependencies(self):
        """Compute inverse dependencies to the current :class:`Linked`

        Returns:
            :obj:`list` of :class:`Linked`: List of descendants of the current :class:`Linked`
        """
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
        """Compute dependencies to the current :class:`Linked`

        Returns:
            :obj:`list` of :class:`Linked`: List of ancestors of the current :class:`Linked`
        """
        deps = list()
        dep = self.parent
        while dep is not None:
            deps.append(dep)
            dep = dep.parent
        return tuple(deps)


class Deploy(object):
    """Class to wrap a given function and its arguments
        to call it without arguments

    Args:
        func (Callable): Function to wrap
        *args: Arguments of a given function
        **kwargs: Keyword arguments of a given function
    """
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
    """Class to execute a sequence of Callable

    Args:
        *args (:obj:`Callable` or :class: `Deploy`): Sequence of callable objects to execute
    """
    def __init__(self, *args: Union[Callable, Deploy]):
        self._run_flow = args

    def __call__(
        self,
        *args,
        kwargs: Optional[Union[Dict[str, Any], Tuple[Dict[str, Any]]]] = None
    ):
        """_summary_

        Args:
            kwargs (:obj:`dict` of :obj:`str` and obj:`Any`
                or :obj:`tuple` of :obj:`dict` of :obj:`str` and :obj:`Any`, optional):
                Keyword arguments to each function of a pipline. Defaults to None.

        Raises:
            ValueError:
                * If kwargs are not :obj:`Iterable` of :obj:Callable
                * If kwargs are not given to all pipeline members

        Returns:
            Any: Output of a pipeline
        """
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
        """:obj:`list` of :obj:`Callanle`: execution queue of a pipeline
        """
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
        """Adds :obj:`Callable` members to a pipeline

        Raises:
            AttributeError: If any of the given objects is not callable
        """
        if any([not isinstance(el, Callable) for el in steps]):
            raise AttributeError('All elements to append must be callable')
        else:
            self.run_flow += steps


class NumberedDict(UserDict):
    """A dictionary which comprises some :obj:`list` properties
    """
    def __getitem__(self, key: Union[int, Hashable]):
        """Returns item by key

        Args:
            key (:obj:`int` or :obj:`Hashable`): If hashable is given, key plays a role
                of a dictionary key, if int is goven, key plays a role of index

        Returns:
            Any: Content corresponding to the given key
        """
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
        """appends elements to the end of :class:`NumberedDict`

        Args:
            *args: If one argument is given, appends it at the kay equal to length of
                :class:`NumberedDict`. If two arguments are given, the second one
                plays a role of a key
            **kwargs: Keyword arguments are used to update the dictionary

        Raises:
            TypeError:
                * If both args and kwargs are given
                * If more than two arguments are given in args
                * If no any arguments are given
        """
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
    """A dictionary-like class that allows to refer its fields via
        both square brackets and dot
    """
    def __init__(self):
        self.__data = dict()
        self.__field_properties = namedtuple('FieldPropertes', 'readonly writable')(list(), list())

    def __setattr__(self, name: str, value: Union[Any, tuple[Any, str]]):
        """Sets attributes

        Args:
            name (str): Key and field name used to access new value
            value (:obj:`Any` or :obj:`tuple` of obj:`Any` and obj:`str`):
                Any value to set under the given name. If tuple of length 2,
                the second value is considered as a mode name.

        Raises:
            AttributeError: When trying to change a read-only field.

        Note:
            There are two types of modes for values:
                * writable - allows both to write and read the field
                * readonly - prevents field from changes from outside
        """

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
        """Returns existing keys / field-names

        Args:
            mode (str, optional): Which type of keys to return. Defaults to 'all'.

        Returns:
            list[Any]: :obj:`list` of keys

        Note:
            There are three types of keys:
            * 'all' - all keys in the object
            * 'readonly' - keys only from read-only fields
            * 'writable' - keys only from writable fields
        """

        if mode == 'all':
            return list(self.__data.keys())
        elif mode == 'writable':
            return self.__field_properties.writable
        elif mode == 'readonly':
            return self.__field_properties.readonly

    def values(self, mode: Optional[str] = 'all') -> list[Any]:
        """Returns existing values

        Args:
            mode (str, optional): Which type of values to return. Defaults to 'all'.

        Returns:
            list[Any]: :obj:`list` of values
        """

        if mode == 'all':
            return list(self.__data.values())
        elif mode == 'writable':
            return [self.__data[key] for key in self.__field_properties.writable]
        elif mode == 'readonly':
            return [self.__data[key] for key in self.__field_properties.readonly]

    def items(self, mode: Optional[str] = 'all') -> list[tuple[Any, Any]]:
        """Returns pairs of existing keys and values

        Args:
            mode (str, optional): Which type of items to return. Defaults to 'all'.

        Returns:
            :obj:`list` of :obj:`tuple` of :obj:`Any` and :obj:`Any`: :obj:`list` of values
        """

        return list(zip(self.keys(mode), self.values(mode)))

    def is_writable(self, key: Any) -> bool:
        """Checks if key corresponds to a writable field

        Args:
            key (Any): Key to check

        Returns:
            bool: True, if the given field is writable
        """
        return key in self.__field_properties.writable

    def is_readonly(self, key: Any) -> bool:
        """Checks if key corresponds to a readonly field

        Args:
            key (Any): Key to check

        Returns:
            bool: True, if the given field is readonly
        """
        return key in self.__field_properties.readonly
