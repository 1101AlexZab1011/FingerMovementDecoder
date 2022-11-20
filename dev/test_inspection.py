import _thread
import numpy as np
import re
from typing import Callable, Iterable, Any
from dataclasses import dataclass


@dataclass
class CrossRunsTFScorer():
    tf_scores: Any
    accuracy_cache: Any
    csp: Any

    def mean(self):
        return np.mean(self.tf_scores, axis=0)

    def std(self):
        return np.std(self.tf_scores, axis=0)

    def tf_windows_mean(self):
        return {freq: {time: np.mean(np.array(self.accuracy_cache[freq][time])) for time in self.accuracy_cache[freq]} for freq in self.accuracy_cache}

    def tf_windows_std(self):
        return {freq: {time: np.std(np.array(self.accuracy_cache[freq][time])) for time in self.accuracy_cache[freq]} for freq in self.accuracy_cache}

class Linked(object):

    def __init__(self, name='Linked', parent=None, meta=None, options=None, child_options_generator=None, children=None):
        self.name = name
        self.__siblings = None
        self.parent = parent
        if ((not isinstance(children, list)) and issubclass(type(children), Linked)):
            self.children = [children]
        elif (isinstance(children, list) or (children is None)):
            self.children = children
        else:
            raise AttributeError(f'Children must be either subclass of Linked or list of it {issubclass(type(children), Linked)}')
        if (self.children is not None):
            if issubclass(type(self.children), Linked):
                self.children._parent = self
            elif isinstance(self.children, list):
                for child in self.children:
                    child._parent = self
        if ((options is None) or isinstance(options, Iterable)):
            self._options = options
        else:
            raise AttributeError('Options must be an iterable object')
        self.child_options_generator = child_options_generator
        self.__selected_option = None
        self.meta = (dict() if (not isinstance(meta, dict)) else meta)

    def __call__(self, options=None, option_index_to_choose=0):
        if ((options is not None) and (options != []) and (options != ())):
            self._options = options
            self.__selected_option = self.options[option_index_to_choose]
        if ((self.children is not None) and (self.child_options_generator is not None)):
            if (issubclass(type(self.children), Linked) and isinstance(self.child_options_generator, Callable)):
                self.children(self.child_options_generator(self))
            elif (isinstance(self.children, list) and isinstance(self.child_options_generator, Callable)):
                for child in self.children:
                    child(self.child_options_generator(self))
            elif isinstance(self.child_options_generator, list):
                if isinstance(self.children, list):
                    if (len(self.children) > len(self.child_options_generator)):
                        gens = (self.child_options_generator + [self.child_options_generator[(- 1)] for _ in range((len(self.children) - len(self.child_options_generator)))])
                        children = self.children
                    else:
                        gens = self.child_options_generator
                        children = self.children
                elif isinstance(self.children, Linked):
                    gens = self.child_options_generator
                    children = [self.children]
                else:
                    raise ValueError(f'Children can not belong to this type: {type(self.children)}')
                for (child, gen) in zip(children, gens):
                    child(gen(self))

    def __iter__(self):
        return iter(([self] + self.inverse_dependencies()))

    def __str__(self):
        return f'{self.name}'

    def __getitem__(self, i):
        chain = ([self] + list(self.inverse_dependencies()))
        return chain[i]

    @property
    def parent(self):
        return self._parent

    @parent.setter
    def parent(self, parent):
        if ((parent is not None) and (not issubclass(type(parent), Linked))):
            raise AttributeError(f'Parent of Linked must be Linked or subclass of it, but it is: {type(parent)}')
        self._parent = parent
        if (self._parent is not None):
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
                if (not issubclass(type(child), Linked)):
                    valid_children = False
                    break
        if ((children is not None) and (not valid_children)):
            raise AttributeError(f'Children of Linked must be list of Linked or Linked or subclass of it, but it is: {type(children)}')
        self._children = children
        self.__introduce_children()

    @property
    def siblings(self):
        return self.__siblings

    @siblings.setter
    def siblings(self, value):
        raise AttributeError(f'Siblings of Linked cannot be set')

    @property
    def selected_option(self):
        return self.__selected_option

    @selected_option.setter
    def selected_option(self, value):
        raise AttributeError("Options can be selected only via 'select' method")

    @property
    def options(self):
        return self._options

    @options.setter
    def options(self, options):
        if (self.parent is None):
            self._options = options
        else:
            raise AttributeError(f'Can not set options to Linked with dependencies. This Linked depends on: {self.dependencies()}')

    def __introduce_children(self):
        if isinstance(self.children, list):
            for child in self.children:
                child.__siblings = [sib for sib in self.children if (sib != child)]

    def add_children(self, children):
        if (not isinstance(self.children, list)):
            self.children = ([self.children] if (self.children is not None) else [])
        if issubclass(type(children), Linked):
            self.children.append(children)
        elif isinstance(children, list):
            for child in children:
                if (not issubclass(child, Linked)):
                    raise AttributeError(f'All the children must be Linked, but {type(child)} found in the list of given children')
            self.children += children
        else:
            raise AttributeError(f'All the children must be Linked, but {type(children)} is given')
        self.__introduce_children()

    def remove(self):
        self.parent.children = [child for child in self.parent.children if (child != self)]
        for dep in self.inverse_dependencies():
            del dep
        del self

    def select(self, option, index=False, child_option_index_to_choose=0):
        if index:
            self.__selected_option = self.options[option]
        else:
            self.__selected_option = option
        self(option_index_to_choose=child_option_index_to_choose)

    def inverse_dependencies(self):
        if (self.children is None):
            return []
        elif (isinstance(self.children, list) and self.children):
            invdep = [child for child in self.children]
            for child in invdep:
                invdep += [dep for dep in child.inverse_dependencies() if (dep not in invdep)]
            return invdep
        elif issubclass(type(self.children), Linked):
            return [self.children]

    def dependencies(self):
        deps = list()
        dep = self.parent
        while (dep is not None):
            deps.append(dep)
            dep = dep.parent
        return tuple(deps)

def dict2str(dictionary, *, space=1, tabulation=2, first_space=True):
    tab = (lambda tabulation: (' ' * tabulation))
    string = f'''{(tab(tabulation) if first_space else '')}{{
'''
    tabulation += space
    for (key, value) in zip(dictionary.keys(), dictionary.values()):
        if (not isinstance(key, str)):
            key = f'{key}'
        if ('\n' in key):
            key = key.replace('\n', '')
        string += f'{tab(tabulation)}{key}: '
        if (not isinstance(value, dict)):
            string += f'''{value},
'''
        else:
            string += dict2str(value, space=space, tabulation=(tabulation + space), first_space=False)
    string += f'''{tab((tabulation - space))}}}
'''
    return string

def clean_styles(styled_text):

    def find_sublist(sublist, in_list):
        sublist_length = len(sublist)
        for i in range((len(in_list) - sublist_length)):
            if (sublist == in_list[i:(i + sublist_length)]):
                return (i, (i + sublist_length))
        return None

    def remove_sublist_from_list(in_list, sublist):
        indices = find_sublist(sublist, in_list)
        if (not (indices is None)):
            return (in_list[0:indices[0]] + in_list[indices[1]:])
        else:
            return in_list
    pure_text = str(styled_text.encode('ascii'))
    found_styles = re.findall('\\\\x1b\\[[\\d*;]*m', pure_text)
    clean_text = [char for char in pure_text]
    for style in found_styles:
        style = [char for char in style]
        clean_text = remove_sublist_from_list(clean_text, style)
    out = ''.join(clean_text[2:(- 1)])
    return (out if (out != '\\\\') else '\\')

def bold(msg, **kwargs):
    print(ColoredText().color().style('b')(msg), **kwargs)


class ColoredText(object):

    def __init__(self):
        self.styles: list = [30]
        self.__current_style: int = 0
        self._text = ''

    def __call__(self, text):
        code = ';'.join([str(style) for style in self.styles])
        self._text = text
        return f'[{code}m{text}[0m'

    @property
    def text(self):
        return self._text

    @text.setter
    def text(self, value):
        raise AttributeError('Text can not be set directly')

    def color(self, color_name='normal'):
        self.styles[self.__current_style] = 30
        self.styles[self.__current_style] += {'black': 0, 'red': 1, 'r': 1, 'green': 2, 'g': 2, 'yellow': 3, 'y': 3, 'blue': 4, 'b': 4, 'violet': 5, 'v': 5, 'cyan': 6, 'c': 6, 'grey': 7, 'white': 7, 'normal': 8}[color_name]
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

    def style(self, style_name):
        self.styles = ([{'bold': 1, 'b': 1, 'italic': 3, 'i': 3, 'underline': 4, 'u': 4, 'reverse': 7, 'r': 7}[style_name]] + self.styles)
        self.__current_style += 1
        return self

####################################################################################################

