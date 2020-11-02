import functools
from contextlib import contextmanager
import inspect
import numpy as np
import types
import numbers
import random
import string
import os
import re
import sys
import pickle
import matplotlib.pyplot as plt
from functools import total_ordering
import dateutil


def random_string(N=16):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=N))


def in_dict_and_has_value(key, dictionary, value, compare_func=None):
    if key in dictionary:
        if compare_func is not None:
            return compare_func(dictionary[key], value)
        else:
            return dictionary[key] == value
    else:
        return False


def in_dict_and_not_none(key, dictionary):
    return in_dict_and_has_value(key=key,
                                 dictionary=dictionary,
                                 value=None,
                                 compare_func=lambda val1, val2: val1 is not val2)


def in_dict_and_true(key, dictionary):
    return in_dict_and_has_value(key=key,
                                 dictionary=dictionary,
                                 value=True,
                                 compare_func=lambda val1, val2: val1 is True)


def add_singleton(singleton_object_name="_singleton",
                  singleton_subclass_name="Singleton",
                  check_function=None,
                  check_function_name="check_function"):
    """
    Class decorator to add a singleton object
    :param singleton_object_name:
    :param singleton_subclass_name:
    :param check_function:
    :param check_function_name:
    :return:
    """
    def decorator(Cls):
        # helper method to add static method to class
        def add_static(cls, func_name, func):
            setattr(cls, func_name, func)
            setattr(cls, func_name, staticmethod(getattr(cls, func_name)))
        # add singleton attribute to class
        setattr(Cls, singleton_object_name, None)
        # add singleton subclass
        class SingletonTMP:
            pass
        setattr(SingletonTMP, "__name__", singleton_subclass_name)
        setattr(Cls, singleton_subclass_name, SingletonTMP)
        singleton_subclass = getattr(Cls, singleton_subclass_name)
        # add check function
        if check_function is None:
            def _check_function():
                if not getattr(Cls, singleton_object_name, False):
                    setattr(Cls, singleton_object_name, Cls())
        else:
            _check_function = check_function
        add_static(singleton_subclass, check_function_name, _check_function)
        # add static methods to singleton subclass that first check and
        # then call respective function on singleton object
        for name, method in inspect.getmembers(Cls, predicate=inspect.isfunction):
            def f(*args, **kwargs):
                getattr(singleton_subclass, check_function_name)()
                return method(getattr(Cls, singleton_object_name), *args, **kwargs)
            add_static(singleton_subclass, name, f)
        return Cls
    return decorator


def create_singleton_class(Cls,
                           singleton_object_name="_singleton",
                           singleton_class_name="Singleton"):
    # helper method to add static method to class
    def add_static(cls, func_name, func):
        setattr(cls, func_name, func)
        setattr(cls, func_name, staticmethod(getattr(cls, func_name)))
    # create singleton class
    class SingletonTMP:
        pass
    # add singleton object
    setattr(SingletonTMP, singleton_object_name, Cls())
    singleton_object = getattr(SingletonTMP, singleton_object_name)
    # add static methods for every bound method in singleton object
    for name, method in inspect.getmembers(singleton_object, predicate=inspect.ismethod):
        def f(*args, **kwargs):
            return method(*args, **kwargs)
        add_static(SingletonTMP, name, f)
    setattr(SingletonTMP, "__name__", singleton_class_name)
    return SingletonTMP


class MetaIndex(type):
    """
    Meta class for Index class.
    """
    def __getitem__(self, item):
        return item


class Index(object, metaclass=MetaIndex):
    """
    A helper class to create index objects using conventional syntax by returning the index object.
    The Index class can be used as follows:

        from pyulib import Index

        l = [1, 2, 3, 4, 5, 6]
        i = Index[1:6:2]
        assert(l[i] == [2, 4, 6])

    """
    pass


class NestedOutput:
    """Class to handle nested output with indentation"""

    def __init__(self,
                 print_func=None,
                 indent_func=None,
                 debug_func=None,
                 same_line_func=None,
                 prepend_debug=False,
                 prepend_debug_level=1):
        self.indent_list = []
        self.active_list = []
        self.same_line_list = []
        # print_func
        if print_func is None:
            self.print_func = lambda *args, same_line=False, **kwargs: print(*args,
                                                                             **kwargs,
                                                                             **{'end': '',
                                                                                'flush': True} if same_line else {})
        else:
            self.print_func = print_func
        # indent_func
        if indent_func is None:
            self.indent_func = lambda *args, **kwargs: print(*args, **kwargs, end='')
        else:
            self.indent_func = print_func
        # debug_func
        if debug_func is None:
            self.debug_func = lambda *args, **kwargs: print(*args, **kwargs, end='')
        else:
            self.debug_func = debug_func
        # same_line_func
        if same_line_func is None:
            self.same_line_func = lambda: print('\r', end='')
        else:
            self.same_line_func = same_line_func
        # prepend debug variables
        self._prepend_debug = prepend_debug
        self._prepend_debug_level = prepend_debug_level

    def open(self, indent="│  ", active=True, same_line=False):
        """open new output context with given indentation"""
        self.indent_list.append(indent)
        self.active_list.append(active)
        self.same_line_list.append(same_line)

    def close(self):
        """close current output context"""
        self.indent_list.pop()
        self.active_list.pop()
        self.same_line_list.pop()

    def print(self, *args, **kwargs):
        """print into current output context using indentation provided via previous open() calls"""
        if not self.active_list or self.active_list[-1]:
            # go to beginning of line
            same_line = {}
            if self.same_line_list and self.same_line_list[-1]:
                same_line = {'same_line': True}
                self.same_line_func()
            # prepending debug information
            if self._prepend_debug:
                caller = inspect.getframeinfo(inspect.stack()[self._prepend_debug_level][0])
                self.debug_func(f"{caller.filename}:{caller.lineno} ")
            # print indentation
            self.indent_func(
                "".join([indent if active else "" for indent, active in zip(self.indent_list, self.active_list)])
            )
            # print actual message
            self.print_func(*args, **kwargs, **same_line)

    def prepend_debug(self, prepend=True):
        self._prepend_debug = prepend

    def deco(self, indent="│  ", active=True, add_print_func=False):
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                self.open(indent=indent, active=active)
                ret = func(*args, **kwargs)
                self.close()
                return ret
            if add_print_func:
                wrapper.print = self.print
            return wrapper
        return decorator

    @contextmanager
    def __call__(self, *args, **kwargs):
        self.open(*args, **kwargs)
        yield None
        self.close()


class NestedOutputSingleton:
    """Class to provide global nested output handling. The class uses static methods to provide the same interface
    as a NestedOutput objects and hands calls over a static class-level object."""

    _singleton = NestedOutput(prepend_debug_level=2)

    @staticmethod
    def open(*args, **kwargs):
        NestedOutputSingleton._singleton.open(*args, **kwargs)

    @staticmethod
    def close():
        NestedOutputSingleton._singleton.close()

    @staticmethod
    def print(*args, **kwargs):
        NestedOutputSingleton._singleton.print(*args, **kwargs)

    @staticmethod
    def prepend_debug(*args, **kwargs):
        NestedOutputSingleton._singleton.prepend_debug(*args, **kwargs)

    @staticmethod
    def deco(*args, **kwargs):
        return NestedOutputSingleton._singleton.deco(*args, **kwargs)

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def __enter__(self):
        NestedOutputSingleton.open(*self.args, **self.kwargs)
        return None

    def __exit__(self, *args):
        NestedOutputSingleton.close()


class NestedOutputDummy:
    """Class to provide a dummy object that mocks the Singleton interface without actually doing anything. This is
    to efficiently supressing debug output by reassigning the used class instead of changing the actual code."""

    @staticmethod
    def open(*args, **kwargs):
        pass

    @staticmethod
    def close():
        pass

    @staticmethod
    def print(*args, **kwargs):
        pass

    @staticmethod
    def prepend_debug(*args, **kwargs):
        pass

    @staticmethod
    def deco(*args, **kwargs):
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args_, **kwargs_):
                return func(*args_, **kwargs_)

            return wrapper

        return decorator

    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return None

    def __exit__(self, *args):
        pass


# for debug output of this module
# NO = NestedOutputSingleton  # debug output active
NO = NestedOutputDummy      # debug output inactive


def add_hash(Cls):
    """
    Class decorator that defines a hash function by hashing the tuple of the class name and all key-value-pairs
    in __dict__.
    """

    def __hash__(self):
        t = [self.__class__.__name__]
        for key, val in self.__dict__.items():
            t += [(key, val)]
        return hash(tuple(t))

    Cls.__hash__ = __hash__

    return Cls


def add_eq(Cls):
    """
    Class decorator to add an __eq__ method.

    Equality is defined by comparing all fields (in self.__dict__). Lists, tuples and numpy arrays are compared using
    numpy.array_equal.
    """

    def __eq__(self, other):
        if isinstance(other, type(self)):
            # attributes don't match
            if list(self.__dict__.keys()) != list(other.__dict__.keys()):
                return False
            # attribute values don't match
            for (this_key, this_prop), (other_key, other_prop) in zip(self.__dict__.items(), other.__dict__.items()):
                try:
                    if isinstance(this_prop, (list, tuple, np.ndarray)) \
                            and isinstance(other_prop, (list, tuple, np.ndarray)):
                        if not np.array_equal(this_prop, other_prop):
                            return False
                    else:
                        if this_prop != other_prop:
                            return False
                except Exception as e:
                    raise UserWarning(f"Failed to compute equality between objects {self} and {other}\n"
                                      f"Failed while evaluating {this_key} != {other_key} with\n"
                                      f"    {this_key} (type: {type(this_prop)}): {this_prop}\n"
                                      f"    {other_key} (type: {type(other_prop)}): {other_prop}"
                                      f"Exception: {repr(e)}")
            return True
        else:
            return NotImplemented

    Cls.__eq__ = __eq__

    return Cls


class AddRepr:
    indent_list = []
    idx_list = []
    new_object_list = []
    object_set = set()
    single_line = True
    max_depth = 10

    @staticmethod
    def to_string(obj):
        NO.print(f"Length indent list: {len(AddRepr.indent_list)}/{AddRepr.max_depth}")
        """transform obj into printable string, handling some special cases (numpy arrays, strings)"""
        if AddRepr.max_depth is not None and max(len(AddRepr.indent_list),
                                                 len(AddRepr.idx_list),
                                                 len(AddRepr.new_object_list),
                                                 len(AddRepr.object_set)) > AddRepr.max_depth:
            NO.print(f"Length indent list: {len(AddRepr.indent_list)}")
            return "<...>"
        if AddRepr.in_object_set(obj):
            # recursive call
            return ">>>"
        elif isinstance(obj, np.ndarray):
            return str(obj.tolist())
        elif isinstance(obj, str):
            return f"'{obj}'"
        else:
            return str(obj)

    @staticmethod
    def add_to_set(obj):
        if id(obj) in AddRepr.object_set:
            AddRepr.new_object_list.append(False)
        else:
            AddRepr.new_object_list.append(True)
            AddRepr.object_set.add(id(obj))

    @staticmethod
    def remove_from_set(obj):
        if AddRepr.new_object_list.pop():
            AddRepr.object_set.remove(id(obj))

    @staticmethod
    def in_object_set(obj):
        return id(obj) in AddRepr.object_set

    @staticmethod
    def add_multi_line_repr(Cls, name="__repr__"):
        """
        Class decorator to add a __repr__ method
         - returns a multi-line sting with the class name and all fields in self.__dict__
         - handles indentation for nested calls
         - will be added as attribute 'name' (default: __repr__)
        """
        def __repr__(self, exclude):
            # "open" context for this object
            AddRepr.indent_list.append("│  ")
            AddRepr.idx_list.append(0)
            AddRepr.add_to_set(self)
            # initialize string with this class name
            s = self.__class__.__name__
            # add fields to string
            print_subfield_separator = True
            for key, val in self.__dict__.items():
                if key in exclude:
                    continue
                # add new line (adding subfield separator the first time)
                if print_subfield_separator:
                    s += ":\n"
                    print_subfield_separator = False
                else:
                    s += "\n"
                # connect next field and adjust indent after last field
                if AddRepr.idx_list[-1] == len(self.__dict__.items()) - 1:
                    # last field
                    AddRepr.indent_list[-1] = "   "
                    s += "".join(AddRepr.indent_list[:-1])
                    s += "╰→ "
                else:
                    # there are more fields to come
                    AddRepr.idx_list[-1] += 1
                    s += "".join(AddRepr.indent_list[:-1])
                    s += "├→ "
                s += f"{key}: {AddRepr.to_string(val)}"
            # "close" context for this object
            AddRepr.indent_list.pop()
            AddRepr.idx_list.pop()
            AddRepr.remove_from_set(self)
            # return
            return s

        setattr(Cls, name, __repr__)

        return Cls

    @staticmethod
    def add_single_line_repr(Cls, name='__repr__'):
        """
        Class decorator to add a __repr__ method
         - returns a single-line sting with the class name and all fields in self.__dict__
         - will be added as attribute 'name' (default: __repr__)
        """
        def __repr__(self, exclude):
            AddRepr.add_to_set(self)
            s = self.__class__.__name__ + "("
            first_field = True
            for key, val in self.__dict__.items():
                if key in exclude:
                    continue
                if not first_field:
                    s += ", "
                else:
                    first_field = False
                s += f"{key}: {AddRepr.to_string(val)}"
            s += ")"
            AddRepr.remove_from_set(self)
            return s

        setattr(Cls, name, __repr__)

        return Cls


def add_repr(exclude=(), single_line=AddRepr.single_line):
    """class decorator that adds a __repr__ function to the class"""
    def add_repr_decorator(Cls):
        def __repr__(self, exclude=exclude, single_line=single_line):
            if single_line:
                return self.__single_line_repr__(exclude=exclude)
            else:
                return self.__multi_line_repr__(exclude=exclude)
        Cls = AddRepr.add_single_line_repr(Cls, "__single_line_repr__")
        Cls = AddRepr.add_multi_line_repr(Cls, "__multi_line_repr__")
        setattr(Cls, "__repr__", __repr__)
        return Cls
    return add_repr_decorator


@add_repr()
@add_eq
class NDSlicable:
    """
    This is a wrapper class that implements slicing for immutable objects that can be indexed. Unlike regular
    slicing, negative indices are processed with the same arithmetics as positive indices. Indexing of the underlying
    object is only performed when accessing a specific element.

    The class relies on the underlying object having a get_item_at(*args) method (first choice) or an __getitem__(item)
    method implemented. This order of preference allows to implement the __getitem__ method of a new class by first
    implementing get_item_at and then returning an NDSlicable object in __getitem__. The get_item_at method should
    raise an IndexError if out of bounds to allow for iteration.

    A word on stopping with repeated slicing: When a dimension is sliced repeatedly, the start, stop, and step values
    need to be updated accordingly. The old slicing defines a grid w.r.t. the underlying object, which has to be
    adapted based on the new slicing. With start and step there is an obvious way to do that: The new start value
    (w.r.t. the underlying object) is chosen w.r.t. the old grid, and the new step is the product of the old step and
    the step value provided by the new slicing. A new stop value can be computed in the same way as the new start
    value, however, it is not clear how to handle "conflicts" between a possibly defined old stop value and the newly
    computed value (Which one should take precedence? And what about negative step values?). Therefore, if a stop value
    is provided, it always takes precedence over the old stop value, which is discarded in that case. If no stop values
    is provided (i.e. None is provided) the old stop value stays in place.
    """

    @add_repr()
    @add_eq
    class DimEntry:
        def __init__(self, is_fixed=False, value=None, out_of_bounds=False):
            if value is None:
                value = slice(None, None, None)
            self.is_fixed = is_fixed
            self.value = value
            self.out_of_bounds = out_of_bounds

    def __init__(self, object, dimensionality, tuple_indices=True):
        """
        :param object: The underlying object to create a sliced view of.
        :param dimensionality: Number of dimensions of underlying object.
        :param tuple_indices: If False tuples of indices are applied consecutively (e.g. required for nested lists).
        """
        self._object = object
        self._dimensions = [NDSlicable.DimEntry() for _ in range(dimensionality)]
        self._tuple_indices = tuple_indices

    @NO.deco()
    def adjust_dimension(self, dim_idx, item):
        dim = self._dimensions[dim_idx]
        if isinstance(item, numbers.Integral):
            # a single index
            start_ = 0 if dim.value.start is None else dim.value.start
            step_ = 1 if dim.value.step is None else dim.value.step
            idx = start_ + item * step_
            out_of_bounds = False if dim.value.stop is None else idx >= dim.value.stop
            self._dimensions[dim_idx] = NDSlicable.DimEntry(True, idx, out_of_bounds)
        elif isinstance(item, slice):
            # a single slice object
            start, stop, step = item.start, item.stop, item.step
            old_start, old_stop, old_step = dim.value.start, dim.value.stop, dim.value.step
            # new start
            if old_start is None and start is None:
                new_start = None
            else:
                old_start_ = 0 if old_start is None else old_start
                start_ = 0 if start is None else start
                old_step_ = 1 if old_step is None else old_step
                new_start = old_start_ + start_ * old_step_
            # new stop
            if old_stop is None and stop is None:
                new_stop = None
            else:
                if stop is None:
                    new_stop = old_stop
                else:
                    old_start_ = 0 if old_start is None else old_start
                    old_step_ = 1 if old_step is None else old_step
                    new_stop = old_start_ + stop * old_step_
            # new step
            if old_step is None and step is None:
                new_step = None
            else:
                old_step_ = 1 if old_step is None else old_step
                step_ = 1 if step is None else step
                new_step = old_step_ * step_
            # set new slice
            self._dimensions[dim_idx] = NDSlicable.DimEntry(False, slice(new_start, new_stop, new_step))
        else:
            UserWarning("Unhandled type. This is a bug.")
        NO.print(f"adjusting dimension {dim_idx}: {self._dimensions[dim_idx]}")

    @NO.deco()
    def process_item(self, item):
        NO.print("processing:", item)
        # unpack length-1 tuples
        if isinstance(item, tuple) and len(item) == 1:
            NO.print("unpacking length-1 item")
            item = item[0]
        # adapt indices/slices
        if isinstance(item, tuple):
            # multiple indices/slices
            NO.print("multi-index")
            sub_items = list(item)
            for dim_idx, dim in enumerate(self._dimensions):
                NO.print("remaining sub-items", sub_items)
                if sub_items:
                    if sub_items[0] is None:
                        # skip dimension (None does not change indexing/slicing)
                        NO.print(f"Skipping dimension {dim_idx}: None index")
                        sub_items.pop(0)
                        continue
                    elif dim.is_fixed:
                        # skip dimension (dimension is already fixed)
                        NO.print(f"Skipping dimension {dim_idx}: is fixed")
                        continue
                    else:
                        self.adjust_dimension(dim_idx=dim_idx, item=sub_items.pop(0))
                else:
                    break
            else:
                ValueError(f"Could not process remaining indices/slices '{sub_items}' specified by '{item}' because "
                           f"all free dimensions were fixed.")
        elif isinstance(item, (numbers.Integral, slice)):
            dim_idx = self.first_free_dim()
            if dim_idx is not None:
                self.adjust_dimension(dim_idx=dim_idx, item=item)
            else:
                raise ValueError(f"Cannot further index or slice object with item '{item}'. All dimensions are fixed.")
        elif item is None:
            # nothing to do; indexing/slicing stays as it is
            pass
        else:
            raise TypeError(f"Cannot handle '{item}' of type {type(item)} for indexing")

    def first_free_dim(self):
        for dim_idx, dim in enumerate(self._dimensions):
            if not dim.is_fixed:
                return dim_idx
        return None

    def n_free(self):
        """return number of unfixed/free dimensions"""
        return sum([0 if dim.is_fixed else 1 for dim in self._dimensions])

    def out_of_bounds(self):
        """return True if any dimension is out of bounds"""
        for dim in self._dimensions:
            if dim.out_of_bounds:
                return True
        return False

    @NO.deco()
    def __getitem__(self, item):
        # initialize new view
        NO.print("init new view")
        new_view = NDSlicable(object=self._object,
                              dimensionality=len(self._dimensions),
                              tuple_indices=self._tuple_indices)
        new_view._dimensions = [dim for dim in self._dimensions]
        # process item
        new_view.process_item(item=item)
        # check if any index is out of bounds (w.r.t. slicing, not w.r.t. the underlying object)
        if new_view.out_of_bounds():
            raise IndexError
        # get number of free dimensions
        n = new_view.n_free()
        # return view or item
        if n > 0:
            NO.print("returning view")
            # get item at index (0, 0, ...) to trigger IndexError if out of bounds
            NO.print("checking (0,0,...) index")
            _ = new_view[tuple([0] * n)]
            # return view
            return new_view
        else:
            NO.print("returning item")
            index = tuple([dim.value for dim in new_view._dimensions])
            if new_view._tuple_indices:
                NO.print("use tuple indices")
                try:
                    return new_view._object.get_item_at(*index)
                except AttributeError:
                    return new_view._object.__getitem__(index)
            else:
                NO.print("use consecutive indexing")
                indexed_object = new_view._object
                with NO():
                    for idx in index:
                        NO.print(f"indexing: {idx}")
                        try:
                            indexed_object = indexed_object.get_item_at(idx)
                        except AttributeError:
                            indexed_object = indexed_object.__getitem__(idx)
                        NO.print(f"--> indexed object: {indexed_object}")
                NO.print(f"returning {indexed_object}")
                return indexed_object

    def to_list(self):
        l = []
        for sub in self:
            if isinstance(sub, NDSlicable) and sub.n_free():
                l.append(sub.to_list())
            else:
                l.append(sub)
        return l


def wrap_bound_method(object, method, new_name, pre=lambda: None, post=lambda: None):
    # rename method
    setattr(object, new_name, getattr(object, method))
    # construct new method that calls pre/post before/after calling the original method
    def f(self, *args, **kwargs):
        pre()
        getattr(self, new_name)(*args, **kwargs)
        post()
    # use that new function in place of the old one
    setattr(object, method, types.MethodType(f, object))


def iterate_files(root_dir,
                  depth=-1,
                  path_regex="",
                  file_regex="",
                  return_full_path=False,
                  return_dir_path=False,
                  return_file_name=False):
    # get length of root path
    root_depth = len(root_dir.split(os.path.sep))
    # compile regex
    path_regex = re.compile(path_regex)
    file_regex = re.compile(file_regex)
    # walk through tree
    for dir_path, dir_names, file_names in os.walk(root_dir):
        # break on maximum depth
        if 0 <= depth < len(dir_path.split(os.path.sep)) - root_depth:
            continue
        # skip non-matching directories
        if not path_regex.match(dir_path):
            continue
        for file in file_names:
            # skip non-matching files
            if not file_regex.match(file):
                continue
            # construct tuple to return
            ret = tuple()
            if return_full_path:
                ret = ret + (os.path.join(dir_path, file), )
            if return_dir_path:
                ret = ret + (dir_path, )
            if return_file_name:
                ret = ret + (file, )
            # return
            if len(ret) == 0:
                yield None
            elif len(ret) == 1:
                yield ret[0]
            else:
                yield ret


def nested_enum(data, depth=None, _index=()):
    """
    Iterate through nested iterables returning index tuple and value, that is, like enumerate but nested and thus
    with index tuples.
    :param data: nested iterable
    :param depth: maximum depth to follow nested structures
    :param _index: start index that is appended when traversing to the next level (only for internal use in recursive
    calls)
    :return:
    """
    if depth is not None and depth == 0:
        yield _index, data
    else:
        try:
            for sub_index, sub_data in enumerate(data):
                new_index = _index + (sub_index,)
                yield from nested_enum(data=sub_data, _index=new_index, depth=None if depth is None else depth - 1)
        except TypeError:
            yield _index, data


def append_nested(nested_list, elem, index, check_index=True):
    """
    Append elem to nested_list at index checking if index is actually adjacent
    """
    unnested_list = nested_list
    max_level = len(index) - 1
    for level, idx in enumerate(index):
        if idx == len(unnested_list):
            if level < max_level:
                unnested_list.append([])
            else:
                unnested_list.append(elem)
        if check_index and idx != len(unnested_list) - 1:
            raise UserWarning(f"Index at level {level} is not adjacent")
        unnested_list = unnested_list[idx]


def load(file_path):
    """
    This is a defensive way to write pickle.load, allowing for very large files on all platforms
    https://stackoverflow.com/questions/42653386/does-pickle-randomly-fail-with-oserror-on-large-files?rq=1
    """
    max_bytes = 2 ** 31 - 1
    input_size = os.path.getsize(file_path)
    bytes_in = bytearray(0)
    with open(file_path, 'rb') as f_in:
        for _ in range(0, input_size, max_bytes):
            bytes_in += f_in.read(max_bytes)
    return pickle.loads(bytes_in)


def dump(data, file_path, overwrite=False):
    """
    This is a defensive way to write pickle.write, allowing for very large files on all platforms
    https://stackoverflow.com/questions/42653386/does-pickle-randomly-fail-with-oserror-on-large-files?rq=1
    """
    max_bytes = 2**31 - 1
    bytes_out = pickle.dumps(data)
    n_bytes = sys.getsizeof(bytes_out)
    with open(file_path, 'wb' if overwrite else 'xb') as f_out:
        for idx in range(0, n_bytes, max_bytes):
            f_out.write(bytes_out[idx:idx+max_bytes])


def index_to_tuple(index):
    if isinstance(index, tuple):
        return tuple(elem.__reduce__() if isinstance(elem, slice) else elem for elem in index)
    else:
        if isinstance(index, slice):
            return (index.__reduce__(),)
        else:
            return (index,)


def tuple_to_index(t):
    t = tuple(tt[0](*tt[1]) if isinstance(tt, tuple) and tt[0] == slice else tt for tt in t)
    if len(t) == 1:
        return t[0]
    else:
        return t


@add_repr()
@add_eq
class IndexRecorder:

    def __init__(self):
        self.assignments = []
        self._frozen = False

    def __setitem__(self, key, value):
        if self._frozen:
            raise IndexError("Trying to assign to frozen IndexRecorder")
        self.assignments.append((key, value))

    def freeze(self):
        self._frozen = True
        self.assignments = tuple((index_to_tuple(a[0]), a[1]) for a in self.assignments)

    def apply(self, container):
        for key, val in self.assignments:
            if self._frozen:
                key = tuple_to_index(key)
            container[key] = val
        return container

    def __hash__(self):
        if self._frozen:
            return hash(tuple(self.assignments))
        else:
            raise UserWarning("Non-frozen IndexRecorder objects are not hashable")


def merge_dicts(*dicts, depth=None):
    """
    Recursively merges multiple dictionaries into one. Earlier dictionaries override later ones. If all values for
    a specific key are dictionaries, merging is performed recursively.
    """
    # taken/inspired from: https: // stackoverflow.com / questions / 20656135 / python - deep - merge - dictionary - data
    # collect keys
    key_set = set()
    for d in dicts:
        for key in d.keys():
            key_set.add(key)
    # fill construct new dict
    return_dict = {}
    for key in key_set:
        values = [d[key] for d in dicts if key in d]
        if not values:
            raise UserWarning("Could not find key while merging dicts. This is a bug.")
        if np.all([isinstance(v, dict) for v in values]) and (depth is None or depth > 0):
            # all values are dicts --> merge recursively
            return_dict[key] = merge_dicts(*values, depth=depth - 1 if depth is not None else None)
        else:
            # at least one value is not a dict --> take first value
            return_dict[key] = values[0]
    return return_dict


def dict_from_attributes(attributes, objects, raise_attr_error=True):
    """
    Retrieve dictionary of attribute values from list objects. Attributes are only included if their value is the same
    for all objects.
    :param attributes: the attributes to retrieve (iterable of strings)
    :param objects: list of objects to retrieve the attribute values from
    :return:
    """
    return_dict = {}
    for key in attributes:
        for o in objects:
            delete = True
            try:
                val = getattr(o, key)
                if key not in return_dict:
                    # new key --> add value and don't delete
                    return_dict[key] = val
                    delete = False
                else:
                    try:
                        if return_dict[key] == val:
                            # value is the same --> don't delete
                            delete = False
                    except ValueError:
                        # comparison failed --> delete
                        pass
            except AttributeError:
                if raise_attr_error:
                    raise
                else:
                    # ignore error and delete
                    pass
            if delete:
                # delete attribute from dictionary (if present) and break loop for this attribute,
                # i.e., continue with next attribute
                if key in return_dict:
                    del return_dict[key]
                break
    return return_dict


def default_override_exclude_dict(default_dict, override_dict=(), exclude=()):
    """
    Construct a dictionary using default value
    :param default_dict: default values; only keys contained in default_dict will be considered; their value will be the
    one provided in default_dict, unless found in override_dice, unless found in exclude dict
    :param override_dict: entries in this dictionary override those in default_dict
    :param exclude: keys found in in exclude are excluded from the return dictionary
    :return: dictionary with values from default_dict or override_dict unless present in exclude
    """
    return_dict = {}
    for key, val in default_dict.items():
        if key not in exclude:
            if key in override_dict:
                val = override_dict[key]
            return_dict[key] = val
    return return_dict


def pretty_table(iterable, alignment='l', colsep=' ', keep_rows=None, keep_cells=None, use_cols=None):
    """
    transform content of 2D iterable into padded strings
    :param iterable: 2D iterable
    :param alignment: alignment ('l', 'c', 'r'); either a single letter valid for the whole table or a list with one
    letter for each column
    :param colsep: separator used between columns
    :param keep_rows: return a list of strings with one string for each row
    :param keep_cells: return a 2D list of strings with one string for each cell (implies keep_rows=True)
    :param use_cols: list of bools indicating whether to use the corresponding column [optional]
    :return: string, 1D list of strings [keep_rows=True], or 2D list of strings [keep_cells=True]
    """
    # set default for keep_cells
    if keep_cells is None:
        keep_cells = False
    # force to keep rows if not explicitly set
    if keep_cells and keep_rows is None:
        keep_rows = True
    # set default for keep_rows
    if keep_rows is None:
        keep_rows = False
    if keep_cells and not keep_rows:
        raise UserWarning("Cells can only be kept if also rows are kept.")
    # get nested list of cells and determine minimum column width
    raw_table = []
    col_widths = {}
    for row in iterable:
        raw_table.append([])
        for col_idx, cell in enumerate(row):
            if use_cols is None or use_cols[col_idx]:
                cell_str = str(cell)
            else:
                cell_str = ""
            cell_width = len(cell_str)
            raw_table[-1].append(cell_str)
            try:
                col_widths[col_idx] = max(col_widths[col_idx], cell_width)
            except KeyError:
                col_widths[col_idx] = cell_width
    col_widths = [col_widths[i] for i in range(len(col_widths))]
    # get alignment
    if alignment in ['l', 'c', 'r']:
        alignment = [alignment for _ in col_widths]
    else:
        if len(alignment) != len(col_widths):
            raise UserWarning(f"Number of alignment specifiers does not equal number of columns "
                              f"({len(alignment)} != {len(col_widths)})")
    for col_idx in range(len(alignment)):
        a = alignment[col_idx]
        if a == 'l':
            alignment[col_idx] = str.ljust
        elif a == 'c':
            alignment[col_idx] = str.center
        elif a == 'r':
            alignment[col_idx] = str.rjust
        else:
            raise UserWarning(f"Alignment specifies must be one of ('l', 'r', 'c'), got: {alignment}")
    # write to string
    if not keep_cells:
        for row_idx in range(len(raw_table)):
            raw_table[row_idx] = colsep.join([align(cell, width)
                                              for col_idx, (align, cell, width) in enumerate(zip(alignment,
                                                                                                 raw_table[row_idx],
                                                                                                 col_widths))
                                              if use_cols is None or use_cols[col_idx]])
    if keep_rows:
        return raw_table
    else:
        return '\n'.join(raw_table)


def html_table(iterable, raw=False):
    table = ""
    for row in iterable:
        table += "<tr>"
        for cell in row:
            table += f"<td>{cell}</td>"
        table += "</tr>"
    if raw:
        return table
    else:
        return f"<table>{table}</table>"


def point_list_from_meshgrid(meshgrid):
    return np.array(list(zip(*[dim.flatten() for dim in meshgrid])))


def get_nd_grid(min_max_steps, margin=None, return_meshgrid=False, indexing='ij'):
    """
    Generate an n-dimensional grid
    :param min_max_steps: list of (min, max, steps) for each dimension
    :param margin: list of bools specifying whether the respective dimension should have a margin of half the grid
    spacing; the default is no margin
    :param return_meshgrid: whether to return the numpy meshgrid (True) or an array of points (default/False)
    :param indexing: indexing for grid (corresponds to numpy.meshgrid parameter) 'ij' (default) or 'xy'
    :return: meshgrid or array of points
    """
    dimensions = []
    for idx, (minimum, maximum, N) in enumerate(min_max_steps):
        if margin is not None and margin[idx]:
            margin_width = ((maximum - minimum) / N) / 2
        else:
            margin_width = 0.
        dimensions.append(list(np.linspace(minimum + margin_width, maximum - margin_width, N, endpoint=True)))
    meshgrid = np.meshgrid(*dimensions, indexing=indexing)
    if return_meshgrid:
        return meshgrid
    else:
        return point_list_from_meshgrid(meshgrid)


def eval_on_grid(func, min_max_steps, margin=None, accept_array=False):
    coords = get_nd_grid(min_max_steps=min_max_steps, margin=margin)
    if accept_array:
        data = func(coords)
    else:
        data = np.array([func(c) for c in coords])
    return data.reshape(tuple(x[2] for x in min_max_steps), order='C').transpose()


def plot_heatmap(data, ax, imshow_kwargs=None, colorbar_kwargs=None, colorbar=True):
    if imshow_kwargs is None:
        imshow_kwargs = {}
    if "origin" not in imshow_kwargs:
        imshow_kwargs["origin"] = "lower"
    img = ax.imshow(data, **imshow_kwargs)
    if colorbar:
        if colorbar_kwargs is None:
            colorbar_kwargs = {}
        plt.colorbar(img, ax=ax, **colorbar_kwargs)


def next_color(ax):
    return next(ax._get_lines.prop_cycler)['color']


def reorder_legen_row_wise(ax, ncol):
    # get handles and labels
    handles, labels = ax.get_legend_handles_labels()
    # get number of legend entries
    n_legend = len(handles)
    # compute number of rows needed
    nrow = int(np.ceil(n_legend / ncol))
    # fill index table row-wise
    index = np.full((nrow, ncol), np.nan)
    for idx in range(n_legend):
        index[idx // ncol, idx % ncol] = idx
    # reorder handles and labels by iterating over indices column-wise
    return tuple(
        zip(*[(handles[int(idx)], labels[int(idx)]) for idx in index.transpose().flatten() if not np.isnan(idx)])
    )


def assert_xor(a, b):
    assert (a or b) and not (a and b)


def unique_in_df(data_frame, n_max=np.inf):
    """
    Return list with unique values for each column in a data frame. 1st column: column names; 2nd column: summary of
    unique value for this column (see n_max); 3rd column: number of unique values; 4th column: complete list of unique
    values (got via pandas unique() method)
    :param data_frame: pandas data frame to process
    :param n_max: maximum number of unique values to return as explicit list
    :return: 2D list of shape (n_rows_in_df, 4)
    """
    unique_list = []
    for x in data_frame.columns:
        unique_list.append([x])
        unique = data_frame[x].unique()
        if len(unique) > n_max:
            unique_list[-1].append(len(unique))
        else:
            unique_list[-1].append(unique)
        unique_list[-1].append(len(unique))
        unique_list[-1].append(unique)
    return unique_list


@total_ordering
class LogFloat:

    def __init__(self, value, is_log=False):
        if isinstance(value, LogFloat):
            self._value = value._value
        else:
            if is_log:
                self._value = value
            else:
                if value == 0:
                    self._value = -np.inf
                elif value > 0:
                    try:
                        self._value = np.log(np.float(value))
                    except TypeError:
                        raise TypeError(f"A type error occurred when passing argument '{value}' to np.log(np.float(.))")
                else:
                    raise ValueError(f"Cannot store negative value '{value}' in log representation")

    def __float__(self):
        return float(np.exp(self._value))

    def __str__(self):
        return f"{np.exp(self._value)}"

    def __repr__(self):
        return f"exp({self._value})"

    def __hash__(self):
        return hash(np.exp(self._value))

    def __eq__(self, other):
        return np.exp(self._value) == other

    def __lt__(self, other):
        return np.exp(self._value) < other

    def __add__(self, other):
        if isinstance(other, LogFloat):
            return LogFloat(np.logaddexp(self._value, other._value), is_log=True)
        else:
            return self + LogFloat(other)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, LogFloat):
            return LogFloat(np.exp(self._value) - np.exp(other._value))
        else:
            return self - LogFloat(other)

    def __rsub__(self, other):
        return LogFloat(other) - self

    def __mul__(self, other):
        if isinstance(other, LogFloat):
            return LogFloat(self._value + other._value, is_log=True)
        else:
            return self * LogFloat(other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, LogFloat):
            return LogFloat(self._value - other._value, is_log=True)
        else:
            return self / LogFloat(other)

    def __rtruediv__(self, other):
        return LogFloat(other).__truediv__(self)


def normalize_non_zero(a, axis=None, skip_type_check=False):
    """For the given ND array, normalise each 1D array obtained by indexing the 'axis' dimension if the sum along the
    other dimensions (for that entry) is non-zero. Normalisation is performed in place."""
    # check that dtype is float (in place division of integer arrays silently rounds)
    if not skip_type_check:
        if not np.issubdtype(a.dtype, np.floating):
            raise TypeError(f"Cannot guarantee that normalisation works as expected on array of type '{a.dtype}'. "
                            f"Use 'skip_type_check=True' to skip this check.")
    # normalise along last axis per default
    if axis is None:
        axis = a.ndim - 1
    # make axis a tuple if it isn't
    if not isinstance(axis, tuple):
        axis = (axis,)
    # compute sum along axis, keeping dimensions
    s = a.sum(axis=axis, keepdims=True)
    # check for non-zero entries
    non_zero = (s != 0)
    if not np.any(non_zero):
        # directly return if there are no non-zero entries
        return a
    # construct an index tuple to select the appropriate entries for normalisation (the dimensions specified by axis
    # have to be replaced by full slices ':' to broadcast normalisation along these dimensions)
    non_zero_arr = tuple(slice(None) if idx in axis else n for idx, n in enumerate(non_zero.nonzero()))
    # in-place replace non-zero entries by their normalised values
    a[non_zero_arr] = a[non_zero_arr] / s[non_zero_arr]
    # return array
    return a


def axis_set_invisible(ax, splines=False, ticks=(), patch=False, x=False, y=False):
    if x:
        ax.get_xaxis().set_visible(False)
    if y:
        ax.get_yaxis().set_visible(False)
    if splines:
        plt.setp(ax.spines.values(), visible=False)
    if ticks:
        if 'left' in ticks or 'all' in ticks:
            ax.tick_params(left=False, labelleft=False)
        if 'right' in ticks or 'all' in ticks:
            ax.tick_params(right=False, labelright=False)
        if 'top' in ticks or 'all' in ticks:
            ax.tick_params(top=False, labeltop=False)
        if 'bottom' in ticks or 'all' in ticks:
            ax.tick_params(bottom=False, labelbottom=False)
    if patch:
        ax.patch.set_visible(False)


def pretty_time_diff(t1, t2):
    rd = dateutil.relativedelta.relativedelta(t1, t2)
    l = []
    for att in ["years", "months", "days", "hours", "minutes", "seconds"]:
        val = getattr(rd, att)
        if abs(val) == 1:
            l.append(f"{val} {att[:-1]}")
        elif val != 0:
            l.append(f"{val} {att}")
    if l:
        return ", ".join(l)
    else:
        return "0 seconds"