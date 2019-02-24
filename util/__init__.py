import functools
from contextlib import contextmanager
import inspect
import numpy as np
import types


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


class NestedOutput:
    """Class to handle nested output with indentation"""

    _singleton = None

    def __init__(self, print_func=print, indent_func=None, debug_func=None, prepend_debug=False):
        self.indent_list = []
        self.active_list = []
        if indent_func is None:
            self.indent_func = lambda *args, **kwargs: print(*args, **kwargs, end='')
        else:
            self.indent_func = print_func
        if debug_func is None:
            self.debug_func = lambda *args, **kwargs: print(*args, **kwargs, end='')
        else:
            self.debug_func = debug_func
        self.print_func = print_func
        self._prepend_debug = prepend_debug
        self._prepend_debug_level = 1

    def open(self, indent="│  ", active=True):
        """open new output context with given indentation"""
        self.indent_list.append(indent)
        self.active_list.append(active)

    def close(self):
        """close current output context"""
        self.indent_list.pop()
        self.active_list.pop()

    def print(self, *args, **kwargs):
        """print into current output context using indentation provided via previous open() calls"""
        if not self.active_list or self.active_list[-1]:
            # prepending debug information
            # prepend_debug = self._prepend_debug
            # if in_dict_and_not_none("prepend_debug", kwargs):
            #     prepend_debug = kwargs["prepend_debug"]
            if self._prepend_debug:
            # if prepend_debug:
                caller = inspect.getframeinfo(inspect.stack()[self._prepend_debug_level][0])
                self.debug_func(f"{caller.filename}:{caller.lineno} ")
            # print indentation
            self.indent_func(
                "".join([indent if active else "" for indent, active in zip(self.indent_list, self.active_list)])
            )
            # print actual message
            self.print_func(*args, **kwargs)

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
    as a NestedOutput objects and hands calls over to the singleton object after ensuring its existence."""

    @staticmethod
    def check_singleton():
        if not NestedOutput._singleton:
            NestedOutput._singleton = NestedOutput()
            # level has to be 2 instead of one because of indirect call of print function
            NestedOutput._singleton._prepend_debug_level = 2

    @staticmethod
    def open(*args, **kwargs):
        NestedOutputSingleton.check_singleton()
        NestedOutput._singleton.open(*args, **kwargs)

    @staticmethod
    def close():
        NestedOutputSingleton.check_singleton()
        NestedOutput._singleton.close()

    @staticmethod
    def print(*args, **kwargs):
        NestedOutputSingleton.check_singleton()
        NestedOutput._singleton.print(*args, **kwargs)

    @staticmethod
    def prepend_debug(*args, **kwargs):
        NestedOutputSingleton.check_singleton()
        NestedOutput._singleton.prepend_debug(*args, **kwargs)

    @staticmethod
    def deco(*args, **kwargs):
        NestedOutputSingleton.check_singleton()
        return NestedOutput._singleton.deco(*args, **kwargs)

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
    def check_singleton():
        pass

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


def add_eq(Cls):
    """
    Class decorator to add an __eq__ method.

    Equality is defined by comparing all fields (in self.__dict__) in two steps: 1. directly 2. if that fails
    by using numpy.any in case the field is a numpy array.
    """

    def __eq__(self, other):
        if isinstance(other, type(self)):
            for this_prop, other_prop in zip(self.__dict__.values(), other.__dict__.values()):
                try:
                    if this_prop != other_prop:
                        return False
                except ValueError:
                    if np.any(this_prop != other_prop):
                        return False
                except Exception as e:
                    raise UserWarning("Failed to compute equality between objects {} and {}\n"
                                      "Exception: {}".format(self, other, repr(e)))
            return True
        else:
            return NotImplemented

    Cls.__eq__ = __eq__

    return Cls


def add_repr(Cls):
    def __repr__(self):
        if AddRepr.single_line:
            return self.__single_line_repr__()
        else:
            return self.__multi_line_repr__()
    Cls = AddRepr.add_single_line_repr(Cls, "__single_line_repr__")
    Cls = AddRepr.add_multi_line_repr(Cls, "__multi_line_repr__")
    setattr(Cls, "__repr__", __repr__)
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
        def description(self):
            # "open" context for this object
            AddRepr.indent_list.append("│  ")
            AddRepr.idx_list.append(0)
            AddRepr.add_to_set(self)
            # initialize string with this class name
            s = self.__class__.__name__
            # add fields to string
            print_subfield_separator = True
            for key, val in self.__dict__.items():
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

        setattr(Cls, name, description)

        return Cls

    @staticmethod
    def add_single_line_repr(Cls, name='__repr__'):
        """
        Class decorator to add a __repr__ method
         - returns a single-line sting with the class name and all fields in self.__dict__
         - will be added as attribute 'name' (default: __repr__)
        """
        def description(self):
            AddRepr.add_to_set(self)
            s = self.__class__.__name__ + "("
            first_field = True
            for key, val in self.__dict__.items():
                if not first_field:
                    s += ", "
                else:
                    first_field = False
                s += f"{key}: {AddRepr.to_string(val)}"
            s += ")"
            AddRepr.remove_from_set(self)
            return s

        setattr(Cls, name, description)

        return Cls


@add_repr
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

    @add_repr
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
        if isinstance(item, int):
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
        elif isinstance(item, (int, slice)):
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
