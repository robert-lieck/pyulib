from unittest import TestCase
from pyulib import add_repr, AddRepr
import numpy as np
from pyulib import NestedOutputSingleton as NO


class TestAdd_repr(TestCase):

    def test_add_repr(self):

        @add_repr()
        class A:
            def __init__(self):
              self.a = 1
              self.b = 'x'

        @add_repr()
        class B:
            def __init__(self):
                self.c = 'y'
                self.d = 2
                self.e = A()

        a = A()
        self.assertEqual("A(a: 1, b: 'x')", str(a))

        b = B()
        self.assertEqual("B(c: 'y', d: 2, e: A(a: 1, b: 'x'))", str(b))

    def test_add_repr_exclude(self):

        @add_repr(['a'])
        class A:
            def __init__(self):
                self.a = 1
                self.b = 'x'

        a = A()
        self.assertEqual("A(b: 'x')", str(a))
        self.assertEqual("A(a: 1)", a.__repr__(['b']))

    def test_add_repr_recursion(self):
        @add_repr()
        class X:
            def __init__(self):
                self.x = None

        x1 = X()
        x2 = X()
        x1.x = x2
        x2.x = x1
        self.assertEqual("X(x: X(x: >>>))", str(x1))
        self.assertEqual("X(x: X(x: >>>))", str(x2))

    def test_add_repr_max_depth(self):
        @add_repr()
        class X:
            def __init__(self):
                self.x = None

        # remember old max_depth and set to finite value
        old_max_depth = AddRepr.max_depth
        AddRepr.max_depth = 10

        x = X()
        next_x = x
        for _ in range(AddRepr.max_depth):
            new_x = X()
            next_x.x = new_x
            next_x = new_x
        self.assertEqual(str(x), "X(x: " * (AddRepr.max_depth + 1) + "<...>" + ")" * (AddRepr.max_depth + 1))

        # reset to old value
        AddRepr.max_depth = old_max_depth
