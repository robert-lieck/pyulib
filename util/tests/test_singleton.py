from unittest import TestCase
from util import add_singleton, create_singleton_class, random_string


class TestAdd_singleton(TestCase):

    def test_singleton(self):
        # random names
        singleton_object_name = random_string()
        singleton_subclass_name = random_string()
        # create class and add singleton
        @add_singleton(singleton_object_name=singleton_object_name, singleton_subclass_name=singleton_subclass_name)
        class TMP:
            def f(self):
                return 1
        # create separate singleton class
        Sing = create_singleton_class(TMP,
                                      singleton_object_name=singleton_object_name,
                                      singleton_class_name=singleton_subclass_name)
        # asser that singleton object was createt but is still none
        self.assertIsNone(getattr(TMP, singleton_object_name))
        # create class object
        t = TMP()
        # check that function can be called on either object itself, via static method of
        # singleton subclass, or via separate singleton class)
        self.assertEqual(1, t.f())
        self.assertEqual(1, getattr(TMP, singleton_subclass_name).f())
        self.assertEqual(1, Sing.f())
