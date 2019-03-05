from unittest import TestCase
from util import NestedOutput, NestedOutputSingleton


class TestNestedOutput(TestCase):

    def test_output(self):
        # test for regular object and singleton
        for active_deco_func in [True, False]:
            for add_print_func in [True, False]:
                for singleton in [True, False]:

                    # collect output
                    l = []
                    # special print and indent functions to collect output
                    def print_func(out):
                        l.append(out)
                    def indent_func(out):
                        l.append(out)
                    # set object
                    if singleton:
                        no = NestedOutputSingleton
                    else:
                        no = NestedOutput(indent_func=indent_func, print_func=print_func)

                    # define decorated function
                    @no.deco(indent="----", active=active_deco_func, add_print_func=add_print_func)
                    def f():
                        if add_print_func:
                            f.print("function output")
                        else:
                            self.assertRaises(AttributeError, lambda: f.print("function output"))
                            no.print("function output")

                    # set print and indent of singleton (was created when decorating function)
                    if singleton:
                        NestedOutputSingleton._singleton.print_func = print_func
                        NestedOutputSingleton._singleton.indent_func = indent_func

                    no.open("    ")
                    no.print("outer 1")
                    with no(indent="    "):
                        no.print("inner 1")
                        with no(indent="    "):
                            no.print("inner inner 1")
                            f()
                            no.print("inner inner 2")
                        no.print("inner 2")
                    no.print("outer 2")
                    no.close()

                    self.assertEqual(l,
                                     ["    ", "outer 1", "        ", "inner 1", "            ", "inner inner 1"] +
                                     (["            ----", "function output"] if active_deco_func else []) +
                                     ["            ", "inner inner 2", "        ", "inner 2", "    ", "outer 2"],
                                     f"singleton: {singleton}, deactivate: {active_deco_func}")

    def test_context_manager(self):
        l = []
        def pfunc(out):
            l.append(out)
        no = NestedOutput(print_func=pfunc, indent_func=pfunc)
        with no(indent=" "):
            no.print("x")
            for _ in range(2):
                with no(indent=" "):
                    no.print("y")
                    for _ in range(3):
                        with no(indent=" "):
                            no.print("z")
                    no.print("yy")
            no.print("xx")
        self.assertEqual(l, [" ", "x",
                             "  ", "y",
                             "   ", "z", "   ", "z", "   ", "z",
                             "  ", "yy",
                             "  ", "y",
                             "   ", "z", "   ", "z", "   ", "z",
                             "  ", "yy",
                             " ", "xx"])
