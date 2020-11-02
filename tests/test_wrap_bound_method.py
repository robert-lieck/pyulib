from unittest import TestCase
from util import wrap_bound_method


class TestWrap_bound_method(TestCase):

    def test_wrap_bound_method(self):

        class TMP:

            def __init__(self):
                self.counter = 0

            def f(self):
                self.counter += 1

        for pre in [True, False]:
            for post in [True, False]:

                t = TMP()

                def pre_func():
                    t.counter += 1

                def post_func():
                    t.counter += 1

                self.assertEqual(t.counter, 0)
                t.f()
                self.assertEqual(t.counter, 1)
                wrap_bound_method(t,
                                  "f",
                                  "f_",
                                  pre_func if pre else lambda: None,
                                  post_func if post else lambda: None)
                t.f()
                self.assertEqual(t.counter, 2 + pre + post)
