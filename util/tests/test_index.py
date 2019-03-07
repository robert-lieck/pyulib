from unittest import TestCase
from util import Index
import numpy as np


class TestIndex(TestCase):
    def test___getitem__(self):
        I = Index()

        def f(l, i):
            return l[i]

        l = [1, 2, 3, 4, 5, 6]
        i = I[1:6:2]
        assert(f(l, i) == [2, 4, 6])

        for _ in range(10):
            N = 10
            l = list(np.random.uniform(0, 1, N))
            min_, max_ = tuple(sorted(np.random.randint(0, N, 2)))
            step = np.random.randint(1, 3)
            self.assertEqual(l[min_:max_:step], l[I[min_:max_:step]])
