from unittest import TestCase
import numpy as np
from numpy.testing import assert_array_equal
from pyulib import nested_enum, append_nested


class TestNested_enum(TestCase):
    def test_nested_enum(self):
        N = 3
        dim = 3
        it1 = np.array(range(int(np.power(N, dim)))).reshape(tuple(3 for _ in range(dim)))
        it1_ = [
            ((0, 0, 0), 0),
            ((0, 0, 1), 1),
            ((0, 0, 2), 2),
            ((0, 1, 0), 3),
            ((0, 1, 1), 4),
            ((0, 1, 2), 5),
            ((0, 2, 0), 6),
            ((0, 2, 1), 7),
            ((0, 2, 2), 8),
            ((1, 0, 0), 9),
            ((1, 0, 1), 10),
            ((1, 0, 2), 11),
            ((1, 1, 0), 12),
            ((1, 1, 1), 13),
            ((1, 1, 2), 14),
            ((1, 2, 0), 15),
            ((1, 2, 1), 16),
            ((1, 2, 2), 17),
            ((2, 0, 0), 18),
            ((2, 0, 1), 19),
            ((2, 0, 2), 20),
            ((2, 1, 0), 21),
            ((2, 1, 1), 22),
            ((2, 1, 2), 23),
            ((2, 2, 0), 24),
            ((2, 2, 1), 25),
            ((2, 2, 2), 26),
        ]
        it1__ = [
            ((0, 0), [0, 1, 2]),
            ((0, 1), [3, 4, 5]),
            ((0, 2), [6, 7, 8]),
            ((1, 0), [9, 10, 11]),
            ((1, 1), [12, 13, 14]),
            ((1, 2), [15, 16, 17]),
            ((2, 0), [18, 19, 20]),
            ((2, 1), [21, 22, 23]),
            ((2, 2), [24, 25, 26]),
        ]

        nested = []
        for (index, data), (index_, data_) in zip(nested_enum(it1), it1_):
            self.assertEqual(index, index_)
            self.assertEqual(data, data_)
            append_nested(nested, data, index)
        assert_array_equal(it1, np.array(nested))

        nested = []
        for (index, data), (index_, data_) in zip(nested_enum(it1, depth=2), it1__):
            self.assertEqual(index, index_)
            assert_array_equal(data, data_)
            append_nested(nested, data, index)
        assert_array_equal(it1, np.array(nested))

        it2 = [[[1, 2], [3, 4, 5], [6, 7, 8, 9]], [[10, 11], [12]], [13, 14], 15]
        it2_ = [
            ((0, 0, 0), 1),
            ((0, 0, 1), 2),
            ((0, 1, 0), 3),
            ((0, 1, 1), 4),
            ((0, 1, 2), 5),
            ((0, 2, 0), 6),
            ((0, 2, 1), 7),
            ((0, 2, 2), 8),
            ((0, 2, 3), 9),
            ((1, 0, 0), 10),
            ((1, 0, 1), 11),
            ((1, 1, 0), 12),
            ((2, 0), 13),
            ((2, 1), 14),
            ((3,), 15)
        ]
        it2__ = [
            ((0, 0), [1, 2]),
            ((0, 1), [3, 4, 5]),
            ((0, 2), [6, 7, 8, 9]),
            ((1, 0), [10, 11]),
            ((1, 1), [12]),
            ((2, 0), 13),
            ((2, 1), 14),
            ((3,), 15)
        ]
        nested = []
        for (index, data), (index_, data_) in zip(nested_enum(it2), it2_):
            self.assertEqual(index, index_)
            self.assertEqual(data, data_)
            append_nested(nested, data, index)
        self.assertEqual(it2, nested)

        nested = []
        for (index, data), (index_, data_) in zip(nested_enum(it2, depth=2), it2__):
            self.assertEqual(index, index_)
            self.assertEqual(data, data_)
            append_nested(nested, data, index)
        self.assertEqual(it2, nested)


