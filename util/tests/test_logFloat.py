from unittest import TestCase
from util import LogFloat
import numpy as np
from numpy.testing import assert_allclose


class TestLogFloat(TestCase):
    def test___new__(self):
        self.assertEqual(LogFloat(1), 1)
        self.assertEqual(LogFloat(1)._value, 0)
        self.assertEqual(LogFloat(0), 0)
        self.assertEqual(LogFloat(0)._value, -np.inf)
        self.assertEqual(LogFloat(0) + 1, 1)
        self.assertEqual((LogFloat(0) + 1)._value, 0)
        n = 100
        for x, y in np.random.uniform(0, 2, (100, 2)):
            x_ = LogFloat(x)
            y_ = LogFloat(y)
            for a, b, c, d in [(x, y, x_, y_),
                               (x, y, x_, y),
                               (x, y, x, y_),
                               (x_, y_, x_, y),
                               (x_, y_, x, y_),
                               (x_, y, x, y_)]:
                # print(a, b, c, d)
                # print(type(a), type(b), type(c), type(d))
                # comparison
                if x == y:
                    self.assertTrue(a == b)
                    self.assertTrue(c == d)
                if x != y:
                    self.assertTrue(a != b)
                    self.assertTrue(c != d)
                if x < y:
                    self.assertTrue(a < b)
                    self.assertTrue(c < d)
                if x > y:
                    self.assertTrue(a > b)
                    self.assertTrue(c > d)
                if x <= y:
                    self.assertTrue(a <= b)
                    self.assertTrue(c <= d)
                if x >= y:
                    self.assertTrue(a >= b)
                    self.assertTrue(c >= d)
                # arithmetics
                assert_allclose(float(a + b), float(c + d))
                if x >= y:
                    assert_allclose(float(a - b), float(c - d))
                else:
                    with self.assertRaises(ValueError):
                        print(float(a - b), float(c - d))
                assert_allclose(float(a * b), float(c * d))
                assert_allclose(float(a / b), float(c / d))
