from unittest import TestCase
import numpy as np
from numpy.testing import assert_allclose
from pyulib import normalize_non_zero


class TestNormalize_non_zero(TestCase):
    def test_normalize_non_zero(self):
        # 1D
        assert_allclose(normalize_non_zero(np.array([0., 1, 0, 1])), [0, 0.5, 0, 0.5])
        assert_allclose(normalize_non_zero(np.array([0., 0, 0, 0])), [0, 0, 0, 0])
        # 2D
        arr = np.array([[0, 0, 1, 1],
                        [0, 0, 0, 0],
                        [0, 1, 0, 1]])
        # should raise TypeError because arr is of integert typen
        self.assertRaises(TypeError, lambda arr: normalize_non_zero(arr), arr)
        # after concersion everything should be fine
        arr = arr.astype(np.float)
        assert_allclose(normalize_non_zero(arr),
                        [[0, 0, 0.5, 0.5],
                         [0, 0, 0, 0],
                         [0, 0.5, 0, 0.5]])
        assert_allclose(normalize_non_zero(arr, axis=0),
                        [[0, 0, 1, 0.5],
                         [0, 0, 0, 0],
                         [0, 1, 0, 0.5]])
        # 3D
        arr = np.array([[[0, 0, 1, 1],
                         [0, 0, 0, 0],
                         [0, 1, 0, 1]],
                        [[0, 0, 0, 1],
                         [1, 0, 0, 0],
                         [0, 1, 0, 1]]], dtype=float)
        assert_allclose(normalize_non_zero(arr.copy(), axis=0),
                        [[[0, 0, 1, 0.5],
                          [0, 0, 0, 0],
                          [0, 0.5, 0, 0.5]],
                         [[0, 0, 0, 0.5],
                          [1, 0, 0, 0],
                          [0, 0.5, 0, 0.5]]])
        assert_allclose(normalize_non_zero(arr.copy(), axis=1),
                        [[[0, 0, 1, 0.5],
                          [0, 0, 0, 0],
                          [0, 1, 0, 0.5]],
                         [[0, 0, 0, 0.5],
                          [1, 0, 0, 0],
                          [0, 1, 0, 0.5]]])
        assert_allclose(normalize_non_zero(arr.copy(), axis=2),
                        [[[0, 0, 0.5, 0.5],
                          [0, 0, 0, 0],
                          [0, 0.5, 0, 0.5]],
                         [[0, 0, 0, 1],
                          [1, 0, 0, 0],
                          [0, 0.5, 0, 0.5]]])
        assert_allclose(normalize_non_zero(arr.copy(), axis=(0, 1)),
                        [[[0, 0, 1, 0.25],
                          [0, 0, 0, 0],
                          [0, 0.5, 0, 0.25]],
                         [[0, 0, 0, 0.25],
                          [1, 0, 0, 0],
                          [0, 0.5, 0, 0.25]]])
        assert_allclose(normalize_non_zero(arr.copy(), axis=(1, 2)),
                        [[[0, 0, 0.25, 0.25],
                          [0, 0, 0, 0],
                          [0, 0.25, 0, 0.25]],
                         [[0, 0, 0, 0.25],
                          [0.25, 0, 0, 0],
                          [0, 0.25, 0, 0.25]]])
        assert_allclose(normalize_non_zero(arr.copy(), axis=(2, 0)),
                        [[[0, 0, 1/3, 1/3],
                          [0, 0, 0, 0],
                          [0, 0.25, 0, 0.25]],
                         [[0, 0, 0, 1/3],
                          [1, 0, 0, 0],
                          [0, 0.25, 0, 0.25]]])
