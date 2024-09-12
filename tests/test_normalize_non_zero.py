from unittest import TestCase
from itertools import product

import numpy as np
from numpy.testing import assert_allclose
import torch

from pyulib import normalize_non_zero


class TestNormalize_non_zero(TestCase):
    def test_normalize_non_zero(self):
        # test raise for unknown array type
        self.assertRaises(TypeError, lambda: normalize_non_zero([ 1, 1, 1]))
        # test raise for negative values
        self.assertRaises(TypeError, lambda: normalize_non_zero([1, 1, -1]))
        # test for both numpy and pytorch
        for lib, uniform in product([
            "numpy",
            "pytorch",
        ], [
            True,
            False,
        ]):
            if lib == "numpy":
                make_arr = np.array
                to_float = lambda arr: arr.astype(float)
                copy = lambda arr: arr.copy()
            elif lib == "pytorch":
                make_arr = torch.tensor
                to_float = lambda arr: arr.type(torch.FloatTensor)
                copy = lambda arr: arr.clone()
            else:
                raise  ValueError(f"Unknown lib {lib}")
            self.assertWarns(DeprecationWarning, lambda: normalize_non_zero(make_arr([0., 1, 0, 1])))
            # 1D
            assert_allclose(normalize_non_zero(make_arr([0., 1, 0, 1]), axis=None), [0, 0.5, 0, 0.5])
            assert_allclose(normalize_non_zero(make_arr([0., 0, 0, 0]), axis=None), [0, 0, 0, 0])
            # 2D
            arr = make_arr([[0, 0, 1, 1],
                            [0, 0, 0, 0],
                            [0, 1, 0, 1]])
            # should raise TypeError because arr is of integer type
            self.assertRaises(TypeError, lambda arr: normalize_non_zero(arr, make_zeros_uniform=uniform), arr)
            # after concersion everything should be fine
            arr = to_float(arr)
            assert_allclose(normalize_non_zero(arr, axis=None, make_zeros_uniform=uniform),
                            [[0, 0, 0.25, 0.25],
                             [0, 0, 0, 0],
                             [0, 0.25, 0, 0.25]])
            assert_allclose(normalize_non_zero(arr, axis=1, make_zeros_uniform=uniform),
                            [[0, 0, 0.5, 0.5],
                             [0.25, 0.25, 0.25, 0.25],
                             [0, 0.5, 0, 0.5]] if uniform else
                            [[0, 0, 0.5, 0.5],
                             [0, 0, 0, 0],
                             [0, 0.5, 0, 0.5]])
            assert_allclose(normalize_non_zero(arr, axis=0, make_zeros_uniform=uniform),
                            [[1/3, 0, 1, 0.5],
                             [1/3, 0, 0, 0],
                             [1/3, 1, 0, 0.5]] if uniform else
                            [[0, 0, 1, 0.5],
                             [0, 0, 0, 0],
                             [0, 1, 0, 0.5]])
            # 3D
            arr = make_arr([[[0, 0, 1, 1],
                             [0, 0, 0, 0],
                             [0, 1, 0, 1]],
                            [[0, 0, 0, 1],
                             [1, 0, 0, 0],
                             [0, 1, 0, 1]]], dtype=float)
            assert_allclose(normalize_non_zero(copy(arr), axis=0, make_zeros_uniform=uniform),
                            [[[0.5, 0.5, 1, 0.5],
                              [0, 0.5, 0.5, 0.5],
                              [0.5, 0.5, 0.5, 0.5]],
                             [[0.5, 0.5, 0, 0.5],
                              [1, 0.5, 0.5, 0.5],
                              [0.5, 0.5, 0.5, 0.5]]] if uniform else
                            [[[0, 0, 1, 0.5],
                              [0, 0, 0, 0],
                              [0, 0.5, 0, 0.5]],
                             [[0, 0, 0, 0.5],
                              [1, 0, 0, 0],
                              [0, 0.5, 0, 0.5]]])
            assert_allclose(normalize_non_zero(copy(arr), axis=1, make_zeros_uniform=uniform),
                            [[[1 / 3, 0, 1, 0.5],
                              [1 / 3, 0, 0, 0],
                              [1 / 3, 1, 0, 0.5]],
                             [[0, 0, 1 / 3, 0.5],
                              [1, 0, 1 / 3, 0],
                              [0, 1, 1 / 3, 0.5]]] if uniform else
                            [[[0, 0, 1, 0.5],
                              [0, 0, 0, 0],
                              [0, 1, 0, 0.5]],
                             [[0, 0, 0, 0.5],
                              [1, 0, 0, 0],
                              [0, 1, 0, 0.5]]])
            assert_allclose(normalize_non_zero(copy(arr), axis=2, make_zeros_uniform=uniform),
                            [[[0, 0, 0.5, 0.5],
                              [0.25, 0.25, 0.25, 0.25],
                              [0, 0.5, 0, 0.5]],
                             [[0, 0, 0, 1],
                              [1, 0, 0, 0],
                              [0, 0.5, 0, 0.5]]] if uniform else
                            [[[0, 0, 0.5, 0.5],
                              [0, 0, 0, 0],
                              [0, 0.5, 0, 0.5]],
                             [[0, 0, 0, 1],
                              [1, 0, 0, 0],
                              [0, 0.5, 0, 0.5]]])
            assert_allclose(normalize_non_zero(copy(arr), axis=(0, 1), make_zeros_uniform=uniform),
                            [[[0, 0, 1, 0.25],
                              [0, 0, 0, 0],
                              [0, 0.5, 0, 0.25]],
                             [[0, 0, 0, 0.25],
                              [1, 0, 0, 0],
                              [0, 0.5, 0, 0.25]]])
            assert_allclose(normalize_non_zero(copy(arr), axis=(1, 2), make_zeros_uniform=uniform),
                            [[[0, 0, 0.25, 0.25],
                              [0, 0, 0, 0],
                              [0, 0.25, 0, 0.25]],
                             [[0, 0, 0, 0.25],
                              [0.25, 0, 0, 0],
                              [0, 0.25, 0, 0.25]]])
            assert_allclose(normalize_non_zero(copy(arr), axis=(2, 0), make_zeros_uniform=uniform),
                            [[[0, 0, 1/3, 1/3],
                              [0, 0, 0, 0],
                              [0, 0.25, 0, 0.25]],
                             [[0, 0, 0, 1/3],
                              [1, 0, 0, 0],
                              [0, 0.25, 0, 0.25]]])
