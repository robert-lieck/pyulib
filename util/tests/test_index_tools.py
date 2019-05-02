from unittest import TestCase
import numpy as np
from numpy.testing import assert_array_equal
from util import IndexRecorder, Index, index_to_tuple, tuple_to_index


def random_index(max_idx=1, min_idx=0, p_None=0.2):
    if np.random.random() < p_None:
        return None
    else:
        return np.random.randint(min_idx, max_idx + 1)


def random_slice_or_index(max_idx, p_slice=0.5):
    if np.random.random() < p_slice:
        start = random_index(max_idx)
        min_idx = 0 if start is None else start
        stop = random_index(min_idx=min_idx, max_idx=max_idx)
        max_step = max(min_idx, 0 if stop is None else stop, 1)
        step = random_index(min_idx=1, max_idx=max_step)
        return slice(start, stop, step)
    else:
        return random_index(max_idx=max_idx)


def random_slice_tuple(max_dim, max_idx):
    dim = np.random.randint(1, max_dim + 1)
    if dim == 1:
        return random_slice_or_index(max_idx=max_idx)
    else:
        return tuple(random_slice_or_index(max_idx=max_idx) for _ in range(dim))


class TestIndexRecorder(TestCase):

    def test_recording(self):
        np.random.seed(0)
        recorder_set = set()
        for _ in range(10):
            # define dimensions
            max_dim = 3
            max_idx = 20
            # create recorder object and array
            recorder = IndexRecorder()
            arr1 = np.zeros(tuple(max_idx + 1 for _ in range(max_dim)))
            # assign random stuff
            for idx in [random_slice_tuple(max_dim=max_dim, max_idx=max_idx) for _ in range(100)]:
                val = np.random.random()
                # print(f"{idx}: {val}")
                recorder[idx] = val
                arr1[idx] = val
            # create array of same dimensions, apply recorded assignments, and check for equality
            arr2 = np.zeros_like(arr1)
            recorder.apply(arr2)
            assert_array_equal(arr1, arr2)
            # freeze recorder and put into set
            recorder.freeze()
            recorder_set.add(recorder)
            # check assignment to frozen recorder raises exception
            try:
                recorder[0] = 0
            except IndexError:
                # as expected
                pass
            else:
                self.fail("should raise exception")
            # same as above with frozen recorder
            arr3 = np.zeros_like(arr1)
            recorder.apply(arr3)
            assert_array_equal(arr1, arr3)


class TestIndexConversion(TestCase):

    def test_conversion(self):
        np.random.seed(0)
        max_dim = 3
        max_idx = 20
        for idx in [random_slice_tuple(max_dim=max_dim, max_idx=max_idx) for _ in range(1000)]:
            # convert to tuple
            t = index_to_tuple(idx)
            # check that it is int or tuple
            self.assertTrue(isinstance(t, (int, tuple)), f"Is not int or tuple: {t}")
            # make sure hashing works
            hash(t)
            # convert back
            idx_ = tuple_to_index(t)
            # check for equality
            self.assertEqual(idx, idx_)
