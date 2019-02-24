from unittest import TestCase
import numpy as np
from util import NDSlicable, NestedOutput
from util import NestedOutputDummy as NO


class TestNDSlicable(TestCase):

    def test___getitem__(self):
        for iteration in range(10):
            arr = np.random.randint(0, 10, (2, 3, 4))
            arr_view = NDSlicable(object=arr, dimensionality=len(arr.shape))
            with NO():
                NO.print(f"iteration: {iteration}")
                with NO():
                    index = tuple([np.random.randint(0, n - 1) for n in arr.shape])
                    NO.print(f"index: {index}")
                    NO.print("checking explicit indexing")
                    with NO():
                        self.assertEqual(arr[index], arr_view[index])
                        NO.print(f"{arr[index]} == {arr_view[index]}")
                    NO.print("checking stepwise indexing")
                    with NO():
                        new_view = arr_view
                        for idx in index:
                            NO.print(new_view._dimensions)
                            new_view = new_view[idx]
                        self.assertEqual(arr[index], new_view)
                        NO.print(f"{arr[index]} == {new_view}")
                    NO.print("checking iteration")
                    with NO():
                        for idx_1, slice_1 in enumerate(arr_view):
                            NO.print(f"index 1: {idx_1}")
                            with NO():
                                for idx_2, slice_2 in enumerate(slice_1):
                                    NO.print(f"index 2: {idx_2}")
                                    with NO():
                                        for idx_3, slice_3 in enumerate(slice_2):
                                            NO.print(f"index 3: {idx_3}")
                                            with NO():
                                                self.assertEqual(slice_3, arr[idx_1, idx_2, idx_3])
                                                NO.print(f"{slice_3} == {arr[idx_1, idx_2, idx_3]}")

    def test_slicing(self):
        seq = [list(range(i)) for i in range(2, 12)]
        view = NDSlicable(object=seq, dimensionality=2, tuple_indices=False)
        NO.print(view.to_list())
        self.assertEqual(str(view.to_list()), str(seq))
        view = view[1:8:2, 1::2]
        NO.print(view.to_list())
        self.assertEqual(view.to_list(), [list(range(1, i, 2)) for i in range(2, 9, 2)])
        view = view[1:]
        NO.print(view.to_list())
        self.assertEqual(view.to_list(), [list(range(1, i, 2)) for i in range(4, 9, 2)])
        view = view[None, 1:]
        NO.print(view.to_list())
        self.assertEqual(view.to_list(), [list(range(3, i, 2)) for i in range(4, 9, 2)])
        view = view[1:, ::2]
        NO.print(view.to_list())
        self.assertEqual(view.to_list(), [list(range(3, i, 4)) for i in range(6, 9, 2)])

    def test_to_list(self):
        for dim in range(1, 4):
            for _ in range(5):
                N = 10
                arr = np.random.randint(0, 100, (N,) * dim)
                low_high = [
                    [int(i + idx) for idx, i in enumerate(sorted(np.random.randint(0, N - 1, 2)))]
                    for _ in range(dim)]
                low = tuple(i[0] for i in low_high)
                slice_idx = tuple(slice(i[0], i[1], None) for i in low_high)
                self.assertTrue(np.all(
                    np.array(NDSlicable(arr, dim).to_list()) == arr
                ))
                self.assertEqual(NDSlicable(arr, dim)[low],
                                 arr[low])
                self.assertTrue(np.all(
                    np.array(NDSlicable(arr, dim)[slice_idx].to_list()) == arr[slice_idx]
                ))
