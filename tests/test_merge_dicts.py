from unittest import TestCase
from pyulib import merge_dicts


class TestMerge_dicts(TestCase):
    def test_merge_dicts(self):
        d1 = {'1': 1}
        d2 = {'2': 2, '3': {'x': 1, 'z': 30, 'A': {'B': 'C'}}}
        d3 = {'2': 20, '3': {'x': 10, 'y': 20, 'A': {'B': 'D', 'E': 'F'}}}
        # infinite depth
        self.assertEqual(merge_dicts(d1, d2),
                         {'1': 1, '2': 2, '3': {'x': 1, 'z': 30, 'A': {'B': 'C'}}})
        self.assertEqual(merge_dicts(d1, d3),
                         {'1': 1, '2': 20, '3': {'x': 10, 'y': 20, 'A': {'B': 'D', 'E': 'F'}}})
        self.assertEqual(merge_dicts(d2, d3),
                         {'2': 2, '3': {'x': 1, 'y': 20, 'z': 30, 'A': {'B': 'C', 'E': 'F'}}})
        self.assertEqual(merge_dicts(d1, d2, d3),
                         {'1': 1, '2': 2, '3': {'x': 1, 'y': 20, 'z': 30, 'A': {'B': 'C', 'E': 'F'}}})
        # depth = 1
        self.assertEqual(merge_dicts(d1, d2, depth=1),
                         {'1': 1, '2': 2, '3': {'x': 1, 'z': 30, 'A': {'B': 'C'}}})
        self.assertEqual(merge_dicts(d1, d3, depth=1),
                         {'1': 1, '2': 20, '3': {'x': 10, 'y': 20, 'A': {'B': 'D', 'E': 'F'}}})
        self.assertEqual(merge_dicts(d2, d3, depth=1),
                         {'2': 2, '3': {'x': 1, 'y': 20, 'z': 30, 'A': {'B': 'C'}}})
        self.assertEqual(merge_dicts(d1, d2, d3, depth=1),
                         {'1': 1, '2': 2, '3': {'x': 1, 'y': 20, 'z': 30, 'A': {'B': 'C'}}})
        # depth = 0
        self.assertEqual(merge_dicts(d1, d2, depth=0),
                         {'1': 1, '2': 2, '3': {'x': 1, 'z': 30, 'A': {'B': 'C'}}})
        self.assertEqual(merge_dicts(d1, d3, depth=0),
                         {'1': 1, '2': 20, '3': {'x': 10, 'y': 20, 'A': {'B': 'D', 'E': 'F'}}})
        self.assertEqual(merge_dicts(d2, d3, depth=0),
                         {'2': 2, '3': {'x': 1, 'z': 30, 'A': {'B': 'C'}}})
        self.assertEqual(merge_dicts(d1, d2, d3, depth=0),
                         {'1': 1, '2': 2, '3': {'x': 1, 'z': 30, 'A': {'B': 'C'}}})
