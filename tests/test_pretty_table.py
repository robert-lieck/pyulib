from unittest import TestCase
from util import pretty_table


class TestPretty_table(TestCase):
    def test_pretty_table(self):
        t = [['a', 'b', 'c'],
             ['aa', 'bb', 'cc'],
             ['aaa', 'bbb', 'ccc']]
        self.assertEqual(pretty_table(t), 'a   b   c  \naa  bb  cc \naaa bbb ccc')
        self.assertEqual(pretty_table(t, alignment='r'), '  a   b   c\n aa  bb  cc\naaa bbb ccc')
        self.assertEqual(pretty_table(t, alignment=['l', 'r', 'l']), 'a     b c  \naa   bb cc \naaa bbb ccc')
