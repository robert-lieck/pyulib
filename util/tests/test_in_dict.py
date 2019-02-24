from unittest import TestCase
from util import in_dict_and_has_value, in_dict_and_true, in_dict_and_not_none


class TestIn_dict(TestCase):

    def test_in_dict(self):
        d = {'key1': 'val1', 'key2': 'val2', 'key3': True, 'key4': False, 'key5': None}
        for key in list(d.keys()) + ['key6', 'key7']:
            # general value check
            if key == 'key1':
                self.assertTrue(in_dict_and_has_value(key=key,
                                                      dictionary=d,
                                                      value='val1'))
            else:
                self.assertFalse(in_dict_and_has_value(key=key,
                                                       dictionary=d,
                                                       value='val1'))
            # True check
            if key == 'key3':
                self.assertTrue(in_dict_and_true(key=key, dictionary=d))
            else:
                self.assertFalse(in_dict_and_true(key=key, dictionary=d))
            # None check
            if key not in d or key == 'key5':
                self.assertFalse(in_dict_and_not_none(key=key, dictionary=d))
            else:
                self.assertTrue(in_dict_and_not_none(key=key, dictionary=d))
