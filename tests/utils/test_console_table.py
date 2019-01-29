import sys
import unittest
from collections import OrderedDict
from io import StringIO

import pytest

from tfsnippet.utils import ConsoleTable, print_as_table


class ConsoleTableTestCase(unittest.TestCase):

    def test_console_table(self):
        t = ConsoleTable(3, col_space=2, col_align=['<', '^', '>'])
        t.add_skip()
        t.add_hr('=')
        t.add_title('The title')
        t.add_hr('-')
        t.add_row(['application', 'b', 'c'])
        t.add_skip()
        t.add_row(['a', 'business', 'c'])
        t.add_hr('*')
        t.add_skip()
        t.add_row(['a', 'b', 'console'])
        t.add_hr('=')
        t.add_skip()
        t.add_title('Another title')

        self.assertEqual(
            str(t),
            '''==============================
The title
------------------------------
application     b            c

a            business        c
******************************
a               b      console
==============================
Another title'''
        )

        with pytest.raises(ValueError, match='`col_count` must be at least 1'):
            _ = ConsoleTable(0)
        with pytest.raises(ValueError, match='`col_space` must be at least 1'):
            _ = ConsoleTable(1, 0)
        with pytest.raises(ValueError, match='Invalid alignment: ?'):
            _ = ConsoleTable(2, col_align=['?', '<'])
        with pytest.raises(ValueError, match='The length of `col_align` must '
                                             'equal to `col_count`'):
            _ = ConsoleTable(2, col_align=['<'])

        with pytest.raises(ValueError, match='Expect exactly 3 columns'):
            t = ConsoleTable(3)
            t.add_row(['a', 'b'])

        with pytest.raises(ValueError, match='`c` must be exactly one character'):
            t = ConsoleTable(1)
            t.add_hr('==')

    def test_long_title(self):
        t = ConsoleTable(2)
        t.add_title('x' * 50)
        t.add_row(['a', 'b'])
        self.assertEqual(
            str(t),
            'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n'
            'a                                                b'
        )

        t = ConsoleTable(2, expand_col=1)
        t.add_title('x' * 50)
        t.add_row(['a', 'b'])
        self.assertEqual(
            str(t),
            'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n'
            'a   b'
        )

    def test_add_key_values(self):
        # test not sort keys
        t = ConsoleTable(2)
        t.add_key_values(
            OrderedDict([
                ('a', 123),
                ('b', 'the string'),
                ('c', [1, 2, 3]),
            ])
        )
        self.assertEqual(
            str(t),
            '''a   123
b   the string
c   [1, 2, 3]'''
        )

        # test sort keys
        t = ConsoleTable(2)
        t.add_key_values(
            [
                ('a', 123),
                ('c', [1, 2, 3]),
                ('b', 'the string'),
            ],
            sort_keys=True
        )
        self.assertEqual(
            str(t),
            '''a   123
b   the string
c   [1, 2, 3]'''
        )

        t = ConsoleTable(3)
        with pytest.raises(TypeError, match='cannot add a key-value sequence'):
            t.add_key_values({})

    def test_print_as_table(self):
        buf = StringIO()
        original_stdout = sys.stdout
        try:
            sys.stdout = buf
            print_as_table(
                'The config',
                [
                    ('a', 123),
                    ('b', 'the string'),
                    ('c', [1, 2, 3]),
                ],
                '*'
            )
        finally:
            sys.stdout = original_stdout
        self.assertEqual(
            buf.getvalue(),
            '''The config
**************
a   123
b   the string
c   [1, 2, 3]
'''
        )
