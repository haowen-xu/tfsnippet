import os
import unittest

from mock import Mock

from tfsnippet.utils import *


class HumanizeDurationTestCase(unittest.TestCase):
    cases = [
        (0.0, '0 sec'),
        (1e-8, '1e-08 sec'),
        (0.1, '0.1 sec'),
        (1.0, '1 sec'),
        (1, '1 sec'),
        (1.1, '1.1 secs'),
        (59, '59 secs'),
        (59.9, '59.9 secs'),
        (60, '1 min'),
        (61, '1 min 1 sec'),
        (62, '1 min 2 secs'),
        (119, '1 min 59 secs'),
        (120, '2 mins'),
        (121, '2 mins 1 sec'),
        (122, '2 mins 2 secs'),
        (3599, '59 mins 59 secs'),
        (3600, '1 hr'),
        (3601, '1 hr 1 sec'),
        (3661, '1 hr 1 min 1 sec'),
        (86399, '23 hrs 59 mins 59 secs'),
        (86400, '1 day'),
        (86401, '1 day 1 sec'),
        (172799, '1 day 23 hrs 59 mins 59 secs'),
        (259199, '2 days 23 hrs 59 mins 59 secs'),
    ]

    def test_positive(self):
        for seconds, answer in self.cases:
            result = humanize_duration(seconds)
            self.assertEqual(
                result, answer,
                msg='humanize_duraion({!r}) is expected to be {!r}, '
                    'but got {!r}.'.format(seconds, answer, result)
            )

    def test_negative(self):
        for seconds, answer in self.cases[1:]:
            seconds = -seconds
            answer = answer + ' ago'
            result = humanize_duration(seconds)
            self.assertEqual(
                result, answer,
                msg='humanize_duraion({!r}) is expected to be {!r}, '
                    'but got {!r}.'.format(seconds, answer, result)
            )


class CamelToUnderscoreTestCase(unittest.TestCase):
    def assert_convert(self, camel, underscore):
        self.assertEqual(
            camel_to_underscore(camel),
            underscore,
            msg='{!r} should be converted to {!r}'.format(camel, underscore)
        )

    def test_camel_to_underscore(self):
        examples = [
            ('simpleTest', 'simple_test'),
            ('easy', 'easy'),
            ('HTML', 'html'),
            ('simpleXML', 'simple_xml'),
            ('PDFLoad', 'pdf_load'),
            ('startMIDDLELast', 'start_middle_last'),
            ('AString', 'a_string'),
            ('Some4Numbers234', 'some4_numbers234'),
            ('TEST123String', 'test123_string'),
        ]
        for camel, underscore in examples:
            self.assert_convert(camel, underscore)
            self.assert_convert(underscore, underscore)
            self.assert_convert('_{}_'.format(camel),
                                '_{}_'.format(underscore))
            self.assert_convert('_{}_'.format(underscore),
                                '_{}_'.format(underscore))
            self.assert_convert('__{}__'.format(camel),
                                '__{}__'.format(underscore))
            self.assert_convert('__{}__'.format(underscore),
                                '__{}__'.format(underscore))
            self.assert_convert(
                '_'.join([s.capitalize() for s in underscore.split('_')]),
                underscore
            )
            self.assert_convert(
                '_'.join([s.upper() for s in underscore.split('_')]),
                underscore
            )


class NotSetTestCase(unittest.TestCase):

    def test_repr(self):
        self.assertEqual(repr(NOT_SET), 'NOT_SET')


class _CachedPropertyHelper(object):

    def __init__(self, value):
        self.value = value

    @cached_property('_cached_value')
    def cached_value(self):
        return self.value


class CachedPropertyTestCase(unittest.TestCase):

    def test_cached_property(self):
        o = _CachedPropertyHelper(0)
        self.assertFalse(hasattr(o, '_cached_value'))
        o.value = 123
        self.assertEqual(o.cached_value, 123)
        self.assertTrue(hasattr(o, '_cached_value'))
        self.assertEqual(o._cached_value, 123)
        o.value = 456
        self.assertEqual(o.cached_value, 123)
        self.assertEqual(o._cached_value, 123)

    def test_clear_cached_property(self):
        o = _CachedPropertyHelper(123)
        _ = o.cached_value
        clear_cached_property(o, '_cached_value')
        o.value = 456
        self.assertFalse(hasattr(o, '_cached_value'))
        self.assertEqual(o.cached_value, 456)
        self.assertEqual(o._cached_value, 456)


class MaybeCloseTestCase(unittest.TestCase):

    def test_maybe_close(self):
        # test having `close()`
        f = Mock(close=Mock(return_value=None))
        with maybe_close(f):
            self.assertFalse(f.close.called)
        self.assertTrue(f.close.called)

        # test having not `close()`
        with maybe_close(1):
            pass


class IterFilesTestCase(unittest.TestCase):

    def test_iter_files(self):
        names = ['a/1.txt', 'a/2.txt', 'a/b/1.txt', 'a/b/2.txt',
                 'b/1.txt', 'b/2.txt', 'c.txt']

        with TemporaryDirectory() as tempdir:
            for name in names:
                f_path = os.path.join(tempdir, name)
                f_dir = os.path.split(f_path)[0]
                makedirs(f_dir, exist_ok=True)
                with open(f_path, 'wb') as f:
                    f.write(b'')

            self.assertListEqual(names, sorted(iter_files(tempdir)))
            self.assertListEqual(names, sorted(iter_files(tempdir + '/a/../')))


if __name__ == '__main__':
    unittest.main()
