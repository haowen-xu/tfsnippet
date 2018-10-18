import os
import unittest

from mock import Mock
import numpy as np

from tfsnippet.utils import *


class HumanizeDurationTestCase(unittest.TestCase):
    cases = [
        (0.0, '0 second'),
        (1e-8, '1e-08 second'),
        (0.1, '0.1 second'),
        (1.0, '1 second'),
        (1, '1 second'),
        (1.1, '1.1 seconds'),
        (59, '59 seconds'),
        (59.9, '59.9 seconds'),
        (60, '1 minute'),
        (61, '1 minute 1 second'),
        (62, '1 minute 2 seconds'),
        (119, '1 minute 59 seconds'),
        (120, '2 minutes'),
        (121, '2 minutes 1 second'),
        (122, '2 minutes 2 seconds'),
        (3599, '59 minutes 59 seconds'),
        (3600, '1 hour'),
        (3601, '1 hour 1 second'),
        (3661, '1 hour 1 minute 1 second'),
        (86399, '23 hours 59 minutes 59 seconds'),
        (86400, '1 day'),
        (86401, '1 day 1 second'),
        (172799, '1 day 23 hours 59 minutes 59 seconds'),
        (259199, '2 days 23 hours 59 minutes 59 seconds'),
    ]

    @staticmethod
    def long_to_short(s):
        for u in ('day', 'hour', 'minute', 'second'):
            s = s.replace(' ' + u + 's', u[0]).replace(' ' + u, u[0])
        return s

    def test_positive(self):
        for seconds, answer in self.cases:
            result = humanize_duration(seconds, short_units=False)
            self.assertEqual(
                result, answer,
                msg='humanize_duraion({!r}) is expected to be {!r}, '
                    'but got {!r}.'.format(seconds, answer, result)
            )
            result = humanize_duration(seconds, short_units=True)
            answer = self.long_to_short(answer)
            self.assertEqual(
                result, answer,
                msg='humanize_duraion({!r}) is expected to be {!r}, '
                    'but got {!r}.'.format(seconds, answer, result)
            )

    def test_negative(self):
        for seconds, answer in self.cases[1:]:
            seconds = -seconds
            answer = answer + ' ago'
            result = humanize_duration(seconds, short_units=False)
            self.assertEqual(
                result, answer,
                msg='humanize_duraion({!r}) is expected to be {!r}, '
                    'but got {!r}.'.format(seconds, answer, result)
            )
            result = humanize_duration(seconds, short_units=True)
            answer = self.long_to_short(answer)
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


class ETATestCase(unittest.TestCase):

    def test_snapshot(self):
        eta = ETA(take_initial_snapshot=False)
        self.assertListEqual([], eta._times)
        self.assertListEqual([], eta._progresses)

        eta = ETA(take_initial_snapshot=True)
        self.assertEquals(1, len(eta._times))
        self.assertListEqual([0.], eta._progresses)

        eta.take_snapshot(.5)
        self.assertEquals(2, len(eta._times))
        self.assertGreaterEqual(eta._times[1], eta._times[0])
        self.assertListEqual([0., .5], eta._progresses)

        eta.take_snapshot(.50001)
        self.assertEquals(2, len(eta._times))
        self.assertListEqual([0., .5], eta._progresses)

        eta.take_snapshot(1., 12345)
        self.assertEquals(3, len(eta._times))
        self.assertEquals(12345, eta._times[-1])
        self.assertListEqual([0., .5, 1.], eta._progresses)

    def test_get_eta(self):
        self.assertIsNone(ETA(take_initial_snapshot=False).get_eta(0.))

        eta = ETA(take_initial_snapshot=False)
        eta.take_snapshot(0., 0)

        self.assertListEqual([0], eta._times)
        self.assertListEqual([0.], eta._progresses)
        np.testing.assert_allclose(3., eta.get_eta(.25, 1, take_snapshot=False))
        self.assertListEqual([0], eta._times)
        self.assertListEqual([0.], eta._progresses)

        np.testing.assert_allclose(99., eta.get_eta(.01, 1))
        self.assertListEqual([0, 1], eta._times)
        self.assertListEqual([0., .01], eta._progresses)

        np.testing.assert_allclose(57.0, eta.get_eta(.05, 3))
        self.assertListEqual([0, 1, 3], eta._times)
        self.assertListEqual([0., .01, .05], eta._progresses)


if __name__ == '__main__':
    unittest.main()
