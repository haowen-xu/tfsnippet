import os
import unittest
from threading import Thread

import pytest
import six
import numpy as np
import tensorflow as tf
from mock import Mock

from tfsnippet.utils import *

if six.PY2:
    LONG_MAX = long(1) << 63 - long(1)
else:
    LONG_MAX = 1 << 63 - 1


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
        self.assertEqual(1, len(eta._times))
        self.assertListEqual([0.], eta._progresses)

        eta.take_snapshot(.5)
        self.assertEqual(2, len(eta._times))
        self.assertGreaterEqual(eta._times[1], eta._times[0])
        self.assertListEqual([0., .5], eta._progresses)

        eta.take_snapshot(.50001)
        self.assertEqual(2, len(eta._times))
        self.assertListEqual([0., .5], eta._progresses)

        eta.take_snapshot(1., 12345)
        self.assertEqual(3, len(eta._times))
        self.assertEqual(12345, eta._times[-1])
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


class ContextStackTestCase(unittest.TestCase):

    def test_thread_local_and_initial_factory(self):
        stack = ContextStack(dict)
        thread_top = [None] * 10

        def thread_job(i):
            thread_top[i] = stack.top()

        threads = [
            Thread(target=thread_job, args=(i,))
            for i, _ in enumerate(thread_top)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        for i, top in enumerate(thread_top):
            for j, top2 in enumerate(thread_top):
                if i != j:
                    self.assertIsNot(top, top2)

    def test_push_and_pop(self):
        stack = ContextStack()
        self.assertIsNone(stack.top())

        # push the first layer
        first_layer = object()
        stack.push(first_layer)
        self.assertIs(stack.top(), first_layer)

        # push the second layer
        second_layer = object()
        stack.push(second_layer)
        self.assertIs(stack.top(), second_layer)

        # pop the second layer
        stack.pop()
        self.assertIs(stack.top(), first_layer)

        # pop the first layer
        stack.pop()
        self.assertIsNone(stack.top())


class ValidateNSamplesArgTestCase(tf.test.TestCase):

    def test_static_values(self):
        # type checks
        for o in [object(), 1.2, LONG_MAX]:
            with pytest.raises(
                    TypeError, match='xyz cannot be converted to int32'):
                _ = validate_n_samples_arg(o, 'xyz')

        # value checks
        self.assertIsNone(validate_n_samples_arg(None, 'xyz'))
        self.assertEqual(validate_n_samples_arg(1, 'xyz'), 1)
        with pytest.raises(ValueError, match='xyz must be positive'):
            _ = validate_n_samples_arg(0, 'xyz')
        with pytest.raises(ValueError, match='xyz must be positive'):
            _ = validate_n_samples_arg(-1, 'xyz')

    def test_dynamic_values(self):
        # type checks
        for o in [tf.constant(1.2, dtype=tf.float32),
                  tf.constant(LONG_MAX, dtype=tf.int64)]:
            with pytest.raises(
                    TypeError, match='xyz cannot be converted to int32'):
                _ = validate_n_samples_arg(o, 'xyz')

        # value checks
        with self.test_session():
            self.assertEqual(
                validate_n_samples_arg(
                    tf.constant(1, dtype=tf.int32), 'xyz').eval(), 1)
            with pytest.raises(Exception, match='xyz must be positive'):
                _ = validate_n_samples_arg(
                    tf.constant(0, dtype=tf.int32), 'xyz').eval()
            with pytest.raises(Exception, match='xyz must be positive'):
                _ = validate_n_samples_arg(
                    tf.constant(-1, dtype=tf.int32), 'xyz').eval()


class ValidateGroupNdimsTestCase(tf.test.TestCase):

    def test_static_values(self):
        # type checks
        for o in [object(), None, 1.2, LONG_MAX]:
            with pytest.raises(
                    TypeError,
                    match='group_ndims cannot be converted to int32'):
                _ = validate_group_ndims_arg(o)

        # value checks
        self.assertEqual(validate_group_ndims_arg(0), 0)
        self.assertEqual(validate_group_ndims_arg(1), 1)
        with pytest.raises(
                ValueError, match='group_ndims must be non-negative'):
            _ = validate_group_ndims_arg(-1)

    def test_dynamic_values(self):
        # type checks
        for o in [tf.constant(1.2, dtype=tf.float32),
                  tf.constant(LONG_MAX, dtype=tf.int64)]:
            with pytest.raises(
                    TypeError,
                    match='group_ndims cannot be converted to int32'):
                _ = validate_group_ndims_arg(o)

        # value checks
        with self.test_session():
            self.assertEqual(
                validate_group_ndims_arg(
                    tf.constant(0, dtype=tf.int32)).eval(),
                0
            )
            self.assertEqual(
                validate_group_ndims_arg(
                    tf.constant(1, dtype=tf.int32)).eval(),
                1
            )
            with pytest.raises(
                    Exception, match='group_ndims must be non-negative'):
                _ = validate_group_ndims_arg(
                    tf.constant(-1, dtype=tf.int32)).eval()


class ArgValidationTestCase(unittest.TestCase):

    def test_validate_enum_arg(self):
        # test good cases
        def good_case(value, choices, nullable):
            self.assertEqual(
                validate_enum_arg('arg', value, choices, nullable),
                value
            )

        good_case(1, (1, 2, 3), True)
        good_case(None, (1, 2, 3), True)
        good_case(1, (1, 2, None), False)
        good_case(None, (1, 2, None), False)

        # test error cases
        def error_case(value, choices, nullable):
            choices = tuple(choices)
            err_msg = 'Invalid value for argument `arg`: expected to be one ' \
                      'of {!r}, but got {!r}'.format(choices, value)
            err_msg = err_msg.replace('(', '\\(')
            err_msg = err_msg.replace(')', '\\)')
            with pytest.raises(ValueError, match=err_msg):
                _ = validate_enum_arg('arg', value, choices, nullable)

        error_case(4, (1, 2, 3), True)
        error_case(4, (1, 2, 3), False)
        error_case(None, (1, 2, 3), False)

    def test_validate_positive_int_arg(self):
        # test good cases
        self.assertEqual(validate_positive_int_arg('arg', 1), 1)

        # test error cases
        def error_case(value):
            err_msg = 'Invalid value for argument `arg`: expected to be a ' \
                      'positive integer, but got {!r}'.format(value)
            err_msg = err_msg.replace('(', '\\(')
            err_msg = err_msg.replace(')', '\\)')
            with pytest.raises(ValueError, match=err_msg):
                _ = validate_positive_int_arg('arg', value)

        error_case(None)
        error_case('x')
        error_case(0)
        error_case(-1)

    def test_validate_int_tuple_arg(self):
        # test good cases
        def good_case(value, expected):
            self.assertEqual(validate_int_tuple_arg('arg', value),
                             expected)

        good_case(1, (1,))
        good_case((1, 2), (1, 2))
        good_case(iter((1, 2)), (1, 2))
        good_case([1, 2], (1, 2))

        # test error cases
        def error_case(value):
            err_msg = 'Invalid value for argument `arg`: expected to be a ' \
                      'tuple of integers, but got {!r}'.format(value)
            err_msg = err_msg.replace('(', '\\(')
            err_msg = err_msg.replace(')', '\\)')
            with pytest.raises(ValueError, match=err_msg):
                _ = validate_int_tuple_arg('arg', value)

        error_case(None)
        error_case('x')
        error_case(('x', 1))

        # test good case for nullable
        self.assertIsNone(
            validate_int_tuple_arg('arg', None, nullable=True))
