from contextlib import contextmanager

import pytest
import numpy as np
import tensorflow as tf

from tfsnippet.utils import *


@contextmanager
def scoped_set_config(value, getter, setter):
    old_value = getter()
    try:
        setter(value)
        yield
    finally:
        setter(old_value)


class AssertionTestCase(tf.test.TestCase):

    def test_default_settings(self):
        self.assertTrue(is_assertion_enabled())

    def test_assertion_enabled(self):
        ph = tf.placeholder(dtype=tf.int32, shape=())
        with scoped_set_config(True,
                               getter=is_assertion_enabled,
                               setter=set_assertion_enabled):
            self.assertTrue(is_assertion_enabled())
            assert_op = maybe_assert(
                tf.assert_equal, 1, ph, message='assertion is enabled')
            with control_deps([assert_op]):
                x = tf.constant(1, dtype=tf.int32)
        with pytest.raises(Exception, match='assertion is enabled'):
            with self.test_session() as sess:
                _ = sess.run(x, feed_dict={ph: 2})

    def test_assertion_disabled(self):
        with scoped_set_config(False,
                               getter=is_assertion_enabled,
                               setter=set_assertion_enabled):
            self.assertFalse(is_assertion_enabled())
            assert_op = maybe_assert(
                tf.assert_equal, 1, 2, message='assertion is enabled')
            with control_deps([assert_op]):
                x = tf.constant(1, dtype=tf.int32)
        with self.test_session() as sess:
            self.assertEqual(sess.run(x), 1)


class CheckNumericsTestCase(tf.test.TestCase):

    def test_default_settings(self):
        self.assertFalse(should_check_numerics())

    def test_check_numerics(self):
        ph = tf.placeholder(dtype=tf.float32, shape=())
        with scoped_set_config(True,
                               getter=should_check_numerics,
                               setter=set_check_numerics):
            self.assertTrue(should_check_numerics())
            x = check_numerics(ph, message='numerical issues')
        with pytest.raises(Exception, match='numerical issues'):
            with self.test_session() as sess:
                _ = sess.run(x, feed_dict={ph: np.nan})

    def test_not_check_numerics(self):
        with scoped_set_config(False,
                               getter=should_check_numerics,
                               setter=set_check_numerics):
            self.assertFalse(should_check_numerics())
            x = check_numerics(tf.constant(np.nan), message='numerical issues')
        with self.test_session() as sess:
            self.assertTrue(np.isnan(sess.run(x)))
