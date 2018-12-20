import pytest
import numpy as np
import tensorflow as tf

from tfsnippet.utils import *


class AssertionTestCase(tf.test.TestCase):

    def test_set_assertion_enabled(self):
        self.assertTrue(is_assertion_enabled())
        with scoped_set_assertion_enabled(False):
            self.assertFalse(is_assertion_enabled())
        self.assertTrue(is_assertion_enabled())

    def test_assertion_enabled(self):
        ph = tf.placeholder(dtype=tf.int32, shape=())
        with scoped_set_assertion_enabled(True):
            assert_op = maybe_assert(
                tf.assert_equal, 1, ph, message='assertion is enabled')
            with control_deps([assert_op]):
                x = tf.constant(1, dtype=tf.int32)
        with pytest.raises(Exception, match='assertion is enabled'):
            with self.test_session() as sess:
                _ = sess.run(x, feed_dict={ph: 2})

    def test_assertion_disabled(self):
        with scoped_set_assertion_enabled(False):
            assert_op = maybe_assert(
                tf.assert_equal, 1, 2, message='assertion is enabled')
            with control_deps([assert_op]):
                x = tf.constant(1, dtype=tf.int32)
        with self.test_session() as sess:
            self.assertEqual(sess.run(x), 1)


class CheckNumericsTestCase(tf.test.TestCase):

    def test_set_check_numerics(self):
        self.assertFalse(should_check_numerics())
        with scoped_set_check_numerics(True):
            self.assertTrue(should_check_numerics())
        self.assertFalse(should_check_numerics())

    def test_check_numerics(self):
        ph = tf.placeholder(dtype=tf.float32, shape=())
        with scoped_set_check_numerics(True):
            x = maybe_check_numerics(ph, message='numerical issues')
        with pytest.raises(Exception, match='numerical issues'):
            with self.test_session() as sess:
                _ = sess.run(x, feed_dict={ph: np.nan})

    def test_not_check_numerics(self):
        with scoped_set_check_numerics(False):
            x = maybe_check_numerics(
                tf.constant(np.nan), message='numerical issues')
        with self.test_session() as sess:
            self.assertTrue(np.isnan(sess.run(x)))
