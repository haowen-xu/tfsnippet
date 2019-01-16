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

    def test_assert_deps(self):
        ph = tf.placeholder(dtype=tf.bool, shape=())
        op = tf.assert_equal(ph, True, message='abcdefg')

        # test ops are empty
        with assert_deps([None]) as asserted:
            self.assertFalse(asserted)

        # test assertion enabled, and ops are not empty
        with self.test_session() as sess, \
                scoped_set_assertion_enabled(True):
            with assert_deps([op, None]) as asserted:
                self.assertTrue(asserted)
                out = tf.constant(1.)
            with pytest.raises(Exception, match='abcdefg'):
                self.assertEqual(sess.run(out, feed_dict={ph: False}), 1.)

        # test assertion disabled
        with self.test_session() as sess, \
                scoped_set_assertion_enabled(False):
            with assert_deps([op, None]) as asserted:
                self.assertFalse(asserted)
                out = tf.constant(1.)
            self.assertEqual(sess.run(out, feed_dict={ph: False}), 1.)


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
