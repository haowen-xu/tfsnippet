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
        var = tf.get_variable('var', dtype=tf.float32, shape=(),
                              initializer=tf.zeros_initializer())
        op = tf.assign(var, 1.)

        # test ops are empty
        with assert_deps([None]) as asserted:
            self.assertFalse(asserted)

        # test assertion enabled, and ops are not empty
        with self.test_session() as sess, \
                scoped_set_assertion_enabled(True):
            sess.run(var.initializer)
            with assert_deps([op, None]) as asserted:
                self.assertTrue(asserted)
                out = tf.identity(var)
            self.assertEqual(sess.run(out), 1.)

        # test assertion disabled
        with self.test_session() as sess, \
                scoped_set_assertion_enabled(False):
            sess.run(var.initializer)
            with assert_deps([op, None]) as asserted:
                self.assertFalse(asserted)
                out = tf.identity(var)
            self.assertEqual(sess.run(out), 0.)


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
