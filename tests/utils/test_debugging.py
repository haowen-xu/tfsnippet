import pytest
import numpy as np
import tensorflow as tf

from tfsnippet.utils import *


class AssertionTestCase(tf.test.TestCase):

    def test_set_enable_assertions(self):
        self.assertTrue(settings.enable_assertions)
        with scoped_set_config(settings, enable_assertions=False):
            self.assertFalse(settings.enable_assertions)
        self.assertTrue(settings.enable_assertions)

    def test_assert_deps(self):
        ph = tf.placeholder(dtype=tf.bool, shape=())
        op = tf.assert_equal(ph, True, message='abcdefg')

        # test ops are empty
        with assert_deps([None]) as asserted:
            self.assertFalse(asserted)

        # test assertion enabled, and ops are not empty
        with self.test_session() as sess, \
                scoped_set_config(settings, enable_assertions=True):
            with assert_deps([op, None]) as asserted:
                self.assertTrue(asserted)
                out = tf.constant(1.)
            with pytest.raises(Exception, match='abcdefg'):
                self.assertEqual(sess.run(out, feed_dict={ph: False}), 1.)

        # test assertion disabled
        with self.test_session() as sess, \
                scoped_set_config(settings, enable_assertions=False):
            with assert_deps([op, None]) as asserted:
                self.assertFalse(asserted)
                out = tf.constant(1.)
            self.assertEqual(sess.run(out, feed_dict={ph: False}), 1.)


class CheckNumericsTestCase(tf.test.TestCase):

    def test_set_check_numerics(self):
        self.assertFalse(settings.check_numerics)
        with scoped_set_config(settings, check_numerics=True):
            self.assertTrue(settings.check_numerics)
        self.assertFalse(settings.check_numerics)

    def test_check_numerics(self):
        ph = tf.placeholder(dtype=tf.float32, shape=())
        with scoped_set_config(settings, check_numerics=True):
            x = maybe_check_numerics(ph, message='numerical issues')
        with pytest.raises(Exception, match='numerical issues'):
            with self.test_session() as sess:
                _ = sess.run(x, feed_dict={ph: np.nan})

    def test_not_check_numerics(self):
        with scoped_set_config(settings, check_numerics=False):
            x = maybe_check_numerics(
                tf.constant(np.nan), message='numerical issues')
        with self.test_session() as sess:
            self.assertTrue(np.isnan(sess.run(x)))
