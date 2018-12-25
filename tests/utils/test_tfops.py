import pytest
import numpy as np
import tensorflow as tf

from tfsnippet.utils import *


class TFOpsTestCase(tf.test.TestCase):

    def test_add_n_broadcast(self):
        # test zero input
        with pytest.raises(ValueError, match='`tensors` must not be empty'):
            _ = add_n_broadcast([])

        with self.test_session() as sess:
            a = np.array([[1., 2.], [3., 4.]])
            b = np.array([5., 6.])
            c = np.array([[7., 8.]])
            a_tensor, b_tensor, c_tensor = \
                tf.constant(a), tf.constant(b), tf.constant(c)

            # test one input
            np.testing.assert_allclose(
                sess.run(add_n_broadcast([a_tensor])), a)

            # test two inputs
            np.testing.assert_allclose(
                sess.run(add_n_broadcast([a_tensor, b_tensor])), a + b)

            # test three inputs
            np.testing.assert_allclose(
                sess.run(add_n_broadcast([a_tensor, b_tensor, c_tensor])),
                a + b + c
            )

    def test_smart_cond(self):
        with self.test_session() as sess:
            # test static condition
            self.assertEqual(1, smart_cond(True, (lambda: 1), (lambda: 2)))
            self.assertEqual(2, smart_cond(False, (lambda: 1), (lambda: 2)))

            # test dynamic condition
            cond_in = tf.placeholder(dtype=tf.bool, shape=())
            value = smart_cond(
                cond_in, lambda: tf.constant(1), lambda: tf.constant(2))
            self.assertIsInstance(value, tf.Tensor)
            self.assertEqual(sess.run(value, feed_dict={cond_in: True}), 1)
            self.assertEqual(sess.run(value, feed_dict={cond_in: False}), 2)

    def test_control_deps(self):
        with self.test_session() as sess:
            # test empty control inputs
            with control_deps([]) as b:
                self.assertFalse(b)

            # test None control inputs reduce to empty
            with control_deps([None]) as b:
                self.assertFalse(b)

            # test run control inputs
            w = tf.get_variable('w', shape=(), dtype=tf.float32)
            with control_deps([tf.assign(w, 123.), None]) as b:
                self.assertTrue(b)
                v = tf.constant(456.)
            self.assertEqual(sess.run(v), 456.)
            self.assertEqual(sess.run(w), 123.)

    def test_get_variable_ddi(self):
        @global_reuse
        def f(initial_value, initializing=False):
            return get_variable_ddi(
                'x', initial_value, shape=(), initializing=initializing)

        with self.test_session() as sess:
            x_in = tf.placeholder(dtype=tf.float32, shape=())
            x = f(x_in, initializing=True)
            self.assertEqual(sess.run(x, feed_dict={x_in: 123.}), 123.)
            x = f(x_in, initializing=False)
            self.assertEqual(sess.run(x, feed_dict={x_in: 456.}), 123.)

    def test_assert_scalar_equal(self):
        with self.test_session() as sess, scoped_set_assertion_enabled(True):
            # test static comparison
            msg = 'Assertion failed for a == b: 1 != 2; abcdefg'
            _ = assert_scalar_equal(1, 1)
            with pytest.raises(ValueError, match=msg):
                _ = assert_scalar_equal(1, 2, message='abcdefg')

            # test static comparison without message
            msg = 'Assertion failed for a == b: 1 != 2'
            with pytest.raises(ValueError, match=msg):
                _ = assert_scalar_equal(1, 2)

            # prepare dynamic comparison
            msg = 'abcdefg'
            x_in = tf.placeholder(dtype=tf.int32, shape=())
            assert_1 = assert_scalar_equal(1, x_in, message='abcdefg')
            assert_2 = assert_scalar_equal(x_in, 2, message='abcdefg')

            with control_deps([assert_1]):
                v1 = tf.constant(1.)
            with control_deps([assert_2]):
                v2 = tf.constant(2.)

            # test dynamic on a
            self.assertEqual(sess.run(v1, feed_dict={x_in: 1}), 1.)
            with pytest.raises(Exception, match=msg):
                _ = sess.run(v1, feed_dict={x_in: 2})

            # test dynamic on b
            self.assertEqual(sess.run(v2, feed_dict={x_in: 2}), 2.)
            with pytest.raises(Exception, match=msg):  # dynamic on b
                _ = sess.run(v2, feed_dict={x_in: 1})
