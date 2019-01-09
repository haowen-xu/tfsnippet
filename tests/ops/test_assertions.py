import pytest
import numpy as np
import tensorflow as tf

from tfsnippet.ops import *


class AssertOpsTestCase(tf.test.TestCase):

    def test_assert_scalar_equal(self):
        with self.test_session() as sess:
            # test static comparison
            self.assertIsNone(assert_scalar_equal(1, 1))
            with pytest.raises(AssertionError,
                               match='Assertion failed for a == b: '
                                     '1 != 2; abcdefg'):
                _ = assert_scalar_equal(1, 2, message='abcdefg')

            # prepare dynamic comparison
            x_in = tf.placeholder(dtype=tf.int32, shape=())
            assert_1 = assert_scalar_equal(1, x_in, message='abcdefg')
            assert_2 = assert_scalar_equal(x_in, 2, message='abcdefg')

            with tf.control_dependencies([assert_1]):
                v1 = tf.constant(1.)
            with tf.control_dependencies([assert_2]):
                v2 = tf.constant(2.)

            # test dynamic on a
            self.assertEqual(sess.run(v1, feed_dict={x_in: 1}), 1.)
            with pytest.raises(Exception, match='abcdefg'):
                _ = sess.run(v1, feed_dict={x_in: 2})

            # test dynamic on b
            self.assertEqual(sess.run(v2, feed_dict={x_in: 2}), 2.)
            with pytest.raises(Exception, match='abcdefg'):  # dynamic on b
                _ = sess.run(v2, feed_dict={x_in: 1})

    def test_assert_rank(self):
        with self.test_session() as sess:
            # test static comparison
            x = np.zeros([2, 3, 4])
            self.assertIsNone(assert_rank(x, 3))
            with pytest.raises(AssertionError,
                               match=r'Assertion failed for rank\(x\) == ndims'
                                     r': 3 != 2; abcdefg'):
                _ = assert_rank(x, 2, message='abcdefg')

            # prepare dynamic comparison
            x_in = tf.placeholder(dtype=tf.int32, shape=None)
            assert_1 = assert_rank(x_in, 3, message='abcdefg')
            assert_2 = assert_rank(x_in, 2, message='abcdefg')

            with tf.control_dependencies([assert_1]):
                v1 = tf.constant(1.)
            with tf.control_dependencies([assert_2]):
                v2 = tf.constant(2.)

            self.assertEqual(sess.run(v1, feed_dict={x_in: x}), 1.)
            with pytest.raises(Exception, match='abcdefg'):
                _ = sess.run(v2, feed_dict={x_in: x})

    def test_assert_rank_at_least(self):
        with self.test_session() as sess:
            # test static comparison
            x = np.zeros([2, 3, 4])
            self.assertIsNone(assert_rank_at_least(x, 2))
            with pytest.raises(AssertionError,
                               match=r'Assertion failed for rank\(x\) >= ndims'
                                     r': 3 < 4; abcdefg'):
                _ = assert_rank_at_least(x, 4, message='abcdefg')

            # prepare dynamic comparison
            x_in = tf.placeholder(dtype=tf.int32, shape=None)
            assert_1 = assert_rank_at_least(x_in, 2, message='abcdefg')
            assert_2 = assert_rank_at_least(x_in, 4, message='abcdefg')

            with tf.control_dependencies([assert_1]):
                v1 = tf.constant(1.)
            with tf.control_dependencies([assert_2]):
                v2 = tf.constant(2.)

            self.assertEqual(sess.run(v1, feed_dict={x_in: x}), 1.)
            with pytest.raises(Exception, match='abcdefg'):
                _ = sess.run(v2, feed_dict={x_in: x})

    def test_assert_shape_equal(self):
        with self.test_session() as sess:
            # test static comparison
            x1 = np.random.normal(size=[2, 3, 4])
            x2 = np.random.normal(size=[2, 1, 4])
            self.assertIsNone(assert_shape_equal(x1, np.copy(x1)))
            with pytest.raises(AssertionError, match=r'Assertion failed for '
                                                     r'x.shape == y.shape.*'
                                                     r'abcdefg'):
                _ = assert_shape_equal(x1, x2, message='abcdefg')

            # prepare dynamic comparison
            x1_in = tf.placeholder(dtype=tf.int32, shape=None)
            x2_in = tf.placeholder(dtype=tf.int32, shape=None)
            assert_op = assert_shape_equal(x1_in, x2_in, message='abcdefg')

            with tf.control_dependencies([assert_op]):
                v = tf.constant(1.)

            self.assertEqual(sess.run(v, feed_dict={x1_in: x1, x2_in: x1}), 1.)
            with pytest.raises(Exception, match='abcdefg'):
                _ = sess.run(v, feed_dict={x1_in: x1, x2_in: x2})
