import pytest
import numpy as np
import tensorflow as tf

from tfsnippet.utils import flatten, unflatten, int_shape, get_batch_size


class IntShapeTestCase(tf.test.TestCase):

    def test_int_shape(self):
        self.assertEqual(int_shape(tf.zeros([1, 2, 3])), (1, 2, 3))
        self.assertEqual(int_shape(tf.placeholder(tf.float32, [None, 2, 3])),
                         (None, 2, 3))
        self.assertIsNone(int_shape(tf.placeholder(tf.float32, None)))


class FlattenUnflattenTestCase(tf.test.TestCase):

    def test_flatten_and_unflatten(self):
        def run_check(x, k, dynamic_shape):
            if dynamic_shape:
                t = tf.placeholder(tf.int32, [None] * len(x.shape))
                run = lambda sess, *args: sess.run(*args, feed_dict={t: x})
            else:
                t = tf.constant(x, dtype=tf.int32)
                run = lambda sess, *args: sess.run(*args)

            if len(x.shape) == k:
                self.assertEqual(flatten(t, k), (t, None, None))
                self.assertEqual(unflatten(t, None, None), t)

            else:
                if k == 1:
                    front_shape = tuple(x.shape)
                    static_front_shape = int_shape(t)
                    xx = x.reshape([-1])
                else:
                    front_shape = tuple(x.shape)[: -(k-1)]
                    static_front_shape = int_shape(t)[: -(k-1)]
                    xx = x.reshape([-1] + list(x.shape)[-(k-1):])

                with self.test_session() as sess:
                    tt, s1, s2 = flatten(t, k)
                    self.assertEqual(s1, static_front_shape)
                    if not dynamic_shape:
                        self.assertEqual(s2, front_shape)
                    else:
                        self.assertEqual(tuple(run(sess, s2)), front_shape)
                    np.testing.assert_equal(run(sess, tt), xx)
                    np.testing.assert_equal(
                        run(sess, unflatten(tt, s1, s2)),
                        x
                    )

        x = np.arange(120).reshape([2, 3, 4, 5]).astype(np.int32)
        run_check(x, 1, dynamic_shape=False)
        run_check(x, 1, dynamic_shape=True)
        run_check(x, 2, dynamic_shape=False)
        run_check(x, 2, dynamic_shape=True)
        run_check(x, 3, dynamic_shape=False)
        run_check(x, 3, dynamic_shape=True)
        run_check(x, 4, dynamic_shape=False)
        run_check(x, 4, dynamic_shape=True)

    def test_flatten_errors(self):
        with pytest.raises(ValueError,
                           match='`k` must be greater or equal to 1'):
            _ = flatten(tf.constant(0.), 0)
        with pytest.raises(ValueError,
                           match='`x` is required to have known number of '
                                 'dimensions'):
            _ = flatten(tf.placeholder(tf.float32, None), 1)
        with pytest.raises(ValueError,
                           match='`k` is 2, but `x` only has rank 1'):
            _ = flatten(tf.zeros([3]), 2)

    def test_unflatten_errors(self):
        with pytest.raises(ValueError,
                           match='`x` is required to have known number of '
                                 'dimensions'):
            _ = unflatten(tf.placeholder(tf.float32, None), (1,), (1,))
        with pytest.raises(ValueError,
                           match='`x` only has rank 0, required at least 1'):
            _ = unflatten(tf.constant(0.), (1,), (1,))


class GetBatchSizeTestCase(tf.test.TestCase):

    def test_get_batch_size(self):
        def run_check(sess, x, axis, x_in=None, dynamic=True):
            if x_in is None:
                x_in = tf.constant(x)
                dynamic = False
            batch_size = get_batch_size(x_in, axis)
            if dynamic:
                self.assertIsInstance(batch_size, tf.Tensor)
                self.assertEqual(sess.run(batch_size, feed_dict={x_in: x}),
                                 x.shape[axis])
            else:
                self.assertEqual(batch_size, x.shape[axis])

        with self.test_session() as sess:
            x = np.zeros([2, 3, 4], dtype=np.float32)

            # check when shape is totally static
            run_check(sess, x, 0)
            run_check(sess, x, 1)
            run_check(sess, x, 2)
            run_check(sess, x, -1)

            # check when some shape is dynamic, but the batch axis is not
            run_check(sess, x, 0, tf.placeholder(tf.float32, [2, None, None]),
                      dynamic=False)
            run_check(sess, x, 1, tf.placeholder(tf.float32, [None, 3, None]),
                      dynamic=False)
            run_check(sess, x, 2, tf.placeholder(tf.float32, [None, None, 4]),
                      dynamic=False)
            run_check(sess, x, -1, tf.placeholder(tf.float32, [None, None, 4]),
                      dynamic=False)

            # check when the batch axis is dynamic
            run_check(sess, x, 0, tf.placeholder(tf.float32, [None, 3, 4]),
                      dynamic=True)
            run_check(sess, x, 1, tf.placeholder(tf.float32, [2, None, 4]),
                      dynamic=True)
            run_check(sess, x, 2, tf.placeholder(tf.float32, [2, 3, None]),
                      dynamic=True)
            run_check(sess, x, -1, tf.placeholder(tf.float32, [2, 3, None]),
                      dynamic=True)

            # check when the shape is totally dynamic
            x_in = tf.placeholder(tf.float32, None)
            run_check(sess, x, 0, x_in, dynamic=True)
            run_check(sess, x, 1, x_in, dynamic=True)
            run_check(sess, x, 2, x_in, dynamic=True)
            run_check(sess, x, -1, x_in, dynamic=True)
