import pytest
import numpy as np
import tensorflow as tf

from tfsnippet.utils import *
from tfsnippet.utils.shape_utils import broadcast_to_shape_sub


class IntShapeTestCase(tf.test.TestCase):

    def test_int_shape(self):
        self.assertEqual(int_shape(tf.zeros([1, 2, 3])), (1, 2, 3))
        self.assertEqual(int_shape(tf.placeholder(tf.float32, [None, 2, 3])),
                         (None, 2, 3))
        self.assertIsNone(int_shape(tf.placeholder(tf.float32, None)))


class ResolveNegativeAxisTestCase(tf.test.TestCase):

    def test_resolve_negative_axis(self):
        # good case
        self.assertEqual(resolve_negative_axis(4, (0, 1, 2)), (0, 1, 2))
        self.assertEqual(resolve_negative_axis(4, (0, -1, -2)), (0, 3, 2))

        # bad case
        with pytest.raises(ValueError, match='`axis` out of range: \\(-5,\\) '
                                             'vs ndims 4.'):
            _ = resolve_negative_axis(4, (-5,))


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


class GetRankTestCase(tf.test.TestCase):

    def test_get_rank(self):
        with self.test_session() as sess:
            # test static shape
            ph = tf.placeholder(tf.float32, (1, 2, 3))
            self.assertEqual(get_rank(ph), 3)

            # test partially dynamic shape
            ph = tf.placeholder(tf.float32, (1, None, 3))
            self.assertEqual(get_rank(ph), 3)

            # test totally dynamic shape
            ph = tf.placeholder(tf.float32, None)
            self.assertEqual(
                sess.run(get_rank(ph), feed_dict={
                    ph: np.arange(6, dtype=np.float32).reshape((1, 2, 3))
                }),
                3
            )


class GetDimensionSizeTestCase(tf.test.TestCase):

    def test_get_dimensions_size(self):
        with self.test_session() as sess:
            # test empty query
            ph = tf.placeholder(tf.float32, None)
            self.assertTupleEqual(get_dimensions_size(ph, ()), ())

            # test static shape
            ph = tf.placeholder(tf.float32, (1, 2, 3))
            self.assertTupleEqual(get_dimensions_size(ph), (1, 2, 3))
            self.assertTupleEqual(get_dimensions_size(ph, [0]), (1,))
            self.assertTupleEqual(get_dimensions_size(ph, [1]), (2,))
            self.assertTupleEqual(get_dimensions_size(ph, [2]), (3,))
            self.assertTupleEqual(get_dimensions_size(ph, [2, 0, 1]), (3, 1, 2))

            # test dynamic shape, but no dynamic axis is queried
            ph = tf.placeholder(tf.float32, (1, None, 3))
            self.assertTupleEqual(get_dimensions_size(ph, [0]), (1,))
            self.assertTupleEqual(get_dimensions_size(ph, [2]), (3,))
            self.assertTupleEqual(get_dimensions_size(ph, [2, 0]), (3, 1))

            # test dynamic shape
            def _assert_equal(a, b):
                ph_in = np.arange(6, dtype=np.float32).reshape((1, 2, 3))
                self.assertIsInstance(a, tf.Tensor)
                np.testing.assert_equal(sess.run(a, feed_dict={ph: ph_in}), b)

            ph = tf.placeholder(tf.float32, (1, None, 3))
            _assert_equal(get_dimensions_size(ph), (1, 2, 3))
            _assert_equal(get_dimensions_size(ph, [1]), (2,))
            _assert_equal(get_dimensions_size(ph, [2, 0, 1]), (3, 1, 2))

            # test fully dynamic shape
            ph = tf.placeholder(tf.float32, None)
            _assert_equal(get_dimensions_size(ph), (1, 2, 3))
            _assert_equal(get_dimensions_size(ph, [0]), (1,))
            _assert_equal(get_dimensions_size(ph, [1]), (2,))
            _assert_equal(get_dimensions_size(ph, [2]), (3,))
            _assert_equal(get_dimensions_size(ph, [2, 0, 1]), (3, 1, 2))

    def test_get_shape(self):
        with self.test_session() as sess:
            # test static shape
            ph = tf.placeholder(tf.float32, (1, 2, 3))
            self.assertTupleEqual(get_shape(ph), (1, 2, 3))

            # test dynamic shape
            def _assert_equal(a, b):
                ph_in = np.arange(6, dtype=np.float32).reshape((1, 2, 3))
                self.assertIsInstance(a, tf.Tensor)
                np.testing.assert_equal(sess.run(a, feed_dict={ph: ph_in}), b)

            ph = tf.placeholder(tf.float32, (1, None, 3))
            _assert_equal(get_shape(ph), (1, 2, 3))

            # test fully dynamic shape
            ph = tf.placeholder(tf.float32, None)
            _assert_equal(get_shape(ph), (1, 2, 3))


class ConcatShapesTestCase(tf.test.TestCase):

    def test_concat_shapes(self):
        with self.test_session() as sess:
            # test empty
            self.assertTupleEqual(concat_shapes(()), ())

            # test static shapes
            self.assertTupleEqual(
                concat_shapes(iter([
                    (1, 2),
                    (3,),
                    (),
                    (4, 5)
                ])),
                (1, 2, 3, 4, 5)
            )

            # test having dynamic shape
            shape = concat_shapes([
                (1, 2),
                tf.constant([3], dtype=tf.int32),
                (),
                tf.constant([4, 5], dtype=tf.int32),
            ])
            self.assertIsInstance(shape, tf.Tensor)
            np.testing.assert_equal(sess.run(shape), (1, 2, 3, 4, 5))


class BroadcastTestCase(tf.test.TestCase):

    def test_broadcast_to_shape_sub(self):
        def check(x, shape, x_ph=None, shape_ph=None, static_shape=None):
            # compute the expected answer
            try:
                y = x * np.ones(tuple(shape), dtype=x.dtype)
                if len(shape) and y.shape[-len(shape):] != shape:
                    raise ValueError()
            except ValueError:
                y = None

            # call the function and get output
            feed_dict = {}
            if x_ph is not None:
                feed_dict[x_ph] = x
                x = x_ph
            if shape_ph is not None:
                feed_dict[shape_ph] = np.asarray(shape)
                shape = shape_ph

            if y is None:
                with pytest.raises(Exception, match='abcdefg'):
                    t = broadcast_to_shape_sub(x, shape, 'abcdefg')
                    _ = sess.run(t, feed_dict=feed_dict)
            else:
                t = broadcast_to_shape_sub(x, shape, 'should never trigger')
                if static_shape is not None:
                    self.assertTupleEqual(int_shape(t), static_shape)

                out = sess.run(t, feed_dict=feed_dict)
                self.assertTupleEqual(out.shape, y.shape)
                np.testing.assert_equal(out, y)

        with self.test_session() as sess:
            np.random.seed(1234)
            x = np.random.random([2, 1, 3]).astype(np.float32)

            # -- fully static shapes --
            # good cases
            check(x, (3, 2, 5, 3), static_shape=(3, 2, 5, 3))
            check(x, (2, 5, 3), static_shape=(2, 5, 3))
            check(x, (5, 3), static_shape=(2, 5, 3))

            # error cases
            check(x, (1, 1, 1, 1))
            check(x, (1, 1, 1))
            check(x, (1, 1))

            # -- partially dynamic shapes on broadcast axis --
            x_ph = tf.placeholder(shape=(2, None, 3), dtype=tf.float32)

            # good cases
            check(x, (3, 2, 5, 3), x_ph=x_ph, static_shape=(3, 2, 5, 3))
            check(x, (2, 5, 3), x_ph=x_ph, static_shape=(2, 5, 3))
            check(x, (5, 3), x_ph=x_ph, static_shape=(2, 5, 3))

            # error cases
            check(x, (1, 1, 1, 1), x_ph=x_ph)
            check(x, (1, 1, 1), x_ph=x_ph)
            check(x, (1, 1), x_ph=x_ph)

            # -- partially dynamic shapes on non-broadcast axis --
            x_ph = tf.placeholder(shape=(None, 1, 3), dtype=tf.float32)

            # good cases
            check(x, (3, 2, 5, 3), x_ph=x_ph, static_shape=(3, 2, 5, 3))
            check(x, (2, 5, 3), x_ph=x_ph, static_shape=(2, 5, 3))
            check(x, (5, 3), x_ph=x_ph, static_shape=(None, 5, 3))

            # error cases
            check(x, (1, 1, 1, 1), x_ph=x_ph)
            check(x, (1, 1, 1), x_ph=x_ph)
            check(x, (1, 1), x_ph=x_ph)

            # -- partially dynamic shapes on all axis --
            x_ph = tf.placeholder(shape=(None, None, None), dtype=tf.float32)

            # good cases
            check(x, (3, 2, 5, 3), x_ph=x_ph, static_shape=(3, 2, 5, 3))
            check(x, (2, 5, 3), x_ph=x_ph, static_shape=(2, 5, 3))
            check(x, (5, 3), x_ph=x_ph, static_shape=(None, 5, 3))

            # error cases
            check(x, (1, 1, 1, 1), x_ph=x_ph)
            check(x, (1, 1, 1), x_ph=x_ph)
            check(x, (1, 1), x_ph=x_ph)

            # -- fully dynamic shapes --
            x_ph = tf.placeholder(shape=None, dtype=tf.float32)
            shape_ph = tf.placeholder(shape=None, dtype=tf.int32)

            # good cases
            check(x, (3, 2, 5, 3), x_ph=x_ph, shape_ph=shape_ph)
            check(x, (2, 5, 3), x_ph=x_ph, shape_ph=shape_ph)
            check(x, (5, 3), x_ph=x_ph, shape_ph=shape_ph)

            # error cases
            check(x, (1, 1, 1, 1), x_ph=x_ph, shape_ph=shape_ph)
            check(x, (1, 1, 1), x_ph=x_ph, shape_ph=shape_ph)
            check(x, (1, 1), x_ph=x_ph, shape_ph=shape_ph)

    def test_broadcast_to_shape(self):
        def check(x, shape, x_ph=None, shape_ph=None, static_shape=None):
            # compute the expected answer
            try:
                y = x * np.ones(tuple(shape), dtype=x.dtype)
                if y.shape != shape:
                    raise ValueError()
            except ValueError:
                y = None

            # call the function and get output
            feed_dict = {}
            if x_ph is not None:
                feed_dict[x_ph] = x
                x = x_ph
            if shape_ph is not None:
                feed_dict[shape_ph] = np.asarray(shape)
                shape = shape_ph

            if y is None:
                with pytest.raises(Exception, match='`x` cannot be broadcasted '
                                                    'to match `shape`'):
                    t = broadcast_to_shape(x, shape)
                    _ = sess.run(t, feed_dict=feed_dict)
            else:
                t = broadcast_to_shape(x, shape)
                if static_shape is not None:
                    self.assertTupleEqual(int_shape(t), static_shape)

                out = sess.run(t, feed_dict=feed_dict)
                self.assertTupleEqual(out.shape, y.shape)
                np.testing.assert_equal(out, y)

        with self.test_session() as sess:
            np.random.seed(1234)
            x = np.random.random([2, 1, 3]).astype(np.float32)

            # -- fully static shapes --
            # good cases
            check(x, (3, 2, 5, 3), static_shape=(3, 2, 5, 3))
            check(x, (2, 5, 3), static_shape=(2, 5, 3))

            # bad cases
            check(x, (5, 3))
            check(x, (1, 1, 1, 1))
            check(x, (1, 1, 1))
            check(x, (1, 1))

            # -- partially dynamic shapes on all axis --
            x_ph = tf.placeholder(shape=(None, None, None), dtype=tf.float32)

            # good cases
            check(x, (3, 2, 5, 3), x_ph=x_ph, static_shape=(3, 2, 5, 3))
            check(x, (2, 5, 3), x_ph=x_ph, static_shape=(2, 5, 3))

            # error cases
            check(x, (5, 3), x_ph=x_ph)
            check(x, (1, 1, 1, 1), x_ph=x_ph)
            check(x, (1, 1, 1), x_ph=x_ph)
            check(x, (1, 1), x_ph=x_ph)

            # -- fully dynamic shapes on x --
            x_ph = tf.placeholder(shape=None, dtype=tf.float32)

            # good cases
            check(x, (3, 2, 5, 3), x_ph=x_ph)
            check(x, (2, 5, 3), x_ph=x_ph)

            # error cases
            check(x, (5, 3), x_ph=x_ph)
            check(x, (1, 1, 1, 1), x_ph=x_ph)
            check(x, (1, 1, 1), x_ph=x_ph)
            check(x, (1, 1), x_ph=x_ph)

            # -- fully dynamic shapes on both x and shape --
            x_ph = tf.placeholder(shape=None, dtype=tf.float32)
            shape_ph = tf.placeholder(shape=None, dtype=tf.int32)

            # good cases
            check(x, (3, 2, 5, 3), x_ph=x_ph, shape_ph=shape_ph)
            check(x, (2, 5, 3), x_ph=x_ph, shape_ph=shape_ph)

            # error cases
            check(x, (5, 3), x_ph=x_ph, shape_ph=shape_ph)
            check(x, (1, 1, 1, 1), x_ph=x_ph, shape_ph=shape_ph)
            check(x, (1, 1, 1), x_ph=x_ph, shape_ph=shape_ph)
            check(x, (1, 1), x_ph=x_ph, shape_ph=shape_ph)
