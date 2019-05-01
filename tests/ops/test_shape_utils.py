import numpy as np
import pytest
import tensorflow as tf

from tfsnippet.ops import *
from tfsnippet.utils import get_static_shape


class PrependDimsTestCase(tf.test.TestCase):

    def test_prepend_dims(self):
        with pytest.raises(ValueError, match='`ndims` must be >= 0: got -1'):
            _ = prepend_dims(tf.constant(0.), ndims=-1)

        x = tf.zeros([2, 3])
        self.assertIs(prepend_dims(x, ndims=0), x)

        with self.test_session() as sess:
            # test static shape
            x = np.random.normal(size=[2, 3])
            y = prepend_dims(x, ndims=1)
            self.assertEqual(get_static_shape(y), (1, 2, 3))
            np.testing.assert_allclose(sess.run(y), x.reshape([1, 2, 3]))

            # test partially dynamic shape
            t = tf.placeholder(shape=[2, None], dtype=tf.float64)
            y = prepend_dims(t, ndims=2)
            self.assertEqual(get_static_shape(y), (1, 1, 2, None))
            np.testing.assert_allclose(
                sess.run(y, feed_dict={t: x}),  x.reshape([1, 1, 2, 3]))

            # test fully dynamic shape
            t = tf.placeholder(shape=None, dtype=tf.float64)
            y = prepend_dims(t, ndims=3)
            self.assertEqual(get_static_shape(y), None)
            np.testing.assert_allclose(
                sess.run(y, feed_dict={t: x}), x.reshape([1, 1, 1, 2, 3]))


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
                self.assertEqual(flatten_to_ndims(t, k), (t, None, None))
                self.assertEqual(unflatten_from_ndims(t, None, None), t)

            else:
                if k == 1:
                    front_shape = tuple(x.shape)
                    static_front_shape = get_static_shape(t)
                    xx = x.reshape([-1])
                else:
                    front_shape = tuple(x.shape)[: -(k-1)]
                    static_front_shape = get_static_shape(t)[: -(k - 1)]
                    xx = x.reshape([-1] + list(x.shape)[-(k-1):])

                with self.test_session() as sess:
                    tt, s1, s2 = flatten_to_ndims(t, k)
                    self.assertEqual(s1, static_front_shape)
                    if not dynamic_shape:
                        self.assertEqual(s2, front_shape)
                    else:
                        self.assertEqual(tuple(run(sess, s2)), front_shape)
                    np.testing.assert_equal(run(sess, tt), xx)
                    np.testing.assert_equal(
                        run(sess, unflatten_from_ndims(tt, s1, s2)),
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
            _ = flatten_to_ndims(tf.constant(0.), 0)
        with pytest.raises(ValueError,
                           match='`x` is required to have known number of '
                                 'dimensions'):
            _ = flatten_to_ndims(tf.placeholder(tf.float32, None), 1)
        with pytest.raises(ValueError,
                           match='`k` is 2, but `x` only has rank 1'):
            _ = flatten_to_ndims(tf.zeros([3]), 2)

    def test_unflatten_errors(self):
        with pytest.raises(ValueError,
                           match='`x` is required to have known number of '
                                 'dimensions'):
            _ = unflatten_from_ndims(tf.placeholder(tf.float32, None), (1,), (1,))
        with pytest.raises(ValueError,
                           match='`x` only has rank 0, required at least 1'):
            _ = unflatten_from_ndims(tf.constant(0.), (1,), (1,))


class BroadcastTestCase(tf.test.TestCase):

    def test_broadcast_to_shape(self):
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
                with pytest.raises(Exception, match='`x` cannot be broadcasted '
                                                    'to match `shape`'):
                    t = broadcast_to_shape(x, shape)
                    _ = sess.run(t, feed_dict=feed_dict)
            else:
                t = broadcast_to_shape(x, shape)
                if static_shape is not None:
                    self.assertTupleEqual(get_static_shape(t), static_shape)

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

    def test_broadcast_to_shape_strict(self):
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
                    t = broadcast_to_shape_strict(x, shape)
                    _ = sess.run(t, feed_dict=feed_dict)
            else:
                t = broadcast_to_shape_strict(x, shape)
                if static_shape is not None:
                    self.assertTupleEqual(get_static_shape(t), static_shape)

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


class BroadcastConcatTestCase(tf.test.TestCase):

    def test_broadcast_concat(self):
        a_ph = tf.placeholder(dtype=tf.float32, shape=[None, 3, 1, None, 1])
        b_ph = tf.placeholder(dtype=tf.float32, shape=[6, None, 1, 5, 7, None])
        ph = tf.placeholder(dtype=tf.float32, shape=None)

        def check(x, y, axis, static_shape):
            ndims = max(len(x.shape), len(y.shape))
            xx = np.reshape(x, [1] * (ndims - len(x.shape)) + list(x.shape))
            yy = np.reshape(y, [1] * (ndims - len(y.shape)) + list(y.shape))
            if axis < 0:
                axis += ndims
            b_shape = [1] * ndims
            for i in range(ndims):
                if i != axis:
                    b_shape[i] = max(xx.shape[i], yy.shape[i])

            xx = xx * np.ones(b_shape)
            yy = yy * np.ones(b_shape)
            ans = np.concatenate([xx, yy], axis=axis)

            out = broadcast_concat(a_ph, b_ph, axis=axis)
            self.assertEqual(get_static_shape(out), static_shape)

            np.testing.assert_allclose(
                sess.run(out, feed_dict={a_ph: x, b_ph: y}),
                ans
            )

        with self.test_session() as sess:
            # test can broadcast
            static_shapes = [
                (7, None, 3, 5, 7, None),
                (6, None, 3, 5, 7, None),
                (6, None, 4, 5, 7, None),
                (6, None, 3, 6, 7, None),
                (6, None, 3, 5, None, None),
                (6, None, 3, 5, 7, None)
            ] * 2

            for axis, static_shape in zip(
                    [-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5], static_shapes):
                check(
                    np.random.normal(size=[4, 3, 1, 7, 1]),
                    np.random.normal(size=[6, 4, 1, 5, 7, 8]),
                    axis=axis,
                    static_shape=static_shape
                )

            for axis in [-5, 1]:
                check(
                    np.random.normal(size=[1, 3, 1, 7, 1]),
                    np.random.normal(size=[6, 4, 1, 5, 7, 8]),
                    axis=axis,
                    static_shape=(6, None, 3, 5, 7, None)
                )

            for axis in [-2, 4]:
                check(
                    np.random.normal(size=[4, 3, 1, 1, 1]),
                    np.random.normal(size=[6, 4, 1, 5, 7, 8]),
                    axis=axis,
                    static_shape=(6, None, 3, 5, None, None)
                )

            for axis in [-1, 5]:
                check(
                    np.random.normal(size=[4, 3, 1, 7, 1]),
                    np.random.normal(size=[6, 4, 1, 5, 7, 1]),
                    axis=axis,
                    static_shape=(6, None, 3, 5, 7, None)
                )

            # test cannot broadcast
            with pytest.raises(ValueError, match='`x` with non-deterministic '
                                                 'shape is not supported'):
                _ = broadcast_concat(ph, b_ph, axis=0)
            with pytest.raises(ValueError, match='`y` with non-deterministic '
                                                 'shape is not supported'):
                _ = broadcast_concat(a_ph, ph, axis=0)
            with pytest.raises(ValueError, match='Invalid axis: must >= -6 and '
                                                 '<= 5, got -7'):
                _ = broadcast_concat(a_ph, b_ph, axis=-7)
            with pytest.raises(ValueError, match='Invalid axis: must >= -6 and '
                                                 '<= 5, got 6'):
                _ = broadcast_concat(a_ph, b_ph, axis=6)
            with pytest.raises(ValueError, match='`x` and `y` cannot be '
                                                 'broadcast concat'):
                _ = broadcast_concat(tf.zeros([2, 2]), tf.zeros([3, 3]), axis=0)

            # runtime check
            t = broadcast_concat(a_ph, b_ph, axis=-1)
            with pytest.raises(Exception, match='`x` and `y` cannot be '
                                                'broadcast concat'):
                _ = sess.run(t, feed_dict={
                    a_ph: np.random.normal(size=[3, 3, 1, 7, 1]),
                    b_ph: np.random.normal(size=[6, 4, 1, 5, 7, 8]),
                })


class TransposeConv2dAxisTestCase(tf.test.TestCase):

    def test_transpose_conv2d_axis(self):
        np.random.seed(1234)
        x = np.random.normal(size=[17, 11, 32, 31, 5]).astype(np.float32)
        x_ph = tf.placeholder(tf.float32, [None, None, None, None, 5])
        y = np.transpose(x, [0, 1, 4, 2, 3])
        self.assertEqual(y.shape, (17, 11, 5, 32, 31))
        y_ph = tf.placeholder(tf.float32, [None, None, 5, None, None])

        g = lambda x, f, t, ph=None: sess.run(
            transpose_conv2d_axis(tf.constant(x), f, t),
            feed_dict=({ph: x} if ph is not None else None)
        )

        with self.test_session() as sess:
            # test static shape
            np.testing.assert_allclose(g(x, True, True), x)
            np.testing.assert_allclose(g(x, True, False), y)
            np.testing.assert_allclose(g(y, False, True), x)
            np.testing.assert_allclose(g(y, False, False), y)

            # test dynamic shape
            np.testing.assert_allclose(g(x, True, True, x_ph), x)
            np.testing.assert_allclose(g(x, True, False, x_ph), y)
            np.testing.assert_allclose(g(y, False, True, y_ph), x)
            np.testing.assert_allclose(g(y, False, False, y_ph), y)

    def test_transpose_conv2d_channels_x_to_x(self):
        np.random.seed(1234)
        x = np.random.normal(size=[17, 11, 32, 31, 5]).astype(np.float32)
        y = np.transpose(x, [0, 1, 4, 2, 3])
        self.assertEqual(y.shape, (17, 11, 5, 32, 31))

        with self.test_session() as sess:
            # test conv2d_channels_last_to_x
            g = lambda t, c: sess.run(
                transpose_conv2d_channels_last_to_x(tf.constant(t), c))
            np.testing.assert_allclose(g(x, True), x)
            np.testing.assert_allclose(g(x, False), y)

            # test conv2d_channels_x_to_last
            g = lambda t, c: sess.run(
                transpose_conv2d_channels_x_to_last(tf.constant(t), c))
            np.testing.assert_allclose(g(x, True), x)
            np.testing.assert_allclose(g(y, False), x)


class ReshapeTailTestCase(tf.test.TestCase):

    def test_reshape_tail(self):
        def check(x, ndims, shape, expected_shape, static_shape=None,
                  x_ph=None, shape_ph=None):
            # compute the answer
            assert(len(x.shape) >= ndims)
            if ndims > 0:
                y = np.reshape(x, x.shape[:-ndims] + tuple(shape))
            else:
                y = np.reshape(x, x.shape + tuple(shape))
            self.assertEqual(y.shape, expected_shape)

            # validate the output
            feed_dict = {}
            if x_ph is not None:
                feed_dict[x_ph] = x
                x = x_ph
            if shape_ph is not None:
                feed_dict[shape_ph] = shape
                shape = shape_ph

            y_tensor = reshape_tail(x, ndims, shape)
            if static_shape is not None:
                self.assertTupleEqual(get_static_shape(y_tensor), static_shape)
            y_out = sess.run(y_tensor, feed_dict=feed_dict)

            self.assertTupleEqual(y_out.shape, y.shape)
            np.testing.assert_equal(y_out, y)

        x = np.random.normal(size=[4, 5, 6]).astype(np.float32)

        with self.test_session() as sess:
            # check static shape
            check(x, 0, [], (4, 5, 6), (4, 5, 6))
            check(x, 0, [1, 1], (4, 5, 6, 1, 1), (4, 5, 6, 1, 1))
            check(x, 1, [-1], (4, 5, 6), (4, 5, 6))
            check(x, 1, [2, 3], (4, 5, 2, 3), (4, 5, 2, 3))
            check(x, 2, [-1], (4, 30), (4, 30))
            check(x, 2, [6, 5], (4, 6, 5), (4, 6, 5))
            check(x, 2, [3, 2, 5], (4, 3, 2, 5), (4, 3, 2, 5))
            check(x, 3, [-1], (120,), (120,))
            check(x, 3, [3, -1], (3, 40), (3, 40))

            # check dynamic shape #1
            x_ph = tf.placeholder(dtype=tf.float32, shape=[None, 5, 6])

            check(x, 0, [], (4, 5, 6), (None, 5, 6), x_ph=x_ph)
            check(x, 0, [1, 1], (4, 5, 6, 1, 1), (None, 5, 6, 1, 1),
                  x_ph=x_ph)
            check(x, 1, [-1], (4, 5, 6), (None, 5, 6), x_ph=x_ph)
            check(x, 1, [2, -1], (4, 5, 2, 3), (None, 5, 2, 3), x_ph=x_ph)
            check(x, 2, [-1], (4, 30), (None, 30), x_ph=x_ph)
            check(x, 2, [-1, 5], (4, 6, 5), (None, 6, 5), x_ph=x_ph)
            check(x, 2, [3, -1, 5], (4, 3, 2, 5), (None, 3, 2, 5), x_ph=x_ph)
            check(x, 3, [-1], (120,), (None,), x_ph=x_ph)
            check(x, 3, [3, -1], (3, 40), (3, None), x_ph=x_ph)

            # check dynamic shape #2
            x_ph = tf.placeholder(dtype=tf.float32, shape=[None, 5, None])

            check(x, 0, [], (4, 5, 6), (None, 5, None), x_ph=x_ph)
            check(x, 0, [1, 1], (4, 5, 6, 1, 1), (None, 5, None, 1, 1),
                  x_ph=x_ph)
            check(x, 1, [-1], (4, 5, 6), (None, 5, None), x_ph=x_ph)
            check(x, 1, [2, 3], (4, 5, 2, 3), (None, 5, 2, 3), x_ph=x_ph)
            check(x, 2, [-1], (4, 30), (None, None), x_ph=x_ph)
            check(x, 2, [6, 5], (4, 6, 5), (None, 6, 5), x_ph=x_ph)
            check(x, 2, [3, 2, 5], (4, 3, 2, 5), (None, 3, 2, 5), x_ph=x_ph)
            check(x, 3, [-1], (120,), (None,), x_ph=x_ph)
            check(x, 3, [3, -1], (3, 40), (3, None), x_ph=x_ph)

            # check fully dynamic shape
            x_ph = tf.placeholder(dtype=tf.float32, shape=None)
            shape_ph = tf.placeholder(dtype=tf.int32, shape=None)

            check(x, 0, [], (4, 5, 6), x_ph=x_ph, shape_ph=shape_ph)
            check(x, 0, [1, 1], (4, 5, 6, 1, 1), x_ph=x_ph, shape_ph=shape_ph)
            check(x, 1, [-1], (4, 5, 6), x_ph=x_ph, shape_ph=shape_ph)
            check(x, 1, [2, 3], (4, 5, 2, 3), x_ph=x_ph, shape_ph=shape_ph)
            check(x, 2, [-1], (4, 30), x_ph=x_ph, shape_ph=shape_ph)
            check(x, 2, [6, 5], (4, 6, 5), x_ph=x_ph, shape_ph=shape_ph)
            check(x, 2, [3, 2, 5], (4, 3, 2, 5), x_ph=x_ph, shape_ph=shape_ph)
            check(x, 3, [-1], (120,), x_ph=x_ph, shape_ph=shape_ph)
            check(x, 3, [3, -1], (3, 40), x_ph=x_ph, shape_ph=shape_ph)

            # check errors
            with pytest.raises(ValueError,
                               match='`shape` is not a valid shape: at most '
                                     'one `-1` can be specified'):
                _ = reshape_tail(x, 1, [-1, -1])

            with pytest.raises(ValueError,
                               match='`shape` is not a valid shape: 0 is not '
                                     'allowed'):
                _ = reshape_tail(x, 1, [0])

            with pytest.raises(Exception,
                               match=r'rank\(input\) must be at least ndims'):
                _ = sess.run(reshape_tail(x, 5, [-1]))

            with pytest.raises(Exception,
                               match=r'rank\(input\) must be at least ndims'):
                _ = sess.run(reshape_tail(x_ph, 5, [-1]), feed_dict={x_ph: x})

            with pytest.raises(Exception,
                               match=r'Cannot reshape the tail dimensions of '
                                     r'`input` into `shape`'):
                _ = sess.run(reshape_tail(x, 2, [7, -1]))
