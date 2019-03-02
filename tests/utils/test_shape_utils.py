import pytest
import numpy as np
import tensorflow as tf

from tfsnippet.utils import *


class IntShapeTestCase(tf.test.TestCase):

    def test_int_shape(self):
        self.assertEqual(get_static_shape(tf.zeros([1, 2, 3])), (1, 2, 3))
        self.assertEqual(
            get_static_shape(tf.placeholder(tf.float32, [None, 2, 3])),
            (None, 2, 3)
        )
        self.assertIsNone(get_static_shape(tf.placeholder(tf.float32, None)))


class ResolveNegativeAxisTestCase(tf.test.TestCase):

    def test_resolve_negative_axis(self):
        # good case
        self.assertEqual(resolve_negative_axis(4, (0, 1, 2)), (0, 1, 2))
        self.assertEqual(resolve_negative_axis(4, (0, -1, -2)), (0, 3, 2))

        # bad case
        with pytest.raises(ValueError, match='`axis` out of range: \\(-5,\\) '
                                             'vs ndims 4.'):
            _ = resolve_negative_axis(4, (-5,))

        with pytest.raises(ValueError, match='`axis` has duplicated elements '
                                             'after resolving negative axis.'):
            _ = resolve_negative_axis(4, (0, -4))


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

    def test_get_dimension_size(self):
        with self.test_session() as sess:
            # test static shape
            ph = tf.placeholder(tf.float32, (1, 2, 3))
            self.assertEqual(get_dimension_size(ph, 0), 1)
            self.assertEqual(get_dimension_size(ph, 1), 2)
            self.assertEqual(get_dimension_size(ph, 2), 3)
            self.assertEqual(get_dimension_size(ph, -1), 3)

            # test dynamic shape, but no dynamic axis is queried
            ph = tf.placeholder(tf.float32, (1, None, 3))
            self.assertEqual(get_dimension_size(ph, 0), 1)
            self.assertEqual(get_dimension_size(ph, 2), 3)
            self.assertEqual(get_dimension_size(ph, -1), 3)

            # test dynamic shape
            def _assert_equal(a, b):
                self.assertIsInstance(a, tf.Tensor)
                self.assertEqual(sess.run(a, feed_dict={ph: ph_in}), b)

            ph = tf.placeholder(tf.float32, (1, None, 3))
            ph_in = np.arange(6, dtype=np.float32).reshape((1, 2, 3))
            _assert_equal(get_dimension_size(ph, 1), 2)
            _assert_equal(get_dimension_size(ph, -2), 2)

            axis_ph = tf.placeholder(tf.int32, None)
            self.assertEqual(
                sess.run(get_dimension_size(ph, axis_ph),
                         feed_dict={ph: ph_in, axis_ph: 1}),
                2
            )

            # test fully dynamic shape
            ph = tf.placeholder(tf.float32, None)
            _assert_equal(get_dimension_size(ph, 0), 1)
            _assert_equal(get_dimension_size(ph, 1), 2)
            _assert_equal(get_dimension_size(ph, 2), 3)
            _assert_equal(get_dimension_size(ph, -2), 2)

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


class IsShapeEqualTestCase(tf.test.TestCase):

    def test_is_shape_equal(self):
        def check(x, y, x_ph=None, y_ph=None):
            ans = x.shape == y.shape
            feed_dict = {}
            if x_ph is not None:
                feed_dict[x_ph] = x
                x = x_ph
            if y_ph is not None:
                feed_dict[y_ph] = y
                y = y_ph

            result = is_shape_equal(x, y)
            if is_tensor_object(result):
                result = sess.run(result, feed_dict=feed_dict)

            self.assertEqual(result, ans)

        with self.test_session() as sess:
            # check static shapes
            x1 = np.random.normal(size=[2, 3, 4])
            x2 = np.random.normal(size=[2, 1, 4])
            x3 = np.random.normal(size=[1, 2, 3, 4])
            check(x1, np.copy(x1))
            check(x1, x2)
            check(x1, x3)

            # check partial dynamic shapes
            x1_ph = tf.placeholder(dtype=tf.float32, shape=[2, None, 4])
            x2_ph = tf.placeholder(dtype=tf.float32, shape=[2, None, 4])
            x3_ph = tf.placeholder(dtype=tf.float32, shape=[None] * 4)
            check(x1, np.copy(x1), x1_ph, x2_ph)
            check(x1, x2, x1_ph, x2_ph)
            check(x1, x3, x1_ph, x3_ph)

            # check fully dimension shapes
            x1_ph = tf.placeholder(dtype=tf.float32, shape=None)
            x2_ph = tf.placeholder(dtype=tf.float32, shape=None)
            x3_ph = tf.placeholder(dtype=tf.float32, shape=None)
            check(x1, np.copy(x1), x1_ph, x2_ph)
            check(x1, x2, x1_ph, x2_ph)
            check(x1, x3, x1_ph, x3_ph)
