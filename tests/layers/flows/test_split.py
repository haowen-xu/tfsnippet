import numpy as np
import pytest
import tensorflow as tf

from tests.layers.flows.helper import (quadratic_transform, QuadraticFlow,
                                       npyops, invertible_flow_standard_check)
from tfsnippet.layers import SplitFlow, SequentialFlow, ReshapeFlow


class SplitFlowTestCase(tf.test.TestCase):

    def test_equal_value_ndims(self):
        def split_transform(x, split_axis, value_ndims, a1, b1, a2=None,
                            b2=None):
            n1 = x.shape[split_axis] // 2
            n2 = x.shape[split_axis] - n1
            x1, x2 = np.split(x, [n1], axis=split_axis)
            y1, log_det1 = quadratic_transform(npyops, x1, a1, b1)
            if a2 is not None:
                y2, log_det2 = quadratic_transform(npyops, x2, a2, b2)
            else:
                y2, log_det2 = x2, np.zeros_like(x, dtype=np.float64)
            y = np.concatenate([y1, y2], axis=split_axis)
            if value_ndims > 0:
                reduce_axis = tuple(range(-value_ndims, 0))
                log_det1 = np.sum(log_det1, axis=reduce_axis)
                log_det2 = np.sum(log_det2, axis=reduce_axis)
            log_det = log_det1 + log_det2
            return y, log_det

        with self.test_session() as sess:
            np.random.seed(1234)
            x = 10. * np.random.normal(size=[3, 4, 5, 6]).astype(np.float64)

            # static input, split_axis = -1, value_ndims = 1, right = None
            flow = SplitFlow(-1, QuadraticFlow(2., 5., value_ndims=1))
            self.assertEqual(flow.x_value_ndims, 1)
            self.assertEqual(flow.y_value_ndims, 1)

            y, log_det = split_transform(
                x, split_axis=-1, value_ndims=1, a1=2., b1=5.)
            y_out, log_det_out = sess.run(flow.transform(x))

            np.testing.assert_allclose(y_out, y)
            np.testing.assert_allclose(log_det_out, log_det)

            invertible_flow_standard_check(
                self, flow, sess, x, rtol=1e-4, atol=1e-5)

            # dynamic input, split_axis = -2, value_ndims = 2, right = None
            x_ph = tf.placeholder(dtype=tf.float64, shape=[None] * 4)
            flow = SplitFlow(-2, QuadraticFlow(2., 5., value_ndims=2))
            self.assertEqual(flow.x_value_ndims, 2)
            self.assertEqual(flow.y_value_ndims, 2)

            y, log_det = split_transform(
                x, split_axis=-2, value_ndims=2, a1=2., b1=5.)
            y_out, log_det_out = sess.run(
                flow.transform(x_ph), feed_dict={x_ph: x})

            np.testing.assert_allclose(y_out, y)
            np.testing.assert_allclose(log_det_out, log_det)

            invertible_flow_standard_check(
                self, flow, sess, x_ph, feed_dict={x_ph: x})

            # dynamic input, split_axis = 2, value_ndims = 3
            x_ph = tf.placeholder(dtype=tf.float64, shape=[None] * 4)
            flow = SplitFlow(split_axis=2,
                             left=QuadraticFlow(2., 5., value_ndims=3),
                             right=QuadraticFlow(1.5, 3., value_ndims=3))
            self.assertEqual(flow.x_value_ndims, 3)
            self.assertEqual(flow.y_value_ndims, 3)

            y, log_det = split_transform(
                x, split_axis=2, value_ndims=3, a1=2., b1=5.,
                a2=1.5, b2=3.
            )
            y_out, log_det_out = sess.run(
                flow.transform(x_ph), feed_dict={x_ph: x})

            np.testing.assert_allclose(y_out, y)
            np.testing.assert_allclose(log_det_out, log_det)

            invertible_flow_standard_check(
                self, flow, sess, x_ph, feed_dict={x_ph: x})

    def test_different_value_ndims(self):
        def reshape_tail(x, value_ndims, shape):
            batch_shape = x.shape
            if value_ndims > 0:
                batch_shape = batch_shape[:-value_ndims]
            return np.reshape(x, batch_shape + tuple(shape))

        def split_transform(x, split_axis, join_axis, x_value_ndims, y_shape,
                            a1, b1, a2=None, b2=None):
            n1 = x.shape[split_axis] // 2
            n2 = x.shape[split_axis] - n1
            x1, x2 = np.split(x, [n1], axis=split_axis)
            y1, log_det1 = quadratic_transform(npyops, x1, a1, b1)
            if a2 is not None:
                y2, log_det2 = quadratic_transform(npyops, x2, a2, b2)
            else:
                y2, log_det2 = x2, np.zeros_like(x, dtype=np.float64)

            y1 = reshape_tail(y1, x_value_ndims, y_shape)
            y2 = reshape_tail(y2, x_value_ndims, y_shape)
            y = np.concatenate([y1, y2], axis=join_axis)

            if x_value_ndims > 0:
                reduce_axis = tuple(range(-x_value_ndims, 0))
                log_det1 = np.sum(log_det1, axis=reduce_axis)
                log_det2 = np.sum(log_det2, axis=reduce_axis)
            log_det = log_det1 + log_det2

            return y, log_det

        with self.test_session() as sess:
            np.random.seed(1234)
            x = 10. * np.random.normal(size=[3, 4, 5, 12]).astype(np.float64)

            # 2 -> 3, x_value_ndims = 3, y_value_ndims = 4
            x_ph = tf.placeholder(dtype=tf.float64, shape=[None] * 4)
            flow = SplitFlow(
                split_axis=-2,
                join_axis=2,
                left=SequentialFlow([
                    QuadraticFlow(2., 5., value_ndims=3),
                    ReshapeFlow(3, [4, -1, 2, 6]),
                ]),
                right=SequentialFlow([
                    QuadraticFlow(1.5, 3., value_ndims=3),
                    ReshapeFlow(3, [4, -1, 2, 6]),
                ])
            )
            self.assertEqual(flow.x_value_ndims, 3)
            self.assertEqual(flow.y_value_ndims, 4)

            y, log_det = split_transform(
                x, split_axis=-2, join_axis=-3, x_value_ndims=3,
                y_shape=[4, -1, 2, 6], a1=2., b1=5., a2=1.5, b2=3.
            )
            y_out, log_det_out = sess.run(
                flow.transform(x_ph), feed_dict={x_ph: x})

            np.testing.assert_allclose(y_out, y)
            np.testing.assert_allclose(log_det_out, log_det)

            invertible_flow_standard_check(
                self, flow, sess, x_ph, feed_dict={x_ph: x})

    def test_errors(self):
        # errors from the constructor
        with pytest.raises(TypeError, match='`left` must be an instance of '
                                            '`BaseFlow`'):
            _ = SplitFlow(-1, object())

        with pytest.raises(TypeError, match='`right` must be an instance of '
                                            '`BaseFlow`'):
            _ = SplitFlow(-1, QuadraticFlow(2., 3.), right=object())

        with pytest.raises(ValueError,
                           match='`left` and `right` must have same `x_value_'
                                 'ndims` and `y_value_ndims`'):
            _ = SplitFlow(-1, left=QuadraticFlow(2., 3., value_ndims=2),
                          right=QuadraticFlow(2., 3., value_ndims=3))

        with pytest.raises(ValueError,
                           match='`x_value_ndims` != `y_value_ndims`, thus '
                                 '`join_axis` must be specified.'):
            _ = SplitFlow(-2,
                          left=SequentialFlow([
                              QuadraticFlow(2., 3., value_ndims=2),
                              ReshapeFlow(2, [-1])
                          ]))

        with pytest.raises(ValueError,
                           match='`x_value_ndims` != `y_value_ndims`, thus '
                                 '`right` must be specified.'):
            _ = SplitFlow(-2,
                          left=SequentialFlow([
                              QuadraticFlow(2., 3., value_ndims=2),
                              ReshapeFlow(2, [-1])
                          ]),
                          join_axis=-1)

        # errors from `build`
        with pytest.raises(ValueError,
                           match='`split_axis` out of range, or not covered '
                                 'by `x_value_ndims`'):
            flow = SplitFlow(split_axis=-3,
                             left=QuadraticFlow(2., 3., value_ndims=2))
            flow.build(tf.zeros([3, 4, 5, 12]))

        with pytest.raises(ValueError,
                           match='`split_axis` out of range, or not covered '
                                 'by `x_value_ndims`'):
            flow = SplitFlow(split_axis=-5,
                             left=QuadraticFlow(2., 3., value_ndims=2))
            flow.build(tf.zeros([3, 4, 5, 12]))

        with pytest.raises(ValueError, match='The split axis of `input` must '
                                             'be at least 2'):
            flow = SplitFlow(split_axis=-1,
                             left=QuadraticFlow(2., 3., value_ndims=1))
            flow.build(tf.zeros([3, 4, 5, 1]))

        # errors from `transform`
        with pytest.raises(RuntimeError,
                           match='`y_left.ndims` != `y_right.ndims`'):
            f1 = ReshapeFlow(x_value_ndims=1, y_value_shape=[-1, 1])
            f1._y_value_ndims = 1  # hack for passing constructor
            flow = SplitFlow(split_axis=-1,
                             join_axis=-1,
                             left=QuadraticFlow(2., 3., value_ndims=1),
                             right=SequentialFlow([
                                 QuadraticFlow(1.5, 3., value_ndims=1),
                                 f1
                             ]))
            flow.transform(tf.zeros([3, 4, 5, 12]))

        with pytest.raises(ValueError, match='`join_axis` out of range, or not '
                                             'covered by `y_value_ndims`'):
            flow = SplitFlow(split_axis=-1,
                             join_axis=-5,
                             left=QuadraticFlow(2., 3., value_ndims=1))
            flow.transform(tf.zeros([3, 4, 5, 12]))
