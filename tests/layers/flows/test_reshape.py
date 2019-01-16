import numpy as np
import pytest
import tensorflow as tf
from mock import mock

from tests.layers.flows.helper import invertible_flow_standard_check
from tests.ops.test_convolution import (naive_space_to_depth,
                                        patched_space_to_depth,
                                        patched_depth_to_space)
from tfsnippet.layers import ReshapeFlow, SpaceToDepthFlow


class ReshapeFlowTestCase(tf.test.TestCase):

    def test_reshape_flow(self):
        def check(x, x_value_ndims, y_value_shape, x_ph=None):
            flow = ReshapeFlow(x_value_ndims=x_value_ndims,
                               y_value_shape=y_value_shape)
            if x_value_ndims > 0:
                batch_shape = list(x.shape[:-x_value_ndims])
            else:
                batch_shape = list(x.shape)
            y_ans = np.reshape(x, batch_shape + list(y_value_shape))
            log_det_ans = np.zeros(batch_shape)

            feed_dict = None
            if x_ph is not None:
                feed_dict = {x_ph: x}
                x = x_ph

            y, log_det = sess.run(flow.transform(x), feed_dict=feed_dict)
            np.testing.assert_allclose(y, y_ans)
            np.testing.assert_allclose(log_det, log_det_ans)

            invertible_flow_standard_check(self, flow, sess, x, feed_dict)

        with self.test_session() as sess:
            # test 3 -> [-1], static shape
            x = np.random.normal(size=[2, 3, 4, 5])
            check(x, 3, [-1])

            # test 3 -> [-1], dynamic shape
            x = np.random.normal(size=[2, 3, 4, 5])
            x_ph = tf.placeholder(dtype=tf.float32, shape=[None] * 4)
            check(x, 3, [-1], x_ph=x_ph)

            # test 1 -> [-1, 2, 3], static shape
            x = np.random.normal(size=[5, 12])
            check(x, 1, [-1, 2, 3])

            # test 1 -> [-1, 2, 3], dynamic shape
            x = np.random.normal(size=[5, 12])
            x_ph = tf.placeholder(dtype=tf.float32, shape=[None] * 2)
            check(x, 1, [-1, 2, 3], x_ph=x_ph)

            # test 0 -> [], dynamic shape
            x = np.random.normal(size=[5, 12])
            x_ph = tf.placeholder(dtype=tf.float32, shape=[None] * 2)
            check(x, 0, [], x_ph=x_ph)

            # test 0 -> [-1], dynamic shape
            x = np.random.normal(size=[5, 12])
            x_ph = tf.placeholder(dtype=tf.float32, shape=[None] * 2)
            check(x, 0, [-1], x_ph=x_ph)

        # test 3 -> [7], static shape, error
        with pytest.raises(ValueError, match='Cannot reshape the tail '
                                             'dimensions of `x` into `y`'):
            x = np.random.normal(size=[2, 3, 4, 5])
            _ = ReshapeFlow(3, [7]).transform(x)

        # test 3 -> [-1, 7], static shape, error
        with pytest.raises(ValueError, match='Cannot reshape the tail '
                                             'dimensions of `x` into `y`'):
            x = np.random.normal(size=[2, 3, 4, 5])
            _ = ReshapeFlow(3, [-1, 7]).transform(x)

        # test 0 -> [-1], static shape, error
        with pytest.raises(ValueError, match='Cannot reshape the tail '
                                             'dimensions of `x` into `y`'):
            x = np.random.normal(size=[2, 3, 4, 5])
            _ = ReshapeFlow(0, [2]).transform(x)

        # test validate require batch ndims
        with pytest.raises(Exception,
                           match=r'`x.ndims` must be known and >= `x_value_'
                                 r'ndims \+ 1`'):
            flow = ReshapeFlow(3, [-1], require_batch_dims=True)
            _ = flow.transform(np.random.normal(size=[2, 3, 4]))

        # test validate require batch ndims
        with pytest.raises(Exception,
                           match='The shape of `y` is invalid'):
            flow = ReshapeFlow(3, [-1], require_batch_dims=True)
            flow.build(np.random.normal(size=[2, 3, 4, 5]))
            _ = flow.inverse_transform(np.random.normal(size=[60]))

        # test the shape error in construction
        with pytest.raises(ValueError, match='`shape` is not a valid shape'):
            _ = ReshapeFlow(x_value_ndims=3, y_value_shape=[-2, 2, 3])

        with pytest.raises(ValueError, match='`shape` is not a valid shape'):
            _ = ReshapeFlow(x_value_ndims=3, y_value_shape=[-1, -1, 2])


class SpaceToDepthFlowTestCase(tf.test.TestCase):

    def test_space_to_depth_flow(self):
        def check(x, y_shape, bs, channels_last=True, x_ph=None):
            # compute the answer
            y = naive_space_to_depth(x, bs, channels_last)
            log_det = np.zeros(y.shape[:-3], dtype=np.float32)

            self.assertTupleEqual(y.shape, y_shape)

            # get feed dict
            feed_dict = None
            if x_ph is not None:
                feed_dict = {x_ph: x}
                x = x_ph

            flow = SpaceToDepthFlow(bs, channels_last=channels_last)
            y_out, log_det_out = sess.run(
                flow.transform(x), feed_dict=feed_dict)

            np.testing.assert_allclose(y_out, y)
            np.testing.assert_allclose(log_det_out, log_det)

            invertible_flow_standard_check(self, flow, sess, x, feed_dict)

        with self.test_session() as sess, \
                mock.patch('tensorflow.space_to_depth', patched_space_to_depth), \
                mock.patch('tensorflow.depth_to_space', patched_depth_to_space):
            # test static shape, bs=2, channels_last = False
            x = np.random.normal(size=[7, 8, 12, 5])
            check(x, (7, 4, 6, 20), 2, channels_last=True)

            # test dynamic shape, bs=3, channels_last = True
            x = np.random.normal(size=[4, 7, 5, 6, 9])
            x_ph = tf.placeholder(shape=[None, None, 5, None, None],
                                  dtype=tf.float32)
            check(x, (4, 7, 45, 2, 3), 3, channels_last=False, x_ph=x_ph)
