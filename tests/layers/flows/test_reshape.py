import numpy as np
import pytest
import tensorflow as tf

from tests.layers.flows.helper import invertible_flow_standard_check
from tfsnippet.layers import ReshapeFlow


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
