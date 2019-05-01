import numpy as np
import tensorflow as tf

from tests.helper import assert_variables
from tests.layers.flows.helper import invertible_flow_standard_check
from tfsnippet.layers import FeatureShufflingFlow


class FeatureShufflingFlowTestCase(tf.test.TestCase):

    def test_feature_shuffling_flow(self):
        np.random.seed(1234)

        with self.test_session() as sess:
            # axis = -1, value_ndims = 1
            x = np.random.normal(size=[3, 4, 5, 6]).astype(np.float32)
            x_ph = tf.placeholder(dtype=tf.float32, shape=[None, None, None, 6])
            permutation = np.arange(6, dtype=np.int32)
            np.random.shuffle(permutation)
            y = x[..., permutation]
            log_det = np.zeros([3, 4, 5]).astype(np.float32)

            layer = FeatureShufflingFlow(axis=-1, value_ndims=1)
            y_out, log_det_out = layer.transform(x_ph)
            sess.run(tf.assign(layer._permutation, permutation))
            y_out, log_det_out = sess.run(
                [y_out, log_det_out], feed_dict={x_ph: x})

            np.testing.assert_equal(y_out, y)
            np.testing.assert_equal(log_det_out, log_det)

            invertible_flow_standard_check(
                self, layer, sess, x_ph, feed_dict={x_ph: x})

            assert_variables(['permutation'], trainable=False,
                             scope='feature_shuffling_flow',
                             collections=[tf.GraphKeys.MODEL_VARIABLES])

            # axis = -2, value_ndims = 3
            x = np.random.normal(size=[3, 4, 5, 6]).astype(np.float32)
            x_ph = tf.placeholder(dtype=tf.float32, shape=[None, None, 5, None])
            permutation = np.arange(5, dtype=np.int32)
            np.random.shuffle(permutation)
            y = x[..., permutation, :]
            log_det = np.zeros([3]).astype(np.float32)

            layer = FeatureShufflingFlow(axis=-2, value_ndims=3)
            y_out, log_det_out = layer.transform(x_ph)
            sess.run(tf.assign(layer._permutation, permutation))
            y_out, log_det_out = sess.run(
                [y_out, log_det_out], feed_dict={x_ph: x})

            np.testing.assert_equal(y_out, y)
            np.testing.assert_equal(log_det_out, log_det)

            invertible_flow_standard_check(
                self, layer, sess, x_ph, feed_dict={x_ph: x})
