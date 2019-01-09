import numpy as np
import tensorflow as tf

from tfsnippet.layers import BaseCouplingLayer
from tfsnippet.utils import flatten_to_ndims, unflatten_from_ndims


class BaseCouplingLayerTestCase(tf.test.TestCase):

    def test_coupling_layer(self):
        np.random.seed(1234)
        kernel1 = np.random.normal(size=[3, 2]).astype(np.float32)
        kernel2 = np.random.normal(size=[2, 3]).astype(np.float32)

        class MyCouplingLayer(BaseCouplingLayer):
            def __init__(self, no_scale=False, *args, **kwargs):
                self.no_scale = no_scale
                super(MyCouplingLayer, self).__init__(*args, **kwargs)

            def _compute_shift_and_scale(o, x1, n2):
                kernel = kernel1 if n2 == 2 else kernel2
                assert(kernel.shape[-1] == n2)
                x1, s1, s2 = flatten_to_ndims(x1, 2)
                scale = unflatten_from_ndims(tf.matmul(x1, kernel), s1, s2)
                shift = scale + 10.
                return shift, scale

        with self.test_session() as sess:
            # test linear scale, primary
            x = np.random.normal(size=[3, 4, 5])
            x_ph = tf.placeholder(dtype=tf.float32, shape=[None, None, 5])
            layer = MyCouplingLayer(axis=-1, value_ndims=2)
            y, log_det = layer.transform(x_ph)
            y_out, log_det_out = sess.run([y, log_det], feed_dict={x_ph: x})

            print(y_out, log_det_out)
