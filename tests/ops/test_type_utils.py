import numpy as np
import tensorflow as tf

from tfsnippet.ops import convert_to_tensor_and_cast


class ConvertToTensorAndCastTestCase(tf.test.TestCase):

    def test_convert_to_tensor_and_cast(self):
        def check(x, dtype=None):
            z = tf.convert_to_tensor(x)
            y = convert_to_tensor_and_cast(x, dtype)
            self.assertIsInstance(y, tf.Tensor)
            if dtype is not None:
                self.assertEqual(y.dtype, dtype)
            else:
                self.assertEqual(y.dtype, z.dtype)

        check(np.arange(10, dtype=np.float32))
        check(np.arange(10, dtype=np.float32), np.float64)
        check(np.arange(10, dtype=np.float32), tf.float64)
        check(tf.range(10, dtype=tf.float32))
        check(tf.range(10, dtype=tf.float32), np.float64)
        check(tf.range(10, dtype=tf.float32), tf.float64)
