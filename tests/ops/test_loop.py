import numpy as np
import tensorflow as tf

from tfsnippet.ops import pixelcnn_2d_sample, convert_to_tensor_and_cast


class PixelCNN2DSampleTestCase(tf.test.TestCase):

    def test_pixelcnn_2d_sample(self):
        height, width = 31, 32
        self.assertLess(height * width, 10000)

        def make_x(dtype, channels_last=True):
            x = tf.range(height * width, dtype=dtype)
            if channels_last:
                x = tf.reshape(x, [1, height, width, 1])
            else:
                x = tf.reshape(x, [1, 1, height, width])
            return x

        with self.test_session() as sess:
            # test static args
            def f(i, x):
                offset = convert_to_tensor_and_cast((i + 1) * 10000,
                                                    dtype=dtype)
                return offset + make_x(dtype=dtype)

            dtype = tf.int32
            x = make_x(dtype=dtype)
            ans = sess.run(x + (x + 1) * 10000)
            y = pixelcnn_2d_sample(f, x, height, width, channels_last=True)

            np.testing.assert_equal(sess.run(y), ans)

            # test dynamic args
            def f(i, x):
                offset = convert_to_tensor_and_cast((i + 1) * 10000,
                                                    dtype=dtype)
                return offset + make_x(dtype=dtype, channels_last=False)

            dtype = tf.int32
            x = make_x(dtype=dtype, channels_last=False)
            mask = tf.reshape(
                tf.concat([tf.ones([10], dtype=tf.int32),
                           tf.zeros([height * width - 110], dtype=tf.int32),
                           tf.ones([100], dtype=tf.int32)],
                          axis=0),
                [1, 1, height, width]
            )
            ans = sess.run(x * mask + (x + (x + 1) * 10000) * (1 - mask))
            height_t = tf.placeholder(dtype=tf.int32, shape=())
            width_t = tf.placeholder(dtype=tf.int32, shape=())
            y = pixelcnn_2d_sample(
                f, x, height_t, width_t,
                start=tf.constant(10),
                end=tf.constant(height * width - 100),
                channels_last=False
            )

            np.testing.assert_equal(
                sess.run(y, feed_dict={height_t: height, width_t: width}),
                ans
            )
