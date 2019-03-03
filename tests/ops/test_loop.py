import numpy as np
import pytest
import tensorflow as tf

from tfsnippet.ops import pixelcnn_2d_sample, convert_to_tensor_and_cast


class PixelCNN2DSampleTestCase(tf.test.TestCase):

    def test_pixelcnn_2d_sample(self):
        height, width = 31, 32
        self.assertLess(height * width, 10000)

        def make_x(channels_last=True):
            x = tf.range(height * width, dtype=tf.int32)
            if channels_last:
                x = tf.reshape(x, [1, height, width, 1])
            else:
                x = tf.reshape(x, [1, 1, height, width])
            return x

        with self.test_session() as sess:
            # test static args
            def f(i, inputs):
                offset = convert_to_tensor_and_cast((i + 1) * 10000,
                                                    dtype=tf.int32)
                return [offset + make_x()]

            x = make_x()
            ans = sess.run(x + (x + 1) * 10000)
            y = pixelcnn_2d_sample(f, [x], height, width, channels_last=True)[0]

            np.testing.assert_equal(sess.run(y), ans)

            # test dynamic args
            def f(i, inputs):
                offset = convert_to_tensor_and_cast((i + 1) * 10000,
                                                    dtype=tf.int32)
                return [offset + make_x(channels_last=False)]

            x = make_x(channels_last=False)
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
                f, [x], height_t, width_t,
                start=tf.constant(10),
                end=tf.constant(height * width - 100),
                channels_last=False
            )[0]

            np.testing.assert_equal(
                sess.run(y, feed_dict={height_t: height, width_t: width}),
                ans
            )

    def test_errors(self):
        height, width = 31, 32

        def fn(i, inputs):
            return inputs

        with pytest.raises(ValueError, match='`inputs` must not be empty'):
            _ = pixelcnn_2d_sample(fn, [], height, width)

        with pytest.raises(ValueError,
                           match=r'The shape of `inputs\[1\]` is invalid'):
            inputs = [tf.zeros([1, height, width, 1]),
                      tf.zeros([2, 1, height, width, 1])]
            _ = pixelcnn_2d_sample(fn, inputs, height, width)

        def fn(i, inputs):
            return [inputs[0]]

        with pytest.raises(ValueError,
                           match='The length of outputs != inputs: 1 vs 2'):
            inputs = [tf.zeros([1, height, width, 1]),
                      tf.zeros([1, height, width, 1])]
            _ = pixelcnn_2d_sample(fn, inputs, height, width)

        def fn(i, inputs):
            return [tf.cast(inputs[0], dtype=tf.float64)]

        with pytest.raises(TypeError,
                           match=r'`outputs\[0\].dtype` != `inputs\[0\].dtype`'
                                 r': .* vs .*'):
            inputs = [tf.zeros([1, height, width, 1], dtype=tf.float32)]
            _ = pixelcnn_2d_sample(fn, inputs, height, width)
