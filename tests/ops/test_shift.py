import numpy as np
import pytest
import tensorflow as tf

from tfsnippet.ops import shift


class ShiftTestCase(tf.test.TestCase):

    def test_shift(self):
        x = np.random.normal(size=[3, 4, 5, 6])
        t = tf.convert_to_tensor(x)

        with self.test_session() as sess:
            # test shift a scalar will do nothing
            t0 = tf.constant(0.)
            self.assertIs(shift(t0, []), t0)

            # test shift by zeros should result in `t`
            self.assertIs(shift(t, [0, 0, 0, 0]), t)

            # test shift all contents outside the size
            y = np.zeros_like(x)
            np.testing.assert_allclose(
                sess.run(shift(x, [3, 4, 5, 6])),
                y
            )
            np.testing.assert_allclose(
                sess.run(shift(x, [-3, -4, -5, -6])),
                y
            )

            # test shift by various distances
            y = np.zeros_like(x)
            y[1:, :-2, 3:, :] = x[:-1, 2:, :-3, :]
            np.testing.assert_allclose(
                sess.run(shift(x, [1, -2, 3, 0])),
                y
            )
            for i in range(4):
                s = [0] * 4
                with pytest.raises(ValueError, match='Cannot shift `input`: '
                                                     'input .* vs shift .*'):
                    s[i] = 4 + i
                    _ = shift(x, s)
                with pytest.raises(ValueError, match='Cannot shift `input`: '
                                                     'input .* vs shift .*'):
                    s[i] = -(4 + i)
                    _ = shift(x, s)

            # test shift dynamic shape
            ph = tf.placeholder(dtype=tf.float64, shape=[None] * 4)
            np.testing.assert_allclose(
                sess.run(shift(ph, [1, -2, 3, 0]), feed_dict={ph: x}),
                y
            )
            for i in range(4):
                s = [0] * 4
                s[i] = 4 + i

                output = shift(ph, s)
                with pytest.raises(Exception, match='Cannot shift `input`: '
                                                    'input .* vs shift .*'):
                    _ = sess.run(output, feed_dict={ph: x})

                s[i] = -(4 + i)
                output = shift(ph, s)
                with pytest.raises(Exception, match='Cannot shift `input`: '
                                                    'input .* vs shift .*'):
                    _ = sess.run(output, feed_dict={ph: x})

        with pytest.raises(ValueError, match='The rank of `shape` is required '
                                             'to be deterministic:'):
            _ = shift(tf.placeholder(dtype=tf.float64, shape=None), [0])

        with pytest.raises(ValueError,
                           match='The length of `shift` is required to equal '
                                 'the rank of `input`: shift .* vs input .*'):
            _ = shift(x, [0, 1, 2])

        with pytest.raises(ValueError,
                           match='The length of `shift` is required to equal '
                                 'the rank of `input`: shift .* vs input .*'):
            _ = shift(tf.constant(0.), [0])
