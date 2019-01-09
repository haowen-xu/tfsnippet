import numpy as np
import pytest
import tensorflow as tf

from tfsnippet.layers.flows.utils import (is_log_det_shape_matches_input,
                                          assert_log_det_shape_matches_input,
                                          broadcast_log_det_against_input,
                                          SigmoidScale,
                                          ExpScale,
                                          LinearScale)


class IsLogDetShapeMatchesInputTestCase(tf.test.TestCase):

    def test_is_log_det_shape_matches_input(self):
        g = lambda size: np.random.random(size=size)
        input = g([2, 1, 3])

        input_3 = tf.placeholder(shape=[None, None, None], dtype=tf.float32)
        input_ph = tf.placeholder(shape=None, dtype=tf.float32)
        log_det_3 = tf.placeholder(shape=[None, None, None], dtype=tf.float32)
        log_det_2 = tf.placeholder(shape=[None, None], dtype=tf.float32)
        log_det_1 = tf.placeholder(shape=[None], dtype=tf.float32)
        log_det_ph = tf.placeholder(shape=None, dtype=tf.float32)

        # static result, good cases
        self.assertTrue(is_log_det_shape_matches_input(g([2, 1, 3]), input, 0))
        self.assertTrue(is_log_det_shape_matches_input(g([2, 1]), input, 1))
        self.assertTrue(is_log_det_shape_matches_input(g([2]), input, 2))
        self.assertTrue(is_log_det_shape_matches_input(g([]), input, 3))

        # static result, bad cases
        self.assertFalse(is_log_det_shape_matches_input(g([2, 5, 3]), input, 0))
        self.assertFalse(is_log_det_shape_matches_input(g([2, 1, 3]), input, 1))
        self.assertFalse(is_log_det_shape_matches_input(g([2, 1]), input, 0))
        self.assertFalse(is_log_det_shape_matches_input(g([2]), input, 4))

        with self.test_session() as sess:
            # good cases
            self.assertTrue(sess.run(
                is_log_det_shape_matches_input(log_det_3, input_3, 0),
                feed_dict={log_det_3: g([2, 1, 3]), input_3: input}
            ))
            self.assertTrue(sess.run(
                is_log_det_shape_matches_input(log_det_2, input_3, 1),
                feed_dict={log_det_2: g([2, 1]), input_3: input}
            ))
            self.assertTrue(sess.run(
                is_log_det_shape_matches_input(log_det_1, input_3, 2),
                feed_dict={log_det_1: g([2]), input_3: input}
            ))

            # good cases for one fully dynamic shape
            self.assertTrue(sess.run(
                is_log_det_shape_matches_input(log_det_ph, input_3, 0),
                feed_dict={log_det_ph: g([2, 1, 3]), input_3: input}
            ))
            self.assertTrue(sess.run(
                is_log_det_shape_matches_input(log_det_3, input_ph, 0),
                feed_dict={log_det_3: g([2, 1, 3]), input_ph: input}
            ))

            # good cases for both fully dynamic shape
            self.assertTrue(sess.run(
                is_log_det_shape_matches_input(log_det_ph, input_ph, 0),
                feed_dict={log_det_ph: g([2, 1, 3]), input_ph: input}
            ))
            self.assertTrue(sess.run(
                is_log_det_shape_matches_input(log_det_ph, input_ph, 1),
                feed_dict={log_det_ph: g([2, 1]), input_ph: input}
            ))
            self.assertTrue(sess.run(
                is_log_det_shape_matches_input(log_det_ph, input_ph, 2),
                feed_dict={log_det_ph: g([2]), input_ph: input}
            ))
            self.assertTrue(sess.run(
                is_log_det_shape_matches_input(log_det_ph, input_ph, 3),
                feed_dict={log_det_ph: g([]), input_ph: input}
            ))

            # bad cases for both fully dynamic shape
            self.assertFalse(sess.run(
                is_log_det_shape_matches_input(log_det_ph, input_ph, 0),
                feed_dict={log_det_ph: g([2, 5, 3]), input_ph: input}
            ))
            self.assertFalse(sess.run(
                is_log_det_shape_matches_input(log_det_ph, input_ph, 1),
                feed_dict={log_det_ph: g([2, 1, 3]), input_ph: input}
            ))
            self.assertFalse(sess.run(
                is_log_det_shape_matches_input(log_det_ph, input_ph, 0),
                feed_dict={log_det_ph: g([2, 1]), input_ph: input}
            ))
            self.assertFalse(sess.run(
                is_log_det_shape_matches_input(log_det_ph, input_ph, 4),
                feed_dict={log_det_ph: g([2]), input_ph: input}
            ))

    def test_assert_log_det_shape_matches_input(self):
        g = lambda size: np.random.random(size=size)
        input = g([2, 1, 3])
        input_ph = tf.placeholder(shape=None, dtype=tf.float32)
        log_det_ph = tf.placeholder(shape=None, dtype=tf.float32)

        # static assertion
        self.assertIsNone(
            assert_log_det_shape_matches_input(g([2, 1, 3]), input, 0))
        self.assertIsNone(
            assert_log_det_shape_matches_input(g([]), input, 3))

        with pytest.raises(AssertionError,
                           match='The shape of `log_det` does not match the '
                                 'shape of `input`'):
            _ = assert_log_det_shape_matches_input(g([2, 5, 3]), input, 0)
        with pytest.raises(AssertionError,
                           match='The shape of `log_det` does not match the '
                                 'shape of `input`'):
            _ = assert_log_det_shape_matches_input(g([2, 1, 3]), input, 1)

        # dynamic assertion
        assert_op = assert_log_det_shape_matches_input(log_det_ph, input_ph, 0)

        with self.test_session() as sess:
            with tf.control_dependencies([assert_op]):
                x = tf.constant(123)

            self.assertEqual(sess.run(x, feed_dict={
                input_ph: input, log_det_ph: g([2, 1, 3])}), 123)

            with pytest.raises(Exception,
                               match='The shape of `log_det` does not match '
                                     'the shape of `input`'):
                _ = sess.run(x, feed_dict={
                    input_ph: input, log_det_ph: g([2, 5, 3])})
            with pytest.raises(Exception,
                               match='The shape of `log_det` does not match '
                                     'the shape of `input`'):
                _ = sess.run(x, feed_dict={
                    input_ph: input, log_det_ph: g([2, 1])})


class BroadcastLogDetAgainstInputTestCase(tf.test.TestCase):

    def test_broadcast_log_det_against_input(self):
        g = lambda size: np.random.random(size=size)
        log_det = g([2, 1, 3])
        input = g([4, 2, 5, 3])
        input_ph = tf.placeholder(shape=None, dtype=tf.float32)
        log_det_ph = tf.placeholder(shape=None, dtype=tf.float32)

        with self.test_session() as sess:
            # test static shape, value_ndims = 0
            out = sess.run(
                broadcast_log_det_against_input(log_det, input, 0))
            self.assertTupleEqual(out.shape, (4, 2, 5, 3))
            np.testing.assert_allclose(
                out,
                np.tile(np.reshape(log_det, [1, 2, 1, 3]), [4, 1, 5, 1])
            )
            # test static shape, value_ndims = 1
            out = sess.run(
                broadcast_log_det_against_input(log_det[..., 0], input, 1))
            self.assertTupleEqual(out.shape, (4, 2, 5))
            np.testing.assert_allclose(
                out,
                np.tile(np.reshape(log_det[..., 0], [1, 2, 1]), [4, 1, 5])
            )
            # test static shape, rank too small assertion
            with pytest.raises(Exception, match='Cannot broadcast `log_det` '
                                                'against `input`'):
                _ = sess.run(
                    broadcast_log_det_against_input(log_det, input, 5))

            # test dynamic shape, value_ndims = 0
            out = sess.run(
                broadcast_log_det_against_input(log_det_ph, input_ph, 0),
                feed_dict={input_ph: input, log_det_ph: log_det}
            )
            self.assertTupleEqual(out.shape, (4, 2, 5, 3))
            np.testing.assert_allclose(
                out,
                np.tile(np.reshape(log_det, [1, 2, 1, 3]), [4, 1, 5, 1])
            )
            # test dynamic shape, value_ndims = 1
            out = sess.run(
                broadcast_log_det_against_input(log_det_ph, input_ph, 1),
                feed_dict={input_ph: input, log_det_ph: log_det[..., 0]}
            )
            self.assertTupleEqual(out.shape, (4, 2, 5))
            np.testing.assert_allclose(
                out,
                np.tile(np.reshape(log_det[..., 0], [1, 2, 1]), [4, 1, 5])
            )
            # test dynamic shape, rank too small assertion
            with pytest.raises(Exception, match='Cannot broadcast `log_det` '
                                                'against `input`'):
                _ = sess.run(
                    broadcast_log_det_against_input(log_det_ph, input_ph, 5),
                    feed_dict={input_ph: input, log_det_ph: log_det[..., 0]}
                )


class ScaleTestCase(tf.test.TestCase):

    def test_scale(self):
        def check(f, cls, x, pre_scale):
            o = cls(pre_scale)

            np.testing.assert_allclose(
                sess.run(o.apply(x)),
                x * f(pre_scale),
                rtol=1e-5
            )
            np.testing.assert_allclose(
                sess.run(o.apply(x, reverse=True)),
                x / f(pre_scale),
                rtol=1e-5
            )
            np.testing.assert_allclose(
                sess.run(o.log_scale()),
                np.log(np.abs(f(pre_scale))),
                rtol=1e-5
            )
            np.testing.assert_allclose(
                sess.run(o.log_scale(reverse=True)),
                -np.log(np.abs(f(pre_scale))),
                rtol=1e-5
            )

        with self.test_session() as sess:
            pre_scale = np.random.normal(size=[2, 3, 4, 5])
            x = np.random.normal(size=pre_scale.shape)

            check((lambda x: 1. / (1 + np.exp(-x))), SigmoidScale,
                  x, pre_scale)
            check(np.exp, ExpScale, x, pre_scale)
            check((lambda x: x), LinearScale, x, pre_scale)
