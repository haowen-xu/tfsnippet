import pytest
import numpy as np
import tensorflow as tf

from tfsnippet.utils import *


class AssertionTestCase(tf.test.TestCase):

    def test_assert_deps(self):
        ph = tf.placeholder(dtype=tf.bool, shape=())
        op = tf.assert_equal(ph, True, message='abcdefg')

        # test ops are empty
        with assert_deps([None]) as asserted:
            self.assertFalse(asserted)

        # test assertion enabled, and ops are not empty
        with self.test_session() as sess, \
                scoped_set_config(settings, enable_assertions=True):
            with assert_deps([op, None]) as asserted:
                self.assertTrue(asserted)
                out = tf.constant(1.)
            with pytest.raises(Exception, match='abcdefg'):
                self.assertEqual(sess.run(out, feed_dict={ph: False}), 1.)

        # test assertion disabled
        with self.test_session() as sess, \
                scoped_set_config(settings, enable_assertions=False):
            with assert_deps([op, None]) as asserted:
                self.assertFalse(asserted)
                out = tf.constant(1.)
            self.assertEqual(sess.run(out, feed_dict={ph: False}), 1.)


class CheckNumericsTestCase(tf.test.TestCase):

    def test_check_numerics(self):
        # test enabled
        ph = tf.placeholder(dtype=tf.float32, shape=())
        with scoped_set_config(settings, check_numerics=True):
            x = maybe_check_numerics(ph, message='numerical issues')
        with pytest.raises(Exception, match='numerical issues'):
            with self.test_session() as sess:
                _ = sess.run(x, feed_dict={ph: np.nan})

        # test disabled
        with scoped_set_config(settings, check_numerics=False):
            x = maybe_check_numerics(
                tf.constant(np.nan), message='numerical issues')
        with self.test_session() as sess:
            self.assertTrue(np.isnan(sess.run(x)))


class AddHistogramTestCase(tf.test.TestCase):

    def test_add_histogram(self):
        with tf.name_scope('parent'):
            x = tf.constant(0., name='x')
            y = tf.constant(1., name='y')
            z = tf.constant(2., name='z')
            w = tf.constant(3., name='w')

        # test enabled
        with scoped_set_config(settings, auto_histogram=True):
            maybe_add_histogram(x, strip_scope=True)
            maybe_add_histogram(y, summary_name='the_tensor')
            maybe_add_histogram(z, collections=[tf.GraphKeys.SUMMARIES])

        # test disabled
        with scoped_set_config(settings, auto_histogram=False):
            maybe_add_histogram(w)

        self.assertListEqual(
            [op.name for op in tf.get_collection(GraphKeys.AUTO_HISTOGRAM)],
            ['maybe_add_histogram/x:0', 'maybe_add_histogram_1/the_tensor:0']
        )
        self.assertListEqual(
            [op.name for op in tf.get_collection(tf.GraphKeys.SUMMARIES)],
            ['maybe_add_histogram_2/parent/z:0']
        )
