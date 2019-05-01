import os
import tensorflow as tf

from tests.helper import assert_variables
from tfsnippet.scaffold import ScheduledVariable, AnnealingVariable
from tfsnippet.utils import ensure_variables_initialized, TemporaryDirectory


class ScheduledVariableTestCase(tf.test.TestCase):

    def test_ScheduledVariable(self):
        v = ScheduledVariable('v', 123., dtype=tf.int32, model_var=True,
                              collections=['my_variables'])
        assert_variables(['v'], trainable=False, collections=['my_variables'])

        with TemporaryDirectory() as tmpdir:
            saver = tf.train.Saver(var_list=[v.variable])
            save_path = os.path.join(tmpdir, 'saved_var')

            with self.test_session() as sess:
                ensure_variables_initialized()

                self.assertEqual(v.get(), 123)
                self.assertEqual(sess.run(v), 123)
                self.assertEqual(v.set(456), 456)
                self.assertEqual(v.get(), 456)

                saver.save(sess, save_path)

            with self.test_session() as sess:
                saver.restore(sess, save_path)
                self.assertEqual(v.get(), 456)

                sess.run(v.assign_op, feed_dict={v.assign_ph: 789})
                self.assertEqual(v.get(), 789)

    def test_AnnealingDynamicValue(self):
        with self.test_session() as sess:
            # test without min_value
            v = AnnealingVariable('v', 1, 2)
            ensure_variables_initialized()
            self.assertEqual(v.get(), 1)

            self.assertEqual(v.anneal(), 2)
            self.assertEqual(v.get(), 2)
            self.assertEqual(v.anneal(), 4)
            self.assertEqual(v.get(), 4)

            self.assertEqual(v.set(2), 2)
            self.assertEqual(v.get(), 2)
            self.assertEqual(v.anneal(), 4)
            self.assertEqual(v.get(), 4)

            # test with min_value
            v = AnnealingVariable('v2', 1, .5, 2)
            ensure_variables_initialized()
            self.assertEqual(v.get(), 2)

            v = AnnealingVariable('v3', 1, .5, .5)
            ensure_variables_initialized()
            self.assertEqual(v.get(), 1)
            self.assertEqual(v.anneal(), .5)
            self.assertEqual(v.get(), .5)
            self.assertEqual(v.anneal(), .5)
            self.assertEqual(v.get(), .5)
