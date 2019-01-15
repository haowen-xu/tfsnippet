import os
import tensorflow as tf

from tests.helper import assert_variables
from tfsnippet.trainer import *
from tfsnippet.utils import ensure_variables_initialized, TemporaryDirectory


class ScheduledVariableTestCase(tf.test.TestCase):

    def test_ScheduledVariable(self):
        v = ScheduledVariable('v', 123., dtype=tf.int32)
        assert_variables(['v'], trainable=False)

        with TemporaryDirectory() as tmpdir:
            saver = tf.train.Saver(var_list=[v.variable])
            save_path = os.path.join(tmpdir, 'saved_var')

            with self.test_session() as sess:
                ensure_variables_initialized()

                self.assertEqual(v.get(), 123)
                self.assertEqual(sess.run(v), 123)
                v.set(456)
                self.assertEqual(v.get(), 456)

                saver.save(sess, save_path)

            with self.test_session() as sess:
                saver.restore(sess, save_path)
                self.assertEqual(v.get(), 456)

    def test_AnnealingDynamicValue(self):
        with self.test_session() as sess:
            # test without min_value
            v = AnnealingVariable('v', 1, 2)
            ensure_variables_initialized()
            self.assertEqual(1, v.get())

            v.anneal()
            self.assertEqual(2, v.get())
            v.anneal()
            self.assertEqual(4, v.get())

            v.set(2)
            self.assertEqual(2, v.get())
            v.anneal()
            self.assertEqual(4, v.get())

            # test with min_value
            v = AnnealingVariable('v2', 1, .5, 2)
            ensure_variables_initialized()
            self.assertEqual(2, v.get())

            v = AnnealingVariable('v3', 1, .5, .5)
            ensure_variables_initialized()
            self.assertEqual(1, v.get())
            v.anneal()
            self.assertEqual(.5, v.get())
            v.anneal()
            self.assertEqual(.5, v.get())
