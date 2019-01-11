import random

import numpy as np
import tensorflow as tf

from tfsnippet.utils import VarScopeRandomState, set_random_seed


class SetRandomSeedTestCase(tf.test.TestCase):

    def test_set_random_seed(self):
        with tf.Graph().as_default():
            with self.test_session() as sess:
                set_random_seed(0)
                np_x = np.random.randn()
                tf_x = sess.run(tf.random_normal(shape=[], seed=0))
                vsrs_seed = VarScopeRandomState._global_seed

                set_random_seed(1)
                self.assertNotEqual(np.random.randn(), np_x)
                self.assertNotEqual(
                    sess.run(tf.random_normal(shape=[], seed=0)), tf_x)
                self.assertNotEqual(VarScopeRandomState._global_seed, vsrs_seed)

        with tf.Graph().as_default():
            with self.test_session() as sess:
                set_random_seed(0)
                self.assertEqual(np.random.randn(), np_x)
                self.assertEqual(
                    sess.run(tf.random_normal(shape=[], seed=0)), tf_x)
                self.assertEqual(VarScopeRandomState._global_seed, vsrs_seed)


class VarScopeRandomStateTestCase(tf.test.TestCase):

    def test_VarScopeRandomState(self):
        def get_seq():
            state = VarScopeRandomState(tf.get_variable_scope())
            return state.randint(0, 0xffffffff, size=[100])

        with tf.Graph().as_default():
            VarScopeRandomState.set_global_seed(0)

            with tf.variable_scope('a'):
                a = get_seq()

            with tf.variable_scope('a'):
                np.testing.assert_equal(get_seq(), a)

            with tf.variable_scope('b'):
                self.assertFalse(np.all(get_seq() == a))

        with tf.Graph().as_default():
            VarScopeRandomState.set_global_seed(0)

            with tf.variable_scope('a'):
                np.testing.assert_equal(get_seq(), a)

            VarScopeRandomState.set_global_seed(1)

            with tf.variable_scope('a'):
                self.assertFalse(np.all(get_seq() == a))

            VarScopeRandomState.set_global_seed(0)

            with tf.variable_scope('a'):
                np.testing.assert_equal(get_seq(), a)
