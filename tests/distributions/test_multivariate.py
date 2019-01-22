import numpy as np
import tensorflow as tf

from tfsnippet.distributions import *


class OnehotCategoricalTestCase(tf.test.TestCase):

    def test_props(self):
        logits = np.arange(24, dtype=np.float32).reshape([2, 3, 4])
        with self.test_session():
            one_hot_categorical = OnehotCategorical(logits=tf.constant(logits))
            self.assertEqual(one_hot_categorical.value_ndims, 1)
            self.assertEqual(one_hot_categorical.n_categories, 4)
            np.testing.assert_allclose(
                one_hot_categorical.logits.eval(), logits)


class ConcreteCategoricalTestCase(tf.test.TestCase):

    def test_props(self):
        logits = np.arange(24, dtype=np.float32).reshape([2, 3, 4])
        with self.test_session():
            concrete = Concrete(temperature=.5, logits=tf.constant(logits))
            self.assertEqual(concrete.value_ndims, 1)
            self.assertEqual(concrete.temperature.eval(), .5)
            self.assertEqual(concrete.n_categories, 4)
            np.testing.assert_allclose(concrete.logits.eval(), logits)


class ExpConcreteCategoricalTestCase(tf.test.TestCase):

    def test_props(self):
        logits = np.arange(24, dtype=np.float32).reshape([2, 3, 4])
        with self.test_session():
            exp_concrete = ExpConcrete(
                temperature=.5, logits=tf.constant(logits))
            self.assertEqual(exp_concrete.value_ndims, 1)
            self.assertEqual(exp_concrete.temperature.eval(), .5)
            self.assertEqual(exp_concrete.n_categories, 4)
            np.testing.assert_allclose(exp_concrete.logits.eval(), logits)
