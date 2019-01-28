import os

import pytest
import numpy as np
import tensorflow as tf

from tfsnippet.scaffold import EarlyStopping
from tfsnippet.utils import TemporaryDirectory, get_default_session_or_error


def get_variable_values(variables):
    sess = get_default_session_or_error()
    return sess.run(variables)


def set_variable_values(variables, values):
    sess = get_default_session_or_error()
    sess.run([tf.assign(v, a) for v, a in zip(variables, values)])


def _populate_variables():
    a = tf.get_variable('a', shape=(), dtype=tf.int32)
    b = tf.get_variable('b', shape=(), dtype=tf.int32)
    c = tf.get_variable('c', shape=(), dtype=tf.int32)
    set_variable_values([a, b, c], [1, 2, 3])
    return [a, b, c]


class EarlyStoppingTestCase(tf.test.TestCase):

    def test_prerequisites(self):
        with self.test_session():
            a, b, c = _populate_variables()
            self.assertEqual(get_variable_values([a, b, c]), [1, 2, 3])

    def test_param_vars_must_not_be_empty(self):
        with self.test_session():
            with pytest.raises(
                    ValueError, match='`param_vars` must not be empty'):
                with EarlyStopping([]):
                    pass

    def test_early_stopping_context_without_updating_loss(self):
        with self.test_session():
            a, b, c = _populate_variables()
            with EarlyStopping([a, b]) as es:
                set_variable_values([a], [10])
            self.assertFalse(es.ever_updated)
            self.assertEqual(get_variable_values([a, b, c]), [10, 2, 3])

    def test_the_first_loss_will_always_cause_saving(self):
        with self.test_session():
            a, b, c = _populate_variables()
            with EarlyStopping([a, b]) as es:
                set_variable_values([a], [10])
                self.assertTrue(es.update(1.))
                set_variable_values([a, b], [100, 20])
            self.assertTrue(es.ever_updated)
            self.assertAlmostEqual(es.best_metric, 1.)
            self.assertEqual(get_variable_values([a, b, c]), [10, 2, 3])

    def test_memorize_the_best_loss(self):
        with self.test_session():
            a, b, c = _populate_variables()
            with EarlyStopping([a, b]) as es:
                set_variable_values([a], [10])
                self.assertTrue(es.update(1.))
                self.assertAlmostEqual(es.best_metric, 1.)
                set_variable_values([a, b], [100, 20])
                self.assertTrue(es.update(.5))
                self.assertAlmostEqual(es.best_metric, .5)
                set_variable_values([a, b, c], [1000, 200, 30])
                self.assertFalse(es.update(.8))
                self.assertAlmostEqual(es.best_metric, .5)
            self.assertTrue(es.ever_updated)
            self.assertAlmostEqual(es.best_metric, .5)
            self.assertEqual(get_variable_values([a, b, c]), [100, 20, 30])

    def test_initial_loss(self):
        with self.test_session():
            a, b, c = _populate_variables()
            with EarlyStopping([a, b], initial_metric=.6) as es:
                set_variable_values([a], [10])
                self.assertFalse(es.update(1.))
                self.assertAlmostEqual(es.best_metric, .6)
                set_variable_values([a, b], [100, 20])
                self.assertTrue(es.update(.5))
                self.assertAlmostEqual(es.best_metric, .5)
            self.assertEqual(get_variable_values([a, b, c]), [100, 20, 3])

    def test_initial_loss_is_tensor(self):
        with self.test_session():
            a, b, c = _populate_variables()
            with EarlyStopping([a, b], initial_metric=tf.constant(.5)) as es:
                np.testing.assert_equal(es.best_metric, .5)

    def test_do_not_restore_on_error(self):
        with self.test_session():
            a, b, c = _populate_variables()
            with pytest.raises(ValueError, match='value error'):
                with EarlyStopping([a, b], restore_on_error=False) as es:
                    self.assertTrue(es.update(1.))
                    set_variable_values([a, b], [10, 20])
                    raise ValueError('value error')
            self.assertAlmostEqual(es.best_metric, 1.)
            self.assertEqual(get_variable_values([a, b, c]), [10, 20, 3])

    def test_restore_on_error(self):
        with self.test_session():
            a, b, c = _populate_variables()
            with pytest.raises(ValueError, match='value error'):
                with EarlyStopping([a, b], restore_on_error=True) as es:
                    self.assertTrue(es.update(1.))
                    set_variable_values([a, b], [10, 20])
                    raise ValueError('value error')
            self.assertAlmostEqual(es.best_metric, 1.)
            self.assertEqual(get_variable_values([a, b, c]), [1, 2, 3])

    def test_restore_on_keyboard_interrupt(self):
        with self.test_session():
            a, b, c = _populate_variables()
            with pytest.raises(KeyboardInterrupt):
                with EarlyStopping([a, b]) as es:
                    self.assertTrue(es.update(1.))
                    set_variable_values([a, b], [10, 20])
                    raise KeyboardInterrupt()
            self.assertAlmostEqual(es.best_metric, 1.)
            self.assertEqual(get_variable_values([a, b, c]), [1, 2, 3])

    def test_bigger_is_better(self):
        with self.test_session():
            a, b, c = _populate_variables()
            with EarlyStopping([a, b], smaller_is_better=False) as es:
                set_variable_values([a], [10])
                self.assertTrue(es.update(.5))
                self.assertAlmostEqual(es.best_metric, .5)
                set_variable_values([a, b], [100, 20])
                self.assertTrue(es.update(1.))
                self.assertAlmostEqual(es.best_metric, 1.)
                set_variable_values([a, b, c], [1000, 200, 30])
                self.assertFalse(es.update(.8))
                self.assertAlmostEqual(es.best_metric, 1.)
            self.assertAlmostEqual(es.best_metric, 1.)
            self.assertEqual(get_variable_values([a, b, c]), [100, 20, 30])

    def test_cleanup_checkpoint_dir(self):
        with self.test_session():
            a, b, c = _populate_variables()
            with TemporaryDirectory() as tempdir:
                checkpoint_dir = os.path.join(tempdir, '1')
                with EarlyStopping([a, b], checkpoint_dir=checkpoint_dir) as es:
                    self.assertTrue(es.update(1.))
                    self.assertTrue(os.path.exists(
                        os.path.join(checkpoint_dir, 'checkpoint')))
                self.assertFalse(os.path.exists(checkpoint_dir))

    def test_not_cleanup_checkpoint_dir(self):
        with self.test_session():
            a, b, c = _populate_variables()
            with TemporaryDirectory() as tempdir:
                checkpoint_dir = os.path.join(tempdir, '2')
                with EarlyStopping([a, b], checkpoint_dir=checkpoint_dir,
                                   cleanup=False) as es:
                    self.assertTrue(es.update(1.))
                    self.assertTrue(os.path.exists(
                        os.path.join(checkpoint_dir, 'checkpoint')))
                self.assertTrue(os.path.exists(
                    os.path.join(checkpoint_dir, 'checkpoint')))
