import unittest

import tensorflow as tf
from mock import Mock
from tfsnippet.trainer import *
from tfsnippet.utils import ensure_variables_initialized


class LossTrainerTestCase(tf.test.TestCase):

    def test_props(self):
        loop = Mock()
        loss = Mock()
        train_op = Mock()

        t = LossTrainer(loop, loss, train_op, [], Mock(), metric_name='loss_x')
        self.assertIs(loop, t.loop)
        self.assertIs(loss, t.loss)
        self.assertIs(train_op, t.train_op)
        self.assertEquals('loss_x', t.metric_name)

        t = LossTrainer(loop, loss, train_op, [], Mock())
        self.assertEquals('loss', t.metric_name)

    def test_fit_step(self):
        loop = Mock(collect_metrics=Mock(return_value=None))

        with self.test_session() as session:
            ph = tf.placeholder(tf.int32, ())
            var = tf.get_variable('var', shape=[], dtype=tf.int32,
                                  initializer=tf.zeros_initializer())
            t = LossTrainer(loop, ph, tf.assign(var, 456), [], Mock(),
                            metric_name='loss_x')

            ensure_variables_initialized()
            t._fit_step(session, {ph: 123})

            self.assertEquals(
                {'loss_x': 123}, loop.collect_metrics.call_args[0][0])
            self.assertEquals(456, session.run(var))


if __name__ == '__main__':
    unittest.main()
