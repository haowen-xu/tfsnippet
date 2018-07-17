import unittest

import numpy as np
import pytest
import tensorflow as tf
from mock import Mock

from tfsnippet.dataflow import DataFlow
from tfsnippet.scaffold import TrainLoop
from tfsnippet.trainer import *
from tfsnippet.utils import ensure_variables_initialized


class TrainerTestCase(tf.test.TestCase):

    def test_props(self):
        loop = Mock(max_epoch=1, max_step=None)
        loss = Mock()
        train_op = Mock()
        df = Mock()

        t = Trainer(loop, train_op, [12, 34], df, feed_dict={'a': 56},
                    metrics={'loss_x': loss})
        self.assertIs(loop, t.loop)
        self.assertIs(train_op, t.train_op)
        self.assertEquals((12, 34), t.inputs)
        self.assertIs(df, t.data_flow)
        self.assertEquals({'a': 56}, t.feed_dict)
        self.assertDictEqual({'loss_x': loss}, t.metrics)

        t = Trainer(loop, train_op, [], Mock())
        self.assertDictEqual({}, t.metrics)

        with pytest.raises(
                ValueError, match='At least one of `max_epoch`, `max_step` '
                                  'should be configured for `loop`'):
            loop = Mock(max_epoch=None, max_step=None)
            _ = Trainer(loop, train_op, [], df)

    def test_run(self):
        ph = tf.placeholder(tf.int32, [5])
        var = tf.get_variable('var', shape=[5], dtype=tf.int32,
                              initializer=tf.zeros_initializer())
        train_op = tf.assign(var, ph)
        df = DataFlow.arrays([np.arange(10, 15, dtype=np.int32)], batch_size=5)
        with self.test_session() as session, \
                TrainLoop([var], max_epoch=1, early_stopping=False) as loop:
            loop.collect_metrics = Mock(wraps=loop.collect_metrics)
            t = Trainer(loop, train_op, [ph], df,
                        metrics={'loss_x': tf.reduce_sum(ph)})
            ensure_variables_initialized()
            t.run()

            self.assertEquals(
                {'loss_x': 60}, loop.collect_metrics.call_args_list[0][0][0])
            np.testing.assert_equal([10, 11, 12, 13, 14], session.run(var))


class LossTrainerTestCase(tf.test.TestCase):

    def test_props(self):
        loop = Mock(max_epoch=1, max_step=None)
        loss = Mock()
        train_op = Mock()
        df = Mock()

        t = LossTrainer(loop, loss, train_op, [12, 34], df, feed_dict={'a': 56},
                        metric_name='loss_x')
        self.assertIs(loop, t.loop)
        self.assertIs(loss, t.loss)
        self.assertIs(train_op, t.train_op)
        self.assertEquals((12, 34), t.inputs)
        self.assertIs(df, t.data_flow)
        self.assertEquals({'a': 56}, t.feed_dict)
        self.assertEquals('loss_x', t.metric_name)

        t = LossTrainer(loop, loss, train_op, [], Mock())
        self.assertEquals('loss', t.metric_name)

    def test_run(self):
        ph = tf.placeholder(tf.int32, [5])
        var = tf.get_variable('var', shape=[5], dtype=tf.int32,
                              initializer=tf.zeros_initializer())
        train_op = tf.assign(var, ph)
        df = DataFlow.arrays([np.arange(10, 15, dtype=np.int32)], batch_size=5)
        with self.test_session() as session, \
                TrainLoop([var], max_epoch=1, early_stopping=False) as loop:
            loop.collect_metrics = Mock(wraps=loop.collect_metrics)
            t = LossTrainer(loop, tf.reduce_sum(ph), train_op, [ph], df,
                            metric_name='loss_x')
            ensure_variables_initialized()
            t.run()

            self.assertEquals(
                {'loss_x': 60}, loop.collect_metrics.call_args_list[0][0][0])
            np.testing.assert_equal([10, 11, 12, 13, 14], session.run(var))


if __name__ == '__main__':
    unittest.main()
