import numpy as np
import pytest
import tensorflow as tf
from mock import Mock

from tfsnippet.dataflows import DataFlow
from tfsnippet.scaffold import TrainLoop
from tfsnippet.trainer import *
from tfsnippet.utils import ensure_variables_initialized, TemporaryDirectory


class TrainerTestCase(tf.test.TestCase):

    def test_props(self):
        loop = Mock(max_epoch=1, max_step=None)
        loss = Mock()
        train_op = Mock()
        df = Mock()
        summary = tf.constant(1.)

        t = Trainer(loop, train_op, [12, 34], df, feed_dict={'a': 56},
                    metrics={'loss_x': loss}, summaries=[summary])
        self.assertIs(loop, t.loop)
        self.assertIs(train_op, t.train_op)
        self.assertEqual((12, 34), t.inputs)
        self.assertIs(df, t.data_flow)
        self.assertEqual({'a': 56}, t.feed_dict)
        self.assertDictEqual({'loss_x': loss}, t.metrics)
        self.assertListEqual([summary], t.summaries)

        t = Trainer(loop, train_op, [], Mock())
        self.assertDictEqual({}, t.metrics)

        t = Trainer(loop, train_op, [], Mock(), summaries=summary)
        self.assertListEqual([summary], t.summaries)

        with pytest.raises(
                ValueError, match='At least one of `max_epoch`, `max_step` '
                                  'should be configured for `loop`'):
            loop = Mock(max_epoch=None, max_step=None)
            _ = Trainer(loop, train_op, [], df)

    def test_run(self):
        ph = tf.placeholder(tf.int32, [5])
        var = tf.get_variable('var', shape=[5], dtype=tf.int32,
                              initializer=tf.zeros_initializer())
        summary = tf.summary.histogram(var.name, var)
        train_op = tf.assign(var, ph)
        df = DataFlow.arrays([np.arange(10, 15, dtype=np.int32)], batch_size=5)

        with TemporaryDirectory() as tmpdir:
            with self.test_session() as session, \
                    TrainLoop([var], max_epoch=1, early_stopping=False,
                              summary_dir=tmpdir) as loop:
                loop.collect_metrics = Mock(wraps=loop.collect_metrics)
                loop.add_summary = Mock(wraps=loop.add_summary)
                t = Trainer(loop, train_op, [ph], df,
                            metrics={'loss_x': tf.reduce_sum(ph)},
                            summaries=[summary])
                ensure_variables_initialized()
                t.run()

                self.assertEqual(
                    {'loss_x': 60},
                    loop.collect_metrics.call_args_list[0][0][0]
                )
                self.assertTrue(loop.add_summary.called)
                np.testing.assert_equal([10, 11, 12, 13, 14], session.run(var))

        # test to specify summaries, but without loop.summary_dir, so no
        # summary should run
        with self.test_session() as session, \
                TrainLoop([var], max_epoch=1, early_stopping=False) as loop:
            loop.collect_metrics = Mock(wraps=loop.collect_metrics)
            loop.add_summary = Mock(wraps=loop.add_summary)
            t = Trainer(loop, train_op, [ph], df,
                        metrics={'loss_x': tf.reduce_sum(ph)},
                        summaries=[summary])
            ensure_variables_initialized()
            t.run()

            self.assertEqual(
                {'loss_x': 60},
                loop.collect_metrics.call_args_list[0][0][0]
            )
            self.assertFalse(loop.add_summary.called)


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
        self.assertEqual((12, 34), t.inputs)
        self.assertIs(df, t.data_flow)
        self.assertEqual({'a': 56}, t.feed_dict)
        self.assertEqual('loss_x', t.metric_name)

        t = LossTrainer(loop, loss, train_op, [], Mock())
        self.assertEqual('loss', t.metric_name)

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

            self.assertEqual(
                {'loss_x': 60}, loop.collect_metrics.call_args_list[0][0][0])
            np.testing.assert_equal([10, 11, 12, 13, 14], session.run(var))
