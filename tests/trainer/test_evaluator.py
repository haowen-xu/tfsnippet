import unittest

import numpy as np
import pytest
import tensorflow as tf
from mock import Mock

from tfsnippet.dataflows import DataFlow
from tfsnippet.scaffold import TrainLoop
from tfsnippet.trainer import *
from tfsnippet.utils import EventSource


class AutoBatchWeightTestCase(unittest.TestCase):

    def test_auto_loss_weight(self):
        self.assertEqual(5., auto_batch_weight(np.arange(5)))
        self.assertEqual(7., auto_batch_weight(np.arange(7), np.arange(5)))
        self.assertEqual(1., auto_batch_weight(None))


class EvaluatorTestCase(tf.test.TestCase):

    def test_props(self):
        loop = Mock(valid_metric_name='valid_loss')
        df = Mock()

        v = Evaluator(loop, 12, [34, 56], df)
        self.assertIs(loop, v.loop)
        with self.test_session():
            self.assertEqual(12, v.metrics['valid_loss'].eval())
        self.assertEqual([34, 56], v.inputs)
        self.assertIs(v.data_flow, df)
        self.assertEqual({}, v.feed_dict)
        self.assertEqual('eval_time', v.time_metric_name)
        self.assertIs(auto_batch_weight, v.batch_weight_func)
        self.assertIsInstance(v.events, EventSource)

        batch_weight_func = Mock(return_value=123.)
        v = Evaluator(loop, {'valid_loss_x': 12}, [34, 56], df,
                      feed_dict={'a': 1},
                      time_metric_name='valid_time_x',
                      batch_weight_func=batch_weight_func)
        with self.test_session():
            self.assertEqual(12, v.metrics['valid_loss_x'].eval())
        self.assertEqual('valid_time_x', v.time_metric_name)
        self.assertIs(batch_weight_func, v.batch_weight_func)

    def test_error(self):
        with pytest.raises(ValueError, match='Metric is not a scalar tensor'):
            _ = Evaluator(Mock(), {'x': tf.constant([1, 2])}, [], Mock())

    def test_run(self):
        with self.test_session() as session:
            df = DataFlow.arrays([np.arange(6, dtype=np.float32)], batch_size=4)
            ph = tf.placeholder(tf.float32, shape=[None])
            ph2 = tf.placeholder(tf.float32, shape=[])
            ph3 = tf.placeholder(tf.float32, shape=[])

            # test default loss weight and merged feed dict
            with TrainLoop([], max_epoch=1) as loop:
                v = Evaluator(loop, tf.reduce_mean(ph), [ph], df,
                              feed_dict={ph2: 34})
                v._run_batch = Mock(wraps=v._run_batch)

                for epoch in loop.iter_epochs():
                    v.run({ph3: 56})
                    np.testing.assert_almost_equal(
                        2.5, loop._epoch_metrics._metrics['valid_loss'].mean)
                    np.testing.assert_almost_equal(
                        2.5, v.last_metrics_dict['valid_loss'])
                    self.assertIn('eval_time', loop._epoch_metrics._metrics)

                self.assertEqual(2, len(v._run_batch.call_args_list))
                for i, call_args in enumerate(v._run_batch.call_args_list):
                    call_session, call_feed_dict = call_args[0]
                    self.assertIs(session, call_session)
                    np.testing.assert_equal(
                        np.arange(6, dtype=np.float32)[i*4: (i+1)*4],
                        call_feed_dict[ph]
                    )
                    self.assertEqual(34, call_feed_dict[ph2])
                    self.assertEqual(56, call_feed_dict[ph3])

            # test None loss weight and None time metric and override feed dict
            with TrainLoop([], max_epoch=1) as loop:
                v = Evaluator(loop, {'valid_loss_x': tf.reduce_mean(ph)},
                              [ph], df,
                              feed_dict={ph2: 34},
                              batch_weight_func=None,
                              time_metric_name=None)
                v._run_batch = Mock(wraps=v._run_batch)

                for epoch in loop.iter_epochs():
                    v.run({ph2: 56})
                    np.testing.assert_almost_equal(
                        3.0, loop._epoch_metrics._metrics['valid_loss_x'].mean)
                    np.testing.assert_almost_equal(
                        3.0, v.last_metrics_dict['valid_loss_x'])
                    self.assertNotIn('eval_time', loop._epoch_metrics._metrics)

                for i, call_args in enumerate(v._run_batch.call_args_list):
                    call_session, call_feed_dict = call_args[0]
                    self.assertEqual(56, call_feed_dict[ph2])
                    self.assertNotIn(ph3, call_feed_dict)
