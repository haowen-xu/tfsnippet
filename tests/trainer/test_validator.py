import unittest

import numpy as np
import tensorflow as tf
from mock import Mock

from tfsnippet.dataflow import DataFlow
from tfsnippet.scaffold import TrainLoop
from tfsnippet.trainer import *


class AutoLossWeightTestCase(unittest.TestCase):

    def test_auto_loss_weight(self):
        self.assertEquals(5., auto_loss_weight(np.arange(5)))
        self.assertEquals(7., auto_loss_weight(np.arange(7), np.arange(5)))
        self.assertEquals(1., auto_loss_weight(None))


class ValidatorTestCase(tf.test.TestCase):

    def test_props(self):
        loop = Mock(valid_metric_name='valid_loss')
        df = Mock()

        v = Validator(loop, 12, [34, 56], df)
        self.assertIs(loop, v.loop)
        self.assertEquals(12, v.loss)
        self.assertEquals([34, 56], v.inputs)
        self.assertIs(v.data_flow, df)
        self.assertEquals({}, v.feed_dict)
        self.assertEquals('valid_time', v.time_metric_name)
        self.assertEquals('valid_loss', v.loss_metric_name)
        self.assertIs(auto_loss_weight, v.loss_weight_func)

        loss_weight_func = Mock(return_value=123.)
        v = Validator(loop, 12, [34, 56], df,
                      feed_dict={'a': 1},
                      time_metric_name='valid_time_x',
                      loss_metric_name='valid_loss_x',
                      loss_weight_func=loss_weight_func)
        self.assertEquals('valid_time_x', v.time_metric_name)
        self.assertEquals('valid_loss_x', v.loss_metric_name)
        self.assertIs(loss_weight_func, v.loss_weight_func)

    def test_run(self):
        with self.test_session() as session:
            df = DataFlow.arrays([np.arange(6, dtype=np.float32)], batch_size=4)
            ph = tf.placeholder(tf.float32, shape=[None])
            ph2 = tf.placeholder(tf.float32, shape=[])
            ph3 = tf.placeholder(tf.float32, shape=[])

            # test default loss weight and merged feed dict
            with TrainLoop([], max_epoch=1) as loop:
                v = Validator(loop, tf.reduce_mean(ph), [ph], df,
                              feed_dict={ph2: 34})
                v._run_batch = Mock(wraps=v._run_batch)

                for epoch in loop.iter_epochs():
                    v.run({ph3: 56})
                    np.testing.assert_almost_equal(
                        2.5, loop._epoch_metrics._metrics['valid_loss'].mean)

                self.assertEquals(2, len(v._run_batch.call_args_list))
                for i, call_args in enumerate(v._run_batch.call_args_list):
                    call_session, call_feed_dict = call_args[0]
                    self.assertIs(session, call_session)
                    np.testing.assert_equal(
                        np.arange(6, dtype=np.float32)[i*4: (i+1)*4],
                        call_feed_dict[ph]
                    )
                    self.assertEquals(34, call_feed_dict[ph2])
                    self.assertEquals(56, call_feed_dict[ph3])

            # test None loss weight and override feed dict
            with TrainLoop([], max_epoch=1) as loop:
                v = Validator(loop, tf.reduce_mean(ph), [ph], df,
                              feed_dict={ph2: 34},
                              loss_weight_func=None,
                              loss_metric_name='valid_loss_x')
                v._run_batch = Mock(wraps=v._run_batch)

                for epoch in loop.iter_epochs():
                    v.run({ph2: 56})
                    np.testing.assert_almost_equal(
                        3.0, loop._epoch_metrics._metrics['valid_loss_x'].mean)

                for i, call_args in enumerate(v._run_batch.call_args_list):
                    call_session, call_feed_dict = call_args[0]
                    self.assertEquals(56, call_feed_dict[ph2])
                    self.assertNotIn(ph3, call_feed_dict)


if __name__ == '__main__':
    unittest.main()
