import unittest

import pytest
import tensorflow as tf
from mock import Mock

from tfsnippet.examples.generative_adversarial_nets.gan_trainer import GANTrainer


class GANTrainerTestCase(tf.test.TestCase):

    def test_props(self):
        loop = Mock(max_epoch=1, max_step=1000)
        g_loss = Mock()
        g_train_op = Mock()
        d_loss = Mock()
        d_train_op = Mock()

        df = Mock()

        t = GANTrainer(
            loop, g_loss, g_train_op, 7, d_loss, d_train_op,
            [12, 34], df, feed_dict={'a': 1}, g_feed_dict={'a': 11},
            d_feed_dict={'a': 111}, g_metric_name='gen_cost',
            d_metric_name='d_cost'
        )
        self.assertIs(loop, t.loop)
        self.assertIs(g_loss, t.g_loss)
        self.assertIs(g_train_op, t.g_train_op)
        self.assertEquals(7, t.d_iters)
        self.assertIs(d_loss, t.d_loss)
        self.assertIs(d_train_op, t.d_train_op)
        self.assertEquals((12, 34), t.inputs)
        self.assertIs(df, t.data_flow)
        self.assertEquals({'a': 1}, t.feed_dict)
        self.assertEquals({'a': 11}, t.g_feed_dict)
        self.assertEquals({'a': 111}, t.d_feed_dict)
        self.assertEquals('gen_cost', t.g_metric_name)
        self.assertEquals('d_cost', t.d_metric_name)

        t = GANTrainer(loop, g_loss, g_train_op, 7, d_loss,
                       d_train_op, [12, 34], df)
        self.assertEquals('g_loss', t.g_metric_name)
        self.assertEquals('d_loss', t.d_metric_name)

        with pytest.raises(
                ValueError, match='`max_epoch` must be set to 1 for `loop`'):
            loop = Mock(max_epoch=2, max_step=1000)
            _ = GANTrainer(loop, g_loss, g_train_op, 7, d_loss,
                           d_train_op, [12, 34], df)
        with pytest.raises(
                ValueError, match='`max_step` must be configured for `loop`'):
            loop = Mock(max_epoch=1, max_step=None)
            _ = GANTrainer(loop, g_loss, g_train_op, 7, d_loss,
                           d_train_op, [12, 34], df)
        with pytest.raises(
                ValueError, match='`d_iters` must be at least 1'):
            loop = Mock(max_epoch=1, max_step=1000)
            _ = GANTrainer(loop, g_loss, g_train_op, 0, d_loss,
                           d_train_op, [12, 34], df)


if __name__ == '__main__':
    unittest.main()
