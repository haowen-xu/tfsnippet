import numpy as np
import tensorflow as tf
from mock import Mock

from tfsnippet.variational import VariationalChain, VariationalInference


class VariationalChainTestCase(tf.test.TestCase):

    def prepare_model(self):
        variational_local_log_probs = Mock(
            return_value=[tf.constant(1.), tf.constant(2.)])
        variational = Mock(
            local_log_probs=Mock(
                wraps=lambda names: variational_local_log_probs(tuple(names))),
            __iter__=Mock(return_value=iter(['a', 'b'])),
        )
        model_local_log_probs = Mock(
            return_value=[tf.constant(3.), tf.constant(4.)])
        model = Mock(
            local_log_probs=Mock(
                wraps=lambda names: model_local_log_probs(tuple(names))),
            __iter__=Mock(return_value=iter(['c', 'd'])),
        )
        return (variational_local_log_probs, variational,
                model_local_log_probs, model)

    def test_default_args(self):
        (variational_local_log_probs, variational,
         model_local_log_probs, model) = self.prepare_model()

        chain = VariationalChain(variational, model)
        self.assertEqual(variational_local_log_probs.call_args,
                         ((('a', 'b'),),))
        self.assertEqual(model_local_log_probs.call_args,
                         ((('c', 'd'),),))

        self.assertIs(chain.variational, variational)
        self.assertIs(chain.model, model)
        self.assertEqual(chain.latent_names, ('a', 'b'))
        self.assertIsNone(chain.latent_axis)
        self.assertIsInstance(chain.vi, VariationalInference)

        with self.test_session() as sess:
            np.testing.assert_allclose(chain.log_joint.eval(), 7.)
            np.testing.assert_allclose(chain.vi.log_joint.eval(), 7.)
            np.testing.assert_allclose(sess.run(chain.vi.latent_log_probs),
                                       [1., 2.])

    def test_log_joint_arg(self):
        (variational_local_log_probs, variational,
         model_local_log_probs, model) = self.prepare_model()

        chain = VariationalChain(variational, model, log_joint=tf.constant(-1.))
        self.assertEqual(variational_local_log_probs.call_args,
                         ((('a', 'b'),),))
        self.assertFalse(model_local_log_probs.called)

        with self.test_session():
            np.testing.assert_allclose(chain.log_joint.eval(), -1.)
            np.testing.assert_allclose(chain.vi.log_joint.eval(), -1.)

    def test_latent_names_arg(self):
        (variational_local_log_probs, variational,
         model_local_log_probs, model) = self.prepare_model()

        chain = VariationalChain(variational, model, latent_names=iter(['a']))
        self.assertEqual(variational_local_log_probs.call_args,
                         ((('a',),),))
        self.assertEqual(model_local_log_probs.call_args,
                         ((('c', 'd'),),))
        self.assertEqual(chain.latent_names, ('a',))

    def test_latent_axis_arg(self):
        (variational_local_log_probs, variational,
         model_local_log_probs, model) = self.prepare_model()

        chain = VariationalChain(variational, model, latent_axis=1)
        self.assertEqual(chain.latent_axis, 1)
        self.assertEqual(chain.vi.axis, 1)
