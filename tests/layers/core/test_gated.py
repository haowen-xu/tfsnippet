import copy

import numpy as np
import pytest
import tensorflow as tf

from tfsnippet.layers import as_gated


def safe_sigmoid(x):
    return np.where(x < 0, np.exp(x) / (1. + np.exp(x)), 1. / (1. + np.exp(-x)))


class AsGatedHelper(object):
    def __init__(self, main_ret, gate_ret):
        self.main_args = None
        self.gate_args = None
        self.main_ret = main_ret
        self.gate_ret = gate_ret

    def __call__(self, *args, **kwargs):
        scope = kwargs['scope']
        if scope == 'main':
            assert(self.main_args is None)
            self.main_args = (args, copy.copy(kwargs))
            return self.main_ret
        elif scope == 'gate':
            assert(self.gate_args is None)
            self.gate_args = (args, copy.copy(kwargs))
            return self.gate_ret
        else:
            raise RuntimeError()


class TestAsGated(tf.test.TestCase):

    def test_as_gated(self):
        main_ret = np.random.normal(size=[2, 3, 4]).astype(np.float32)
        gate_ret = np.random.normal(size=[2, 3, 4]).astype(np.float32)
        activation_fn = object()

        # default_name infer failed
        with pytest.raises(ValueError,
                           match='`default_name` cannot be inferred'):
            g = as_gated(AsGatedHelper(main_ret, gate_ret))

        with self.test_session() as sess:
            # test infer default name
            f = AsGatedHelper(main_ret, gate_ret)
            f.__name__ = 'f'
            g = as_gated(f)
            g_ret = g(1, xyz=2, activation_fn=activation_fn)
            np.testing.assert_allclose(
                sess.run(g_ret), main_ret * safe_sigmoid(gate_ret + 2.))

            self.assertTrue(g_ret.name, 'gated_f/')
            self.assertEqual(
                f.main_args,
                (
                    (1,),
                    {'xyz': 2, 'activation_fn': activation_fn, 'scope': 'main'}
                )
            )
            self.assertEqual(
                f.gate_args,
                (
                    (1,),
                    {'xyz': 2, 'scope': 'gate'}
                )
            )

            # test specify default name
            f = AsGatedHelper(main_ret, gate_ret)
            g = as_gated(f, sigmoid_bias=1., default_name='ff')
            g_ret = g(1, xyz=2, activation_fn=activation_fn)
            np.testing.assert_allclose(
                sess.run(g_ret), main_ret * safe_sigmoid(gate_ret + 1.))

            self.assertTrue(g_ret.name, 'gated_ff/')
            self.assertEqual(
                f.main_args,
                (
                    (1,),
                    {'xyz': 2, 'activation_fn': activation_fn, 'scope': 'main'}
                )
            )
            self.assertEqual(
                f.gate_args,
                (
                    (1,),
                    {'xyz': 2, 'scope': 'gate'}
                )
            )

            # test using `name`
            f = AsGatedHelper(main_ret, gate_ret)
            g = as_gated(f, default_name='f')
            g_ret = g(1, xyz=2, activation_fn=activation_fn, name='name')
            np.testing.assert_allclose(
                sess.run(g_ret), main_ret * safe_sigmoid(gate_ret + 2.))
            self.assertTrue(g_ret.name, 'name/')

            # test using `scope`
            f = AsGatedHelper(main_ret, gate_ret)
            g = as_gated(f, default_name='f')
            g_ret = g(1, xyz=2, activation_fn=activation_fn, scope='scope')
            np.testing.assert_allclose(
                sess.run(g_ret), main_ret * safe_sigmoid(gate_ret + 2.))
            self.assertTrue(g_ret.name, 'scope/')
