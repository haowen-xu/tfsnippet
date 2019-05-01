import mock
import pytest
import tensorflow as tf

from tfsnippet.layers.utils import validate_weight_norm_arg


class ValidateWeightNormArgTestCase(tf.test.TestCase):

    def test_validate_weight_norm_arg(self):
        # noinspection PyUnresolvedReferences
        import tfsnippet.layers.utils

        # test callable
        f = lambda t: t
        self.assertIs(validate_weight_norm_arg(f, -1, True), f)

        # test True: should generate a function that wraps true weight_norm
        with mock.patch('tfsnippet.layers.normalization.weight_norm') as m:
            f = validate_weight_norm_arg(True, -2, False)
            t = tf.reshape(tf.range(6, dtype=tf.float32), [1, 2, 3])
            _ = f(t)
            self.assertEqual(
                m.call_args, ((t,), {'axis': -2, 'use_scale': False}))

        # test False: should return None
        self.assertIsNone(validate_weight_norm_arg(False, -1, True))
        self.assertIsNone(validate_weight_norm_arg(None, -1, True))

        # test others: should raise error
        with pytest.raises(TypeError,
                           match='Invalid value for argument `weight_norm`: '
                                 'expected a bool or a callable function'):
            _ = validate_weight_norm_arg(123, -1, True)
